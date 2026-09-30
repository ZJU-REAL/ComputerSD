"""Post-rollout logging hook that surfaces GiGPO internals to wandb/tensorboard.

Wired in via slime ``--custom-rollout-log-function-path reward.gigpo_metrics.log_gigpo_rollout``.
slime calls it once per rollout with ``(rollout_id, args, samples, rollout_extra_metrics, rollout_time)``;
returning False keeps slime's own reward/perf logging intact, and we additionally
log GiGPO-specific metrics through ``logging_utils.log`` (the same wandb channel).

GiGPO writes a per-sample breakdown onto ``metadata["gigpo"]["gigpo_diag"]`` in
reward/gigpo.py during reward computation (which runs in rollout workers). By
the time this hook runs, all flattened training samples carry that breakdown, so
we can recover the full group structure here without any cross-process state.

Logged metrics (all keyed by ``rollout/step`` like slime's defaults):
  gigpo/avg_step_group_size  — mean # samples sharing an anchor state (==1 means
                               no state-relative comparison fired). Want >1.
  gigpo/num_step_groups      — distinct anchor sub-groups in this rollout.
  gigpo/num_episodes         — distinct prompt groups (group_index values).
  gigpo/empty_anchor_ratio   — fraction of samples whose a11y anchor was empty.
                               Hybrid mode keeps these samples singleton. Want ~0.
  gigpo/anchor_singleton_ratio — fraction of samples without an anchor peer.
  gigpo/singleton_zeroed_step_ratio — fraction whose step advantage is zeroed
                                       because its state group has one sample.
  gigpo/anchor_exact_grouped_ratio — samples grouped by canonical hash.
  gigpo/anchor_fuzzy_grouped_ratio — samples admitted by hybrid fuzzy fallback.
  gigpo/trajectory_reward_mean/std — raw task outcome attached to each step.
  gigpo/trajectory_advantage_mean/std — prompt-level GRPO outcome advantage.
  gigpo/trajectory_advantage_scaled_mean/std — trajectory advantage after
                                                dynamic-history T scaling.
  gigpo/discounted_step_return_mean/std — analyzer-only discounted return.
  gigpo/step_advantage_mean/std — discounted analyzer return normalized within
                                  its anchor micro-group.
  gigpo/weighted_step_advantage_mean/std — step advantage after its w weight.
  gigpo/advantage_mean/std       — value written to sample.reward["score"].
  gigpo/prm_step_mean        — mean per-step PRM score (step reward r_t, when PRM
                               populated step_wise.step_scores; 0 if PRM off).
  gigpo/prm_step_sample_count — step samples with an explicit PRM status.
  gigpo/prm_step_failure_count — step samples whose PRM status is not ``ok``
                                 (including timeout, exception, invalid output,
                                 or JSON parse failure).
  gigpo/prm_step_failure_ratio — ``prm_step_failure_count`` divided by
                                 ``prm_step_sample_count``.
  gigpo/prm_max_concurrency — configured per-worker PRM semaphore limit.
  gigpo/task_acc              — fraction of samples from successful episodes
                               (outcome_reward == +1). The headline learning signal.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

logger = logging.getLogger("gui.gigpo_metrics")


def _log_dynamic_batch_summary(rollout_id: int, samples: list) -> None:
    """Log the pre-clean fan-out shape once per rollout batch."""
    group_children: Counter = Counter()
    trajectory_children: Counter = Counter()
    trajectory_keys: set[tuple[Any, Any]] = set()
    non_child_samples = 0
    removed_samples = 0
    aborted_samples = 0

    for sample in samples:
        group_index = getattr(sample, "group_index", None)
        sample_index = getattr(sample, "index", None)
        trajectory_key = (group_index, sample_index)
        trajectory_keys.add(trajectory_key)
        trajectory_children.setdefault(trajectory_key, 0)

        metadata = getattr(sample, "metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("dynamic_step_index") is not None:
            group_children[group_index] += 1
            trajectory_children[trajectory_key] += 1
        else:
            non_child_samples += 1
        if bool(getattr(sample, "remove_sample", False)):
            removed_samples += 1
        status = getattr(getattr(sample, "status", None), "value", getattr(sample, "status", None))
        if status == "aborted":
            aborted_samples += 1

    counts = list(trajectory_children.values())
    group_keys = {key[0] for key in trajectory_keys}
    count_range = (
        f"{min(counts)}/{sum(counts) / len(counts):.1f}/{max(counts)}"
        if counts
        else "0/0.0/0"
    )
    group_text = "{" + ", ".join(
        f"{group}:{group_children[group]}" for group in sorted(group_keys, key=str)
    ) + "}"
    logger.info(
        "[dynamic-batch] rollout=%s samples=%d groups=%d trajectories=%d "
        "group_children=%s trajectory_children[min/mean/max]=%s "
        "non_child=%d removed=%d aborted=%d",
        rollout_id,
        len(samples),
        len(group_keys),
        len(trajectory_keys),
        group_text,
        count_range,
        non_child_samples,
        removed_samples,
        aborted_samples,
    )


def _compute_gigpo_metrics(samples: list, args: Any | None = None) -> dict[str, float]:
    n = len(samples)
    if n == 0:
        return {}

    trajectory_rewards: list[float] = []
    trajectory_advs: list[float] = []
    scaled_trajectory_advs: list[float] = []
    discounted_step_returns: list[float] = []
    step_advs: list[float] = []
    weighted_step_advs: list[float] = []
    advs: list[float] = []
    prm_steps: list[float] = []
    task_accs: list[float] = []
    prm_failure_count = 0
    prm_request_count = 0
    prm_step_sample_count = 0
    prm_step_failure_count = 0
    opd_requested = 0
    opd_attached = 0
    opd_guidance_chars = 0
    opd_topk_values: list[int] = []
    opd_gate_means: list[float] = []
    opd_logprob_shifts: list[float] = []
    opd_polarity_metrics: dict[str, dict[str, list[float]]] = {
        "positive": {
            "raw_gate_mean": [],
            "raw_gate_active_ratio": [],
            "effective_gate_mean": [],
            "effective_gate_active_ratio": [],
            "logprob_shift_mean": [],
        },
        "corrective": {
            "raw_gate_mean": [],
            "raw_gate_active_ratio": [],
            "effective_gate_mean": [],
            "effective_gate_active_ratio": [],
            "logprob_shift_mean": [],
        },
    }
    empty_anchor = 0
    singleton_anchor = 0
    singleton_zeroed = 0
    exact_anchor = 0
    fuzzy_anchor = 0
    episodes: set = set()
    step_groups: set = set()
    prm_trajectories: set[tuple[Any, Any]] = set()
    analyzer_details: dict[tuple[Any, Any, int], dict[str, Any]] = {}
    trajectory_latencies: dict[tuple[Any, Any], float] = {}
    has_gigpo = False  # False under GRPO (no gigpo_diag) -> only log task_acc/prm.

    for s in samples:
        meta = getattr(s, "metadata", None) or {}
        gm = meta.get("gigpo") if isinstance(meta, dict) else None
        diag = gm.get("gigpo_diag") if isinstance(gm, dict) else None
        sw = meta.get("step_wise") if isinstance(meta, dict) else None
        prm = meta.get("prm") if isinstance(meta, dict) else None
        opd = meta.get("opd") if isinstance(meta, dict) else None

        if isinstance(diag, dict):
            has_gigpo = True
            trajectory_rewards.append(float(diag.get("trajectory_reward", 0.0)))
            trajectory_advs.append(float(diag.get("trajectory_advantage", 0.0)))
            scaled_trajectory_advs.append(float(diag.get("trajectory_advantage_scaled", 0.0)))
            discounted_step_returns.append(float(diag.get("discounted_step_return", 0.0)))
            step_advs.append(float(diag.get("step_advantage", 0.0)))
            weighted_step_advs.append(float(diag.get("weighted_step_advantage", 0.0)))
            advs.append(float(diag.get("advantage", 0.0)))
            if diag.get("anchor_source") == "missing" or not diag.get("anchor_key"):
                empty_anchor += 1
            if int(diag.get("step_group_size", 0) or 0) == 1:
                singleton_anchor += 1
            match_kind = diag.get("anchor_match_kind")
            if match_kind == "exact":
                exact_anchor += 1
            elif match_kind == "fuzzy":
                fuzzy_anchor += 1
            if bool(diag.get("singleton_step_advantage_zeroed", False)):
                singleton_zeroed += 1
            if diag.get("step_group_uid"):
                step_groups.add(diag["step_group_uid"])
            if diag.get("group_index") is not None:
                episodes.add(diag["group_index"])

        if isinstance(sw, dict):
            prm_status = sw.get("prm_status")
            if prm_status is not None:
                prm_step_sample_count += 1
                if str(prm_status).lower() != "ok":
                    prm_step_failure_count += 1
            raw = sw.get("step_scores", [])
            if isinstance(raw, list) and raw:
                try:
                    prm_steps.append(float(raw[0]))
                except (TypeError, ValueError):
                    pass
            outcome = sw.get("outcome_reward")
            if outcome is not None:
                try:
                    task_accs.append(1.0 if float(outcome) >= 0.5 else 0.0)
                except (TypeError, ValueError):
                    pass
        if isinstance(prm, dict):
            trajectory_key = (
                getattr(s, "group_index", None),
                getattr(s, "index", None),
            )
            if trajectory_key not in prm_trajectories:
                prm_trajectories.add(trajectory_key)
                prm_failure_count += int(prm.get("failure_count", 0) or 0)
                prm_request_count += int(prm.get("request_count", 0) or 0)
            for detail in prm.get("step_details", []):
                if isinstance(detail, dict) and detail.get("step_index") is not None:
                    analyzer_details[(trajectory_key[0], trajectory_key[1], int(detail["step_index"]))] = detail
        latency = meta.get("trajectory_e2e_latency_s") if isinstance(meta, dict) else None
        if latency is not None:
            try:
                value = float(latency)
                if value == value and value not in (float("inf"), float("-inf")):
                    trajectory_latencies.setdefault((getattr(s, "group_index", None), getattr(s, "index", None)), value)
            except (TypeError, ValueError):
                pass
        if isinstance(opd, dict) and opd.get("status") in {"ok", "dropped", "masked"}:
            opd_requested += 1
            if opd.get("status") == "ok":
                opd_attached += 1
                opd_guidance_chars += len(str(opd.get("guidance", "")))
                opd_topk_values.append(int(opd.get("topk", 0) or 0))
                opd_gate_means.append(float(opd.get("gate_mean", 0.0) or 0.0))
                opd_logprob_shifts.append(float(opd.get("logprob_shift_mean", 0.0) or 0.0))
                polarity_metrics = opd_polarity_metrics.get(str(opd.get("step_polarity", "")))
                if polarity_metrics is not None:
                    for key in polarity_metrics:
                        polarity_metrics[key].append(float(opd.get(key, 0.0) or 0.0))

    out: dict[str, float] = {"gigpo/num_samples": float(n)}
    if has_gigpo:
        out["gigpo/num_step_groups"] = float(len(step_groups))
        out["gigpo/num_episodes"] = float(len(episodes))
        out["gigpo/empty_anchor_ratio"] = empty_anchor / n
        out["gigpo/anchor_singleton_ratio"] = singleton_anchor / n
        out["gigpo/singleton_zeroed_step_ratio"] = singleton_zeroed / n
        out["gigpo/anchor_exact_grouped_ratio"] = exact_anchor / n
        out["gigpo/anchor_fuzzy_grouped_ratio"] = fuzzy_anchor / n
        out["gigpo/avg_step_group_size"] = (n / len(step_groups)) if step_groups else 0.0

        def _stats(key: str, vals: list[float]) -> None:
            if not vals:
                out[f"{key}_mean"] = 0.0
                out[f"{key}_std"] = 0.0
                return
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = var ** 0.5

        _stats("gigpo/trajectory_reward", trajectory_rewards)
        _stats("gigpo/trajectory_advantage", trajectory_advs)
        _stats("gigpo/trajectory_advantage_scaled", scaled_trajectory_advs)
        _stats("gigpo/discounted_step_return", discounted_step_returns)
        _stats("gigpo/step_advantage", step_advs)
        _stats("gigpo/weighted_step_advantage", weighted_step_advs)
        _stats("gigpo/advantage", advs)
    if prm_steps:
        out["gigpo/prm_step_mean"] = sum(prm_steps) / len(prm_steps)
    if task_accs:
        out["gigpo/task_acc"] = sum(task_accs) / len(task_accs)
    if prm_request_count:
        out["gigpo/prm_failure_ratio"] = prm_failure_count / prm_request_count
    if has_gigpo:
        out["gigpo/prm_step_sample_count"] = float(prm_step_sample_count)
        out["gigpo/prm_step_failure_count"] = float(prm_step_failure_count)
        out["gigpo/prm_step_failure_ratio"] = (
            prm_step_failure_count / prm_step_sample_count
            if prm_step_sample_count
            else 0.0
        )
    if opd_requested:
        out["opd/target_ready_ratio"] = opd_attached / opd_requested
        out["opd/masked_steps"] = float(opd_requested - opd_attached)
        if opd_attached:
            out["opd/guidance_chars_mean"] = opd_guidance_chars / opd_attached
            out["opd/gate_mean"] = sum(opd_gate_means) / len(opd_gate_means)
            for polarity, values_by_key in opd_polarity_metrics.items():
                count = len(values_by_key["effective_gate_mean"])
                for key, values in values_by_key.items():
                    if values:
                        out[f"opd/{polarity}_{key}"] = sum(values) / len(values)
    if analyzer_details:
        details = list(analyzer_details.values())
        labels = [
            (detail.get("consistency_majority"), detail.get("effectiveness_majority"))
            for detail in details
            if detail.get("consistency_majority") in (0, 1)
            and detail.get("effectiveness_majority") in (0, 1)
        ]
        for consistency, effectiveness in ((1, 1), (1, 0), (0, 1), (0, 0)):
            out[f"opd/joint_label_{consistency}{effectiveness}_ratio"] = (
                sum(label == (consistency, effectiveness) for label in labels) / len(labels)
                if labels else 0.0
            )
        total = len(details)
        for reason in ("format_error", "http_failure", "empty_guidance"):
            out[f"opd/{reason}_ratio"] = sum(
                detail.get("failure_reason") == reason for detail in details
            ) / total

        shift_sum = directed_shift_sum = 0.0
        shift_count = 0
        for sample in samples:
            if not getattr(sample, "opd_q0_valid", False):
                continue
            q0 = getattr(sample, "opd_q0_log_probs", None)
            qplus = getattr(sample, "teacher_log_probs", None)
            mask = getattr(sample, "loss_mask", None) or []
            if not q0 or not qplus or len(q0) != len(qplus) or len(mask) != len(q0):
                continue
            direction = int(getattr(sample, "opd_step_polarity", 0) or 0)
            for old, new, active in zip(q0, qplus, mask, strict=True):
                if int(active) == 0:
                    continue
                shift_sum += float(new) - float(old)
                directed_shift_sum += direction * (float(new) - float(old))
                shift_count += 1
        out["opd/guidance_shift_mean"] = shift_sum / shift_count if shift_count else 0.0
        out["opd/directed_guidance_shift_mean"] = directed_shift_sum / shift_count if shift_count else 0.0
    if trajectory_latencies:
        out["rollout/perf/trajectory_e2e_latency_mean"] = sum(trajectory_latencies.values()) / len(trajectory_latencies)
    if getattr(args, "gui_opd_enable", False):
        for name in (
            "opd/joint_label_11_ratio", "opd/joint_label_10_ratio",
            "opd/joint_label_01_ratio", "opd/joint_label_00_ratio",
            "opd/format_error_ratio", "opd/http_failure_ratio",
            "opd/empty_guidance_ratio", "opd/guidance_shift_mean",
            "opd/directed_guidance_shift_mean",
        ):
            out.setdefault(name, 0.0)
    return out


def log_gigpo_rollout(rollout_id: int, args: Any, samples: list,
                      rollout_extra_metrics: dict | None, rollout_time: float) -> bool:
    """Custom rollout-log hook: log GiGPO diagnostics, then let slime log its
    own reward/perf metrics too (return False).
    """
    samples = samples or []
    if getattr(args, "dynamic_history", False):
        _log_dynamic_batch_summary(rollout_id, samples)
    metrics = _compute_gigpo_metrics(samples, args)
    if rollout_extra_metrics:
        metrics.update(rollout_extra_metrics)
    metrics.setdefault("rollout/perf/trajectory_e2e_latency_mean", 0.0)
    metrics.setdefault("rollout/perf/trajectories_per_hour", 0.0)
    if not metrics:
        return False
    if getattr(args, "prm_enable", False):
        metrics["gigpo/prm_max_concurrency"] = float(
            getattr(args, "prm_max_concurrency", 8) or 8
        )
        metrics["gigpo/prm_m"] = float(getattr(args, "prm_m", 1) or 1)
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logger.info("gigpo metrics %s: %s", rollout_id, metrics)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return False
