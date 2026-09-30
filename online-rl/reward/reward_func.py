"""Reward computation for GUI rollouts (slime ``--custom-rm-path`` entry).

``reward_func`` is the entry slime calls after a rollout to score samples. It
composes the task outcome reward (success -> +1, else -1) with optional PRM
(process reward model) per-step scores. ``mark_aborted_samples`` keeps the
training pipeline stable for broken trajectories.

When ``args.advantage_estimator == "gigpo"`` the entry instead routes the whole
prompt group through :mod:`reward.gigpo`, which computes the GiGPO
episode + step-level advantage per sample (grouping steps by normalized
accessibility-tree anchor) and returns it as the scalar ``score``. Pair this
path with slime ``--disable-rewards-normalization`` so the composed advantage is
not re-normalized before being broadcast to response tokens.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger("gui.reward")


def mark_aborted_samples(samples: list[Sample]) -> None:
    """Give ABORTED samples a default reward and exclude them from training.

    The rollout pipeline skips reward_func for lists containing any ABORTED
    sample, which leaves reward=None and crashes downstream metrics. Setting a
    default reward + remove_sample=True keeps the pipeline stable while ensuring
    these broken trajectories never contribute to the gradient.
    """
    for s in samples:
        if s.status == Sample.Status.ABORTED:
            if s.reward is None:
                s.reward = {"score": 0.0, "acc": 0.0}
            s.remove_sample = True


def single_reward(sample: Sample) -> dict[str, float]:
    """Outcome reward for one sample: ``score`` for GRPO, ``acc`` for logging."""
    outcome_score = 0.0
    raw_acc = 0.0
    if isinstance(sample.reward, dict):
        outcome_score = float(sample.reward.get("score", 0.0))
        raw_acc = float(sample.reward.get("acc", 0.0))
    elif isinstance(sample.metadata, dict):
        raw_acc = float(sample.metadata.get("gui_score", 0.0))
        outcome_score = 1.0 if raw_acc == 1.0 else -1.0
    return {"score": outcome_score, "acc": raw_acc}


def _compose_with_prm(args: Any, s: Sample) -> dict[str, float]:
    """Combine outcome reward with PRM step scores (no-op when PRM is off)."""
    result = single_reward(s)
    if not getattr(args, "prm_enable", False):
        return result

    prm_metadata = s.metadata.get("prm", {}) if isinstance(s.metadata, dict) else {}
    if not isinstance(prm_metadata, dict):
        prm_metadata = {}

    prm_step_mean = float(prm_metadata.get("step_mean_score", 0.0))
    outcome_reward = float(result.get("score", 0.0))
    # GUI step-level OPD consumes analyzer text as privileged teacher context;
    # its consistency/effectiveness scalar must not replace the requested
    # outcome-only trajectory GRPO reward.  GiGPO keeps its own explicit step
    # advantage path and is unaffected by this guard.
    if getattr(args, "gui_opd_enable", False) and not _is_gigpo(args):
        step_coef = 0.0
    else:
        step_coef = float(getattr(args, "prm_step_coef", 1.0))
    final_score = outcome_reward + step_coef * prm_step_mean
    result["base_score"] = outcome_reward
    result["prm_step_score"] = prm_step_mean
    result["score"] = final_score

    # Expose one concrete PRM raw output for quick sanity-checking (like retool).
    prm_example_eval = ""
    step_details = prm_metadata.get("step_details", [])
    if isinstance(step_details, list) and step_details:
        first_step = step_details[0] if isinstance(step_details[0], dict) else {}
        votes = first_step.get("votes", []) if isinstance(first_step, dict) else []
        if isinstance(votes, list) and votes:
            first_vote = votes[0] if isinstance(votes[0], dict) else {}
            raw_text = first_vote.get("raw_text", "") if isinstance(first_vote, dict) else ""
            if isinstance(raw_text, str):
                prm_example_eval = raw_text
    result["prm_example_eval"] = prm_example_eval

    # Populate step_wise composed rewards for step_wise advantage.
    if isinstance(s.metadata, dict):
        step_wise_meta = s.metadata.get("step_wise", {})
        if not isinstance(step_wise_meta, dict):
            step_wise_meta = {}
        step_wise_meta["outcome_reward"] = outcome_reward
        raw_step_scores = step_wise_meta.get("step_scores", [])
        if isinstance(raw_step_scores, list):
            step_wise_meta["step_scores_with_outcome"] = [
                float(step_score) + outcome_reward for step_score in raw_step_scores
            ]
        else:
            step_wise_meta["step_scores_with_outcome"] = []
        s.metadata["step_wise"] = step_wise_meta
    return result


def _gigpo_hparams(args: Any) -> dict[str, Any]:
    """Resolve GiGPO hyperparameters, preferring slime args then env vars."""
    import os

    def _f(name: str, default: str) -> float:
        val = getattr(args, name, None)
        if val is None:
            val = os.environ.get(name.upper())
        return float(val) if val not in (None, "") else float(default)

    def _s(name: str, default: str) -> str:
        val = getattr(args, name, None)
        if val is None:
            val = os.environ.get(name.upper())
        return str(val) if val not in (None, "") else default

    hparams = {
        "step_advantage_w": _f("gigpo_step_advantage_w", "1.0"),
        "gamma": _f("gigpo_gamma", "0.95"),
        "mode": _s("gigpo_mode", "mean_norm"),
        "anchor_mode": _s("gigpo_anchor_mode", "hybrid"),
        "anchor_node_difference_conflict_threshold": int(
            _f("gigpo_anchor_node_difference_conflict_threshold", "3")
        ),
        "platform": _s("gigpo_platform", "ubuntu"),
        "trajectory_advantage_scaling": _s("dynamic_trajectory_advantage_scaling", "linear"),
    }
    if hparams["step_advantage_w"] < 0:
        raise ValueError("gigpo_step_advantage_w must be non-negative")
    if not 0 <= hparams["gamma"] <= 1:
        raise ValueError("gigpo_gamma must be between 0 and 1")
    return hparams


def _is_gigpo(args: Any) -> bool:
    """True iff the configured advantage estimator is GiGPO.

    slime stores the user's ``--advantage-estimator`` choice on
    ``args.advantage_estimator``; ``adv_estimator`` is accepted as an alias.
    """
    val = getattr(args, "advantage_estimator", None) or getattr(args, "adv_estimator", None)
    return str(val).lower() == "gigpo"


async def _compose_with_gigpo(args: Any, samples: list[Sample]) -> list[dict[str, float]]:
    """Score a prompt group with state-relative composite step rewards.

    The group looks like one prompt's N rollouts, each already expanded to
    per-step training samples (dynamic-history). For each sample we read its
    step PRM score (``metadata["step_wise"]["step_scores"]``) and the trajectory
    outcome (``metadata["step_wise"]["outcome_reward"]`` / ``dynamic_outcome_reward``,
    else the legacy ±1 outcome). :mod:`reward.gigpo` discounts only the analyzer
    step scores, computes a trajectory GRPO advantage from the outcome, and
    computes a separate state-relative advantage from analyzer returns. The two
    advantages are summed after the trajectory term is distributed across its
    dynamic-history steps. Pair with slime ``--disable-rewards-normalization``
    so this score is not re-normalized.
    """
    if not samples:
        return []
    group_ids = {getattr(s, "group_index", None) for s in samples}
    trajectory_ids = {
        (getattr(s, "group_index", None), getattr(s, "index", None)) for s in samples
    }
    if len(group_ids) != 1:
        raise RuntimeError(f"GiGPO reward expected one prompt group, got group_index values {group_ids}")
    if int(getattr(args, "n_samples_per_prompt", 2) or 2) > 1 and len(trajectory_ids) < 2:
        raise RuntimeError(
            "GiGPO reward received fewer than two distinct trajectories. "
            "Enable --group-rm so reward runs after the full prompt group is generated."
        )

    try:
        from reward.gigpo import compute_gigpo_advantages
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError("GiGPO reward implementation could not be imported") from e

    hp = _gigpo_hparams(args)

    # Ensure every sample carries an outcome_reward in step_wise metadata so the
    # trajectory GRPO branch has a consistent input on dynamic-history samples.
    base_scores: list[dict[str, float]] = []
    for s in samples:
        composed = _compose_with_prm(args, s)
        # Keep the raw task outcome separate from PRM aggregation. gigpo.py
        # normalizes it once per trajectory, independently of state-group PRM
        # advantages.
        outcome = float(composed.get("base_score", composed.get("score", 0.0)))
        base = dict(composed)
        base["score"] = outcome
        base_scores.append(base)
        if isinstance(s.metadata, dict):
            sw = s.metadata.get("step_wise")
            if not isinstance(sw, dict):
                sw = {}
            sw["outcome_reward"] = outcome
            s.metadata["step_wise"] = sw

    advantages, diag = compute_gigpo_advantages(
        samples,
        step_advantage_w=hp["step_advantage_w"],
        gamma=hp["gamma"],
        mode=hp["mode"],
        anchor_mode=hp["anchor_mode"],
        anchor_node_difference_conflict_threshold=hp[
            "anchor_node_difference_conflict_threshold"
        ],
        platform=hp["platform"],
        trajectory_std_normalization=bool(getattr(args, "grpo_std_normalization", True)),
    )

    results: list[dict[str, float]] = []
    for s, base, adv in zip(samples, base_scores, advantages, strict=False):
        acc = float(base.get("acc", 0.0))
        result = {
            "score": float(adv),
            "acc": acc,
            "base_score": float(base.get("score", 0.0)),
            "gigpo_score": float(adv),
        }
        # Record the GiGPO breakdown for diagnostics / logging.
        if isinstance(s.metadata, dict):
            sw = s.metadata.get("step_wise")
            if not isinstance(sw, dict):
                sw = {}
            sw["gigpo_advantage"] = float(adv)
            s.metadata["step_wise"] = sw
        results.append(result)

    logger.info(
        "gigpo reward: samples=%d episodes=%d step_groups=%d avg_step_group=%.2f "
        "mode=%s anchor_mode=%s w=%s gamma=%s traj_scale=%s "
        "empty_anchor=%d singleton=%d fuzzy_merges=%d",
        diag.get("num_samples", 0),
        diag.get("num_episodes", 0),
        diag.get("num_step_groups", 0),
        diag.get("avg_step_group_size", 0.0),
        hp["mode"], hp["anchor_mode"], hp["step_advantage_w"], hp["gamma"],
        hp["trajectory_advantage_scaling"],
        diag.get("empty_anchor_count", 0),
        diag.get("singleton_count", 0),
        diag.get("fuzzy_bucket_merges", 0),
    )
    return results


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    """slime reward entry: score one sample or a list of samples."""
    if _is_gigpo(args):
        group = sample if isinstance(sample, list) else [sample]
        return await _compose_with_gigpo(args, group)
    if isinstance(sample, list):
        return [_compose_with_prm(args, s) for s in sample]
    return _compose_with_prm(args, sample)
