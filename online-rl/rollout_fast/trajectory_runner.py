"""Single-trajectory logic for the fast multiprocess rollout path.

``run_trajectory`` is the rollout_fast counterpart of the legacy
``rollout/partial_async_rollout_gui.py::_generate_local``. The business logic is
preserved verbatim (task resolution, agent creation, ``Trajectory.run()``,
``build_train_data``, dynamic-history / PRM / timing / abort marking), with two
intentional differences for the multiprocess model:

1. **No trajectory semaphore.** In the legacy single-loop world many trajectories
   share one process, so ``_get_gui_trajectory_semaphore`` bounds concurrent env
   sessions. Here the ProcessPoolExecutor size *is* the concurrency cap and each
   worker process runs exactly one trajectory at a time, so the env-session count
   is already bounded by the pool size. The semaphore would always be uncontended;
   we drop it to keep the hot path clean.

2. **Process-level env-client reuse.** Instead of constructing a fresh
   ``GuiEnvClient`` / ``SessionGuiEnvClient`` per trajectory, we fetch a
   per-process singleton from :func:`rollout_fast.process_pool._get_env_client`, reusing
   its httpx connection pool across tasks. The lease itself is still
   allocate→reset→close per task (no lease reuse) — that lives inside
   ``Trajectory.run()`` and is unchanged.

The shared helpers (`_sample_task_info`, `_build_result_dir`,
`_clear_sample_result_dir`, `_create_gui_agent`, `_attach_gui_timings`) are
imported from ``utils.rollout_helpers`` — the same module the legacy path uses, so
there is no logic fork.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from config import EpisodeConfig
from reward.gui_opd import attach_precomputed_gui_opd_targets
from reward.reward_func import mark_aborted_samples
from rollout_fast.trajectory import Trajectory, build_dynamic_history_samples
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from utils.rollout_helpers import (
    _attach_gui_timings,
    _build_result_dir,
    _clear_sample_result_dir,
    _create_gui_agent,
    _sample_task_info,
)

logger = logging.getLogger(__name__)


async def run_trajectory(args, sample: Sample, sampling_params, evaluation: bool = False, *, phase_gate=None) -> Sample | list[Sample]:
    """Run one GUI trajectory and populate ``sample`` with training data.

    Mirrors ``_generate_local`` minus the env-session semaphore (the pool bounds
    concurrency) and using a process-level reused env client.
    """
    trajectory_t0 = time.perf_counter()
    assert not args.partial_rollout, "Partial rollout is not supported for GUI rollout."
    if getattr(args, "gui_opd_enable", False) and not evaluation:
        if not getattr(args, "dynamic_history", False):
            raise RuntimeError("GUI OPD requires dynamic_history=true")
        if not getattr(args, "prm_enable", False):
            raise RuntimeError("GUI OPD requires prm_enable=true")

    # Imported lazily to avoid a circular import at module load
    # (worker -> trajectory_runner -> worker).
    from rollout_fast.process_pool import _get_env_client

    profile = config.gui_profile()
    # A-group external timings (outside ep.run()). Each measured by perf_counter
    # deltas guarded by `profile`, so there is zero overhead when disabled.
    ext: dict[str, float] = {}
    setup_t0 = time.perf_counter() if profile else 0.0
    instruction, task_config, domain, example_id = _sample_task_info(sample, evaluation=evaluation)
    result_dir = _build_result_dir(args, domain, example_id, sample, evaluation=evaluation)
    _clear_sample_result_dir(result_dir)

    # Per-process singleton env client (connection pool reused across tasks).
    # GUI_ENV_CLIENT=session uses the self-contained /v1/sessions adapter
    # (clients/), default keeps the legacy lease-HTTP GuiEnvClient.
    env_client = _get_env_client()
    state = GenerateState(args)

    ep_cfg = EpisodeConfig.resolve(args, evaluation=evaluation)
    max_steps = ep_cfg.max_steps
    max_image_history_length = ep_cfg.max_image_history_length

    sampling_params = dict(sampling_params)
    if evaluation and getattr(args, "eval_temperature", None) is not None:
        sampling_params["temperature"] = float(args.eval_temperature)
    elif (not evaluation) and getattr(args, "rollout_temperature", None) is not None:
        sampling_params["temperature"] = float(args.rollout_temperature)
    parser = _create_gui_agent(
        args,
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        result_dir=result_dir,
    )
    parser.reset(logging.getLogger("desktopenv.gui_agent.rollout"))

    ep = Trajectory(
        args=args,
        agent=parser,
        env_client=env_client,
        state=state,
        ep_cfg=ep_cfg,
        sampling_params=sampling_params,
        sample=sample,
        instruction=instruction,
        task_config=task_config,
        result_dir=result_dir,
        phased=phase_gate is not None,
    )
    if profile:
        ext["setup"] = round(time.perf_counter() - setup_t0, 4)
        run_t0 = time.perf_counter()
    res = await ep.run()
    if phase_gate is not None:
        await phase_gate.pause("environment")
        await ep.collect_deferred_analyzer(res)
        await phase_gate.pause("analyzer")
        await ep.collect_deferred_teacher(res)
    sample.metadata = sample.metadata or {}
    # Includes environment execution and any OPD target pipeline tail.
    sample.metadata["trajectory_e2e_latency_s"] = time.perf_counter() - trajectory_t0
    if profile:
        ext["episode_run"] = round(time.perf_counter() - run_t0, 4)

    final_status = res.status
    eval_score = res.eval_score
    step_snapshots = res.step_snapshots
    assistant_responses = res.assistant_responses
    prm_step_scores = res.prm.step_scores
    prm_step_details = res.prm.step_details
    if sample.metadata.get("collect_analyzer_guidance"):
        details = {int(d["step_index"]): d for d in prm_step_details}
        sample.metadata["analyzer_guidance_steps"] = [
            {
                "step_index": int(attempt["step_idx"]),
                "guidance": str(details.get(int(attempt["step_idx"]), {}).get("guidance", "") or ""),
                "status": details.get(int(attempt["step_idx"]), {}).get("status", "missing"),
            }
            for attempt in res.generation_attempts
        ]
    if res.error_stage is not None:
        sample.metadata = sample.metadata or {}
        sample.metadata["gui_invalid_reason"] = f"{res.error_stage}_failed"
        sample.metadata["gui_error_stage"] = res.error_stage
        sample.metadata["gui_error_message"] = res.error_message
    if final_status == Sample.Status.ABORTED:
        # An infrastructure/generation abort invalidates the whole trajectory,
        # including any steps collected before the failure. Return a lightweight
        # tombstone instead of spending more time building partial train data.
        # Downstream removes it before reward normalization, so it affects
        # neither group statistics nor gradients.
        sample.status = final_status
        sample.tokens = []
        sample.loss_mask = []
        sample.response = "\n".join(assistant_responses)
        sample.response_length = 0
        sample.multimodal_train_inputs = None
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.metadata = sample.metadata or {}
        sample.metadata["gui_result_dir"] = str(result_dir)
        sample.metadata["gui_score"] = eval_score
        _attach_gui_timings(sample, ext, ep, profile)
        mark_aborted_samples([sample])
        return sample
    train_messages_for_loss = res.train_messages_for_loss
    tool_spec_for_loss = res.tool_spec_for_loss
    if profile:
        btd_t0 = time.perf_counter()
    input_ids, loss_mask, mm_train = parser.build_train_data(
        args=args,
        state=state,
        train_messages=train_messages_for_loss,
        tool_spec=tool_spec_for_loss,
    )
    if profile:
        ext["build_train_data"] = round(time.perf_counter() - btd_t0, 4)
    response_start = None
    active_positions = [i for i in range(len(loss_mask)) if i < len(input_ids) and int(loss_mask[i]) == 1]
    if active_positions:
        response_start = active_positions[0]
        response_length = len(input_ids) - response_start
        loss_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(input_ids))]
    else:
        response_length = 0
        loss_mask = []

    sample.tokens = input_ids
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.response_length = response_length
    sample.multimodal_train_inputs = mm_train
    sample.status = final_status
    sample.metadata = sample.metadata or {}
    sample.metadata["gui_result_dir"] = str(result_dir)
    sample.metadata["gui_score"] = eval_score
    if getattr(args, "prm_enable", False):
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
            "failure_count": sum(
                1 for detail in prm_step_details
                if not isinstance(detail, dict) or detail.get("status") != "ok"
            ),
            "request_count": len(prm_step_details),
        }
        # Current GUI non-dynamic path trains the suffix of one step response;
        # align step_wise metadata to that suffix span.
        if response_start is not None and response_length > 0:
            last_step_idx = int(step_snapshots[-1]["step_idx"]) if step_snapshots else 0
            prm_score_by_step = {int(d.get("step_index", i)): float(d.get("mean_score", 0.0)) for i, d in enumerate(prm_step_details)}
            prm_detail_by_step = {
                int(d.get("step_index", i)): d
                for i, d in enumerate(prm_step_details)
                if isinstance(d, dict)
            }
            last_prm_detail = prm_detail_by_step.get(last_step_idx, {})
            sample.metadata["step_wise"] = {
                "step_scores": [float(prm_score_by_step.get(last_step_idx, 0.0))],
                "step_indices": [int(last_step_idx)],
                "step_token_spans": [[0, int(response_length)]],
                "prm_status": str(last_prm_detail.get("status", "missing")),
                "prm_failed": str(last_prm_detail.get("status", "missing")) != "ok",
            }
    gui_reward = 1.0 if eval_score == 1 else -1.0
    # Keep training reward on `score` (GRPO expects this),
    # and expose raw task accuracy on `acc` for eval logging.
    # Important:
    # - PRM path must go through reward_func so step-wise PRM composition
    #   and prm_example_eval are visible in rollout logs.
    # - Non-PRM path keeps old behavior with prefilled reward.
    if getattr(args, "prm_enable", False):
        sample.reward = None
    else:
        sample.reward = {"score": gui_reward, "acc": float(eval_score)}
    if getattr(args, "dynamic_history", False) and not evaluation:
        prm_score_by_step = None
        prm_status_by_step = None
        if getattr(args, "prm_enable", False) and isinstance(sample.metadata.get("prm"), dict):
            prm_score_by_step = {}
            prm_status_by_step = {}
            for item in sample.metadata["prm"].get("step_details", []):
                if isinstance(item, dict) and "step_index" in item:
                    step_idx = int(item["step_index"])
                    prm_score_by_step[step_idx] = float(item.get("mean_score", 0.0))
                    prm_status_by_step[step_idx] = str(item.get("status", "missing"))
        if profile:
            bdh_t0 = time.perf_counter()
        dynamic_samples = build_dynamic_history_samples(
            args=args,
            state=state,
            agent=parser,
            base_sample=sample,
            step_snapshots=step_snapshots,
            outcome_reward=gui_reward,
            prm_score_by_step=prm_score_by_step,
            prm_status_by_step=prm_status_by_step,
            prepared_step_samples=(res.opd_targets_by_step if getattr(args, "gui_opd_enable", False) else None),
        )
        if profile:
            ext["build_dynamic_history"] = round(time.perf_counter() - bdh_t0, 4)
        if getattr(args, "gui_opd_enable", False) and dynamic_samples:
            opd_t0 = time.perf_counter() if profile else 0.0
            opd_stats = attach_precomputed_gui_opd_targets(
                args=args,
                samples=dynamic_samples,
                targets_by_step=res.opd_targets_by_step,
                errors_by_step=res.opd_target_errors_by_step,
            )
            # Final child samples now own the validated target rows. Drop the
            # temporary samples (including their duplicate full token lists).
            res.opd_targets_by_step.clear()
            res.opd_target_errors_by_step.clear()
            sample.metadata["opd"] = dict(opd_stats)
            if profile:
                ext["opd_teacher_targets"] = round(time.perf_counter() - opd_t0, 4)
        _attach_gui_timings(sample, ext, ep, profile)
        result = dynamic_samples if dynamic_samples else [sample]
        mark_aborted_samples(result)
        # >>> CHANGED (per-sample / cua-style): 不设 rollout_id（保持 None）。
        # 每个 step 样本作为独立训练样本（按样本归一化 + 按样本切 step），对齐
        # computeruseagent。slime 侧 _convert 会对 rollout_id 为 None 的样本自动
        # 分配每样本唯一 id（list(range(len))），_validate_rollout_id_annotated /
        # rollout_mask_sums 聚合已禁用，故无需共享 rollout_id。
        # <<< CHANGED
        return result
    _attach_gui_timings(sample, ext, ep, profile)
    mark_aborted_samples([sample])
    return sample
