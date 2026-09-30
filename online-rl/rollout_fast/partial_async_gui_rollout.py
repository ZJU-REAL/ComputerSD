"""slime entry points for the fast multiprocess GUI rollout (train + eval).

Wire these via the launch script:
- ``--custom-generate-function-path rollout_fast.partial_async_gui_rollout.generate``
- ``--eval-function-path           rollout_fast.partial_async_gui_rollout.fast_eval_rollout``
  (that entry reuses :func:`_fast_eval_rollout_async` below as the eval coroutine)

Both converge on :class:`rollout_fast.process_pool.FastRolloutPool`, which runs each
trajectory in its own worker process. The legacy
``rollout/partial_async_rollout_gui.py`` is left untouched and is simply not
referenced by the fast launch scripts.

Why eval still goes through ``generate_and_rm``: that wrapper owns reward-model
dispatch. For the non-PRM GUI path the trajectory pre-fills ``sample.reward`` so
RM is skipped; for the PRM path it leaves ``reward=None`` and ``generate_and_rm``
runs ``reward_func`` via ``async_rm``/``batched_async_rm``. Crucially,
``generate_and_rm`` resolves ``args.custom_generate_function_path`` — which the
fast scripts point at :func:`generate` below — so it calls into our pool for the
actual trajectory while keeping RM handling correct. We only replace the
*single-trajectory fan-out*, exactly as planned.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from slime.utils.types import Sample
from utils.rollout_helpers import _load_meta_pairs
from utils.utils import load_task_config
from utils.guidance import guidance_record, load_guidance, write_guidance

logger = logging.getLogger(__name__)


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    """Train/eval single-trajectory entry: dispatch to the multiprocess pool.

    Called by slime's ``generate_and_rm`` (via ``custom_generate_function_path``)
    on the RolloutManager event loop; suspends here while a worker process runs
    the trajectory. Inside a fast worker we never recurse — the worker calls
    ``run_trajectory`` directly, not this function.
    """
    if os.getenv("_IN_FAST_WORKER"):
        # Defensive: a worker should never re-enter the slime dispatch layer.
        from rollout_fast.trajectory_runner import run_trajectory

        return await run_trajectory(args, sample, sampling_params, evaluation)

    # Backend A/B: GUI_ROLLOUT_BACKEND=ray routes *training* rollout through the
    # Ray-actor pool (plasma zero-copy result return); default keeps the process pool.
    # Eval stays on the process pool (off the training event loop the actor queue binds to).
    if os.getenv("GUI_ROLLOUT_BACKEND", "process").lower() == "ray" and not evaluation:
        from rollout_fast.ray_actor_pool import RayActorPool

        return await RayActorPool.get(args).submit(sample, sampling_params, evaluation)

    from rollout_fast.process_pool import FastRolloutPool

    return await FastRolloutPool.get(args).submit(sample, sampling_params, evaluation)


def fast_eval_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """Half-async eval entry (``--eval-function-path``). Train falls back to slime's
    default ``generate_rollout``; eval runs our pool-backed organizer on the global loop."""
    if not evaluation:
        from slime.rollout.sglang_rollout import generate_rollout

        return generate_rollout(args, rollout_id, data_source, evaluation=False)

    from slime.utils.async_utils import run

    output, _ = run(_fast_eval_rollout_async(args))
    return output


async def _fast_eval_rollout_async(args):
    """Build every eval task and run them through ``generate_and_rm``.

    Replicates ``rollout/partial_async_rollout_gui.py::_gui_eval_rollout`` exactly;
    the only behavioral change is upstream — ``generate_and_rm`` dispatches each
    trajectory to the process pool via :func:`generate` instead of the legacy
    single-loop ``_generate_local``.
    """
    from slime.rollout.base_types import RolloutFnEvalOutput
    from slime.rollout.sglang_rollout import generate_and_rm

    base_dir = os.getenv(
        "GUI_TEST_CONFIG_BASE_DIR",
        # rollout_fast/ -> parent.parent is the gui-rl/ root (see _sample_task_info).
        str(Path(__file__).resolve().parent.parent / "evaluation_examples"),
    )
    meta_path = os.getenv("GUI_EVAL_META_PATH", str(Path(base_dir) / "test_nochrome.json"))
    pairs = _load_meta_pairs(meta_path)
    if not pairs:
        raise RuntimeError(f"No eval tasks loaded from {meta_path}")

    # Limit unique tasks before n_samples_per_eval_prompt replication. A value
    # of 0 (the default in the common launcher) means evaluate the full split.
    raw_task_limit = os.getenv("GUI_EVAL_TASK_LIMIT", "0").strip()
    try:
        task_limit = int(raw_task_limit or "0")
    except ValueError as exc:
        raise ValueError(
            f"GUI_EVAL_TASK_LIMIT must be a non-negative integer, got {raw_task_limit!r}"
        ) from exc
    if task_limit < 0:
        raise ValueError(f"GUI_EVAL_TASK_LIMIT must be non-negative, got {task_limit}")
    if task_limit:
        pairs = pairs[:task_limit]
        logger.info(
            "Limited GUI eval to %d unique task(s) from %s",
            len(pairs),
            meta_path,
        )

    eval_max_response_len = getattr(args, "eval_max_response_len", None)
    if eval_max_response_len is None:
        eval_max_response_len = getattr(args, "rollout_max_response_len", 512)

    eval_top_p = getattr(args, "eval_top_p", None)
    if eval_top_p is None:
        eval_top_p = getattr(args, "rollout_top_p", 1.0)
    eval_top_k = getattr(args, "eval_top_k", None)
    if eval_top_k is None:
        eval_top_k = getattr(args, "rollout_top_k", -1)

    sampling_params = dict(
        temperature=float(getattr(args, "eval_temperature", 0.0) or 0.0),
        top_p=float(eval_top_p),
        top_k=int(eval_top_k),
        max_new_tokens=int(eval_max_response_len),
        stop=getattr(args, "rollout_stop", None),
        stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
        skip_special_tokens=getattr(args, "rollout_skip_special_tokens", False),
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    n_samples = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    guidance_path = os.getenv("GUI_GUIDANCE_FILE", "").strip()
    guidance_output = os.getenv("GUI_GUIDANCE_OUTPUT_FILE", "").strip()
    guidance = load_guidance(guidance_path) if guidance_path else {}
    if guidance_path:
        logger.info("Offline guidance matches %d/%d eval tasks from %s",
                    sum(example_id in guidance for _, example_id in pairs), len(pairs), guidance_path)
    if guidance_output:
        if guidance_path:
            raise ValueError("Guidance collection and replay cannot be enabled together")
        if n_samples != 1 or sampling_params["temperature"] != 0.0:
            raise ValueError("Guidance collection requires one sample per task and temperature=0")
        if not getattr(args, "prm_enable", False) or getattr(args, "gui_opd_enable", False):
            raise ValueError("Guidance collection requires prm_enable=true and gui_opd_enable=false")
        if getattr(args, "gui_reward_agent_class_path", None) != "reward.analyzer_agent.AnalyzerAgent":
            raise ValueError("Guidance collection requires AnalyzerAgent")
        if len({example_id for _, example_id in pairs}) != len(pairs):
            raise ValueError("Guidance collection requires unique task IDs")
        missing = [f"{domain}/{example_id}" for domain, example_id in pairs
                   if not (Path(base_dir) / "examples" / domain / f"{example_id}.json").is_file()]
        if missing:
            raise ValueError(f"Missing task configurations: {missing}")
        write_guidance(guidance_output, {})
    guidance_records = {}
    tasks = []
    sample_index = 0
    for domain, example_id in pairs:
        cfg_path = Path(base_dir) / "examples" / str(domain) / f"{example_id}.json"
        if not cfg_path.exists():
            continue
        task_config = load_task_config(base_dir, domain, example_id)
        instruction = str(task_config.get("instruction", ""))
        group_index = sample_index // n_samples
        for _ in range(n_samples):
            sample = Sample(
                prompt=instruction,
                label="",
                metadata={
                    "domain": domain,
                    "example_id": example_id,
                    "instruction": instruction,
                    "task_config": task_config,
                    **({"eval_guidance_steps": guidance.get(example_id, {})} if guidance_path else {}),
                    **({"collect_analyzer_guidance": True} if guidance_output else {}),
                },
            )
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
                )
            )

    if not tasks:
        raise RuntimeError(f"No valid eval tasks found from {meta_path}")

    data = []
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        if guidance_output:
            if isinstance(sample, list):
                raise ValueError("Guidance collection must return one trajectory per task")
            record = guidance_record(sample)
            guidance_records[record["task_id"]] = record
            write_guidance(guidance_output, guidance_records)
            logger.info("Saved analyzer guidance for %d/%d tasks to %s",
                        len(guidance_records), len(tasks), guidance_output)

    data.sort(key=lambda s: s.index)
    reward_key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", "score")
    rewards = []
    for sample in data:
        if isinstance(sample.reward, dict):
            rewards.append(float(sample.reward.get(reward_key, 0.0)))
        elif sample.reward is not None:
            rewards.append(float(sample.reward))
        else:
            rewards.append(0.0)

    return RolloutFnEvalOutput(
        data={
            "gui_eval": {
                "rewards": rewards,
                "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
                "samples": data,
            }
        }
    ), []
