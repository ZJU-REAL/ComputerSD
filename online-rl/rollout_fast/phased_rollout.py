"""Batch barriers for the serial OPD efficiency baseline.

Worker coroutines pause after environment rollout and analyzer inference. Their
contexts stay on the owning Ray actor, but the actor is free to run another
trajectory. This supports batches larger than the worker pool without deadlock
or transferring screenshots and tokenizer state between phases.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class PhaseGate:
    def __init__(self):
        self.ready = {name: asyncio.Event() for name in ("environment", "analyzer")}
        self.release = {name: asyncio.Event() for name in self.ready}

    async def pause(self, phase: str) -> None:
        self.ready[phase].set()
        await self.release[phase].wait()


class PhasedWorker:
    """Worker-local suspended trajectories; advanced on its persistent loop."""

    def __init__(self):
        self.pending = {}

    async def advance(self, key, phase, payload, args):
        from rollout_fast.trajectory_runner import run_trajectory

        if phase == "environment":
            if key in self.pending:
                raise RuntimeError(f"Duplicate phased trajectory: {key}")
            sample, sampling_params = payload
            gate = PhaseGate()
            task = asyncio.create_task(
                run_trajectory(args, sample, sampling_params, phase_gate=gate)
            )
            self.pending[key] = (sample, gate, task)
        sample, gate, task = self.pending[key]
        if phase in ("analyzer", "teacher"):
            gate.release["environment" if phase == "analyzer" else "analyzer"].set()
        if phase != "teacher":
            waiter = asyncio.create_task(gate.ready[phase].wait())
            try:
                await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
            # Even failed trajectories participate in later dispatches so every
            # actor is drained and every input gets exactly one output.
            return None
        try:
            return await task
        except Exception as exc:
            from reward.reward_func import mark_aborted_samples
            from slime.utils.types import Sample

            logger.exception("Phased GUI trajectory failed: %s", key)
            sample.status = Sample.Status.ABORTED
            sample.metadata = sample.metadata or {}
            sample.metadata["gui_invalid_reason"] = "fast_worker_exception"
            sample.metadata["gui_error_message"] = str(exc)
            sample.reward = {"score": 0.0, "acc": 0.0}
            mark_aborted_samples([sample])
            return sample
        finally:
            del self.pending[key]

    async def discard(self):
        tasks = [task for _, _, task in self.pending.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()


async def run_phases(jobs, dispatch):
    """Drain every job in one stage before dispatching any job in the next."""
    metrics = {}
    results = None
    for phase in ("environment", "analyzer", "teacher"):
        started = time.perf_counter()
        # Drain failures too: callers may then safely discard suspended contexts.
        results = await asyncio.gather(
            *(dispatch(job, phase) for job in jobs), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        elapsed = time.perf_counter() - started
        metrics[f"rollout/phases/{phase}_seconds"] = elapsed
        logger.info("Serial OPD phase=%s trajectories=%d elapsed=%.3fs", phase, len(jobs), elapsed)
    return results, metrics


async def _generate_rollout_phased_async(args: Any, rollout_id: int, data_source: Any):
    import os
    import uuid

    from rollout_fast.ray_actor_pool import RayActorPool
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.rollout.rm_hub import batched_async_rm
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.misc import load_function
    from slime.utils.types import Sample

    if os.getenv("GUI_ROLLOUT_BACKEND", "ray").lower() != "ray":
        raise ValueError("The phased OPD baseline requires GUI_ROLLOUT_BACKEND=ray")
    if args.group_rm or getattr(args, "dynamic_sampling_filter_path", None):
        raise ValueError("The phased OPD baseline supports GRPO without group RM or dynamic sampling filters")
    assert args.rollout_global_dataset
    groups = data_source.get_samples(args.rollout_batch_size)
    state = GenerateState(args)
    jobs = []
    for group in groups:
        for index, sample in enumerate(group):
            if sample.session_id is None:
                sample.session_id = str(uuid.uuid4())
            params = state.sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                params["sampling_seed"] = state.group_sampling_seeds[index]
            jobs.append((sample, params))

    results, metrics = await RayActorPool.get(args).run_phased_batch(
        jobs, rollout_id, semaphore=state.semaphore
    )
    iterator = iter(results)
    output = [[next(iterator) for _ in group] for group in groups]
    flat = [s for result in results for s in (result if isinstance(result, list) else [result])]
    rewardable = [s for s in flat if s.status != Sample.Status.ABORTED and not s.remove_sample and s.reward is None]
    if rewardable:
        rewards = await batched_async_rm(args, rewardable)
        for sample, reward in zip(rewardable, rewards, strict=True):
            sample.reward = reward
    state.reset()
    if getattr(args, "rollout_sample_filter_path", None):
        load_function(args.rollout_sample_filter_path)(args, output)
    if getattr(args, "rollout_all_samples_process_path", None):
        load_function(args.rollout_all_samples_process_path)(args, output, data_source.get_samples)
    return RolloutFnTrainOutput(samples=output, metrics=metrics)


def generate_rollout_phased(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False):
    from slime.utils.async_utils import run

    if evaluation:
        from rollout_fast.partial_async_gui_rollout import fast_eval_rollout

        return fast_eval_rollout(args, rollout_id, data_source, evaluation=True)
    return run(_generate_rollout_phased_async(args, rollout_id, data_source))
