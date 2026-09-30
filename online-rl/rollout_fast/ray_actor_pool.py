"""Ray-actor rollout pool — drop-in alternative to ``FastRolloutPool`` for A/B.

Same public API as ``process_pool.FastRolloutPool`` (``get(args)`` + ``async submit``),
so ``partial_async_gui_rollout.generate`` can switch backends with one ``if``. No edits
to ``process_pool.py``; execution logic is **reused verbatim** from it.

Why it can beat the process pool on big multimodal payloads: each trajectory result
(Sample/list[Sample] with numpy ``pixel_values``) is written to Ray **plasma** by its own
actor and read back **zero-copy** on the RolloutManager — no single ProcessPoolExecutor
result pipe, no single manager-thread unpickle of the 4.3GB blob (kills dispatch_wait).
Long-lived actors keep process reuse (no per-trajectory spawn).

Honest caveats: plasma must fit in-flight results (``object_store_memory``) or it spills;
actors are co-located with the RolloutManager node (soft) for true intra-node zero-copy;
B2 does NOT shrink volume — pair with image dedup for the full win.

Enable: ``GUI_ROLLOUT_BACKEND=ray`` (read in partial_async_gui_rollout.generate).
"""

from __future__ import annotations

import asyncio
import os

import ray

from slime.utils.types import Sample
from rollout_fast.process_pool import _resolve_pool_size, _run_one, _worker_init


@ray.remote
class _RolloutActor:
    """One long-lived worker == one ProcessPoolExecutor slot."""

    def __init__(self, args) -> None:
        self._args = args
        from rollout_fast.phased_rollout import PhasedWorker

        self._phased = PhasedWorker()
        _worker_init(args)  # http client (use_distributed_post=False) + env client + event loop

    def run_one(self, payload):
        return _run_one(payload)  # run_trajectory on this actor's loop; returns Sample|list[Sample]

    def run_phase(self, key, phase, payload=None):
        from rollout_fast.process_pool import _LOOP

        return _LOOP.run_until_complete(self._phased.advance(key, phase, payload, self._args))

    def discard_phased(self):
        from rollout_fast.process_pool import _LOOP

        _LOOP.run_until_complete(self._phased.discard())


class RayActorPool:
    """N actors + a free-actor queue: exactly N concurrent, work-stealing reuse."""

    _instance: "RayActorPool | None" = None

    def __init__(self, args) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        n = _resolve_pool_size()
        # Forward the full env so actors inherit what a spawn worker would
        # (GUI_* knobs, env-server url, LD_LIBRARY_PATH for cuDNN, ...).
        runtime_env = {"env_vars": {k: v for k, v in os.environ.items()}}
        # Co-locate with this (RolloutManager) node so plasma reads are zero-copy.
        sched = NodeAffinitySchedulingStrategy(ray.get_runtime_context().get_node_id(), soft=True)
        actor_cpus = float(os.getenv("GUI_RAY_ACTOR_CPUS", "1"))
        Actor = _RolloutActor.options(num_cpus=actor_cpus, runtime_env=runtime_env, scheduling_strategy=sched)

        self._actors = [Actor.remote(args) for _ in range(n)]
        self._free: asyncio.Queue = asyncio.Queue()
        for a in self._actors:
            self._free.put_nowait(a)

    @classmethod
    def get(cls, args) -> "RayActorPool":
        if cls._instance is None:
            cls._instance = cls(args)
        return cls._instance

    async def submit(self, sample: Sample, sampling_params: dict, evaluation: bool = False):
        payload = (sample, dict(sampling_params), evaluation)
        actor = await self._free.get()  # blocks (backpressure) until a worker is free
        try:
            return await actor.run_one.remote(payload)  # ObjectRef awaited; result via plasma
        finally:
            self._free.put_nowait(actor)

    async def run_phased_batch(self, payloads, rollout_id, *, semaphore):
        from rollout_fast.phased_rollout import run_phases

        owners = {}
        jobs = list(enumerate(payloads))

        async def dispatch_one(job, phase):
            ordinal, payload = job
            key = (rollout_id, ordinal)
            if phase == "environment":
                actor = await self._free.get()
                owners[ordinal] = actor
                try:
                    return await actor.run_phase.remote(key, phase, payload)
                finally:
                    self._free.put_nowait(actor)
            return await owners[ordinal].run_phase.remote(key, phase)

        async def dispatch(job, phase):
            async with semaphore:
                return await dispatch_one(job, phase)

        try:
            return await run_phases(jobs, dispatch)
        finally:
            await asyncio.gather(*(actor.discard_phased.remote() for actor in self._actors))

    def shutdown(self) -> None:
        for a in self._actors:
            ray.kill(a)
        type(self)._instance = None
