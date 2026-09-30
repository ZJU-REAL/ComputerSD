"""Long-lived multiprocess pool + worker-process side for fast GUI rollout.

(Merged from the former pool.py + worker.py — pool and the per-worker code it
drives are one cohesive unit.)

POOL SIDE — ``FastRolloutPool``: a process-level singleton holding a
``ProcessPoolExecutor`` (spawn context). slime drives rollout by ``await``-ing one
``generate()`` coroutine per sample on the RolloutManager's single event loop;
:meth:`submit` bridges each such await to the pool via
``loop.run_in_executor(pool, _run_one, payload)`` — the coroutine suspends, a
worker process runs the trajectory, and resumes with the result. This gives
sample-level work-stealing, one-process-one-loop (no GIL/loop contention), and
N-way real parallelism (N = pool size = concurrency cap). Shared by train + eval.

WORKER SIDE — each pool process runs :func:`_worker_init` once (executor
``initializer``), then handles many trajectories via :func:`_run_one`. Per-process
state (args, sglang httpx client, env client, asyncio loop) is created once and
reused. Notes:
- **spawn, not fork.** Worker starts clean; ``args`` arrives by pickle (initargs).
- **httpx per process** with ``use_distributed_post=False`` — routing POSTs
  through a Ray actor here would reintroduce the nested-actor deadlock.
- **env client reuse** per process; the lease is still allocate→reset→close per
  task inside ``Trajectory.run`` (no lease reuse).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# =========================================================================
# Pool side (RolloutManager process)
# =========================================================================
def _resolve_pool_size() -> int:
    """Pool size = GUI_FAST_ROLLOUT_PROCS, falling back to the env-session budget."""
    fallback = os.getenv("GUI_TRAJECTORY_CONCURRENCY", os.getenv("GUI_POOL_MAX_ENVS", "16"))
    raw = os.getenv("GUI_FAST_ROLLOUT_PROCS", fallback) or "64"
    return max(1, int(raw))


class FastRolloutPool:
    """Singleton wrapper around a spawn-based ProcessPoolExecutor.

    Shared by both train and eval (eval runs concurrently with training sampling).
    """

    _instance: "FastRolloutPool | None" = None

    def __init__(self, args: Any) -> None:
        self._args = args
        self.num_procs = _resolve_pool_size()
        # spawn: the manager process already holds torch/CUDA state; fork would
        # inherit a broken CUDA context. spawn starts each worker clean and ships
        # ``args`` by pickle through initargs.
        ctx = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_procs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(args,),
        )
        logger.info("FastRolloutPool: %d worker processes (spawn)", self.num_procs)

    @classmethod
    def get(cls, args: Any) -> "FastRolloutPool":
        if cls._instance is None:
            cls._instance = cls(args)
        return cls._instance

    async def submit(self, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample | list[Sample]:
        """Bridge one trajectory await to a worker process. Returns Sample|list."""
        loop = asyncio.get_running_loop()
        payload = (sample, dict(sampling_params), evaluation)
        return await loop.run_in_executor(self._executor, _run_one, payload)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        type(self)._instance = None


# =========================================================================
# Worker side (each pool subprocess)
# =========================================================================
# ---- per-process singletons (one set per worker process) --------------------
_ARGS: Any = None
_LOOP: asyncio.AbstractEventLoop | None = None
_ENV_CLIENT: Any = None


def _worker_init(args) -> None:
    """ProcessPoolExecutor initializer: run once per worker process."""
    global _ARGS, _LOOP

    level = getattr(logging, os.getenv("GUI_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().setLevel(level)

    # Mark this process so any code that branches on the legacy worker flag
    # treats us as an in-worker (no recursive dispatch).
    os.environ["_IN_FAST_WORKER"] = "1"

    _ARGS = args

    # Init the sglang httpx client on THIS process, with distributed POST forced
    # off (see module docstring).
    from slime.utils.http_utils import init_http_client

    worker_args = copy.copy(args)
    worker_args.use_distributed_post = False
    init_http_client(worker_args)

    # Dedicated event loop for this process; reused by every _run_one call.
    _LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_LOOP)

    # Warm up the env client singleton so the first task doesn't pay setup.
    _get_env_client()

    logger.info("rollout_fast worker initialized (pid=%d)", os.getpid())


def _get_env_client():
    """Return this process's reused env client (built once)."""
    global _ENV_CLIENT
    if _ENV_CLIENT is None:
        import config

        if os.getenv("GUI_ENV_CLIENT", "legacy").strip().lower() == "session":
            from clients import SessionGuiEnvClient

            _ENV_CLIENT = SessionGuiEnvClient(config.env_server_url())
        else:
            from env_client import GuiEnvClient

            _ENV_CLIENT = GuiEnvClient(config.env_server_url())
    return _ENV_CLIENT


def _slim_result(result):
    """For eval, drop heavy training payloads (tokens / multimodal inputs) before
    pickling back to the parent — eval only needs reward/status/index/metadata.

    Operates in place on the Sample(s) and returns them. Train results are passed
    through untouched (the parent needs the full training data).
    """
    samples = result if isinstance(result, list) else [result]
    for s in samples:
        s.tokens = []
        s.loss_mask = []
        s.multimodal_train_inputs = None
        s.response_length = 0
    return result


def _run_one(payload):
    """Run one trajectory to completion on this process's event loop.

    ``payload`` is a tuple ``(sample, sampling_params, evaluation)``. ``args`` is
    the per-process singleton set in :func:`_worker_init`. Returns the resulting
    ``Sample`` or ``list[Sample]`` (eval results slimmed). Exceptions are caught
    and returned as an ABORTED sample so a single bad task never poisons the pool.
    """
    sample, sampling_params, evaluation = payload
    from rollout_fast.trajectory_runner import run_trajectory

    try:
        result = _LOOP.run_until_complete(
            run_trajectory(_ARGS, sample, sampling_params, evaluation)
        )
        if evaluation:
            result = _slim_result(result)
        return result
    except Exception as e:  # noqa: BLE001 — must not crash the worker process
        import traceback
        from reward.reward_func import mark_aborted_samples
        from slime.utils.types import Sample

        logger.error(
            "rollout_fast _run_one failed (idx=%s, eval=%s): %s\n%s",
            getattr(sample, "index", None), evaluation, e, traceback.format_exc(),
        )
        sample.status = Sample.Status.ABORTED
        sample.metadata = sample.metadata or {}
        sample.metadata["gui_invalid_reason"] = "fast_worker_exception"
        sample.metadata["gui_error_message"] = str(e)
        sample.reward = {"score": 0.0, "acc": 0.0}
        mark_aborted_samples([sample])
        return sample
