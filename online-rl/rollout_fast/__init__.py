"""rollout_fast: self-contained pure-multiprocess GUI rollout (train + eval).

A drop-in alternative to the Ray-actor ``TrajectoryDispatcher`` in
``rollout/trajectory_worker.py``. Instead of N Ray actors each running M asyncio
coroutines (where per-trajectory synchronous CPU work — tokenize / image base64 /
build_train_data — serializes on one event loop), this package runs each
trajectory on its own OS process via a long-lived ``ProcessPoolExecutor`` (spawn),
giving sample-level work-stealing, one-process-one-loop (no GIL/loop contention),
and per-process env-client reuse.

slime entry points live in :mod:`rollout_fast.partial_async_gui_rollout`:
- ``generate``           -> ``--custom-generate-function-path`` (train)
- ``fast_eval_rollout``  -> ``--eval-function-path`` (eval)

Both converge on the same single-trajectory function
:func:`rollout_fast.trajectory_runner.run_trajectory`, branching only on the
``evaluation`` flag — identical to the legacy ``_generate_local`` contract.
"""
