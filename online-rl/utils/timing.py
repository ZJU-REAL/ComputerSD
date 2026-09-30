"""Lightweight per-trajectory wall-clock profiling for GUI rollout.

Why hand-rolled spans instead of cProfile: the rollout is async with many
concurrent coroutines, where cProfile output is unreadable and, crucially,
cannot attribute time spent *waiting* on remote env / sglang IO — which is
exactly the wall-clock breakdown we care about.

Toggle via ``GUI_PROFILE=1`` (see :func:`config.gui_profile`). When disabled the
span context managers are pure no-ops — no timestamp, no dict write — so the
instrumentation can stay in production code at effectively zero cost.

Intervals use ``time.perf_counter`` (monotonic), not ``time.time``, so a system
clock adjustment cannot corrupt a measurement.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

# Static nesting map of the profiling spans (parent -> ordered children),
# describing the A/B/C/D layer structure. The JSON on disk stays flat (so the
# aggregation script and any consumers keep working); this map is only used to
# render a readable indented tree in scripts/agg_timings.py. A child's time is a
# subset of its parent's. Spans not listed here still show in the flat summary.
SPAN_TREE: dict[str, list[str]] = {
    # B layer — run() lifecycle (one per trajectory)
    "turn_loop": ["heartbeat", "build_policy_messages", "sglang_generate", "parse_response", "env_step"],
    # C layer — per-step turn loop (sums across all steps)
    # D layer — inside one sglang call
    "sglang_generate": ["gen_apply_chat_template", "gen_extract_mm", "gen_post", "gen_decode"],
}


class Timings:
    """Accumulate per-stage wall-clock time for one trajectory.

    A single trajectory runs in one asyncio task and touches these spans
    serially (no concurrent writers), so plain dict accumulation is safe.

    Four context managers, all no-ops when ``enabled`` is False:
    - ``span`` / ``aspan``: accumulate into running totals only.
    - ``measure`` / ``ameasure``: accumulate totals AND store this call's single
      duration into the supplied ``out`` dict, for emitting per-step detail.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.spans: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.steps: list[dict[str, Any]] = []

    def _add(self, name: str, dt: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add(name, time.perf_counter() - t0)

    @asynccontextmanager
    async def aspan(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._add(name, time.perf_counter() - t0)

    @contextmanager
    def measure(self, name: str, out: dict[str, float]):
        """Time a block, accumulate into totals, AND record the single delta.

        Like :meth:`span` but also stores this call's duration into ``out[name]``
        so the caller can emit per-step detail. Writes 0.0 when disabled.
        """
        if not self.enabled:
            out[name] = 0.0
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._add(name, dt)
            out[name] = round(dt, 4)

    @asynccontextmanager
    async def ameasure(self, name: str, out: dict[str, float]):
        """Async counterpart of :meth:`measure` (for blocks containing await)."""
        if not self.enabled:
            out[name] = 0.0
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._add(name, dt)
            out[name] = round(dt, 4)

    def summary(self) -> dict[str, Any]:
        """Per-span ``{total_s, count, avg_s}`` sorted by total descending."""
        return {
            k: {
                "total_s": round(v, 4),
                "count": self.counts[k],
                "avg_s": round(v / self.counts[k], 4) if self.counts[k] else 0.0,
            }
            for k, v in sorted(self.spans.items(), key=lambda x: -x[1])
        }
