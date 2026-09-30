"""Process-reward-model (PRM) hook for GUI rollouts.

Encapsulates the optional per-step reward model that judges each policy action.
Without PRM the rollout uses only the final task score; with PRM enabled, each
completed step is dispatched asynchronously while later GUI steps continue,
and all results are collected at the trajectory boundary.

This keeps the PRM plumbing (lazy agent construction, asynchronous per-step
judging, end-of-trajectory collection, score bookkeeping) out of the turn loop.
``PrmHook.disabled()`` returns a no-op instance so the episode code path is the
same whether or not PRM is configured.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.misc import load_function

import config

logger = logging.getLogger(__name__)


class PrmHook:
    """Owns the PRM reward agent and its pending per-step judging tasks.

    Use :meth:`create` to build the right variant from ``args``; the turn loop
    then calls :meth:`submit_step` after each sampled step and :meth:`collect`
    once at the end. ``submit_step`` creates the task immediately; ``collect``
    only awaits the already-running analyzer tasks. With ``deferred=True``,
    submissions retain immutable contexts until ``dispatch_deferred`` starts
    analyzer work after the batch environment phase.
    """

    def __init__(self, agent: Any | None, *, deferred: bool = False) -> None:
        self._agent = agent
        self.deferred = deferred
        self._pending: list[tuple[int, asyncio.Task]] = []
        self._deferred: list[tuple[Any, dict[str, Any]]] = []
        self.step_scores: list[float] = []
        self.step_details: list[dict[str, Any]] = []

    # --- construction -------------------------------------------------------------

    @classmethod
    def disabled(cls) -> "PrmHook":
        """A no-op hook (PRM not enabled)."""
        return cls(agent=None)

    @classmethod
    def create(cls, args: Any, state: GenerateState, result_dir: str) -> "PrmHook":
        """Build a PRM hook from ``args``; returns a no-op hook if PRM is off."""
        if not getattr(args, "prm_enable", False):
            return cls.disabled()
        # Fixed trajectory guidance supplies the privileged context directly
        # to OPD.  Keep prm_enable true for the existing reward/rollout wiring,
        # but do not construct or query an analyzer model in this mode.
        if getattr(args, "gui_opd_enable", False) and os.getenv("GUI_GUIDANCE_FILE", "").strip():
            logger.info("Fixed GUI guidance enabled; analyzer PRM is disabled")
            return cls.disabled()

        max_hist = getattr(args, "gui_max_reward_image_history_length", None)
        if max_hist is None:
            max_hist = config.get_int("GUI_MAX_REWARD_IMAGE_HISTORY_LENGTH", 2)

        # When the PRM model differs from the policy model, the reward agent must
        # use its own tokenizer/processor rather than sharing the policy's.
        policy_model_path = str(getattr(args, "hf_checkpoint", "") or "")
        prm_model_path = str(getattr(args, "prm_model_path", "") or policy_model_path)
        share_formatter = policy_model_path == prm_model_path
        if not share_formatter:
            logger.info(
                "PRM formatter mismatch; using PRM tokenizer/processor. policy=%s prm=%s",
                policy_model_path,
                prm_model_path,
            )

        reward_cls_path = getattr(args, "gui_reward_agent_class_path", None) or config.reward_agent_class_path()
        if not reward_cls_path:
            raise RuntimeError("GUI_REWARD_AGENT_CLASS_PATH is required when prm_enable=True.")
        reward_cls = load_function(reward_cls_path)
        agent = reward_cls(
            max_reward_image_history_length=int(max_hist),
            example_result_dir=result_dir,
            tokenizer=(state.tokenizer if share_formatter else None),
            processor=(state.processor if share_formatter else None),
        )
        return cls(agent=agent)

    # --- query --------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._agent is not None

    @property
    def reward_trajectory(self) -> list[dict[str, Any]]:
        return self._agent.reward_trajectory if self._agent is not None else []

    # --- turn-loop hooks ----------------------------------------------------------

    def submit_step(
        self,
        args: Any,
        *,
        instruction: str,
        actions_history: list[str],
        policy_response: str,
        step_index: int,
        student_context: list[dict[str, Any]] | None = None,
        executed_actions: Any = None,
        is_final_step: bool = False,
    ) -> asyncio.Task | None:
        """Dispatch one exact step context without blocking the turn loop."""
        if self._agent is None:
            return None
        submit_kwargs = {
            "instruction": instruction,
            "actions_history": actions_history,
            "policy_response": policy_response,
            "step_index": int(step_index),
            "student_context": student_context,
            "executed_actions": executed_actions,
            "is_final_step": bool(is_final_step),
        }
        if self.deferred:
            self._deferred.append((args, copy.deepcopy(submit_kwargs)))
            return None
        task = self._agent.submit_step_judge(
            args,
            **submit_kwargs,
        )
        self._pending.append((int(step_index), task))
        return task

    def dispatch_deferred(self) -> None:
        """Start analyzer work collected during the environment-only phase."""
        if self._agent is None:
            return
        for args, submit_kwargs in self._deferred:
            task = self._agent.submit_step_judge(args, **submit_kwargs)
            self._pending.append((int(submit_kwargs["step_index"]), task))
        self._deferred.clear()

    async def collect(self) -> None:
        """Await all already-dispatched per-step judgments."""
        if self._agent is None:
            return
        try:
            if self._pending:
                self.step_scores, self.step_details = await self._agent.collect_step_results(self._pending)
                failure_count = sum(
                    1 for detail in self.step_details
                    if not isinstance(detail, dict) or detail.get("status") != "ok"
                )
                if failure_count:
                    logger.warning(
                        "PRM failed or returned no valid vote for %d/%d steps",
                        failure_count, len(self.step_details),
                    )
        finally:
            self._pending.clear()
            self._deferred.clear()
            close = getattr(self._agent, "aclose", None)
            if close is not None:
                await close()

    # --- score bookkeeping --------------------------------------------------------

    def score_by_step(self) -> dict[int, float]:
        """Map ``step_index -> mean_score`` from collected step details."""
        out: dict[int, float] = {}
        for i, detail in enumerate(self.step_details):
            if isinstance(detail, dict):
                idx = int(detail.get("step_index", i))
                out[idx] = float(detail.get("mean_score", 0.0))
        return out
