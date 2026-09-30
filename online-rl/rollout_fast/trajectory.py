"""One GUI rollout trajectory: env lifecycle + turn loop + per-step sample build.

Two responsibilities, previously split across episode.py + trajectory.py, now
unified here:

1. :class:`Trajectory` drives one rollout's remote-env lifecycle (acquire ->
   reset -> per-step act -> evaluate -> close) and the multi-turn policy loop
   (query policy, parse action, step env), producing a :class:`TrajectoryResult`.
2. :func:`build_dynamic_history_samples` turns the episode's per-step snapshots
   into training :class:`Sample` objects (dynamic-history GRPO): each snapshot
   becomes one sample whose loss mask covers only that step's response suffix.

PRM (process reward) is delegated to :class:`reward.prm_hook.PrmHook`.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

import config
from clients.osworld_remote_async import merge_reset_observations
from env_client import GuiEnvClient
from reward.prm_hook import PrmHook
from utils.guidance import inject_step_guidance, trajectory_guidance
from utils.timing import Timings
from utils.utils import save_image

# ===== DEBUG: real env occupancy =====
# Opt in with GUI_DEBUG_ENV_INFLIGHT=1. This must stay disabled by default because
# all workers serialize on one file lock, which can block rollout indefinitely
# when GUI_RESULT_DIR is backed by a shared filesystem.
import fcntl as _fcntl
import os as _os
import time as _time

_HELD_LEASES: set[str] = set()
_INFLIGHT_DIR: Path | None = None


def _dump_inflight(action: str = "", task: str = "") -> None:
    """Track in-flight envs across all workers, guarded by one exclusive flock:
    - inflight.json  {pid:[leases]}  : current snapshot (overwritten).
    - counter.txt    timeline log    : APPENDED one line per change:
        "<ts>  total=<N>  <action> <task>  pid=<pid>"
      where <task> = "<example_id>#<sample_index>" (the task that grabbed/released an env).
    """
    if _os.getenv("GUI_DEBUG_ENV_INFLIGHT", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    global _INFLIGHT_DIR
    try:
        if _INFLIGHT_DIR is None:
            _INFLIGHT_DIR = Path(_os.getenv("GUI_RESULT_DIR", "/tmp")) / "_env_inflight"
            _INFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        pid = str(_os.getpid())
        with open(_INFLIGHT_DIR / "inflight.json", "a+", encoding="utf-8") as fh:
            _fcntl.flock(fh, _fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                data = json.loads(raw) if raw.strip() else {}
                if _HELD_LEASES:
                    data[pid] = sorted(_HELD_LEASES)
                else:
                    data.pop(pid, None)  # this process holds nothing -> drop its key
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps(data))
                fh.flush()
                # APPEND a timeline line to counter.txt (total + this change)
                total = sum(len(v) for v in data.values())
                ts = _time.strftime("%H:%M:%S")
                with open(_INFLIGHT_DIR / "counter.txt", "a", encoding="utf-8") as cf:
                    cf.write(f"{ts}  total={total}  {action} {task}  pid={pid}\n")
            finally:
                _fcntl.flock(fh, _fcntl.LOCK_UN)
    except Exception:
        pass  # debug-only, never break rollout

logger = logging.getLogger(__name__)


def _count_message_images(value: Any) -> int:
    if isinstance(value, dict):
        return int(value.get("type") == "image") + sum(
            _count_message_images(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_count_message_images(child) for child in value)
    return 0


def _messages_with_image_paths(
    messages: list[dict[str, Any]], result_dir: Path, step_idx: int
) -> list[dict[str, Any]]:
    """Copy policy messages while replacing embedded image data with absolute PNG paths."""
    copied = copy.deepcopy(messages)
    image_count = _count_message_images(copied)
    first_step = max(0, step_idx - image_count + 1)
    image_number = 0

    def replace(value: Any) -> Any:
        nonlocal image_number
        if isinstance(value, dict):
            if value.get("type") == "image":
                value["image"] = str(
                    (result_dir / f"step_{first_step + image_number}.png").resolve()
                )
                image_number += 1
                return value
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            return [replace(child) for child in value]
        return value

    return replace(copied)


def _absolute_screenshot_path(result_dir: Path, step_idx: int) -> str:
    return str((result_dir / f"step_{step_idx}.png").resolve())


def _response_suffix_mask(token_ids: list[int], loss_mask: list[int]) -> tuple[int, int, list[int]] | None:
    """Convert a full-sequence loss mask to a response-suffix mask.

    Training expects the mask to start at the first trainable (response) token.
    Returns ``(response_start, response_length, suffix_mask)`` or ``None`` when
    the snapshot has no trainable tokens (skip it).
    """
    active_positions = [i for i in range(len(loss_mask)) if i < len(token_ids) and int(loss_mask[i]) == 1]
    if not active_positions:
        return None
    response_start = active_positions[0]
    response_length = len(token_ids) - response_start
    if response_length <= 0:
        return None
    suffix_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(token_ids))]
    return response_start, response_length, suffix_mask


def _build_child_metadata(
    base_sample: Sample,
    step_idx: int,
    outcome_reward: float,
    prm_score_by_step: dict[int, float] | None,
    prm_status_by_step: dict[int, str] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal, explicit metadata for one dynamic-history child sample."""
    meta = copy.deepcopy(base_sample.metadata or {})
    # Do not replicate every vote/raw analyzer response into every child.  Keep
    # the trajectory summary and only this step's detail; the full list remains
    # in the trajectory artifact written by Trajectory._write_results.
    prm_meta = meta.get("prm") if isinstance(meta, dict) else None
    if isinstance(prm_meta, dict):
        current_details = prm_meta.get("step_details", [])
        current_detail = None
        if isinstance(current_details, list):
            for detail in current_details:
                if isinstance(detail, dict) and int(detail.get("step_index", -1)) == int(step_idx):
                    current_detail = detail
                    break
        prm_meta["step_details"] = [current_detail] if current_detail is not None else []
        meta["prm"] = prm_meta
    meta["dynamic_step_index"] = step_idx
    meta["dynamic_outcome_reward"] = float(outcome_reward)
    if prm_score_by_step is not None:
        # rollout infers the span from loss_mask, so one score/index per child.
        step_wise = {
            "step_scores": [float(prm_score_by_step.get(step_idx, 0.0))],
            "step_indices": [int(step_idx)],
        }
        if prm_status_by_step is not None:
            prm_status = str(prm_status_by_step.get(step_idx, "missing"))
            step_wise["prm_status"] = prm_status
            step_wise["prm_failed"] = prm_status != "ok"
        meta["step_wise"] = step_wise
    # Carry the GiGPO anchor for this step's pre-action state. Only the raw a11y
    # text + instruction are stored here; hashing/normalization happens in the
    # reward stage (reward.gigpo) so the rollout worker stays numpy-free.
    gigpo_meta: dict[str, Any] = {"step_index": int(step_idx)}
    if snapshot is not None:
        gigpo_meta["anchor_obs"] = snapshot.get("anchor_obs")
        gigpo_meta["instruction"] = snapshot.get("instruction")
    # Preserve any GiGPO config (e.g. platform) the base sample already carries.
    if isinstance(base_sample.metadata, dict) and isinstance(base_sample.metadata.get("gigpo"), dict):
        gigpo_meta.update(base_sample.metadata["gigpo"])
    meta["gigpo"] = gigpo_meta
    return meta


def build_dynamic_history_samples(
    args: Any,
    state: GenerateState,
    agent: Any,
    base_sample: Sample,
    step_snapshots: list[dict[str, Any]],
    outcome_reward: float,
    prm_score_by_step: dict[int, float] | None = None,
    prm_status_by_step: dict[int, str] | None = None,
    prepared_step_samples: dict[int, Sample] | None = None,
) -> list[Sample]:
    """Turn each step snapshot into one training Sample (dynamic-history GRPO).

    Each child carries the full token sequence for that step's context+response,
    a response-suffix loss mask, and the (outcome or PRM) reward. Snapshots with
    no trainable response tokens are skipped.
    """
    reward_key = getattr(args, "reward_key", None) or "score"
    prm_enabled = getattr(args, "prm_enable", False)
    dynamic_samples: list[Sample] = []
    mask_sums: list[int] = []
    token_counts: list[int] = []
    image_counts: list[int] = []
    last_active_gaps: list[int] = []
    skipped_steps: list[int] = []

    for snapshot in step_snapshots:
        step_idx = int(snapshot["step_idx"])
        messages = snapshot["train_messages"]
        response_text = snapshot["response_text"]
        tool_spec = snapshot.get("tool_spec")

        prepared = (prepared_step_samples or {}).get(step_idx)
        if prepared is not None:
            token_ids = prepared.tokens
            response_length = int(prepared.response_length)
            child_loss_mask = list(prepared.loss_mask or [])
            mm_train = prepared.multimodal_train_inputs
            response_start = len(token_ids) - response_length
            active_positions = [
                response_start + i
                for i, value in enumerate(child_loss_mask)
                if int(value) == 1
            ]
        else:
            token_ids, loss_mask, mm_train = agent.build_train_data(
                args=args,
                state=state,
                train_messages=messages,
                tool_spec=tool_spec,
            )
            active_positions = [
                i
                for i, value in enumerate(loss_mask)
                if i < len(token_ids) and int(value) == 1
            ]
            suffix = _response_suffix_mask(token_ids, loss_mask)
            if suffix is None:
                skipped_steps.append(step_idx)
                continue
            _response_start, response_length, child_loss_mask = suffix

        mask_sums.append(len(active_positions))
        token_counts.append(len(token_ids))
        image_counts.append(_count_message_images(messages))
        last_active_gaps.append(
            len(token_ids) - 1 - active_positions[-1] if active_positions else len(token_ids)
        )

        if response_length <= 0 or len(child_loss_mask) != response_length:
            skipped_steps.append(step_idx)
            continue

        child_reward = {"score": float(outcome_reward), reward_key: float(outcome_reward)}
        # When PRM is on, per-step reward is attached separately; leave it unset here.
        child_reward_for_sample = None if prm_enabled else child_reward

        child = Sample(
            group_index=base_sample.group_index,
            index=base_sample.index,
            prompt=base_sample.prompt,
            tokens=token_ids,
            multimodal_inputs=base_sample.multimodal_inputs,
            multimodal_train_inputs=mm_train,
            response=response_text,
            response_length=response_length,
            label=base_sample.label,
            reward=child_reward_for_sample,
            loss_mask=child_loss_mask,
            weight_versions=list(base_sample.weight_versions),
            rollout_log_probs=None,
            rollout_routed_experts=None,
            remove_sample=base_sample.remove_sample,
            status=base_sample.status,
            metadata=_build_child_metadata(
                base_sample,
                step_idx,
                outcome_reward,
                prm_score_by_step,
                prm_status_by_step=prm_status_by_step,
                snapshot=snapshot,
            ),
            generate_function_path=base_sample.generate_function_path,
            train_metadata=base_sample.train_metadata,
            non_generation_time=base_sample.non_generation_time,
            spec_info=base_sample.spec_info,
            prefix_cache_info=base_sample.prefix_cache_info,
        )
        dynamic_samples.append(child)

    def _range(values: list[int]) -> str:
        if not values:
            return "0/0.0/0"
        return f"{min(values)}/{sum(values) / len(values):.1f}/{max(values)}"

    status = getattr(base_sample.status, "value", str(base_sample.status))
    log = logger.warning if skipped_steps else logger.info
    log(
        "[dynamic-fanout] group=%s sample=%s status=%s snapshots=%d children=%d "
        "skipped_zero_mask=%d skipped_steps=%s mask_sum[min/mean/max]=%s "
        "tokens[min/mean/max]=%s images[min/mean/max]=%s last_active_gap[min/mean/max]=%s",
        base_sample.group_index,
        base_sample.index,
        status,
        len(step_snapshots),
        len(dynamic_samples),
        len(skipped_steps),
        skipped_steps,
        _range(mask_sums),
        _range(token_counts),
        _range(image_counts),
        _range(last_active_gaps),
    )

    return dynamic_samples


@dataclass
class TrajectoryResult:
    """Everything ``generate`` needs to build training samples from one trajectory."""

    status: Sample.Status
    eval_score: float
    # Per-step snapshots consumed by build_dynamic_history_samples.
    step_snapshots: list[dict[str, Any]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    # Every model response, including responses that could not be parsed into
    # an executable action. This keeps format failures diagnosable even when
    # the executed trajectory is empty.
    generation_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Messages/tool_spec to build the single-sample (non-dynamic) loss target.
    train_messages_for_loss: list[dict[str, Any]] = field(default_factory=list)
    tool_spec_for_loss: dict[str, Any] | None = None
    prm: PrmHook = field(default_factory=PrmHook.disabled)
    # OPD targets are submitted when their corresponding PRM task completes,
    # rather than waiting for the trajectory-wide PRM collection barrier.
    opd_target_tasks: list[asyncio.Task] = field(default_factory=list)
    opd_targets_by_step: dict[int, Sample] = field(default_factory=dict)
    opd_target_errors_by_step: dict[int, str] = field(default_factory=dict)
    result_dir: Path | None = None
    # Set when the trajectory failed; recorded onto the sample metadata by caller.
    error_stage: str | None = None
    error_message: str | None = None
    terminal_action_sent: bool = False


class Trajectory:
    """Drive one rollout: hold env/agent/state and execute the turn loop."""

    def __init__(
        self,
        *,
        args: Any,
        agent: Any,
        env_client: GuiEnvClient,
        state: GenerateState,
        ep_cfg: Any,
        sampling_params: dict[str, Any],
        sample: Sample,
        instruction: str,
        task_config: dict[str, Any] | None,
        result_dir: Path,
        phased: bool = False,
    ) -> None:
        self.args = args
        self.phased = phased
        self.agent = agent
        self.env_client = env_client
        self.state = state
        self.cfg = ep_cfg
        self.sampling_params = sampling_params
        self.sample = sample
        self.instruction = instruction
        self.task_config = task_config
        self.result_dir = result_dir
        self.traj_path = result_dir / "traj.jsonl"
        self.domain = str((sample.metadata or {}).get("domain", ""))
        self.example_id = str((sample.metadata or {}).get("example_id", ""))
        self.guidance_steps = (sample.metadata or {}).get("eval_guidance_steps", {})
        self.fixed_guidance_enabled = bool(
            getattr(args, "gui_opd_enable", False)
            and _os.getenv("GUI_GUIDANCE_FILE", "").strip()
        )
        self.fixed_guidance = ""
        if self.fixed_guidance_enabled:
            guidance_path = _os.environ["GUI_GUIDANCE_FILE"].strip()
            self.fixed_guidance = trajectory_guidance(guidance_path, self.example_id)
            if not self.fixed_guidance:
                logger.warning(
                    "No fixed guidance found for task %s in %s; OPD will be masked",
                    self.example_id,
                    guidance_path,
                )

    # --- env lifecycle ------------------------------------------------------------

    async def _acquire(self) -> str:
        """Acquire a lease with bounded retry (env capacity is the bottleneck).

        Returns the lease_id. ``task_config`` is applied later at reset().
        """
        last_error: Exception | None = None
        # A stable episode id gives all retries the same server-side session id.
        # If an acquire response is lost after the server allocated a slot, the
        # retry resolves to that slot instead of leaking another one.
        episode_id = f"{self.domain}:{self.example_id}:{uuid.uuid4().hex[:8]}"
        for attempt in range(self.cfg.allocate_retries):
            try:
                # Match computeruseagent: do not pass task_type; allocate defaults
                # to "training". config.env_mode() returns "train"/"eval" which is
                # NOT the task_type the env server expects.
                lease = await self.env_client.allocate(
                    episode_id=episode_id,
                    user_id=config.user_id(),
                    job_id=config.job_id() or None,
                )
                _HELD_LEASES.add(lease["lease_id"])  # DEBUG: env acquired
                _dump_inflight("acquire", f"{self.example_id}#{self.sample.index}")
                return lease["lease_id"]
            except Exception as e:  # external service call
                last_error = e
                if attempt < self.cfg.allocate_retries - 1:
                    logger.warning(
                        "GUI acquire failed (%d/%d), retry in %.1fs: %s",
                        attempt + 1,
                        self.cfg.allocate_retries,
                        self.cfg.allocate_backoff_seconds,
                        e,
                    )
                    await asyncio.sleep(self.cfg.allocate_backoff_seconds)
        raise RuntimeError(
            f"GUI env acquire failed after {self.cfg.allocate_retries} retries: {last_error}"
        )

    # --- run ----------------------------------------------------------------------

    async def run(self) -> TrajectoryResult:
        res = TrajectoryResult(status=Sample.Status.COMPLETED, eval_score=0.0, result_dir=self.result_dir)
        res.train_messages_for_loss = [self.agent.build_train_system_message()]
        self.timings = Timings(config.gui_profile())
        trace_records: list[dict[str, Any]] = []
        error_stage = "init"
        lease_id: str | None = None

        try:
            error_stage = "acquire"
            logger.info(
                "GUI rollout start sample=%s group=%s domain=%s example=%s",
                self.sample.index, self.sample.group_index, self.domain, self.example_id,
            )
            async with self.timings.aspan("acquire"):
                lease_id = await self._acquire()

            error_stage = "reset"
            async with self.timings.aspan("reset"):
                reset_obs = await self.env_client.reset(
                    lease_id=lease_id, task_config=self.task_config
                )
                obs = reset_obs
                if self.cfg.wait_after_reset > 0:
                    await asyncio.sleep(self.cfg.wait_after_reset)
                    latest_obs = await self.env_client.get_obs(lease_id)
                    obs = merge_reset_observations(reset_obs, latest_obs)
            save_image(obs["screenshot"], self.result_dir / "step_0.png")

            res.prm = PrmHook.create(self.args, self.state, str(self.result_dir))
            res.prm.deferred = self.phased

            async with self.timings.aspan("turn_loop"):
                res.status, obs = await self._turn_loop(lease_id, obs, res, trace_records)

            if not self.phased:
                async with self.timings.aspan("prm_collect"):
                    await res.prm.collect()

            if res.status != Sample.Status.COMPLETED and not res.terminal_action_sent:
                try:
                    await self.env_client.step(lease_id=lease_id, action="FAIL", sleep_after_execution=0)
                except Exception:
                    logger.debug("Failed to send terminal FAIL action", exc_info=True)

            error_stage = "evaluate"
            async with self.timings.aspan("evaluate"):
                res.eval_score = await self.env_client.evaluate(lease_id=lease_id)
            logger.info(
                "GUI rollout end sample=%s status=%s score=%.4f",
                self.sample.index, res.status.value, res.eval_score,
            )
            self._write_results(res, trace_records)
        except Exception as e:
            res.status = Sample.Status.ABORTED
            res.error_stage = error_stage
            res.error_message = str(e)[:500]
            tb = traceback.format_exc()
            try:
                with open(self.traj_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"Error": str(e), "Traceback": tb}, ensure_ascii=False) + "\n")
            except Exception:
                pass
            print(
                f"[GUI_ROLLOUT_ERROR] sample_index={self.sample.index} "
                f"domain={self.domain} example_id={self.example_id}\n{tb}",
                file=sys.stderr, flush=True,
            )
            logger.exception("GUI rollout failed for sample %s", self.sample.index)
        finally:
            if res.error_stage is not None and res.opd_target_tasks:
                for task in res.opd_target_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*res.opd_target_tasks, return_exceptions=True)
            if lease_id is not None:
                try:
                    async with self.timings.aspan("close"):
                        await self.env_client.close(lease_id=lease_id)
                        _HELD_LEASES.discard(lease_id)  # DEBUG: env released
                        _dump_inflight("release", f"{self.example_id}#{self.sample.index}")
                except Exception:
                    logger.exception("Failed to close env lease: %s", lease_id)
            # Preserve dispatched analyzer results on collection failures after
            # releasing the environment; screenshot files remain available.
            if res.error_stage is not None and (self.sample.metadata or {}).get("collect_analyzer_guidance"):
                await res.prm.collect()

        # Do not hold a scarce physical environment slot while draining the
        # remaining teacher prefills. Most targets have already overlapped with
        # later rollout turns; only their tail latency is paid here.
        if res.error_stage is None and not self.phased:
            async with self.timings.aspan("opd_collect"):
                await self._collect_pipelined_opd_targets(res)

        if self.timings.enabled:
            try:
                with open(self.result_dir / "timings.json", "w", encoding="utf-8") as f:
                    json.dump(self.timings.summary(), f, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to write timings.json")

        return res

    async def collect_deferred_analyzer(self, res: TrajectoryResult) -> None:
        """Run only after all environment trajectories in the batch finish."""
        async with self.timings.aspan("prm_collect"):
            if res.error_stage is None and res.status != Sample.Status.ABORTED:
                res.prm.dispatch_deferred()
            await res.prm.collect()
        # Refresh the persisted analyzer trace written at environment completion.
        trace_path = self.result_dir / "trajectory.json"
        if trace_path.exists():
            with trace_path.open(encoding="utf-8") as f:
                trace = json.load(f)
            trace["reward_trajectory"] = res.prm.reward_trajectory
            with trace_path.open("w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

    async def collect_deferred_teacher(self, res: TrajectoryResult) -> None:
        """Run only after every analyzer request in the batch finishes."""
        if res.error_stage is not None or res.status == Sample.Status.ABORTED:
            return
        if self.fixed_guidance_enabled:
            # Fixed guidance creates its OPD futures during the environment
            # phase; there is no deferred analyzer detail to fan out again.
            async with self.timings.aspan("opd_collect"):
                await self._collect_pipelined_opd_targets(res)
            if self.timings.enabled:
                with (self.result_dir / "timings.json").open("w", encoding="utf-8") as f:
                    json.dump(self.timings.summary(), f, ensure_ascii=False, indent=2)
            return
        details = {
            int(d["step_index"]): d for d in res.prm.step_details
            if isinstance(d, dict) and "step_index" in d
        }
        for snapshot in res.step_snapshots:
            ready = asyncio.get_running_loop().create_future()
            ready.set_result(details.get(int(snapshot["step_idx"]), {}))
            self._schedule_pipelined_opd_target(res, snapshot=snapshot, prm_task=ready)
        async with self.timings.aspan("opd_collect"):
            await self._collect_pipelined_opd_targets(res)
        if self.timings.enabled:
            with (self.result_dir / "timings.json").open("w", encoding="utf-8") as f:
                json.dump(self.timings.summary(), f, ensure_ascii=False, indent=2)

    # --- turn loop ----------------------------------------------------------------

    def _schedule_pipelined_opd_target(
        self,
        res: TrajectoryResult,
        *,
        snapshot: dict[str, Any],
        prm_task: asyncio.Future | None,
    ) -> None:
        """Start q0/q+ prefill as soon as this step's PRM task resolves."""
        if not getattr(self.args, "gui_opd_enable", False):
            return

        if self.fixed_guidance_enabled:
            # Use one deterministic teacher record for every step.  The
            # analyzer's majority labels are set to positive so the existing
            # OPD gate/loss computes unchanged against the fixed privileged
            # context; no analyzer request is made.
            prm_task = asyncio.get_running_loop().create_future()
            prm_task.set_result({
                "status": "ok" if self.fixed_guidance else "missing",
                "guidance": self.fixed_guidance,
                "consistency_majority": 1,
                "effectiveness_majority": 1,
            })
        if prm_task is None:
            return

        step_idx = int(snapshot["step_idx"])

        async def _run() -> None:
            try:
                result = await prm_task
                if not isinstance(result, dict):
                    raise ValueError(
                        f"PRM returned {type(result).__name__}, expected per-step detail dict"
                    )
                detail = dict(result)
                detail["step_index"] = step_idx
                if detail.get("status") != "ok":
                    raise ValueError(f"PRM step status is {detail.get('status', 'missing')}")
                if not str(detail.get("guidance", "") or "").strip():
                    raise ValueError("PRM returned no canonical action guidance")
                if detail.get("consistency_majority") not in (0, 1) or detail.get(
                    "effectiveness_majority"
                ) not in (0, 1):
                    raise ValueError(
                        "PRM majority direction is unavailable or tied: "
                        f"consistency={detail.get('consistency_majority')}, "
                        f"effectiveness={detail.get('effectiveness_majority')}"
                    )
                target = self._build_pipelined_opd_sample(snapshot)

                # Import lazily: plain GRPO rollout workers never need OPD code.
                from reward.gui_opd import attach_gui_opd_target_for_step

                await attach_gui_opd_target_for_step(
                    self.args,
                    self.state,
                    self.agent,
                    target,
                    snapshot,
                    detail,
                )
                res.opd_targets_by_step[step_idx] = target
            except Exception as exc:
                # The final fan-out will turn this into a zero-gate OPD target
                # while retaining any valid trajectory-level GRPO contribution.
                res.opd_target_errors_by_step[step_idx] = str(exc)
                logger.warning(
                    "Pipelined GUI OPD target failed sample=%s step=%d: %s",
                    self.sample.index,
                    step_idx,
                    exc,
                )

        res.opd_target_tasks.append(asyncio.create_task(_run()))

    def _build_pipelined_opd_sample(self, snapshot: dict[str, Any]) -> Sample:
        """Materialize the minimal token-aligned object needed for q0/q+."""
        token_ids, loss_mask, mm_train = self.agent.build_train_data(
            args=self.args,
            state=self.state,
            train_messages=snapshot["train_messages"],
            tool_spec=snapshot.get("tool_spec"),
        )
        suffix = _response_suffix_mask(token_ids, loss_mask)
        if suffix is None:
            raise ValueError("step has no trainable response suffix for OPD")
        _response_start, response_length, child_loss_mask = suffix
        step_idx = int(snapshot["step_idx"])
        return Sample(
            group_index=self.sample.group_index,
            index=self.sample.index,
            tokens=token_ids,
            response=snapshot["response_text"],
            response_length=response_length,
            loss_mask=child_loss_mask,
            multimodal_train_inputs=mm_train,
            metadata={"dynamic_step_index": step_idx},
        )

    async def _collect_pipelined_opd_targets(self, res: TrajectoryResult) -> None:
        """Drain the remaining target work after the trajectory's PRM collection."""
        if not res.opd_target_tasks:
            return
        await asyncio.gather(*res.opd_target_tasks, return_exceptions=True)
        ready = len(res.opd_targets_by_step)
        failed = len(res.opd_target_errors_by_step)
        logger.info(
            "Pipelined GUI OPD targets sample=%s ready=%d failed=%d total=%d",
            self.sample.index,
            ready,
            failed,
            len(res.opd_target_tasks),
        )

    async def _turn_loop(
        self,
        lease_id: str,
        obs: dict[str, Any],
        res: TrajectoryResult,
        trace_records: list[dict[str, Any]],
    ) -> tuple[Sample.Status, dict[str, Any]]:
        """Run up to ``max_steps`` policy turns. Returns (final_status, last_obs)."""
        for step_idx in range(self.cfg.max_steps):
            # Per-step phase breakdown (C1-C5). `measure`/`ameasure` also fold
            # each delta into self.timings totals; no-op when profiling is off.
            st: dict[str, float] = {}

            if (
                str(getattr(self.args, "advantage_estimator", "")).lower() == "gigpo"
                and getattr(self.args, "gigpo_require_a11y_tree", True)
                and not obs.get("accessibility_tree")
            ):
                # A transiently missing tree should not invalidate the whole
                # trajectory. The snapshot keeps anchor_obs=None; GiGPO gives
                # this step a unique missing-anchor key and treats it as a
                # singleton during reward computation.
                logger.warning(
                    "GiGPO step sample=%s step=%d has no accessibility_tree; "
                    "keeping the step as a singleton anchor",
                    self.sample.index,
                    step_idx,
                )

            # NOTE: heartbeat disabled — profiling showed it cost ~11.6s/step
            # (~23% of turn_loop), an unexpectedly heavy cost for a keepalive
            # ping. Skipping it; re-enable if the env server starts expiring
            # leases mid-trajectory.
            # async with self.timings.ameasure("heartbeat", st):
            #     await self.env_client.heartbeat(lease_id)
            st["heartbeat"] = 0.0
            with self.timings.measure("build_policy_messages", st):
                parse_ctx = self.agent.build_policy_messages(instruction=self.instruction, obs=obs)
            policy_messages = parse_ctx["messages"]
            # Evaluation-time teacher guidance is attached to the exact current
            # user turn, matching reward.prm_utils.append_privileged_guidance.
            # The ordinary context is preserved when a task/step has no record.
            if self.guidance_steps:
                policy_messages = inject_step_guidance(policy_messages, self.guidance_steps, step_idx)
            tool_spec = parse_ctx.get("tool_spec")
            # Track latest context as the fallback single-sample loss target.
            res.train_messages_for_loss = policy_messages
            res.tool_spec_for_loss = tool_spec

            async with self.timings.ameasure("sglang_generate", st):
                response, finish_type = await self.agent.generate_with_sglang(
                    args=self.args,
                    state=self.state,
                    messages=policy_messages,
                    sampling_params=self.sampling_params,
                    sampling_seed=((int(self.sample.index or 0) + 1) * 1000003 + step_idx * 9973),
                    tool_spec=tool_spec,
                    timings=self.timings,
                )
            self._log_step(step_idx, finish_type, response)

            attempt = {
                "step_idx": step_idx,
                "finish_type": finish_type,
                "response": response,
                "actions": [],
                "parse_info": {},
            }
            res.generation_attempts.append(attempt)

            if finish_type == "abort":
                attempt["failure_reason"] = "generation_aborted"
                return Sample.Status.ABORTED, obs

            # Capture the pre-action GUI state as the GiGPO anchor: snapshot k is
            # the state the model saw before emitting step k's action, so child k
            # (dynamic_step_index=k) gets this state's accessibility tree. Only the
            # raw a11y text is stored here; normalization/hashing is deferred to the
            # reward stage (reward.gigpo) so the turn loop never blocks on it.
            before_accessibility_tree = obs.get("accessibility_tree")
            self._record_snapshot(
                res, step_idx, policy_messages, tool_spec, response,
                anchor_obs=before_accessibility_tree,
                instruction=self.instruction,
                rollout_weight_version=getattr(self.agent, "last_generation_weight_version", None),
            )

            with self.timings.measure("parse_response", st):
                natural_action, actions, info_dict = self.agent.parse_response(
                    response=response,
                    original_width=int(parse_ctx["original_width"]),
                    original_height=int(parse_ctx["original_height"]),
                    processed_width=int(parse_ctx["processed_width"]),
                    processed_height=int(parse_ctx["processed_height"]),
                )
            attempt["actions"] = list(actions)
            attempt["parse_info"] = info_dict
            self.agent.record_policy_turn(
                action_text=natural_action or "Execute action",
                response=response,
                screenshot_bytes=obs["screenshot"],
            )

            if not actions or actions[0] == "":
                attempt["failure_reason"] = "no_executable_action"
                prm_task = res.prm.submit_step(
                    self.args,
                    instruction=self.instruction,
                    actions_history=list(self.agent.actions),
                    policy_response=response,
                    step_index=step_idx,
                    student_context=policy_messages,
                    executed_actions=[],
                    is_final_step=True,
                )
                self._schedule_pipelined_opd_target(
                    res,
                    snapshot=res.step_snapshots[-1],
                    prm_task=prm_task,
                )
                return Sample.Status.FAILED, obs
            agent_reported_failure = str(actions[0]).upper() == "FAIL"
            if agent_reported_failure:
                attempt["failure_reason"] = "agent_reported_failure"

            serialized_messages = _messages_with_image_paths(
                policy_messages, self.result_dir, step_idx
            )
            async with self.timings.ameasure("env_step", st):
                obs, done, step_executed = await self._execute_actions(
                    lease_id,
                    step_idx,
                    actions,
                    response,
                    info_dict,
                    serialized_messages=serialized_messages,
                    before_accessibility_tree=before_accessibility_tree,
                    instruction=self.instruction,
                )
            executed_actions = list(getattr(self, "_last_executed_actions", []))
            if agent_reported_failure and executed_actions:
                res.terminal_action_sent = True

            if self.timings.enabled:
                step_record = {
                    "sample_index": int(self.sample.index) if self.sample.index is not None else -1,
                    "group_index": int(self.sample.group_index) if self.sample.group_index is not None else -1,
                    "step_idx": step_idx,
                    "heartbeat": st["heartbeat"],
                    "build_msg": st["build_policy_messages"],
                    "sglang": st["sglang_generate"],
                    "parse": st["parse_response"],
                    "env_step": st["env_step"],
                    "step_time": round(sum(st.values()), 4),
                }
                self.timings.steps.append(step_record)
                with open(self.result_dir / "step_times.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            trace_records.append({
                "messages": serialized_messages,
                "response": response,
                "actions": executed_actions,
                "step_idx": step_idx,
                "step_executed": step_executed,
                "done": done,
                # Persist the exact data needed by the step-level expert
                # dataset builder.  The a11y tree is the GiGPO state anchor.
                "instruction": self.instruction,
                "before_screenshot": _absolute_screenshot_path(self.result_dir, step_idx),
                "after_screenshot": _absolute_screenshot_path(self.result_dir, step_idx + 1),
                "before_accessibility_tree": before_accessibility_tree,
                "after_accessibility_tree": obs.get("accessibility_tree"),
            })

            # Dispatch after the post-action screenshot is durable. The OPD
            # callback starts q0/q+ as soon as this individual task resolves;
            # PrmHook still collects the complete trajectory summary at the end.
            prm_task = res.prm.submit_step(
                self.args,
                instruction=self.instruction,
                actions_history=list(self.agent.actions),
                policy_response=response,
                step_index=step_idx,
                student_context=policy_messages,
                executed_actions=executed_actions,
                is_final_step=bool(done or step_idx == self.cfg.max_steps - 1),
            )
            self._schedule_pipelined_opd_target(
                res,
                snapshot=res.step_snapshots[-1],
                prm_task=prm_task,
            )

            if agent_reported_failure:
                return Sample.Status.FAILED, obs

            if done:
                return Sample.Status.COMPLETED, obs

        return Sample.Status.TRUNCATED, obs

    async def _execute_actions(
        self,
        lease_id: str,
        step_idx: int,
        actions: list[Any],
        response: str,
        info_dict: dict[str, Any],
        *,
        serialized_messages: list[dict[str, Any]],
        before_accessibility_tree: str | None = None,
        instruction: str = "",
    ) -> tuple[dict[str, Any], bool, bool]:
        """Execute each action of one turn; returns (last_obs, done, step_executed)."""
        obs: dict[str, Any] = {}
        done = False
        step_executed = False
        executed_actions: list[Any] = []
        current_before_path = _absolute_screenshot_path(self.result_dir, step_idx)
        current_before_tree = before_accessibility_tree
        for action_index, action in enumerate(actions):
            obs, reward, done, info = await self.env_client.step(
                lease_id=lease_id, action=action, sleep_after_execution=self.cfg.sleep_after_execution
            )
            step_executed = True
            executed_actions.append(action)
            step_image_path = self.result_dir / f"step_{step_idx + 1}.png"
            save_image(obs["screenshot"], step_image_path)
            # The canonical step_N image is the final state after the complete
            # model response. Preserve intermediate states separately when one
            # response expands to multiple low-level environment actions.
            action_image_path = step_image_path
            if len(actions) > 1:
                action_image_path = self.result_dir / (
                    f"step_{step_idx + 1}_action_{action_index}.png"
                )
                save_image(obs["screenshot"], action_image_path)
            with open(self.traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_index": action_index,
                    "action": action,
                    "natural_language_action": info_dict.get("action"),
                    "response": response,
                    "messages": serialized_messages,
                    "instruction": instruction,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "before_screenshot": current_before_path,
                    "screenshot_file": str(action_image_path.resolve()),
                    "before_accessibility_tree": current_before_tree,
                    "after_accessibility_tree": obs.get("accessibility_tree"),
                }, ensure_ascii=False))
                f.write("\n")
            current_before_path = str(action_image_path.resolve())
            current_before_tree = obs.get("accessibility_tree")
            if done:
                break
        self._last_executed_actions = executed_actions
        return obs, done, step_executed

    # --- helpers ------------------------------------------------------------------

    def _record_snapshot(
        self,
        res: TrajectoryResult,
        step_idx: int,
        policy_messages: list[dict[str, Any]],
        tool_spec: dict[str, Any] | None,
        response: str,
        *,
        anchor_obs: str | None = None,
        instruction: str | None = None,
        rollout_weight_version: Any = None,
    ) -> None:
        """Record one trainable step: history assistant turns masked off, current on.

        ``anchor_obs`` is the pre-action accessibility tree (the GiGPO anchor
        state for this step); ``instruction`` is the task goal, stored so the
        reward stage can fall back to it when a11y is unavailable.
        """
        step_train_messages = copy.deepcopy(policy_messages)
        for msg in step_train_messages:
            if msg.get("role") == "assistant":
                msg["step_loss_mask"] = 0
        step_train_messages.append({"role": "assistant", "content": response, "step_loss_mask": 1})

        res.train_messages_for_loss = step_train_messages
        res.tool_spec_for_loss = tool_spec
        res.assistant_responses.append(response)
        res.step_snapshots.append({
            "step_idx": step_idx,
            "train_messages": step_train_messages,
            "response_text": response,
            "tool_spec": tool_spec,
            "anchor_obs": anchor_obs,
            "instruction": instruction,
            "rollout_weight_version": rollout_weight_version,
        })

    def _log_step(self, step_idx: int, finish_type: str, response: str) -> None:
        preview = response
        n = self.cfg.response_preview_chars
        if n > 0 and len(response) > n:
            preview = response[:n] + "...(truncated)"
        logger.info(
            "step sample=%s step=%s finish=%s response=%s",
            self.sample.index, step_idx, finish_type, preview,
        )

    def _write_results(self, res: TrajectoryResult, trace_records: list[dict[str, Any]]) -> None:
        with open(self.result_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"{res.eval_score}\n")
        guidance_steps = [
            {"step_index": int(detail.get("step_index", i)), "guidance": str(detail.get("guidance", "") or "")}
            for i, detail in enumerate(res.prm.step_details)
            if isinstance(detail, dict)
        ]
        with open(self.result_dir / "trajectory.json", "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"result": res.eval_score},
                "trajectory": trace_records,
                "generation_attempts": res.generation_attempts,
                "reward_trajectory": res.prm.reward_trajectory,
                **({"analyzer_guidance_steps": guidance_steps}
                   if (self.sample.metadata or {}).get("collect_analyzer_guidance") else {}),
            }, f, ensure_ascii=False, indent=2)
