"""Shared GUI rollout helpers.

Extracted verbatim from ``rollout/partial_async_rollout_gui.py`` so that both the
legacy single-loop path and the new ``rollout_fast/`` multiprocess path can reuse
the exact same task-resolution / result-dir / agent-creation / timing helpers
without duplicating logic.

Logic is intentionally byte-for-byte identical to the original definitions; only
the home of the functions changed. Names keep their leading underscore so callers
that imported them by name need no rename.

NOTE on ``__file__``-relative defaults: ``_sample_task_info`` derives the default
``GUI_TEST_CONFIG_BASE_DIR`` from this module's location. The original lived in
``gui-rl/rollout/`` (``parent.parent`` -> ``gui-rl/``); this module lives in
``gui-rl/utils/`` (also ``parent.parent`` -> ``gui-rl/``), so the default resolves
to the same ``gui-rl/`` package root.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from slime.utils.misc import load_function
from slime.utils.types import Sample

from utils.utils import load_task_config

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"
_ANSI_COLORS = {
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "none": "",
}


def _gui_log(message: str, *args: Any) -> None:
    color_name = os.getenv("GUI_LOG_COLOR", "yellow").strip().lower()
    color_code = _ANSI_COLORS.get(color_name, _ANSI_COLORS["yellow"])
    prefix = "[GUI]"
    if color_code:
        prefix = f"{color_code}{prefix}{_ANSI_RESET}"
    logger.info(f"{prefix} {message}", *args)


@lru_cache(maxsize=1)
def _get_gui_trajectory_semaphore() -> asyncio.Semaphore:
    """Cap the number of trajectories concurrently holding a GUI env session.

    This is INDEPENDENT from the sglang request concurrency
    (``sglang_server_concurrency``). One trajectory issues many sglang calls
    (one per step) but holds exactly one env session for its whole lifetime, so
    env-side load is bounded by the trajectory count, not the request count.

    Critical under fully-async rollout: the background worker keeps a fixed pool
    of in-flight trajectories that, without this gate, would each open an env
    session and overwhelm the remote env cluster. ``GUI_TRAJECTORY_CONCURRENCY``
    sets the cap (falling back to ``GUI_POOL_MAX_ENVS``). The semaphore is a
    process-level singleton (lru_cache) shared across all trajectories in this
    rollout worker process.
    """
    max_envs = max(1, int(os.getenv("GUI_POOL_MAX_ENVS", "4")))
    concurrency = max(1, int(os.getenv("GUI_TRAJECTORY_CONCURRENCY", str(max_envs))))
    # Log the resolved cap once per process. Under multi-process rollout each
    # worker should see the SLICED value (global / M); if this prints the global
    # value in a worker, the runtime_env override didn't take effect and env
    # sessions will exceed the global budget (M x global).
    logger.info(
        "GUI trajectory semaphore cap=%d (pid=%d, worker=%s, GUI_TRAJECTORY_CONCURRENCY=%s)",
        concurrency, os.getpid(), os.getenv("_IN_TRAJECTORY_WORKER", "0"),
        os.getenv("GUI_TRAJECTORY_CONCURRENCY"),
    )
    return asyncio.Semaphore(concurrency)


@lru_cache(maxsize=16)
def _load_meta_pairs(meta_path: str) -> list[tuple[str, str]]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    pairs: list[tuple[str, str]] = []
    for domain, examples in meta.items():
        for example_id in examples:
            pairs.append((str(domain), str(example_id)))

    shuffle_seed_str = os.getenv("GUI_TASK_SHUFFLE_SEED", "42")
    if shuffle_seed_str.strip().lower() not in {"", "none", "-1"}:
        import random
        rng = random.Random(int(shuffle_seed_str))
        rng.shuffle(pairs)
        _gui_log("Shuffled %d task pairs with seed=%s", len(pairs), shuffle_seed_str)

    return pairs


def _sample_task_info(sample: Sample, evaluation: bool = False) -> tuple[str, dict[str, Any] | None, str, str]:
    metadata = sample.metadata or {}
    instruction = metadata.get("instruction")
    task_config = metadata.get("task_config")
    domain = metadata.get("domain", "default")
    example_id = metadata.get("example_id", f"sample_{sample.index if sample.index is not None else uuid.uuid4().hex[:8]}")

    if not instruction and isinstance(sample.prompt, str):
        prompt = sample.prompt.strip()
        if prompt.startswith("{") and prompt.endswith("}"):
            try:
                obj = json.loads(prompt)
                instruction = obj.get("instruction") or instruction
                task_config = obj.get("task_config") or task_config
                domain = obj.get("domain", domain)
                example_id = obj.get("example_id", example_id)
            except Exception:
                instruction = prompt
        else:
            instruction = prompt

    if task_config is None:
        base_dir = os.getenv(
            "GUI_TEST_CONFIG_BASE_DIR",
            # This module lives in gui-rl/utils/, so parent.parent points back
            # to the gui-rl/ package root where evaluation_examples/ lives.
            str(Path(__file__).resolve().parent.parent / "evaluation_examples"),
        )
        default_meta = "test_nogdrive.json" if evaluation else "train_nochrome.json"
        env_meta_key = "GUI_EVAL_META_PATH" if evaluation else "GUI_TRAIN_META_PATH"
        meta_path = os.getenv(env_meta_key, str(Path(base_dir) / default_meta))

        if isinstance(sample.prompt, str) and "/" in sample.prompt:
            maybe_domain, maybe_example_id = sample.prompt.split("/", 1)
            domain = maybe_domain
            example_id = maybe_example_id
        elif isinstance(sample.prompt, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", sample.prompt):
            example_id = sample.prompt
        else:
            try:
                pairs = _load_meta_pairs(meta_path)
                sample_key = sample.group_index if sample.group_index is not None else sample.index or 0
                if pairs:
                    domain, example_id = pairs[int(sample_key) % len(pairs)]
            except Exception:
                logger.exception("Failed to resolve task pair from meta file: %s", meta_path)

        try:
            task_config = load_task_config(base_dir, domain, example_id)
            if not instruction:
                instruction = str(task_config.get("instruction", ""))
        except Exception:
            logger.exception("Failed to load task config for %s/%s", domain, example_id)

    return instruction or "", task_config, str(domain), str(example_id)


def _build_result_dir(args: Any, domain: str, example_id: str, sample: Sample, evaluation: bool = False) -> Path:
    base = Path(os.getenv("GUI_RESULT_DIR", "./results"))
    action_space = os.getenv("GUI_ACTION_SPACE", "pyautogui")
    observation_type = os.getenv("GUI_OBSERVATION_TYPE", "screenshot")
    model_name = getattr(args, "hf_checkpoint", "gui-policy-model")
    model_tag = re.sub(r"[\\/]+", "_", str(model_name)).lstrip("_")
    # IMPORTANT: each sample must have an isolated output dir.
    # Using group_index alone causes all samples in the same prompt group to
    # write into one traj.jsonl/step_*.png and corrupt each other's traces.
    if sample.index is not None:
        run_idx = f"s{sample.index}"
    elif sample.group_index is not None:
        run_idx = f"g{sample.group_index}"
    else:
        run_idx = f"u{uuid.uuid4().hex[:8]}"
    # Train and eval reuse the same sample.index space (0,1,2,...), so without a
    # split an eval trajectory and a train trajectory for the same example/index
    # would share a dir — and per-sample clearing would delete each other's
    # results. Put each under its own "train/" or "eval/" subtree (same level).
    split = "eval" if evaluation else "train"
    out = base / action_space / observation_type / model_tag / split / domain / example_id / run_idx
    out.mkdir(parents=True, exist_ok=True)
    return out


def _clear_sample_result_dir(result_dir: Path) -> None:
    """Clear THIS sample's own ``s{index}`` result dir before the trajectory runs.

    Per-sample (not per-example): each trajectory wipes only its own dir, so it is
    safe regardless of how trajectories are routed across worker processes — no
    cross-process clobbering. The dir is isolated per sample.index and
    ``GUI_RESULT_DIR`` is per-run (cleared at launch), so there is no stale-result
    concern from clearing at this granularity.
    """
    if os.getenv("GUI_CLEAR_TASK_RESULT_ON_START", "1").strip().lower() in {"0", "false", "no"}:
        return
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def _create_gui_agent(args: Any, *, max_steps: int, max_image_history_length: int, result_dir: Path):
    agent_cls_path = getattr(args, "gui_agent_class_path", None) or os.getenv("GUI_AGENT_CLASS_PATH")
    if not agent_cls_path:
        raise RuntimeError("GUI_AGENT_CLASS_PATH is required for GUI rollout framework.")
    agent_cls = load_function(agent_cls_path)
    max_history_turns = getattr(args, "gui_max_history_turns", None)
    if os.getenv("GUI_MAX_HISTORY_TURNS"):
        max_history_turns = int(os.environ["GUI_MAX_HISTORY_TURNS"])
    resize_factor = getattr(args, "gui_resize_factor", None)
    if os.getenv("GUI_RESIZE_FACTOR"):
        resize_factor = int(os.environ["GUI_RESIZE_FACTOR"])
    prompt_style = os.getenv("GUI_PROMPT_STYLE") or getattr(args, "gui_prompt_style", None)
    return agent_cls(
        model=getattr(args, "hf_checkpoint", "gui-policy-model"),
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        max_history_turns=max_history_turns,
        resize_factor=resize_factor,
        prompt_style=prompt_style,
        action_space=os.getenv("GUI_ACTION_SPACE", "pyautogui"),
        observation_type=os.getenv("GUI_OBSERVATION_TYPE", "screenshot"),
        coordinate_type=os.getenv("GUI_COORDINATE_TYPE", "relative"),
        example_result_dir=str(result_dir),
    )


def _attach_gui_timings(sample: Sample, ext: dict[str, float], ep: Any, profile: bool) -> None:
    """Merge A-group external timings with the episode's internal span summary
    onto ``sample.metadata['gui_timings']``. No-op when profiling is off.

    ``external`` holds whole-second spans (sem_wait/setup/episode_run/...);
    ``internal`` is ep.timings.summary() with the B/C per-stage totals. Note
    external.episode_run should ~match the sum of internal acquire/reset/
    turn_loop/evaluate/close — a built-in sanity check.
    """
    if not profile:
        return
    sample.metadata = sample.metadata or {}
    internal = ep.timings.summary() if getattr(ep, "timings", None) is not None else {}
    sample.metadata["gui_timings"] = {"external": ext, "internal": internal}
