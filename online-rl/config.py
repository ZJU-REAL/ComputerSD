"""Centralized reading of ``GUI_*`` environment variables.

Single source of truth for env-var names, defaults, and type coercion. Modules
should import the small accessor functions here instead of calling
``os.getenv`` inline, so the full configuration surface is discoverable in one
place.

Lightweight by design: plain accessor functions, no dataclasses. Each function
reads the env var *at call time* (not import time), so tests and per-process
overrides via ``os.environ`` take effect without reimporting.

Scope: covers the variables used by ``env_client.py``,
``data/gui_data_source.py``, ``agents/qwen3vl_agent.py`` and the rollout
entrypoint ``rollout/partial_async_rollout_gui.py`` (the latter via
:class:`EpisodeConfig` and the rollout/episode accessors below).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- typed primitive readers --------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None and raw != "" else default


def get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None and raw != "" else default


def get_flag_raw(name: str, default: str) -> str:
    """Return a normalized (lower/stripped) tri-state flag string.

    Use this when a flag has three states (e.g. ``auto`` / on / off) and the
    caller needs to distinguish them. For plain on/off use :func:`get_bool`.
    """
    return os.getenv(name, default).strip().lower()


def get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {_TRUE | _FALSE}, got {raw!r}")


# --- env_client.py ------------------------------------------------------------

def env_http_max_retries() -> int:
    """Retries for env-control HTTP calls. Kept small to avoid long hangs."""
    return get_int("GUI_ENV_HTTP_MAX_RETRIES", 10)


def evaluate_max_retries() -> int:
    return get_int("GUI_EVALUATE_MAX_RETRIES", 6)


def env_direct_node_data_plane() -> str:
    """Tri-state: ``auto`` (default) / on (1,true,..) / off (0,false,..)."""
    return get_flag_raw("GUI_ENV_DIRECT_NODE_DATA_PLANE", "auto")


# --- data/gui_data_source.py --------------------------------------------------

def test_config_base_dir(default: str) -> str:
    """Base dir holding evaluation_examples; caller supplies the package-relative default."""
    return get_str("GUI_TEST_CONFIG_BASE_DIR", default)


def train_meta_path(default: str) -> str:
    return get_str("GUI_TRAIN_META_PATH", default)


def mw_task_list_path(default: str) -> str:
    """Offline MobileWorld task-list JSON (generated once on the env node)."""
    return get_str("GUI_MW_TASK_LIST_PATH", default)


def mw_enable_mcp() -> bool:
    return get_bool("GUI_MW_ENABLE_MCP", False)


def mw_enable_user_interaction() -> bool:
    return get_bool("GUI_MW_ENABLE_USER_INTERACTION", False)


def agentnet_jsonl_path() -> str | None:
    return os.getenv("AGENTNET_JSONL_PATH")


def agentnet_image_dir() -> str | None:
    return os.getenv("AGENTNET_IMAGE_DIR")


# --- rollout / episode --------------------------------------------------------

def env_server_url() -> str:
    return get_str("GUI_ENV_SERVER_URL", "http://localhost:19000")


def env_runtime() -> str:
    """Cluster ``runtime``/``world_id`` selector for /v1/sessions acquire."""
    return get_str("GUI_ENV_RUNTIME", "osworld")


def env_mode() -> str:
    """Session ``mode`` for cluster bookkeeping/quota (train vs eval)."""
    return get_str("GUI_ENV_MODE", "train")


def agent_class_path() -> str | None:
    return os.getenv("GUI_AGENT_CLASS_PATH")


def reward_agent_class_path() -> str | None:
    return os.getenv("GUI_REWARD_AGENT_CLASS_PATH")


def action_space() -> str:
    return get_str("GUI_ACTION_SPACE", "pyautogui")


def observation_type() -> str:
    return get_str("GUI_OBSERVATION_TYPE", "screenshot")


def coordinate_type() -> str:
    return get_str("GUI_COORDINATE_TYPE", "relative")


# --- GiGPO (reward/gigpo.py + reward/reward_func.py gigpo path) ---------------

def gigpo_enabled_hint() -> bool:
    """Whether to capture the a11y anchor during rollout.

    Pure hint for the rollout's env-config injection (forcing
    ``require_a11y_tree=True``) — the actual advantage-estimator dispatch reads
    ``args.advantage_estimator`` in reward_func. Keeping it separate avoids a
    hard dependency on slime args at rollout time. Flip via
    ``GUI_GIGPO_REQUIRE_A11Y`` (default on when ``advantage_estimator=gigpo``).
    """
    return get_bool("GUI_GIGPO_REQUIRE_A11Y", True)


def gigpo_step_advantage_w() -> float:
    """Weight on analyzer-only discounted return in the composite step reward."""
    return get_float("GIGPO_STEP_ADVANTAGE_W", 1.0)


def gigpo_gamma() -> float:
    """Discount factor for within-trajectory step returns (Eq. 5)."""
    return get_float("GIGPO_GAMMA", 0.95)


def gigpo_mode() -> str:
    """``mean_norm`` (subtract group mean) or ``mean_std_norm`` (also / std)."""
    return get_str("GIGPO_MODE", "mean_norm")


def gigpo_use_similarity() -> bool:
    """Cluster by anchor-string similarity instead of exact hash."""
    return get_bool("GIGPO_USE_SIMILARITY", False)


def gigpo_similarity_thresh() -> float:
    return get_float("GIGPO_SIMILARITY_THRESH", 0.95)


def gigpo_platform() -> str:
    """Platform namespace for a11y parsing (ubuntu/windows)."""
    return get_str("GIGPO_PLATFORM", "ubuntu")


def user_id() -> str:
    return get_str("GUI_USER_ID", "anonymous")


def job_id() -> str:
    return get_str("GUI_JOB_ID", "")


def gui_profile() -> bool:
    """Enable per-trajectory wall-clock profiling (writes timings.json).

    On by default. Set ``GUI_PROFILE=0`` to disable, in which case the Timings
    span context managers become pure no-ops (see rollout/_timing.py), so there
    is zero measurement overhead.
    """
    return get_bool("GUI_PROFILE", True)


# --- sglang weight-update abort retry ---------------------------------------
def abort_retry_max() -> int:
    """Maximum retries for a request aborted by ``update_weights``."""
    return get_int("GUI_ABORT_RETRY_MAX", 3)


def abort_retry_backoff() -> float:
    """Backoff (seconds) before retrying an update-weights abort."""
    return get_float("GUI_ABORT_RETRY_BACKOFF", 0.5)


# --- episode (rollout) config -------------------------------------------------

def _arg_or(args: Any, attr: str, fallback):
    """Return ``args.attr`` if set (not None), else ``fallback``."""
    val = getattr(args, attr, None)
    return val if val is not None else fallback


@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int
    sleep_after_execution: float
    wait_after_reset: float
    max_image_history_length: int
    allocate_retries: int
    allocate_backoff_seconds: float
    response_preview_chars: int

    @classmethod
    def resolve(cls, args: Any, *, evaluation: bool) -> "EpisodeConfig":
        """Resolve all per-episode knobs, honoring train/eval split and env fallbacks."""
        rollout_max_steps = _arg_or(args, "gui_max_steps", get_int("GUI_MAX_STEPS", 15))
        eval_max_steps = getattr(args, "gui_eval_max_steps", None)
        max_steps = int(eval_max_steps if evaluation and eval_max_steps is not None else rollout_max_steps)

        rollout_sleep = _arg_or(args, "gui_sleep_after_execution", get_float("GUI_SLEEP_AFTER_EXECUTION", 0.0))
        eval_sleep = getattr(args, "gui_eval_sleep_after_execution", None)
        sleep_after_execution = float(eval_sleep if evaluation and eval_sleep is not None else rollout_sleep)

        rollout_wait = _arg_or(args, "gui_wait_after_reset", get_float("GUI_WAIT_AFTER_RESET", 0.0))
        eval_wait = getattr(args, "gui_eval_wait_after_reset", None)
        wait_after_reset = float(eval_wait if evaluation and eval_wait is not None else rollout_wait)

        max_image_history_length = int(
            _arg_or(args, "gui_max_image_history_length", get_int("GUI_MAX_IMAGE_HISTORY_LENGTH", max_steps))
        )

        return cls(
            max_steps=max_steps,
            sleep_after_execution=sleep_after_execution,
            wait_after_reset=wait_after_reset,
            max_image_history_length=max_image_history_length,
            allocate_retries=get_int("GUI_ALLOCATE_RETRIES", 10),
            allocate_backoff_seconds=get_float("GUI_ALLOCATE_BACKOFF_SECONDS", 2.0),
            response_preview_chars=get_int("GUI_LOG_RESPONSE_PREVIEW_CHARS", 0),
        )
