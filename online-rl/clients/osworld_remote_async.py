"""OSWorld async remote client — DesktopEnv-compatible, asyncio.

Self-contained copy from env_infra (no ``cluster.*`` imports): the obs/action
conversion helpers that the original imported from ``osworld_remote`` are inlined
here, so the async path carries no dependency on the sync client.

Same DesktopEnv-style obs/action shapes (so an async episode loop sees identical
observations), backed by AsyncClusterSessionClient. Lets one event loop drive
many concurrent OSWorld episodes in a single process.
"""
from __future__ import annotations

import base64
import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple

from .async_session_client import AsyncClusterSessionClient

logger = logging.getLogger("cluster.client.osworld_remote_async")

OBSERVATION_RECOVERY_TIMEOUT = float(
    os.environ.get("GUI_OBSERVATION_RECOVERY_TIMEOUT", "60")
)
_SYSTEM_A11Y_APPLICATION_NAMES = {
    "evolution-alarm-notify",
    "gjs",
    "gnome-shell",
    "org.gnome.software",
    "xdg-desktop-portal-gtk",
}
_SYSTEM_A11Y_APPLICATION_PREFIXES = ("gsd-", "ibus-")
_A11Y_LARGE_SUBTREE_DROP_MIN_NODES = 20
_A11Y_LARGE_SUBTREE_DROP_RATIO = 1.5


# -- format conversion (pure; inlined from osworld_remote) -------------------

def to_osworld_obs(obs_data: Dict[str, Any], instruction: Optional[str] = None) -> Dict[str, Any]:
    """Cluster observation → OSWorld dict format (screenshot decoded to bytes)."""
    modalities = obs_data.get("modalities", obs_data)
    screenshot_b64 = modalities.get("screenshot")
    screenshot_bytes = base64.b64decode(screenshot_b64) if screenshot_b64 else None
    obs = {
        "screenshot": screenshot_bytes,
        "accessibility_tree": modalities.get("accessibility_tree"),
        "terminal": modalities.get("terminal"),
    }
    if instruction is not None:
        obs["instruction"] = instruction
    return obs


def _is_system_a11y_application(name: str) -> bool:
    normalized = " ".join(str(name or "").split()).casefold()
    return normalized in _SYSTEM_A11Y_APPLICATION_NAMES or normalized.startswith(
        _SYSTEM_A11Y_APPLICATION_PREFIXES
    )


def _a11y_application_subtree_sizes(a11y_xml: Any) -> Optional[Dict[str, int]]:
    """Return substantive non-system application subtree sizes, or None if invalid."""
    if not isinstance(a11y_xml, str) or not a11y_xml.strip():
        return None
    try:
        root = ET.fromstring(a11y_xml)
    except ET.ParseError:
        return None

    sizes: Dict[str, int] = {}
    for application in root:
        tag = application.tag.rsplit("}", 1)[-1]
        name = " ".join(application.get("name", "").split()).casefold()
        if tag != "application" or _is_system_a11y_application(name):
            continue
        nodes = list(application.iter())
        has_content_root = any(
            node.tag.rsplit("}", 1)[-1] == "frame"
            or node.tag.rsplit("}", 1)[-1].startswith("document")
            for node in nodes
        )
        if has_content_root:
            key = name or "<unnamed-application>"
            sizes[key] = sizes.get(key, 0) + len(nodes)
    return sizes


def merge_reset_observations(
    earlier: Dict[str, Any],
    latest: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep the latest modalities, but recover a clearly more complete reset a11y."""
    earlier_tree = earlier.get("accessibility_tree")
    latest_tree = latest.get("accessibility_tree")
    earlier_sizes = _a11y_application_subtree_sizes(earlier_tree)
    latest_sizes = _a11y_application_subtree_sizes(latest_tree)
    use_earlier = False
    reason = ""

    if earlier_sizes is not None and latest_sizes is None:
        use_earlier = True
        reason = "latest a11y is missing or malformed"
    elif earlier_sizes is not None and latest_sizes is not None:
        earlier_apps = set(earlier_sizes)
        latest_apps = set(latest_sizes)
        if earlier_apps > latest_apps:
            use_earlier = True
            reason = f"foreground application subtree disappeared: {sorted(earlier_apps - latest_apps)}"
        elif earlier_apps == latest_apps and earlier_apps:
            earlier_nodes = sum(earlier_sizes.values())
            latest_nodes = sum(latest_sizes.values())
            if (
                earlier_nodes - latest_nodes >= _A11Y_LARGE_SUBTREE_DROP_MIN_NODES
                and earlier_nodes >= latest_nodes * _A11Y_LARGE_SUBTREE_DROP_RATIO
            ):
                use_earlier = True
                reason = f"application subtrees collapsed from {earlier_nodes} to {latest_nodes} nodes"

    merged = dict(latest)
    if use_earlier:
        merged["accessibility_tree"] = earlier_tree
        logger.warning("Using the earlier reset a11y because %s", reason)
    return merged


async def _obs_with_screenshot(
    client: AsyncClusterSessionClient,
    obs_data: Dict[str, Any],
    instruction: Optional[str],
    *,
    source: str,
) -> Dict[str, Any]:
    """Convert an observation and recover once when the screenshot is absent.

    A VM screenshot endpoint can transiently fail while the rest of the
    observation envelope is still returned.  Treating that partial response as
    a valid OSWorld observation makes the rollout fail later in ``save_image``
    or the vision prompt with an opaque ``None``/bytes error.  A bounded
    follow-up observe is safe and keeps the normal path unchanged; if it also
    lacks a screenshot, surface a contextual error so the trajectory cleanup
    path can release the session.
    """
    obs = to_osworld_obs(obs_data, instruction)
    if obs.get("screenshot") is not None:
        return obs

    logger.warning("Observation from %s has no screenshot; retrying observe once", source)
    try:
        recovered_data = await asyncio.wait_for(
            client.observe(), timeout=OBSERVATION_RECOVERY_TIMEOUT
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"OSWorld observation recovery timed out ({source}) after "
            f"{OBSERVATION_RECOVERY_TIMEOUT:.0f}s"
        ) from exc
    recovered = to_osworld_obs(recovered_data, instruction)
    if recovered.get("screenshot") is None:
        raise RuntimeError(
            f"OSWorld observation unavailable after recovery ({source}): screenshot is missing"
        )
    return recovered


def to_cluster_action(action, action_space: str = "pyautogui") -> Dict[str, Any]:
    """OSWorld action (pyautogui str / computer_13 dict / control token) → cluster Action."""
    if isinstance(action, str):
        action_stripped = action.strip()
        if action_stripped in ("DONE", "FAIL", "WAIT"):
            return {"kind": "control", "type": action_stripped, "payload": {}}
        return {"kind": "gui", "type": "pyautogui", "payload": {"command": action}}
    elif isinstance(action, dict):
        action_type = action.get("action_type", action_space)
        return {"kind": "gui", "type": action_type, "payload": action}
    return {"kind": "gui", "type": "pyautogui", "payload": {"command": str(action)}}


class _NoOpRecorder:
    """No-op controller — cluster doesn't support VNC recording."""

    def start_recording(self) -> None:
        pass

    def end_recording(self, path: str) -> None:
        pass

    def run_bash_script(self, script: str, timeout: int = 30, working_dir: str | None = None):
        logger.warning("run_bash_script is not supported on OSWorldRemoteClient")
        return ""


class OSWorldAsyncRemoteClient(AsyncClusterSessionClient):
    """Async DesktopEnv-compatible env client backed by the cluster session API.

    Async mirror of OSWorldRemoteClient. Methods are coroutines:
        await env.acquire()                       # or `async with`
        await env.reset(task_config) → dict obs
        await env.get_obs() → dict
        await env.step(action) → (obs, reward, done, info)
        await env.evaluate() → float
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        action_space: str = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        **kwargs,
    ):
        super().__init__(cluster_url=cluster_url, runtime="osworld", **kwargs)
        self.action_space = action_space
        self.screen_width, self.screen_height = screen_size
        self.instruction: Optional[str] = None
        self.is_environment_used = False
        self.task_id: Optional[str] = None
        self.controller = _NoOpRecorder()
        self._traj_no = -1
        self._step_no = 0

    async def reset(self, task_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        self._traj_no += 1
        self._step_no = 0
        self.is_environment_used = False
        task_config = task_config or {}
        self.task_id = task_config.get("id")
        self.instruction = task_config.get("instruction")
        obs_data = await super().reset(task_config)
        return await _obs_with_screenshot(self, obs_data, self.instruction, source="reset")

    async def get_obs(self) -> Dict[str, Any]:
        obs_data = await self.observe()
        return await _obs_with_screenshot(self, obs_data, self.instruction, source="observe")

    async def step(self, action, pause: float | None = None) -> Tuple[Dict[str, Any], float, bool, Dict]:
        # Node settles server-side and returns the post-settle screenshot in this
        # one response — no client-side sleep / second observe (see sync client).
        # `pause` (sleep_after_execution) is forwarded so the node honors the
        # rollout's settle time instead of the adapter's hardcoded default.
        self._step_no += 1
        self.is_environment_used = True
        typed_action = to_cluster_action(action, self.action_space)
        data = await super().step(typed_action, pause=pause)
        obs = await _obs_with_screenshot(
            self, data.get("observation", {}), self.instruction, source="step"
        )
        done = bool(data.get("done", False))
        reward = float(data.get("reward", 0))
        info = data.get("info", {}) or {}
        return obs, reward, done, info

    async def evaluate(self) -> float:
        data = await super().evaluate()
        return float(data.get("score", 0.0))
