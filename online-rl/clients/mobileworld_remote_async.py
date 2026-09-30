"""MobileWorld async remote client — cluster /v1/sessions, asyncio.

Self-contained adaptation of env_infra's ``mobile_remote_async`` (no ``cluster.*``
or ``mobile_world`` imports), shaped to the rollout's reset/step/evaluate
interface — the same pattern ``osworld_remote_async`` follows for OSWorld.

The mobile agent emits protocol-native device actions
(``{"kind": "device", "type": ..., "payload": ...}``), so :meth:`step` forwards
them unchanged; the rollout's terminal control tokens (``DONE``/``FAIL``/``WAIT``)
map to ``kind="control"``. Observations carry the screenshot decoded to bytes,
matching what the policy agent consumes.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional, Tuple

from .async_session_client import AsyncClusterSessionClient

logger = logging.getLogger("cluster.client.mobileworld_remote_async")


# -- format conversion (pure) ------------------------------------------------

def to_mobile_obs(obs_data: Dict[str, Any], instruction: Optional[str] = None) -> Dict[str, Any]:
    """Cluster observation → mobile dict (screenshot decoded to bytes)."""
    modalities = obs_data.get("modalities", obs_data)
    screenshot_b64 = modalities.get("screenshot")
    obs: Dict[str, Any] = {"screenshot": base64.b64decode(screenshot_b64) if screenshot_b64 else None}
    if instruction is not None:
        obs["instruction"] = instruction
    return obs


def to_cluster_action(action) -> Dict[str, Any]:
    """Mobile agent output → cluster ``Action`` dict.

    The agent already speaks the protocol, so a device action dict passes through
    untouched; the rollout's terminal token strings map to ``kind="control"``.
    """
    if isinstance(action, dict):
        return action
    if isinstance(action, str) and action.strip() in ("DONE", "FAIL", "WAIT"):
        return {"kind": "control", "type": action.strip(), "payload": {}}
    raise ValueError(f"unsupported mobile action: {action!r}")


class MobileWorldAsyncRemoteClient(AsyncClusterSessionClient):
    """Async MobileWorld env client backed by the cluster session API.

    Mirror of :class:`OSWorldAsyncRemoteClient` for MobileWorld::

        await env.acquire()
        await env.reset(task_config) → dict obs      # task_config = {"task_name": ...}
        await env.step(action)       → (obs, reward, done, info)
        await env.evaluate()         → float
    """

    def __init__(self, cluster_url: str | None = None, **kwargs):
        # SessionGuiEnvClient supplies this OSWorld-specific option uniformly.
        kwargs.pop("action_space", None)
        super().__init__(cluster_url=cluster_url, runtime="mobileworld", **kwargs)
        self.instruction: Optional[str] = None

    async def reset(self, task_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        task_config = task_config or {}
        self.instruction = task_config.get("instruction")
        # task_config carries {"task_name": ...}; the base forwards it as the reset
        # task_payload, which the node's MobileWorld adapter reads.
        obs_data = await super().reset(task_config)
        return to_mobile_obs(obs_data, self.instruction)

    async def get_obs(self) -> Dict[str, Any]:
        return to_mobile_obs(await self.observe(), self.instruction)

    async def step(self, action, pause: float | None = None) -> Tuple[Dict[str, Any], float, bool, Dict]:
        data = await super().step(to_cluster_action(action), pause=pause)
        obs = to_mobile_obs(data.get("observation", {}), self.instruction)
        return obs, float(data.get("reward", 0.0)), bool(data.get("done", False)), data.get("info", {}) or {}

    async def evaluate(self) -> float:
        data = await super().evaluate()
        return float(data.get("score", 0.0))
