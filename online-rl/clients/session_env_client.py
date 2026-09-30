"""Lease-based GUI env client backed by the cluster /v1/sessions protocol.

Drop-in alternative to :class:`env_client.GuiEnvClient`. It exposes the exact
lease-based surface that ``rollout/trajectory.py`` already calls
(``allocate`` / ``reset`` / ``get_obs`` / ``step`` / ``evaluate`` / ``close`` /
``heartbeat``), so swapping it in is a one-line import change in the rollout —
no changes to the turn loop.

Internally it adapts that lease model onto the (self-contained) session clients
copied from env_infra under this package:

- ``GuiEnvClient``: one client object owns *many* leases; every call carries a
  ``lease_id`` and routes by it.
- :class:`clients.osworld_remote_async.OSWorldAsyncRemoteClient`: one client
  object IS one session (no lease/episode id on calls); concurrency comes from
  holding *many* client objects on one event loop.

The bridge: :meth:`allocate` acquires a fresh ``OSWorldAsyncRemoteClient``
session and mints a ``lease_id`` mapped to it; every later call looks the session
up by ``lease_id``. ``OSWorldAsyncRemoteClient`` already returns OSWorld-shaped
obs (``{"screenshot": <bytes>, ...}``) and accepts the same pyautogui / control
actions (``DONE``/``FAIL``/``WAIT``) the policy emits, so obs/action need no
extra conversion here. Heartbeats are automatic (the session client runs a
background keepalive thread), so :meth:`heartbeat` is a no-op.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import config

from .async_session_client import AsyncClusterSessionClient
from .mobileworld_remote_async import MobileWorldAsyncRemoteClient
from .osworld_remote_async import OSWorldAsyncRemoteClient

logger = logging.getLogger("gui.session_env_client")

# GUI_ENV_RUNTIME -> session client class. Add a row per world; unknown runtimes
# fall back to the OSWorld client.
_CLIENT_BY_RUNTIME = {
    "osworld": OSWorldAsyncRemoteClient,
    "mobileworld": MobileWorldAsyncRemoteClient,
}


class SessionGuiEnvClient:
    """Lease-keyed adapter over per-session ``OSWorldAsyncRemoteClient`` objects.

    API-compatible with :class:`env_client.GuiEnvClient`: one instance is shared
    across all trajectories of a process, and every method is keyed by
    ``lease_id``. Each ``allocate`` acquires its own underlying session client.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.runtime = config.env_runtime()
        self.action_space = config.action_space()
        # lease_id -> live session client. Created on allocate(), removed on close().
        self._sessions: dict[str, AsyncClusterSessionClient] = {}

    def _get(self, lease_id: str) -> AsyncClusterSessionClient:
        client = self._sessions.get(lease_id)
        if client is None:
            raise RuntimeError(f"unknown lease_id (not allocated or already closed): {lease_id}")
        return client

    async def allocate(
        self,
        episode_id: str,
        task_type: str = "training",
        user_id: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Acquire a new session and return a lease handle.

        The cluster ``/v1/sessions`` acquire takes only ``runtime``; the
        episode/user/job identifiers are accepted for GuiEnvClient parity and
        logged for traceability rather than sent on the wire.
        """
        client_cls = _CLIENT_BY_RUNTIME.get(self.runtime, OSWorldAsyncRemoteClient)
        client = client_cls(
            cluster_url=self.base_url,
            action_space=self.action_space,
            requested_session_id=f"sess-{uuid.uuid5(uuid.NAMESPACE_URL, episode_id).hex[:16]}",
        )
        client.runtime = self.runtime
        await client.acquire()
        lease_id = f"sess-{uuid.uuid4().hex}"
        self._sessions[lease_id] = client
        logger.info(
            "GUI session allocated lease=%s session=%s episode=%s user=%s job=%s",
            lease_id, client.session_id, episode_id, user_id, job_id,
        )
        return {"ok": True, "lease_id": lease_id, "session_id": client.session_id}

    async def heartbeat(self, lease_id: str) -> None:
        # Session clients run an automatic background heartbeat thread; nothing to
        # do here. Still validate the lease so a stale id surfaces loudly.
        self._get(lease_id)

    async def reset(self, lease_id: str, task_config: dict[str, Any] | None) -> dict[str, Any]:
        return await self._get(lease_id).reset(task_config or {})

    async def get_obs(self, lease_id: str) -> dict[str, Any]:
        return await self._get(lease_id).get_obs()

    async def step(
        self, lease_id: str, action: Any, sleep_after_execution: float
    ) -> tuple[dict[str, Any], float, bool, dict]:
        # OSWorldAsyncRemoteClient.step settles server-side and returns the
        # post-settle screenshot in one response; `pause` is the client-side
        # settle hint, mapped from the rollout's sleep_after_execution.
        return await self._get(lease_id).step(action, pause=sleep_after_execution)

    async def evaluate(self, lease_id: str) -> float:
        return await self._get(lease_id).evaluate()

    async def start_recording(self, lease_id: str) -> None:
        # Cluster sessions don't support VNC recording (see _NoOpRecorder).
        self._get(lease_id)

    async def end_recording(self, lease_id: str, out_path: str) -> None:
        self._get(lease_id)

    async def close(self, lease_id: str) -> None:
        client = self._sessions.pop(lease_id, None)
        if client is None:
            return
        try:
            await client.release()
        except Exception:
            logger.warning("Failed to release session for lease %s", lease_id, exc_info=True)
