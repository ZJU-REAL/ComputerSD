from __future__ import annotations

import base64
import logging
import os
from typing import Any

from slime.utils.http_utils import post

logger = logging.getLogger("gui.env_client")


def _decode_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out = dict(obs or {})
    screenshot_b64 = out.pop("screenshot_b64", None)
    out["screenshot"] = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
    return out


def _env_flag(name: str, default: str) -> str:
    return os.getenv(name, default).strip().lower()


class GuiEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Keep GUI env-control retries small to avoid long hangs on deterministic failures.
        # 3: deterministic 500s (setup download fail / missing file) won't recover by retrying,
        # so fail fast to release the env slot instead of wasting ~10-20s per stuck trajectory.
        self.default_max_retries = int(os.getenv("GUI_ENV_HTTP_MAX_RETRIES", "3"))
        self.evaluate_max_retries = int(os.getenv("GUI_EVALUATE_MAX_RETRIES", "3"))
        self.direct_node_data_plane = _env_flag("GUI_ENV_DIRECT_NODE_DATA_PLANE", "auto")
        if self.direct_node_data_plane not in {"auto", "1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError(
                "GUI_ENV_DIRECT_NODE_DATA_PLANE must be auto, 1/true/yes/on, or 0/false/no/off"
            )
        self._lease_node_urls: dict[str, str] = {}

    @property
    def _direct_node_forced(self) -> bool:
        return self.direct_node_data_plane in {"1", "true", "yes", "on"}

    @property
    def _direct_node_disabled(self) -> bool:
        return self.direct_node_data_plane in {"0", "false", "no", "off"}

    @staticmethod
    def _check_ok(op: str, out: dict[str, Any]) -> None:
        if not out.get("ok", False):
            raise RuntimeError(f"{op} failed: {out}")

    @staticmethod
    def _node_url_from_allocate(out: dict[str, Any]) -> str | None:
        for key in ("node_url", "worker_url", "env_url"):
            value = out.get(key)
            if isinstance(value, str) and value:
                return value.rstrip("/")
        return None

    def _data_base_url(self, lease_id: str) -> str:
        if self._direct_node_disabled:
            return self.base_url
        node_url = self._lease_node_urls.get(lease_id)
        if node_url:
            return node_url
        if self._direct_node_forced:
            raise RuntimeError(f"direct-node data plane requested, but lease {lease_id} has no node_url")
        return self.base_url

    def _uses_direct_node(self, lease_id: str) -> bool:
        return not self._direct_node_disabled and lease_id in self._lease_node_urls

    async def _post(self, base_url: str, path: str, payload: dict[str, Any], *, max_retries: int) -> dict[str, Any]:
        return await post(f"{base_url}{path}", payload, max_retries=max_retries)

    async def _post_data(self, lease_id: str, path: str, payload: dict[str, Any], *, max_retries: int) -> dict[str, Any]:
        return await self._post(self._data_base_url(lease_id), path, payload, max_retries=max_retries)

    async def allocate(self, episode_id: str, task_type: str = "training", user_id: str | None = None, job_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"episode_id": episode_id, "task_type": task_type}
        if user_id:
            body["user_id"] = user_id
        if job_id:
            body["job_id"] = job_id
        out = await self._post(
            self.base_url,
            "/allocate",
            body,
            max_retries=self.default_max_retries,
        )
        self._check_ok("allocate", out)
        lease_id = out.get("lease_id")
        node_url = self._node_url_from_allocate(out)
        if isinstance(lease_id, str) and node_url and not self._direct_node_disabled:
            self._lease_node_urls[lease_id] = node_url
            logger.info("GUI env lease %s data plane routed to node %s", lease_id, node_url)
        elif self._direct_node_forced:
            if isinstance(lease_id, str):
                try:
                    await self._post(
                        self.base_url,
                        "/close",
                        {"lease_id": lease_id},
                        max_retries=self.default_max_retries,
                    )
                except Exception:
                    logger.warning("Failed to close lease %s after missing node_url", lease_id, exc_info=True)
            raise RuntimeError(f"allocate did not return node_url in direct-node mode: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        path = "/lease/heartbeat" if self._uses_direct_node(lease_id) else "/heartbeat"
        out = await self._post(
            self.base_url,
            path,
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        self._check_ok("heartbeat", out)

    async def reset(self, lease_id: str, task_config: dict[str, Any] | None) -> dict[str, Any]:
        out = await self._post_data(
            lease_id,
            "/reset",
            {"lease_id": lease_id, "task_config": task_config},
            max_retries=self.default_max_retries,
        )
        self._check_ok("reset", out)
        return _decode_obs(out["observation"])

    async def get_obs(self, lease_id: str) -> dict[str, Any]:
        out = await self._post_data(
            lease_id,
            "/get_obs",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        self._check_ok("get_obs", out)
        return _decode_obs(out["observation"])

    async def step(self, lease_id: str, action: Any, sleep_after_execution: float) -> tuple[dict[str, Any], float, bool, dict]:
        out = await self._post_data(
            lease_id,
            "/step",
            {
                "lease_id": lease_id,
                "action": action,
                "sleep_after_execution": sleep_after_execution,
            },
            max_retries=self.default_max_retries,
        )
        self._check_ok("step", out)
        obs = _decode_obs(out["observation"])
        return obs, float(out.get("reward", 0.0)), bool(out.get("done", False)), out.get("info", {})

    async def evaluate(self, lease_id: str) -> float:
        out = await self._post_data(
            lease_id,
            "/evaluate",
            {"lease_id": lease_id},
            max_retries=self.evaluate_max_retries,
        )
        self._check_ok("evaluate", out)
        return float(out["score"])

    async def start_recording(self, lease_id: str) -> None:
        out = await self._post_data(
            lease_id,
            "/start_recording",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        self._check_ok("start_recording", out)

    async def end_recording(self, lease_id: str, out_path: str) -> None:
        out = await self._post_data(
            lease_id,
            "/end_recording",
            {"lease_id": lease_id, "out_path": out_path},
            max_retries=self.default_max_retries,
        )
        self._check_ok("end_recording", out)

    async def close(self, lease_id: str) -> None:
        try:
            out = await self._post(
                self.base_url,
                "/close",
                {"lease_id": lease_id},
                max_retries=self.default_max_retries,
            )
            self._check_ok("close", out)
        finally:
            self._lease_node_urls.pop(lease_id, None)
