"""Async cluster session client — benchmark-agnostic (asyncio + aiohttp).

The asyncio counterpart of :class:`ClusterSessionClient`. Same unified
/v1/sessions protocol (shared via ``_protocol``), same data-plane/control-plane
split, same retry policy — only the transport differs.

One instance binds to one session. To drive many sessions concurrently, the
caller runs ONE event loop and fans out with ``asyncio.gather`` /
``asyncio.Semaphore`` over multiple clients — this is what lets a single process
hold N concurrent rollouts (concurrency decoupled from OS processes), instead of
the eval driver's process-per-session model. Example: ``scripts/python/
async_session_demo.py``.

Usage::

    async with AsyncClusterSessionClient(runtime="osworld") as c:
        obs = await c.reset({"id": task_id, "instruction": "..."})
        data = await c.step({"kind": "gui", "type": "pyautogui", "payload": {...}})
        score = (await c.evaluate()).get("score")
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, Optional

import aiohttp

from . import _protocol as proto

logger = logging.getLogger("cluster.client.async_session")


# Process-level shared aiohttp connector, keyed by event loop. Sharing one
# connector across all session clients on a loop lets keep-alive connections to
# a node be reused across trajectories (vs a fresh connector per trajectory,
# which paid a cold TCP/TLS handshake on every step — ~+2s/step in training).
# Keyed by id(loop) because an aiohttp connector is bound to the loop that
# created it; multi-process rollout workers each have their own loop and thus
# their own pool, matching the per-worker single-process world.
_SHARED_CONNECTORS: dict[int, aiohttp.TCPConnector] = {}


def _shared_connector_limit() -> int:
    """Pool size >= concurrent in-flight requests on this loop.

    Each trajectory may have its main op + heartbeat in flight, so size to ~2x
    the per-process trajectory-concurrency budget (already sliced per worker via
    GUI_TRAJECTORY_CONCURRENCY) with a generous floor.
    """
    conc = os.getenv("GUI_TRAJECTORY_CONCURRENCY") or os.getenv("GUI_POOL_MAX_ENVS") or "64"
    try:
        return max(proto.POOL_SIZE, int(conc) * 2)
    except ValueError:
        return max(proto.POOL_SIZE, 128)


def _shared_connector() -> aiohttp.TCPConnector:
    loop = asyncio.get_event_loop()
    key = id(loop)
    conn = _SHARED_CONNECTORS.get(key)
    if conn is None or conn.closed:
        limit = _shared_connector_limit()
        conn = aiohttp.TCPConnector(limit=limit, limit_per_host=limit)
        _SHARED_CONNECTORS[key] = conn
        logger.info("Created shared aiohttp connector (limit=%d) for loop %d", limit, key)
    return conn


@atexit.register
def _close_shared_connectors() -> None:
    """Close pooled connectors on interpreter shutdown.

    The shared connectors are intentionally process-lived (not owned by any
    ClientSession), so close them here to avoid leaking sockets and the noisy
    'Event loop is closed' warning aiohttp's __del__ emits at GC time.
    """
    for conn in list(_SHARED_CONNECTORS.values()):
        try:
            if conn.closed:
                continue
            # Prefer the synchronous internal close: the async close() is a
            # coroutine and the loop is already gone at interpreter exit.
            sync_close = getattr(conn, "_close", None)
            if callable(sync_close):
                sync_close()
        except Exception:
            pass
    _SHARED_CONNECTORS.clear()


class AsyncClusterSessionClient:
    """Async stateful client bound to one cluster session.

    Construct, then ``await acquire()`` (or use ``async with``, which acquires on
    enter and releases on exit). Not safe to share one instance across tasks
    that call it concurrently — give each concurrent rollout its own client.
    """

    def __init__(
        self,
        cluster_url: str | None = None,
        runtime: str = "osworld",
        *,
        timeout: float = 300,
        requested_session_id: str | None = None,
    ):
        self.cluster_url = proto.normalize_url(cluster_url or proto.DEFAULT_CLUSTER_URL)
        self.runtime = runtime
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._requested_session_id = requested_session_id or f"sess-{uuid.uuid4().hex[:16]}"
        self._node_url: Optional[str] = None
        self._operation_idempotency_version = 0
        # The aiohttp ClientSession is created lazily in acquire(), which always
        # runs inside the event loop (creating it in __init__ would bind to the
        # wrong/absent loop and trigger aiohttp warnings).
        self._http: Optional[aiohttp.ClientSession] = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            # Share ONE process-level TCPConnector across all sessions on this
            # event loop so keep-alive connections to a node are reused across
            # trajectories (a fresh connector per trajectory paid a cold
            # TCP/TLS handshake on every step — measured ~+2s/step vs the legacy
            # global-pool client). Each client keeps its own lightweight
            # ClientSession but does NOT own the connector (connector_owner=False),
            # so release()/close() never tears down the shared pool.
            self._http = aiohttp.ClientSession(
                connector=_shared_connector(),
                connector_owner=False,
            )
        return self._http

    async def _request(
        self,
        method: str,
        url: str,
        json: Dict[str, Any] | None,
        timeout: float,
        *,
        retry_status: bool = True,
        retry_transport: bool = True,
        headers: Dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> Dict[str, Any]:
        """HTTP with bounded backoff on transient failures; returns parsed JSON.

        Mirrors the sync client's policy via ``_protocol``: retry 502/503 and
        transport errors within a bounded budget; surface everything else.
        """
        attempt = 0
        http = self._ensure_http()
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise asyncio.TimeoutError(f"request recovery deadline exceeded for {method} {url}")
            attempt_timeout = min(timeout, remaining) if remaining is not None else timeout
            ct = aiohttp.ClientTimeout(total=attempt_timeout)
            retry_reason: str | None = None
            try:
                async with http.request(
                    method,
                    url,
                    json=json or {},
                    headers=headers,
                    timeout=ct,
                ) as resp:
                    if retry_status and proto.should_retry_status(resp.status, attempt):
                        retry_reason = f"HTTP {resp.status}"
                        await resp.read()
                    else:
                        resp.raise_for_status()
                        return await resp.json()
            except (
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
            ) as exc:
                if not retry_transport or not proto.can_retry_after_error(attempt):
                    raise
                retry_reason = f"{type(exc).__name__}: {exc}"

            if retry_reason is None:
                raise RuntimeError(f"request retry state lost for {method} {url}")
            delay = proto.backoff_delay(attempt)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        f"request recovery deadline exceeded for {method} {url}"
                    )
                delay = min(delay, remaining)
            logger.warning(
                "Retrying request method=%s url=%s operation_id=%s attempt=%d/%d "
                "delay=%.2fs reason=%s",
                method,
                url,
                (headers or {}).get("Idempotency-Key"),
                attempt + 1,
                proto.RETRY_MAX,
                delay,
                retry_reason,
            )
            await asyncio.sleep(delay)
            attempt += 1

    async def acquire(self) -> "AsyncClusterSessionClient":
        url, body = proto.acquire_request(
            self.cluster_url,
            self.runtime,
            self._requested_session_id,
        )
        deadline = time.monotonic() + self.timeout
        pending_attempt = 0
        while True:
            data = await self._request(
                "POST",
                url,
                body,
                self.timeout,
                deadline=deadline,
            )
            if not data.get("pending"):
                break
            delay = proto.pending_delay(
                pending_attempt,
                float(data.get("retry_after_seconds", 0.0) or 0.0),
            )
            remaining = deadline - time.monotonic()
            if remaining <= delay:
                raise asyncio.TimeoutError(
                    f"session acquire recovery deadline exceeded for {self._requested_session_id}"
                )
            logger.info(
                "Session acquire pending session=%s retry_in=%.2fs",
                self._requested_session_id,
                delay,
            )
            await asyncio.sleep(delay)
            pending_attempt += 1

        data = proto.check_envelope("acquire", data)
        self.session_id, self._node_url = proto.parse_acquire(data)
        self._operation_idempotency_version = proto.operation_idempotency_version(data)
        logger.info(
            "Acquired session %s (runtime=%s, node_url=%s, operation_idempotency=%d)",
            self.session_id,
            self.runtime,
            self._node_url,
            self._operation_idempotency_version,
        )
        self._start_heartbeat()
        return self

    async def _call(self, op: str, body: Dict[str, Any] | None = None, timeout: float | None = None) -> Dict[str, Any]:
        method, url = proto.op_request(op, self.session_id, self.cluster_url, self._node_url)
        eff_timeout = timeout if timeout is not None else (proto.OP_TIMEOUTS.get(op) or self.timeout)
        idempotent_server = self._operation_idempotency_version >= 1
        retryable = idempotent_server or op in {"observe", "heartbeat"}
        operation_id = f"{op}-{uuid.uuid4().hex}" if idempotent_server else None
        headers = {"Idempotency-Key": operation_id} if operation_id else None
        deadline = time.monotonic() + proto.OPERATION_RECOVERY_DEADLINE
        pending_attempt = 0

        while True:
            data = await self._request(
                method,
                url,
                body,
                eff_timeout,
                retry_status=retryable,
                retry_transport=retryable,
                headers=headers,
                deadline=deadline,
            )
            if not data.get("pending"):
                return proto.check_envelope(op, data)
            if not operation_id or data.get("operation_id") != operation_id:
                raise RuntimeError(
                    f"invalid pending response for {op}: expected operation {operation_id!r}, "
                    f"got {data.get('operation_id')!r}"
                )
            delay = proto.pending_delay(
                pending_attempt,
                float(data.get("retry_after_seconds", 0.0) or 0.0),
            )
            remaining = deadline - time.monotonic()
            if remaining <= delay:
                raise asyncio.TimeoutError(
                    f"operation recovery deadline exceeded for {operation_id}"
                )
            logger.info(
                "Operation pending session=%s op=%s operation_id=%s retry_in=%.2fs",
                self.session_id,
                op,
                operation_id,
                delay,
            )
            await asyncio.sleep(delay)
            pending_attempt += 1

    # -- heartbeat -------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        if proto.HEARTBEAT_INTERVAL <= 0 or self.session_id is None:
            return
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        session_id = self.session_id
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(session_id,),
            name=f"ClusterSessionHeartbeat-{session_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        logger.info(
            "Started heartbeat session=%s interval=%.1fs",
            session_id,
            proto.HEARTBEAT_INTERVAL,
        )

    def _heartbeat_loop(self, session_id: str) -> None:
        # Heartbeats deliberately run outside the rollout event loop and outside
        # its shared aiohttp connector. CPU-heavy prompt/image preparation or a
        # saturated async connector must not make a live lease look abandoned.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        failures = 0
        delay = proto.HEARTBEAT_INTERVAL
        while not self._heartbeat_stop.wait(delay):
            if self.session_id != session_id:
                return
            _, url = proto.op_request("heartbeat", session_id, self.cluster_url, self._node_url)
            started = time.monotonic()
            try:
                req = urllib.request.Request(
                    url,
                    data=b"{}",
                    headers={
                        "Content-Type": "application/json",
                        "X-Cluster-Heartbeat-Mode": "independent-v1",
                    },
                    method="POST",
                )
                with opener.open(req, timeout=10) as resp:
                    if not 200 <= resp.status < 300:
                        raise RuntimeError(f"heartbeat HTTP {resp.status}")
                    resp.read()
                failures = 0
                delay = proto.HEARTBEAT_INTERVAL
                logger.debug(
                    "Heartbeat ok session=%s latency=%.3fs",
                    session_id,
                    time.monotonic() - started,
                )
            except Exception as exc:  # noqa: BLE001 - keepalive must stay alive
                failures += 1
                delay = min(
                    proto.HEARTBEAT_INTERVAL,
                    proto.backoff_delay(failures - 1),
                )
                logger.warning(
                    "Heartbeat failed session=%s consecutive_failures=%d "
                    "retry_in=%.2fs error=%r",
                    session_id,
                    failures,
                    delay,
                    exc,
                )

    async def _stop_heartbeat(self) -> None:
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        self._heartbeat_stop.set()
        if thread and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
        if thread:
            logger.info("Stopped heartbeat session=%s", self.session_id)

    # -- session operations ----------------------------------------------------

    async def reset(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        data = await self._call("reset", {"task_payload": task_payload})
        return data.get("observation", {})

    async def step(self, action: Dict[str, Any], pause: float | None = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"action": action}
        if pause is not None:
            body["pause"] = pause
        return await self._call("step", body)

    async def observe(self) -> Dict[str, Any]:
        data = await self._call("observe")
        return data.get("observation", {})

    async def evaluate(self) -> Dict[str, Any]:
        return await self._call("evaluate")

    async def release(self) -> None:
        await self._stop_heartbeat()
        if self.session_id is not None and self._http is not None:
            try:
                _, url = proto.op_request("release", self.session_id, self.cluster_url, self._node_url)
                data = await self._request(
                    "DELETE",
                    url,
                    {},
                    30,
                    deadline=time.monotonic() + 60,
                )
                proto.check_envelope("release", data)
                logger.info("Released session %s", self.session_id)
            except Exception as exc:
                logger.warning("Failed to release session %s: %s", self.session_id, exc)
            finally:
                self.session_id = None
                self._node_url = None
                self._operation_idempotency_version = 0
        if self._http is not None and not self._http.closed:
            await self._http.close()

    async def close(self) -> None:
        await self.release()

    async def __aenter__(self) -> "AsyncClusterSessionClient":
        return await self.acquire()

    async def __aexit__(self, *exc) -> None:
        await self.release()
