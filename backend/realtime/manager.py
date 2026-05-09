"""WebSocket connection manager + Redis subscriber loop.

One ConnectionManager instance is created per FastAPI process. It owns
the set of live WebSocket clients and a single asyncio task that
subscribes to the Redis pub/sub channel and broadcasts every message.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Set

import orjson
from fastapi import WebSocket

from backend.config import get_settings
from backend.realtime.events import JOB_EVENTS_CHANNEL

logger = logging.getLogger(__name__)


def _wrap_payload(raw: str) -> str:
    """Wrap a raw JSON event string in the typed channel envelope.

    Shared by both the Redis subscriber loop and the local-bus loop.
    Returns the final string ready for ``broadcast()``.
    """
    return orjson.dumps({"channel": "jobs", "data": orjson.loads(raw)}).decode()


class ConnectionManager:
    """Tracks live WebSocket clients and fans out Redis events to them."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._subscriber_task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.debug("WS client connected (now %d)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.debug("WS client disconnected (now %d)", len(self._clients))

    async def broadcast(self, message: str) -> None:
        """Send `message` to every connected client; drop dead sockets."""
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                # Client gone; harvest on next disconnect call. Don't await
                # close() here — it can block other broadcasts.
                async with self._lock:
                    self._clients.discard(ws)

    async def start(self) -> None:
        """Spawn the event subscriber task (idempotent).

        Picks the transport based on ``USE_CELERY``: the Redis pub/sub
        loop for the multi-process Celery deployment, the in-process
        ``LocalBus`` listener for the embedded BackgroundTasks mode.
        """
        if self._subscriber_task and not self._subscriber_task.done():
            return
        self._stop.clear()
        settings = get_settings()
        target = self._subscriber_loop if settings.USE_CELERY else self._local_loop
        self._subscriber_task = asyncio.create_task(target(), name="realtime-subscriber")

    async def stop(self) -> None:
        """Cancel the subscriber and close all sockets."""
        self._stop.set()
        if self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except (asyncio.CancelledError, Exception):
                pass
            self._subscriber_task = None
        async with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for ws in clients:
            try:
                await ws.close()
            except Exception:
                pass

    async def _drain_pubsub(self, pubsub: Any) -> None:
        """Forward messages from a Redis pubsub stream until stop is set."""
        async for message in pubsub.listen():
            if self._stop.is_set():
                break
            if message.get("type") != "message":
                continue
            data = message.get("data")
            if not isinstance(data, str):
                continue
            await self.broadcast(_wrap_payload(data))

    async def _subscriber_loop(self) -> None:
        """Subscribe to Redis and broadcast messages until stopped.

        Reconnects with backoff on transient Redis failures so the WS
        endpoint stays usable even if the broker bounces.
        """
        backoff = 1.0
        while not self._stop.is_set():
            try:
                # Imported lazily — we don't want to fail app startup
                # just because the optional realtime layer can't reach
                # Redis. Polling fallback on the frontend covers it.
                from redis import asyncio as aioredis

                settings = get_settings()
                client = aioredis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
                pubsub = client.pubsub()
                await pubsub.subscribe(JOB_EVENTS_CHANNEL)
                logger.info("Realtime subscriber attached to %s", JOB_EVENTS_CHANNEL)
                backoff = 1.0  # reset after a successful connect
                await self._drain_pubsub(pubsub)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Realtime subscriber error: %s (retry in %.1fs)", exc, backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)

    async def _local_loop(self) -> None:
        """Drain the in-process bus and broadcast (no-Celery mode)."""
        from backend.realtime.local_bus import local_bus

        queue = local_bus.attach(asyncio.get_running_loop())
        logger.info("Realtime subscriber attached to in-process bus")
        try:
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    await self.broadcast(_wrap_payload(raw))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("LocalBus broadcast error: %s", exc)
        finally:
            local_bus.detach()


# Single process-wide instance. The router and lifespan reach for this.
connection_manager = ConnectionManager()
