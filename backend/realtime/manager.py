"""WebSocket connection manager + Redis subscriber loop.

One ConnectionManager instance is created per FastAPI process. It owns
the set of live WebSocket clients and a single asyncio task that
subscribes to the Redis pub/sub channel and broadcasts every message.
"""

import asyncio
import contextlib
import logging
from typing import Any

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

    SEND_TIMEOUT = 1.0

    def __init__(self) -> None:
        """Start with no clients and no subscriber task running.

        ``_lock`` guards the client set against a connect, disconnect or broadcast
        landing concurrently, and ``_stop`` is the event both subscriber loops poll
        to know when to exit.
        """
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_lock = asyncio.Lock()
        self._subscriber_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def connect(self, ws: WebSocket) -> None:
        """Accept ``ws``, then add it to the broadcast set.

        The accept comes first, so a socket that is refused never enters the set
        and can never be broadcast to.
        """
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.debug("WS client connected (now %d)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        """Drop ``ws`` from the broadcast set without closing it.

        Explicit disconnect is called by the route after its receive loop ends.
        Failed broadcasts also close the socket so the receive loop can exit.
        """
        async with self._lock:
            self._clients.discard(ws)
        logger.debug("WS client disconnected (now %d)", len(self._clients))

    async def broadcast(self, message: str) -> None:
        """Fan out concurrently, preserving order with bounded backpressure.

        Only one broadcast runs at a time. Each send has a deadline; a slow
        peer is disconnected instead of accumulating an unbounded event queue.
        All send tasks are awaited, including when the subscriber is cancelled.
        """
        async with self._broadcast_lock:
            async with self._lock:
                clients = list(self._clients)
            await asyncio.gather(*(self._send(ws, message) for ws in clients))

    async def _close(self, ws: WebSocket) -> None:
        """Close a failed or shutting-down socket within a bounded deadline."""
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(), timeout=self.SEND_TIMEOUT)

    async def _send(self, ws: WebSocket, message: str) -> None:
        """Remove failed or cancelled sends and release their receive loops."""
        try:
            await asyncio.wait_for(ws.send_text(message), timeout=self.SEND_TIMEOUT)
        except asyncio.CancelledError:
            await self.disconnect(ws)
            await self._close(ws)
            raise
        except Exception:  # noqa: BLE001 - failed peers must not stop the subscriber
            await self.disconnect(ws)
            await self._close(ws)

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
            # best-effort task teardown during shutdown
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._subscriber_task
            self._subscriber_task = None
        async with self._broadcast_lock:
            async with self._lock:
                clients = list(self._clients)
                self._clients.clear()
            await asyncio.gather(*(self._close(ws) for ws in clients))

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
            client: Any = None
            pubsub: Any = None
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
            except Exception as exc:  # noqa: BLE001 - subscriber retries with backoff
                logger.warning("Realtime subscriber error: %s (retry in %.1fs)", exc, backoff)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                # Always release the pubsub/connection before reconnecting or
                # exiting the loop — otherwise every retry (or shutdown) leaks
                # a Redis connection.
                if pubsub is not None:
                    with contextlib.suppress(Exception):
                        await pubsub.close()
                if client is not None:
                    with contextlib.suppress(Exception):
                        await client.aclose()

    async def _local_loop(self) -> None:
        """Drain the in-process bus and broadcast (no-Celery mode)."""
        from backend.realtime.local_bus import local_bus

        queue = local_bus.attach(asyncio.get_running_loop())
        logger.info("Realtime subscriber attached to in-process bus")
        try:
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    continue
                try:
                    await self.broadcast(_wrap_payload(raw))
                except Exception as exc:  # noqa: BLE001 - broadcast error must not stop loop  # pragma: no cover - defensive
                    logger.warning("LocalBus broadcast error: %s", exc)
        finally:
            local_bus.detach()


# Single process-wide instance. The router and lifespan reach for this.
connection_manager = ConnectionManager()
