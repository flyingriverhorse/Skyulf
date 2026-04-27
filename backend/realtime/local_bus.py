"""In-process pub/sub fallback for the no-Celery / no-Redis path.

When ``USE_CELERY=False`` the pipeline runs inside the same FastAPI
process via ``BackgroundTasks`` (or a small thread pool for parallel
branches). In that mode we don't want to require Redis just to deliver
WebSocket invalidator events — the publisher and the subscriber are in
the same Python process anyway.

This module provides a minimal thread-safe bridge: workers call the
sync ``publish`` from any thread; the asyncio listener installed by
``ConnectionManager`` consumes from an ``asyncio.Queue`` bound to its
running loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LocalBus:
    """Single-process bridge from sync publishers to one async listener."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue[str]] = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[str]:
        """Bind the bus to a running event loop and return its queue."""
        self._loop = loop
        self._queue = asyncio.Queue()
        return self._queue

    def detach(self) -> None:
        self._loop = None
        self._queue = None

    def publish(self, message: str) -> None:
        """Schedule ``message`` for delivery on the bound loop.

        Safe to call from any thread (BackgroundTasks worker, the main
        event loop, or our small ThreadPoolExecutor for parallel
        branches). No-op if no listener is attached yet — events
        published before startup are simply dropped, which matches the
        Redis path's behavior.
        """
        loop = self._loop
        queue = self._queue
        if loop is None or queue is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except RuntimeError:
            # Loop is closed (shutdown race). Drop silently — same
            # contract as the Redis publisher.
            logger.debug("LocalBus publish skipped: loop closed")


# Single process-wide instance.
local_bus = LocalBus()
