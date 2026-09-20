"""Slow clients cannot indefinitely stall realtime event delivery."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from backend.realtime.manager import ConnectionManager


class Socket:
    """Expose deterministic send blocking and cancellation for fanout tests."""

    def __init__(self, *, blocked: bool = False, close_blocked: bool = False):
        """Keep event delivery observable without a real network socket."""
        self.blocked = blocked
        self.close_blocked = close_blocked
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.received = asyncio.Event()
        self.sent: list[str] = []
        self.cancelled = False
        self.closed = False

    async def accept(self):
        """Accept the test connection."""

    async def send_text(self, message):
        """Block only the slow peer and record completed sends."""
        self.entered.set()
        try:
            if self.blocked:
                await self.release.wait()
            self.sent.append(message)
            self.received.set()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def close(self):
        """Allow shutdown deadlines to be tested independently of sends."""
        self.closed = True
        if self.close_blocked:
            await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_slow_peer_does_not_hold_fast_peer_or_next_redis_event(monkeypatch):
    """A stalled peer must be evicted so the subscriber can deliver later events."""
    monkeypatch.setattr(ConnectionManager, "SEND_TIMEOUT", 0.1, raising=False)
    manager = ConnectionManager()
    slow, fast = Socket(blocked=True), Socket()
    await manager.connect(slow)
    await manager.connect(fast)

    # Fix iteration order to reproduce the old serial-send failure reliably.
    class SlowFirstSet(set):
        """Make the slow-first reproduction independent of hash iteration."""

        def __iter__(self):
            """Yield the blocked socket before any healthy socket."""
            return iter(sorted(super().__iter__(), key=lambda peer: not peer.blocked))

    manager._clients = SlowFirstSet([slow, fast])

    class Pubsub:
        """Supply two broker events without a Redis dependency."""

        async def listen(self) -> AsyncIterator[dict]:
            """Yield ordered events as the real Redis listener does."""
            for value in (1, 2):
                yield {"type": "message", "data": f'{{"sequence":{value}}}'}

    task = asyncio.create_task(manager._drain_pubsub(Pubsub()))
    try:
        await asyncio.wait_for(fast.received.wait(), 0.05)
        await asyncio.wait_for(task, 0.5)
        assert fast.sent == [
            '{"channel":"jobs","data":{"sequence":1}}',
            '{"channel":"jobs","data":{"sequence":2}}',
        ]
        assert slow not in manager._clients
        assert slow.cancelled and slow.closed
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_overlapping_broadcasts_preserve_send_order():
    """A second broadcast must not send concurrently on the same WebSocket."""
    manager = ConnectionManager()
    peer = Socket(blocked=True)
    await manager.connect(peer)
    first = asyncio.create_task(manager.broadcast("first"))
    await peer.entered.wait()
    peer.blocked = False
    second = asyncio.create_task(manager.broadcast("second"))
    await asyncio.sleep(0)
    peer.release.set()
    await asyncio.gather(first, second)
    assert peer.sent == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelled_broadcast_cleans_up_inflight_sends():
    """Cancellation must not leave orphan sends holding the broadcast lock."""
    manager = ConnectionManager()
    peer = Socket(blocked=True)
    await manager.connect(peer)
    task = asyncio.create_task(manager.broadcast("cancelled"))
    await peer.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert peer.cancelled
    assert peer not in manager._clients
    survivor = Socket()
    await manager.connect(survivor)
    await asyncio.wait_for(manager.broadcast("after"), 0.5)
    assert survivor.sent == ["after"]


@pytest.mark.asyncio
async def test_shutdown_bounds_close_and_cancels_subscriber(monkeypatch):
    """An unresponsive socket must not prevent manager shutdown."""
    monkeypatch.setattr(ConnectionManager, "SEND_TIMEOUT", 0.05, raising=False)
    manager = ConnectionManager()
    slow, fast = Socket(close_blocked=True), Socket()
    await manager.connect(slow)
    await manager.connect(fast)
    manager._subscriber_task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.wait_for(manager.stop(), 0.5)
    assert slow.closed and fast.closed
    assert not manager._clients
    assert manager._subscriber_task is None


@pytest.mark.asyncio
async def test_local_bus_progresses_after_slow_peer_timeout(monkeypatch):
    """Embedded-mode events must retain order and detach their bus on shutdown."""
    from backend.realtime import local_bus as bus_module
    from backend.realtime.local_bus import LocalBus

    monkeypatch.setattr(ConnectionManager, "SEND_TIMEOUT", 0.05)
    bus = LocalBus()
    monkeypatch.setattr(bus_module, "local_bus", bus)
    manager = ConnectionManager()
    slow, fast = Socket(blocked=True), Socket()
    await manager.connect(slow)
    await manager.connect(fast)
    manager._subscriber_task = asyncio.create_task(manager._local_loop())
    await asyncio.sleep(0)
    bus.publish('{"sequence":1}')
    bus.publish('{"sequence":2}')
    try:
        async with asyncio.timeout(0.5):
            while len(fast.sent) < 2:
                fast.received.clear()
                await fast.received.wait()
        assert fast.sent == [
            '{"channel":"jobs","data":{"sequence":1}}',
            '{"channel":"jobs","data":{"sequence":2}}',
        ]
        assert slow not in manager._clients
    finally:
        await manager.stop()
    assert bus._queue is None


@pytest.mark.asyncio
async def test_shutdown_during_broadcast_cancels_sends_and_releases_lock(monkeypatch):
    """Lifespan shutdown must cancel a subscriber already blocked in a send."""
    monkeypatch.setattr(ConnectionManager, "SEND_TIMEOUT", 0.05)
    manager = ConnectionManager()
    slow, fast = Socket(blocked=True, close_blocked=True), Socket()
    await manager.connect(slow)
    await manager.connect(fast)
    subscriber = asyncio.create_task(manager.broadcast("pending"))
    manager._subscriber_task = subscriber
    await slow.entered.wait()
    await asyncio.wait_for(manager.stop(), 0.5)
    assert subscriber.cancelled()
    assert slow.cancelled and slow.closed and fast.closed
    assert not manager._clients
    assert not manager._broadcast_lock.locked()
