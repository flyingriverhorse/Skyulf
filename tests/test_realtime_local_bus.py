"""Tests for the in-process LocalBus fallback used when USE_CELERY=False."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from backend.realtime.events import JobEvent, publish_job_event
from backend.realtime.local_bus import LocalBus, local_bus


@pytest.mark.asyncio
async def test_local_bus_delivers_from_same_thread() -> None:
    bus = LocalBus()
    queue = bus.attach(asyncio.get_running_loop())
    bus.publish("hello")
    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert msg == "hello"
    bus.detach()


@pytest.mark.asyncio
async def test_local_bus_delivers_across_threads() -> None:
    """BackgroundTasks may run our publisher in a thread pool."""
    bus = LocalBus()
    queue = bus.attach(asyncio.get_running_loop())

    def _worker() -> None:
        bus.publish("from-thread")

    threading.Thread(target=_worker).start()
    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert msg == "from-thread"
    bus.detach()


def test_local_bus_publish_without_listener_is_noop() -> None:
    """Events published before startup must not raise."""
    bus = LocalBus()
    bus.publish("dropped")  # no attach -> silent no-op


@pytest.mark.asyncio
async def test_publish_job_event_routes_to_local_bus_when_no_celery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When USE_CELERY=False, publish_job_event must hit the in-process bus."""
    queue = local_bus.attach(asyncio.get_running_loop())
    try:
        # Force the no-Celery branch regardless of test config defaults.
        from backend.realtime import events as events_mod

        class _FakeSettings:
            USE_CELERY = False

        monkeypatch.setattr(events_mod, "get_settings", lambda: _FakeSettings())

        publish_job_event(JobEvent(event="progress", job_id="abc", status="running", progress=42))
        raw = await asyncio.wait_for(queue.get(), timeout=1.0)
        payload = json.loads(raw)
        assert payload == {
            "event": "progress",
            "job_id": "abc",
            "status": "running",
            "progress": 42,
        }
    finally:
        local_bus.detach()
