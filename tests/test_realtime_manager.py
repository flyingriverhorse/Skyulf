"""Tests for the realtime ConnectionManager (no live Redis required)."""

from __future__ import annotations

import pytest

from backend.realtime.manager import ConnectionManager


class _FakeSocket:
    """Minimal stand-in for FastAPI's WebSocket."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self.closed = False
        self._fail = fail_on_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, msg: str) -> None:
        if self._fail:
            raise RuntimeError("dead socket")
        self.sent.append(msg)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_accepts_and_tracks_socket() -> None:
    mgr = ConnectionManager()
    ws = _FakeSocket()
    await mgr.connect(ws)  # type: ignore[arg-type]
    assert ws.accepted is True
    assert ws in mgr._clients


@pytest.mark.asyncio
async def test_broadcast_reaches_all_live_clients() -> None:
    mgr = ConnectionManager()
    a, b = _FakeSocket(), _FakeSocket()
    await mgr.connect(a)  # type: ignore[arg-type]
    await mgr.connect(b)  # type: ignore[arg-type]
    await mgr.broadcast("hello")
    assert a.sent == ["hello"]
    assert b.sent == ["hello"]


@pytest.mark.asyncio
async def test_broadcast_drops_dead_sockets() -> None:
    """A failing send must not poison subsequent broadcasts."""
    mgr = ConnectionManager()
    bad = _FakeSocket(fail_on_send=True)
    good = _FakeSocket()
    await mgr.connect(bad)  # type: ignore[arg-type]
    await mgr.connect(good)  # type: ignore[arg-type]
    await mgr.broadcast("ping")
    assert good.sent == ["ping"]
    assert bad not in mgr._clients
    # Subsequent broadcast still works for the survivor.
    await mgr.broadcast("ping2")
    assert good.sent == ["ping", "ping2"]


@pytest.mark.asyncio
async def test_disconnect_removes_client() -> None:
    mgr = ConnectionManager()
    ws = _FakeSocket()
    await mgr.connect(ws)  # type: ignore[arg-type]
    await mgr.disconnect(ws)  # type: ignore[arg-type]
    assert ws not in mgr._clients
