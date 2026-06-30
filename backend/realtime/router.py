"""WebSocket router for realtime job events.

Mounted at `/ws/jobs`. Clients connect, receive JSON messages of the
form `{"channel": "jobs", "data": {...JobEvent...}}`, and reply with
nothing. The endpoint blocks on `receive_text` purely to detect close.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.realtime.manager import connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/jobs")
async def ws_jobs(ws: WebSocket) -> None:
    try:
        await connection_manager.connect(ws)
    except Exception as exc:
        logger.warning("WS jobs: failed to accept connection: %s", exc)
        return

    try:
        # We don't expect inbound messages today; this read keeps the
        # coroutine parked so disconnects propagate cleanly.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WS jobs receive loop ended unexpectedly: %s", exc)
    finally:
        await connection_manager.disconnect(ws)
