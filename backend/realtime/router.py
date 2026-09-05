"""WebSocket router for realtime job events.

Mounted at `/ws/jobs`. Clients connect, receive JSON messages of the
form `{"channel": "jobs", "data": {...JobEvent...}}`, and reply with
nothing. The endpoint blocks on `receive_text` purely to detect close.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.realtime.manager import connection_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/jobs")
async def ws_jobs(ws: WebSocket) -> None:
    """Serve the ``/ws/jobs`` event feed to one client until it disconnects.

    Push-only: inbound frames are never read for their content, the receive loop
    exists just so a client close surfaces as ``WebSocketDisconnect``. A socket
    that fails to be accepted returns before it is ever registered, so it needs no
    cleanup; every other exit path unregisters it in the ``finally``.
    """
    try:
        await connection_manager.connect(ws)
    except Exception as exc:  # noqa: BLE001 - failed accept exits handler
        logger.warning("WS jobs: failed to accept connection: %s", exc)
        return

    try:
        # We don't expect inbound messages today; this read keeps the
        # coroutine parked so disconnects propagate cleanly.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - receive errors end the loop
        logger.warning("WS jobs receive loop ended unexpectedly: %s", exc)
    finally:
        await connection_manager.disconnect(ws)
