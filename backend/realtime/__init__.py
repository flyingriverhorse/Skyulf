"""Realtime job event delivery (WebSocket + Redis pub/sub).

This package replaces the per-component HTTP polling for job status with
a single broadcast WebSocket. Celery workers publish small JSON events
to a Redis pub/sub channel; the FastAPI process subscribes once at
startup and fans messages out to all connected clients.
"""

from backend.realtime.events import JOB_EVENTS_CHANNEL, JobEvent, publish_job_event
from backend.realtime.manager import ConnectionManager, connection_manager
from backend.realtime.router import router

__all__ = [
    "JOB_EVENTS_CHANNEL",
    "JobEvent",
    "publish_job_event",
    "ConnectionManager",
    "connection_manager",
    "router",
]
