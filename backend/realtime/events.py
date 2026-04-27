"""Job event payloads + Redis pub/sub publisher.

Events are intentionally small ("invalidator" pattern): the frontend
treats each event as a hint to refresh, not as the source of truth.
This keeps the WS payload schema stable across job-type changes and
sidesteps the partial-update problem.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Single broadcast channel. Per-user filtering can be added later when
# auth is wired up; today every endpoint already returns every job.
JOB_EVENTS_CHANNEL = "skyulf:jobs:events"

JobEventType = Literal["status", "progress", "created", "deleted"]


class JobEvent(BaseModel):
    """A minimal status hint emitted whenever a job row changes."""

    event: JobEventType
    job_id: str
    status: Optional[str] = None
    progress: Optional[int] = None
    current_step: Optional[str] = None


def _redis_client_sync() -> Any:
    """Sync Redis client used from Celery workers.

    Imported lazily so the module is importable in environments without
    Redis (e.g. unit tests that exercise the engine but not the queue).
    """
    import redis

    settings = get_settings()
    return redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)


def publish_job_event(event: JobEvent) -> None:
    """Publish a job event from sync code.

    Routing is decided by ``USE_CELERY``:

    * ``True``  — Celery workers and the FastAPI process are separate,
      so events go through Redis pub/sub (same broker Celery already
      uses) and the WS subscriber loop fans them out.
    * ``False`` — the pipeline runs inside the FastAPI process via
      ``BackgroundTasks``; we use an in-process bus so Redis isn't
      required just to surface progress in the UI.

    Failures are swallowed and logged: a transport hiccup must never
    crash the actual job execution. Worst case the frontend falls back
    to its polling safety net.
    """
    payload = json.dumps(event.model_dump(exclude_none=True))
    settings = get_settings()
    if not settings.USE_CELERY:
        # Lazy import avoids a hard cycle (manager imports events).
        from backend.realtime.local_bus import local_bus

        local_bus.publish(payload)
        return
    try:
        client = _redis_client_sync()
        client.publish(JOB_EVENTS_CHANNEL, payload)
    except Exception as exc:  # pragma: no cover - depends on live Redis
        logger.warning("publish_job_event failed for %s: %s", event.job_id, exc)
