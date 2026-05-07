"""Periodic Celery tasks for the monitoring module."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from backend.celery_app import celery_app
from backend.config import get_settings
from backend.database.engine import async_session_factory

logger = logging.getLogger(__name__)


@celery_app.task(name="monitoring.cleanup_error_events", bind=True, max_retries=3)
def cleanup_error_events(self) -> dict:  # type: ignore[override]
    """Delete error_events older than ERROR_LOG_RETENTION_DAYS. Runs daily."""
    settings = get_settings()
    retention_days: int = getattr(settings, "ERROR_LOG_RETENTION_DAYS", 30)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

    async def _run() -> int:
        from sqlalchemy import delete
        from backend.database.models import ErrorEvent

        if not async_session_factory:
            raise RuntimeError("Database not initialized")
        async with async_session_factory() as session:
            result = await session.execute(delete(ErrorEvent).where(ErrorEvent.created_at < cutoff))  # ty: ignore[invalid-argument-type]
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

    try:
        deleted: int = asyncio.run(_run())
        logger.info(
            "cleanup_error_events: deleted %d rows older than %d days", deleted, retention_days
        )
        return {"deleted": deleted, "retention_days": retention_days}
    except Exception as exc:
        logger.error("cleanup_error_events failed: %s", exc)
        raise self.retry(exc=exc, countdown=3600)
