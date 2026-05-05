"""
Celery worker bootstrap.

© 2025 Murat Unsal — Skyulf Project
"""

from __future__ import annotations

import asyncio
import logging

from celery.signals import setup_logging, worker_process_init

from backend.celery_app import celery_app
from backend.config import get_settings
from backend.utils.logging_utils import setup_universal_logging

# Ensure tasks register with the shared Celery app
from backend.data_ingestion import tasks as _ingestion_tasks  # noqa: F401
from backend.database.engine import init_db
from backend.ml_pipeline import tasks as _ml_pipeline_tasks  # noqa: F401

__all__ = ["celery_app"]

logger = logging.getLogger(__name__)


@setup_logging.connect
def configure_logging(sender=None, **kwargs):
    """Configure logging when Celery starts."""
    setup_universal_logging(
        log_file="logs/celery_worker.log", log_level="INFO", console_log_level="INFO"
    )


@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize DB connection and optional Sentry for the worker process."""
    asyncio.run(init_db())

    settings = get_settings()
    if settings.SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            release=settings.APP_VERSION,
            traces_sample_rate=0.1,
            integrations=[CeleryIntegration()],
            send_default_pii=False,
        )
        logger.info("Sentry initialised in Celery worker (release=%s)", settings.APP_VERSION)


if __name__ == "__main__":  # pragma: no cover - manual worker entrypoint
    celery_app.start()
