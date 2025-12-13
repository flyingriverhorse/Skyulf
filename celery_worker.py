"""
Celery worker bootstrap.

© 2025 Murat Unsal — Skyulf Project
"""

from __future__ import annotations

from backend.celery_app import celery_app
# Ensure tasks register with the shared Celery app
from backend.data_ingestion import tasks as _ingestion_tasks  # noqa: F401
from backend.ml_pipeline import tasks as _ml_pipeline_tasks  # noqa: F401
from backend.config import setup_universal_logging
from celery.signals import setup_logging, worker_process_init
from backend.database.engine import init_db
import asyncio

__all__ = ["celery_app"]

@setup_logging.connect
def configure_logging(sender=None, **kwargs):
    """Configure logging when Celery starts."""
    setup_universal_logging(
        log_file="logs/celery_worker.log",
        log_level="INFO",
        console_log_level="INFO"
    )

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize database connection for the worker process."""
    asyncio.run(init_db())

if __name__ == "__main__":  # pragma: no cover - manual worker entrypoint
    celery_app.start()
