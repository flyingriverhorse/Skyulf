"""The shared Celery app and the reliability settings every task inherits.

One ``Celery`` instance built from ``Settings`` read at import time, which the
task modules and the worker bootstrap both import. No tasks are declared or
autodiscovered here: registration happens purely by importing the task modules,
which is what ``celery_worker.py`` does for the worker process. A task module
nobody imports is therefore invisible to the worker even though the web process
can still enqueue it.

The configuration trades duplicated work for lost work — ``task_acks_late`` with
``task_reject_on_worker_lost`` re-queues a task whose worker dies mid-run, and
``worker_prefetch_multiplier=1`` stops a worker reserving tasks it has not
started. Serialization is JSON in both directions, so task arguments have to be
JSON-encodable rather than arbitrary Python objects.

``beat_schedule`` prunes the ``error_events`` table once a day.
"""

from celery import Celery

from backend.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "mlops_training",
    broker=_settings.CELERY_BROKER_URL,
    backend=_settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_default_queue=_settings.CELERY_TASK_DEFAULT_QUEUE,
    task_acks_late=True,
    task_reject_on_worker_lost=True,  # re-queue task instead of silently losing it when worker dies mid-execution
    worker_prefetch_multiplier=1,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    beat_schedule={
        # Delete error events older than ERROR_LOG_RETENTION_DAYS once per day.
        "cleanup-error-events-daily": {
            "task": "monitoring.cleanup_error_events",
            "schedule": 86400,  # seconds — every 24 hours
        },
    },
)
