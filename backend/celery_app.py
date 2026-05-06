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
