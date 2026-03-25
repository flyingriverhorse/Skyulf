"""Celery / background task settings."""


class CeleryMixin:
    """Celery broker, backend, and queue configuration."""

    USE_CELERY: bool = False
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"
    CELERY_TASK_DEFAULT_QUEUE: str = "mlops-training"
