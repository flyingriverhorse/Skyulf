"""Celery / background task settings."""


class CeleryMixin:
    """Celery broker, backend, and queue configuration."""

    USE_CELERY: bool = False
    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"
    CELERY_TASK_DEFAULT_QUEUE: str = "mlops-training"

    # Error-log retention: events older than this many days are auto-deleted.
    ERROR_LOG_RETENTION_DAYS: int = 30

    # ── Tuning Parallelism ───────────────────────────────────────────────────
    # Controls joblib parallelism inside hyperparameter searchers.
    #
    # TUNING_N_JOBS
    #   1   = sequential (default — always safe from FastAPI BackgroundTasks on
    #         macOS/Windows; avoids spawn/loky deadlocks).
    #   -1  = all CPUs. Safe only with TUNING_PARALLEL_BACKEND=threading from
    #         FastAPI, or any value inside a dedicated Celery worker.
    #   N   = use exactly N workers.
    #
    # TUNING_PARALLEL_BACKEND
    #   ""         = no override (joblib default: loky/processes). Only safe
    #                from a Celery worker process.
    #   threading  = thread-based. Safe from FastAPI BackgroundTasks. Speeds up
    #                GIL-releasing models (LightGBM, XGBoost, CatBoost).
    #   loky       = process-based (joblib default). Use only from Celery workers.
    TUNING_N_JOBS: int = 1
    TUNING_PARALLEL_BACKEND: str = ""
