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

    # ── Job lifecycle / scheduling tunables ─────────────────────────────────

    # On API startup, any training/tuning job still "running" after this many
    # hours is assumed orphaned (e.g. the process was killed mid-run) and is
    # auto-marked "failed" so it stops showing as stuck in the UI. Raise this
    # if you expect legitimate jobs to run longer than the default.
    JOB_ORPHAN_STALE_HOURS: int = 2

    # De-duplication window: a resubmission of the same pipeline+node within
    # this many seconds of a prior submission is treated as a duplicate (e.g.
    # accidental double-click / client retry) rather than a new job.
    JOB_IDEMPOTENCY_WINDOW_SECONDS: int = 30

    # Ceiling on how many worker threads a single parallel-branch pipeline run
    # may spawn (e.g. `run_pipeline_batch_task`, `run_pipeline.py`'s branch
    # executor). Without a cap, a pipeline with many parallel branches would
    # size its thread pool 1:1 with branch count, risking thread exhaustion on
    # wide graphs.
    MAX_PARALLEL_BRANCH_WORKERS: int = 8

    # ── Monitoring / observability query caps ───────────────────────────────
    # Upper bounds on client-requested time windows for monitoring endpoints,
    # to keep dashboard queries and DB load bounded regardless of what a
    # caller asks for.
    MONITORING_MAX_TIMELINE_HOURS: int = 168  # 7 days
    MONITORING_MAX_SLOWNODES_DAYS: int = 90
