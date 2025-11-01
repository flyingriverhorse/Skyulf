Build Plan

Database & Schemas

Add SQLAlchemy models (TrainingJob, optional TrainingJobEvent) plus Alembic migration.
Define Pydantic payloads/responses (TrainingJobCreate, TrainingJobStatus, version metadata).
Extend existing FeaturePipelineResponse or add new response wrapper for active jobs.
API Surface

Implement POST /ml-workflow/api/training-jobs to create rows, enqueue Celery task, autosave pipeline.
Implement GET /ml-workflow/api/training-jobs/{job_id} and GET /ml-workflow/api/training-jobs (filter by user/pipeline).
Add pipeline hydration endpoint support for returning job summaries.
Celery Integration

Configure Celery app (if empty) with Redis broker/backing; add train_model.delay(job_id).
Task skeleton: load job config/graph, run _run_pipeline_execution to training node, fit model using defaults, persist artifact/metrics, update status/version.
Artifact & Version Registry

Implement MODEL_DEFAULTS registry (hyperparameter defaults + type hints).
Build artifact writer (local FS for now), version increment logic on success, metrics capture.
Frontend Hooks (Phase 1 backend only? maybe but part of plan)

Node settings UI (model selector, hyperparams, run button, status badges).
React Query mutation/polling for job status, autosave behavior, hydration merge.
Monitoring & UX

User dashboard list of training jobs (/users/me/training-jobs).
Canvas warnings when graph changes while job running; pipeline hash change detection.
Hardening

Error handling, cancellation endpoint, retries/timeouts, access control.
Tests: unit tests for schemas/task logic, integration tests for API + Celery (maybe using eager mode).
Will confirm scope for first implementation (likely DB+API+Celery skeleton) before moving to UI.