# Model Training Modules Overview

This document describes the modeling-node modules that power background training:

- `model_training_registry.py`
- `model_training_jobs.py`
- `model_training_tasks.py`

Keeping these files together under `core/feature_engineering/nodes/modeling/` makes it easier to reason about the entire training workflow directly from the node package.

## `model_training_registry.py`

Purpose: central registry of supported estimators and their sane defaults.

Highlights:
- Declares a small `ModelSpec` dataclass (`key`, `problem_type`, `factory`, `default_params`).
- Registers the models the node can launch (logistic regression, random forest classifier/regressor, ridge regression).
- Exposes `get_model_spec(model_type)` for lookups and `list_registered_models()` for discovery.
- Used by the Celery task to instantiate models with consistent defaults when the node omits explicit hyperparameters.

## `model_training_jobs.py`

Purpose: database-facing helpers for the `training_jobs` table.

Highlights:
- Computes monotonically increasing versions per `(dataset_source_id, node_id, model_type)` via `_resolve_next_version(...)`.
- `create_training_job(...)` persists the request payload, graph snapshot, hyperparameters, and metadata before returning the ORM instance.
- `get_training_job(...)`, `list_training_jobs(...)`, and `bulk_mark_cancelled(...)` support API endpoints and operational scripts.
- `update_job_status(...)` centralises status transitions, timestamp updates, metric storage, and error messaging.

## `model_training_tasks.py`

Purpose: Celery worker entry points plus the orchestration logic that executes training asynchronously.

Highlights:
- Configures the Celery application (`celery_app`) using settings from `config.get_settings()`.
- Lazily initialises the async SQLAlchemy engine inside `_ensure_database_ready()` to avoid race conditions.
- `_prepare_training_data(...)`, `_classification_metrics(...)`, `_regression_metrics(...)`, and `_train_and_save_model(...)` handle the full fit/evaluate/persist cycle, delegating model construction to the registry.
- `_resolve_training_inputs(...)` reaches back into `core.feature_engineering.routes` to rebuild the canvas execution order and run all upstream nodes before training starts.
- `_run_training_workflow(...)` is the main coroutine invoked by the Celery task: it updates job status, loads data, merges hyperparameters, generates metrics, writes artifacts under `uploads/models/{pipeline_id}/`, and records successes or failures.
- `train_model` is the Celery task entrypoint; `dispatch_training_job` is the helper the API uses to enqueue work (`train_model.delay(job_id)`).

## Data Flow Summary

1. A client posts to `POST /ml-workflow/api/training-jobs` with the graph and node configuration.
2. The API stores the job via `model_training_jobs.create_training_job(...)`.
3. When `run_training=True`, it calls `dispatch_training_job(job_id)`.
4. Celery picks up the task, runs `_run_training_workflow(...)`, and updates the record through the job helpers.
5. Clients poll `/ml-workflow/api/training-jobs/{job_id}` for live status, metrics, and artifact locations.

Use this document as the quick reference when modifying the training node or adding new model families.
