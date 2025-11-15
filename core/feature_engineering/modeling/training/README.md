# Training Package

Short reference for the asynchronous training workflow.

## Responsibilities
- Queue and run Celery jobs that fit models and persist artifacts.
- Prepare feature/target frames and cross-validation splits.
- Compute metrics and evaluation payloads for downstream nodes.
- Manage database records for job lifecycle updates.

## Key Modules
- `tasks.py` – Celery task entrypoints plus orchestration helpers.
- `jobs.py` – async DB helpers for creating, listing, updating training jobs.
- `registry.py` – catalog of supported estimators and their factories/defaults.
- `evaluation.py` – utilities for building evaluation reports and signals.
- `train_model_draft.py` – lightweight node that validates readiness during previews.
- `shared/` – cross-cutting helpers (data prep, CV configs, artifact persistence, metrics) reused by training and tuning packages.

## Tips
- Import reusable primitives from `core.feature_engineering.modeling.shared` instead of duplicating logic inside training modules.
- Use `jobs.create_training_job` before calling `tasks.dispatch_training_job` to ensure DB state is consistent.
- Keep estimator additions localized to `registry.py` so Celery workers and APIs stay in sync.
