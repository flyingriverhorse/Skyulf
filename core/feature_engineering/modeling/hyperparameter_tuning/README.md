# Hyperparameter Tuning Package

Concise guide to the tuning workflow used by the Hyperparameter Tuning node.

## Responsibilities
- Persist tuning job requests and surface status/summary data.
- Normalize search spaces, strategies, and Optuna integration.
- Reuse shared modeling utilities to prepare data and metrics for candidate models.
- Trigger Celery tasks that run Grid/Random/Halving/Optuna searches and apply results.

## Package Layout
- `tasks/` – modular Celery orchestration. The package keeps the legacy import surface (`core.feature_engineering.modeling.hyperparameter_tuning.tasks`) but splits responsibilities:
  - `workflow.py` owns `_run_hyperparameter_tuning_workflow`, Celery task registration, and result persistence.
  - `searchers.py` builds sklearn/Optuna searchers and prepares sanitized hyperparameter spaces.
  - `optuna_support.py` encapsulates optional Optuna imports, samplers, and integration guards.
  - `execution.py` runs searches, aggregates warnings, and computes metrics via the shared evaluation helpers.
  - `data_bundle.py` isolates the training/validation split preparation borrowed from `shared`.
- `jobs/` – async DB helpers grouped by concern:
  - `repository.py` handles lookups and run-number resolution.
  - `status.py` centralizes status transitions and cancellation utilities.
  - `service.py` exposes `create_tuning_job`/`purge_tuning_jobs` and composes the lower-level helpers.
- `registry/` – strategy defaults, option types, and resolver helpers consumed by schemas and APIs (unchanged).

## Tips
- Import primitives from `core.feature_engineering.modeling.shared` instead of copying data prep or metric logic into workflow modules.
- Use `registry.resolve_strategy_selection` to validate UI payloads before enqueuing jobs.
- When adding a new search strategy, keep the implementation self-contained (typically inside `tasks/searchers.py`) and register it via the registry to avoid touching Celery internals.

Tests under `tests/test_hyperparameter_tuning_*` plus the new `tests/test_shared_*.py` suites exercise the shared helpers that the modular workflow depends on.
