# Hyperparameter Tuning Package

Concise guide to the tuning workflow used by the Hyperparameter Tuning node.

## Responsibilities
- Persist tuning job requests and surface status/summary data.
- Normalize search spaces, strategies, and Optuna integration.
- Reuse training utilities to prepare data and metrics for candidate models.
- Trigger Celery tasks that run Grid/Random/Halving/Optuna searches and apply results.

## Key Modules
- `tasks.py` – Celery orchestration, strategy dispatch, result persistence.
- `jobs.py` – async DB helpers to create/update/list tuning jobs.
- `registry.py` – list of supported strategies plus resolver helpers exposed to schemas and APIs.

## Tips
- Use `registry.resolve_strategy_selection` to validate UI payloads before enqueuing jobs.
- When adding a new search strategy, keep the implementation self-contained and register it via the registry to avoid touching Celery internals.
