# Modeling Config Package

Short reference for shared configuration helpers across modeling modules.

## Responsibilities
- Centralize defaults for training/tuning nodes (data splits, metrics, serialization paths).
- Provide schema-backed settings objects consumed by Celery tasks and FastAPI endpoints.
- Offer helpers for reading/writing persisted modeling configuration artifacts.

## Key Modules
- `hyperparameters/base.py` – defines the reusable `HyperparameterField` descriptor.
- `hyperparameters/<model>.py` – model-specific field lists kept in isolated files (e.g., logistic_regression, random_forest_classifier).
- `hyperparameters/registry.py` – aggregates all models and exposes lookup helpers consumed by APIs/tasks.

## Tips
- Keep new config knobs here first, then wire them into training/tuning modules to avoid duplicated literals.
- Add Pydantic validators for anything user-facing so FastAPI and Celery fail fast with readable errors.
