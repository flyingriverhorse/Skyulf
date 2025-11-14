# Modeling Config Package

Short reference for shared configuration helpers across modeling modules.

## Responsibilities
- Centralize defaults for training/tuning nodes (data splits, metrics, serialization paths).
- Provide schema-backed settings objects consumed by Celery tasks and FastAPI endpoints.
- Offer helpers for reading/writing persisted modeling configuration artifacts.

## Key Modules
- `defaults.py` (or similarly named files) – constants for batch sizes, seeds, cross-validation folds.
- `schemas.py` – Pydantic models describing request/response payloads for modeling routes.
- `io.py` / `storage.py` – load/save utilities for job configs and fitted parameter bundles.

## Tips
- Keep new config knobs here first, then wire them into training/tuning modules to avoid duplicated literals.
- Add Pydantic validators for anything user-facing so FastAPI and Celery fail fast with readable errors.
