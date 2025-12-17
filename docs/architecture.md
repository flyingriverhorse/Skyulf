# Architecture

Skyulf is split into a FastAPI backend (the running application) and a standalone Python SDK (skyulf-core).

## Components

1. Frontend: a React UI for interacting with the platform (feature canvas, dataset views).
2. Backend API: a FastAPI service that manages ingestion, jobs, artifacts, and deployments.
3. Worker (optional): Celery + Redis for long-running tasks (training, tuning). The backend can also run without Celery for simple development setups.
4. Storage: SQLite/PostgreSQL for metadata and an artifact store (local filesystem by default).

## Data flow (high level)

1. Data is ingested and profiled.
2. A job executes a preprocessing + modeling pipeline.
3. Fitted transformers and trained models are stored as artifacts.
4. Deployments load those artifacts to serve predictions.

## Repository layout

```text
backend/        # FastAPI app (routes, ingestion, pipeline execution, registry, deployment)
frontend/       # React UI
skyulf-core/    # Python SDK (imported as skyulf)
tests/          # Backend + SDK tests
```

## Key code areas

- backend/data_ingestion: connectors and profiling (Polars-based ingestion)
- backend/ml_pipeline: pipeline execution, artifacts, model registry, deployments
- skyulf-core/skyulf/preprocessing: calculators/appliers (fit/transform pattern)
- skyulf-core/skyulf/modeling: estimators and model wrappers
