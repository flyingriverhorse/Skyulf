# Skyulf MLOps Platform (FastAPI)

Self-hosted, privacy-first MLOps web application built with FastAPI, Celery/Redis for job taks, and SQLAlchemy. It provides data ingestion, feature engineering, model training/tuning, and simple LLM utilities — designed to run with SQLite by default and scale to PostgreSQL.

Note: This repo hosts the FastAPI app. A separate reusable ML utility library lives under `skyulf-mlflow/`.

## Quick start

Prerequisites
- Python 3.10+

Steps
- Create and activate a virtualenv
- Install minimal runtime deps
- Run the API and visit http://127.0.0.1:8000/health

On Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-fastapi.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Optional: start the Celery worker (Redis required)

```powershell
.\.venv\Scripts\python.exe -m celery -A celery_worker.celery_app worker --pool=solo --loglevel=info --queues mlops-training
```

Health endpoints
- GET /health — basic health
- GET /health/detailed — DB/cache checks

## Features
- Data ingestion (CSV/Excel/JSON/Parquet/SQL)
- Feature engineering (math, binning, selection)
- Model training and hyperparameter search (Grid/Random/Halving; optional Optuna integration)
- Task queue for background training with Celery
- Authentication scaffolding and admin pages

## Development
- Configuration via `config.py` with sane defaults (SQLite, dev CORS)
- Lifespan hooks initialize the async DB engine automatically
- Tests under `tests/` cover core feature engineering and training helpers

## License

Apache-2.0. See `LICENSE`.

## NGI Zero fit (at a glance)
- Open source, self-hostable building blocks
- Clear path to privacy-preserving ML (differential privacy, federated adapter), interoperability (ONNX, MLflow-compatible packaging), and auditability (provenance, SBOM)
- See `docs/ngi-zero-concept-note.md` for proposed milestones and deliverables
