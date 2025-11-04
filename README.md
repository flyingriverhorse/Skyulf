# Skyulf — local‑first MLOps web app

![CI](https://github.com/flyingriverhorse/skyulf-mlflow/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/flyingriverhorse/skyulf-mlflow/actions/workflows/codeql.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)

**© 2025 Murat Unsal — Skyulf Project**

> ⚠️ **Early Alpha (v0.0.1)**: This is a passion project in active development. Expect bugs, incomplete features, and visual inconsistencies. Use at your own risk and please report issues on GitHub!

Skyulf is a self‑hosted, privacy‑first MLOps web application. It runs locally by default (SQLite) and scales to PostgreSQL without requiring Kubernetes or cloud services.

What you get out of the box:
- A web UI for data ingestion and quick EDA
- A low‑code Feature Canvas for feature engineering
- Built‑in model training and hyperparameter search (Grid/Random/Halving; Optuna optional)
- Background jobs with Celery/Redis
- LLM helpers that can run locally (Ollama) or via APIs, with dataset‑aware guardrails - if configured.

Note: This repo hosts the FastAPI app. A separate reusable ML utility library lives under `skyulf-mlflow/`.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow guidance, and read our [Code of Conduct](CODE_OF_CONDUCT.md). Use GitHub Issues for bugs and Discussions for questions. Good first issues are labeled.

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

Or with Docker Compose (recommended for a 1‑command dev run)

```powershell
docker compose up --pull=always --build
```

Then open:
- API health: http://127.0.0.1:8000/health
- Docs (dev mode): http://127.0.0.1:8000/docs

Health endpoints
- GET /health — basic health
- GET /health/detailed — DB/cache checks

## Features
- Data ingestion (CSV/Excel/JSON/Parquet/SQL)
- Feature engineering (math, binning, selection)
- Model training and hyperparameter search (Grid/Random/Halving; optional Optuna integration)
- Task queue for background training with Celery
- Authentication scaffolding and admin pages
 - LLM helpers (OpenAI, Anthropic, DeepSeek, or local Ollama)

## Development
- Configuration via `config.py` with sane defaults (SQLite, dev CORS)
- Lifespan hooks initialize the async DB engine automatically
- Tests under `tests/` cover core feature engineering and training helpers
 - `docker-compose.yml` to run API + Redis (+ Celery worker)

## License

Apache-2.0. See `LICENSE`.

If you’re interested in contributing or sponsoring, see `.github/FUNDING.yml` and open a Discussion. Good first issues will be labeled.

## European dimension
Skyulf directly supports GDPR-aligned data minimisation and data sovereignty by keeping EDA/training on-prem by default, enabling SMEs, municipalities, and public institutions across Europe to work with sensitive datasets without cloud egress.
