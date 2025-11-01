# Celery & Redis Setup Guide

This guide explains how to bootstrap the asynchronous training stack that powers the training node. It covers application configuration, Redis deployment, and process management on Windows with PowerShell.

## 1. Prerequisites
- Python environment: activate the project virtualenv (`.venv`).
- Project dependencies: `pip install -r requirements-fastapi.txt` (already handled in this repo).
- Redis 6+ available locally (Docker is preferred on Windows).

## 2. Configure Environment Variables
Celery reads its settings from `config.py` via environment variables. The defaults work for localhost, but you can override them in a `.env` file at the project root:

```
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
CELERY_TASK_DEFAULT_QUEUE=mlops-training
TRAINING_ARTIFACT_DIR=uploads/models
```

If you change ports or credentials for Redis, update the URLs accordingly.

## 3. Start Redis with Docker (recommended)
First start docker desktop on your machine. Then open a PowerShell terminal and run Redis in a disposable Docker container:

```powershell
docker run --name redis-mlops -p 6379:6379 redis:7-alpine
```
- Leave this terminal running.
- To stop and remove later: `docker stop redis-mlops; docker rm redis-mlops`.

### Alternative: Native Redis on Windows
If you install Redis directly, ensure the service is running and reachable on the same host/port used in the URLs above.

## 4. Initialize the Database Layer
Apply the Alembic migrations to ensure the new `training_jobs` table exists. From the project root run:

```powershell
cd c:\Users\Murat\Desktop\MLops2
alembic upgrade head
```

The Celery worker bootstraps the async engine the first time it runs, but it expects the metadata tables to exist. If you prefer to initialize via the ORM, run the FastAPI app once to let SQLAlchemy create tables:

```powershell
cd c:\Users\Murat\Desktop\MLops2
.\.venv\Scripts\activate
.\.venv\Scripts\uvicorn.exe main:app --reload
```
- `taskkill /IM uvicorn.exe /F` to stop the server from another terminal.
Keep the API process running while you enqueue jobs.

## 5. Launch the Celery Worker (Windows-friendly command)
Celery’s default fork-based pool does not work on Windows. Use the `solo` pool:

```powershell
cd c:\Users\Murat\Desktop\MLops2
.\.venv\Scripts\activate
.\.venv\Scripts\celery.exe -A celery_worker.celery_app worker --pool=solo --loglevel=info
```

Key flags:
- `-A celery_worker.celery_app` points to the Celery instance defined in `celery_worker.py`.
- `--pool=solo` forces a single-process event loop compatible with Windows.
	- If you forget this flag you will see `PermissionError: [WinError 5] Access is denied` from the billiard worker pool. Restart the worker with `--pool=solo` to resolve it.
- `--loglevel=info` surfaces training progress messages.
- `taskkill /IM celery.exe /F` to stop the worker from another terminal when Ctrl+C is unavailable.
## 6. Dispatch Training Jobs
With Redis, FastAPI, and the worker running:
1. Authenticate via the UI or API.
2. Trigger `POST /api/training-jobs` with the desired payload.
3. Monitor worker logs for execution progress. The job status updates in the database.

## 7. Stopping Services
- Stop the Celery worker with `Ctrl+C`.
- Stop Uvicorn with `Ctrl+C`.
- Stop Redis Docker container (from another shell): `docker stop redis-mlops`.

## 8. Troubleshooting Checklist
- "Connection refused" errors → ensure Redis is running and accessible.
- Celery import errors → verify virtualenv is active before running the worker.
- Schema warnings about Pydantic → already resolved; ensure you are on the latest commit.
- Artifact writes fail → confirm `TRAINING_ARTIFACT_DIR` exists or create it manually.
- Stuck or duplicate jobs → query `training_jobs` in SQLite and mark outdated rows as `cancelled` using `core/feature_engineering/nodes/modeling/model_training_jobs.py` utilities (see `core/feature_engineering/doc/training_job_maintenance.md`).

Following these steps will give you a complete async training environment with Celery, Redis, and FastAPI working together.
