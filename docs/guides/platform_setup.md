# Full Platform Setup

This guide walks you through deploying the complete Skyulf platform — backend API, Celery workers, Redis, and the web UI.

> **Just want the Python library?** See [Installation](../user_guide/installation.md) to `pip install skyulf-core` and skip this page.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Docker Desktop** (recommended) | Docker Compose launches everything in containers — no host Python or Redis needed. |
| **Python 3.12** (manual path) | Only if you prefer running services directly on your machine. |
| **Redis** (manual path) | Required for Celery background tasks. Install natively, via WSL, or run `docker run -d -p 6379:6379 redis:7`. |
| **Git** | To clone the repository. |

---

## 1. Clone the Repository

```bash
git clone https://github.com/flyingriverhorse/Skyulf.git
cd Skyulf
```

---

## 2. Configure Environment Variables

Copy the example environment file and review the settings:

```bash
cp .env.example .env
```

Key variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DB_TYPE` | `sqlite` | Database backend (`sqlite` or `postgres`) |
| `USE_CELERY` | `false` | Set `true` to enable Redis-backed background jobs (requires Redis) |
| `CELERY_BROKER_URL` | `redis://127.0.0.1:6379/0` | Redis URL for Celery message broker |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis URL for caching |
| `SECRET_KEY` | *(random, regenerated each restart)* | **Required in production** — set a stable value or JWT tokens will break on restart |
| `DEBUG` | `true` | Set `false` in production |

> **Minimal setup:** For local development with SQLite and no Celery (the default), no changes are needed — just `cp .env.example .env` and start the server.

---

## 3A. Docker Compose (Recommended)

From the repository root:

```bash
docker compose up --build
```

This starts three containers:

| Service | Port | Purpose |
|---|---|---|
| `api` | `8000` | FastAPI backend + static frontend |
| `redis` | `6379` | Message broker for Celery |
| `worker` | — | Celery worker for background training jobs |

**Verify it's running:**

```bash
curl http://127.0.0.1:8000/health
# Expected: {"status": "healthy", ...}
```

**Useful commands:**

```bash
# Run in background
docker compose up --build -d

# Follow API logs
docker compose logs -f api

# Stop everything
docker compose down
```

> **Tip:** Docker Compose mounts your working tree, so local code edits are picked up on reload.

---

## 3B. Manual Setup (No Docker)

### Create a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements-fastapi.txt
```

### Start Redis (if `USE_CELERY=true`)

```bash
# Option 1: Docker (simplest)
docker run --name skyulf-redis -p 6379:6379 -d redis:7

# Option 2: Native install (Linux)
sudo apt install redis-server && sudo systemctl start redis

# Option 3: Skip Redis entirely
# Set USE_CELERY=false in your .env file
```

### Launch the API server

```bash
python run_skyulf.py
```

Or with uvicorn directly:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Start the Celery worker (optional)

In a second terminal with the virtualenv activated:

```bash
python -m celery -A celery_worker.celery_app worker --pool=solo --loglevel=info --queues mlops-training
```

---

## 4. Verify the Stack

Once the server is running, check these URLs:

| URL | What it is |
|---|---|
| [http://127.0.0.1:8000](http://127.0.0.1:8000) | Web dashboard (main UI) |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | **Swagger UI** — interactive API explorer |
| [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) | **ReDoc** — API reference documentation |
| [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | Health check endpoint |
| [http://127.0.0.1:8000/health/detailed](http://127.0.0.1:8000/health/detailed) | Detailed diagnostics (DB connectivity, Redis/Celery availability, Snowflake status) |

---

## 5. First Login

Authentication is **not yet enabled** in this release — the platform is accessible without credentials. When auth is implemented, credentials will be configured via environment variables (`AUTH_FALLBACK_*` settings in `.env`).

> **Heads-up:** Do not expose the API on a public network until authentication is implemented.

---

## 6. First Steps After Setup

1. **Open the dashboard** at [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. **Upload data:** Navigate to **Data Sources** and upload a CSV file.
3. **Explore your data:** Open **EDA** to see automated profiling, distributions, and outlier alerts.
4. **Build a pipeline:** Open the **ML Canvas**, drag nodes, connect them, and hit **Execute**.
5. **Deploy a model:** Register a trained model and test it from the **Deployments** page.

For a detailed step-by-step walkthrough, see [Platform Walkthrough](platform_walkthrough.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 8000 | Is the API running? Check `docker compose logs api` or your terminal. |
| Celery tasks stuck in "pending" | Verify Redis is running: `redis-cli ping` should return `PONG`. |
| `ModuleNotFoundError` on startup | Activate your virtualenv and run `pip install -r requirements-fastapi.txt`. |
| Database errors after update | Delete `mlops_database.db` (SQLite) to start fresh — it auto-recreates on boot. |
| CORS errors in browser | Add your frontend URL to `CORS_ORIGINS` in `.env`. Default allows `localhost:3000` and `localhost:8080`. |

For more issues, see [Troubleshooting](../user_guide/troubleshooting.md).
