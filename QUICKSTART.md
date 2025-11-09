# Skyulf Quickstart

> **Goal:** Get the Skyulf FastAPI workspace, Celery worker, and optional UI assets running locally.

---

## 1. Decide how you want to run it

- **Option A – Docker Compose (recommended):** Needs only Docker Desktop (or Docker Engine). Compose launches the FastAPI API, Redis broker, and Celery worker in containers. You do *not* need to install Python or Redis on your host unless you want to develop locally.
- **Option B – Manual environment:** Requires Python 3.10/3.11 plus a Redis instance you manage (native install, WSL, or Docker). Choose this if you prefer running services directly on your machine or inside an existing virtualenv.

Node.js stays optional for both paths—it’s only necessary if you plan to build the forthcoming frontend bundle yourself.

---

## 2. Clone the project

```powershell
# Windows PowerShell
cd C:\path\to\projects
git clone https://github.com/flyingriverhorse/skyulf.git
cd skyulf
```

```bash
# macOS / Linux
cd ~/projects
git clone https://github.com/flyingriverhorse/skyulf.git
cd skyulf
```

---

## 3. Fast path: Docker Compose (recommended)

From the repository root, run the stack directly in containers. Compose builds the FastAPI image, starts Redis, and boots the Celery worker—no host-side Python needed.

```powershell
docker compose up --pull always --build
```

- Swagger UI (interactive docs): <http://127.0.0.1:8000/docs>
- ReDoc (reference view): <http://127.0.0.1:8000/redoc>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>
- Health probes: <http://127.0.0.1:8000/health> (`/health/detailed` for diagnostics)
- Stop everything: `Ctrl+C` (or `docker compose down` in another terminal)

> **First run tips:** Add `-d` to run in the background, and use `docker compose logs -f api` to follow the FastAPI container output.

> **Tip:** Compose mounts your working tree, so local code edits are picked up on reload.

---

## 4. Manual setup (no Docker)

If you picked Option B, the steps below set up the services directly on your host.

### 4.1 Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Install runtime dependencies

```powershell
pip install -r requirements-fastapi.txt
```

Redis is required for background training. On Windows, install via [WSL](https://learn.microsoft.com/windows/wsl/install) or Docker (`docker run --name redis-mlops -p 6379:6379 -d redis:7`).

### 4.3 Launch the API server

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Navigate to:

- Swagger UI (interactive docs): <http://127.0.0.1:8000/docs>
- ReDoc (reference view): <http://127.0.0.1:8000/redoc>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>
- Health probes: <http://127.0.0.1:8000/health> (`/health/detailed` for diagnostics)

### 4.4 Start the Celery worker *(optional but recommended)*

Open a second terminal with the virtual environment activated and run:

```powershell
python -m celery -A celery_worker.celery_app worker --pool=solo --loglevel=info --queues mlops-training
```

You can tail logs in `logs/` to watch training and feature-engineering jobs progress.

---

## 5. Load sample data

1. Sign in to the web UI (default admin credentials live in `config.py`).
2. Navigate to **Data Ingestion → Upload** and select a CSV from `data/`.
3. Save the inferred schema to reuse columns in future experiments.

---

## 6. Build your first flow

1. Open **Feature Canvas** and drag a **Dataset Source** node onto the grid.
2. Connect it to a **Train/Val/Test Split** node (70/15/15 by default).
3. Finish with a **Model Trainer** node targeting `RandomForestClassifier`.
4. Hit **Save & Run** — the job is queued immediately and tracked via Celery.

You can follow progress from the **Runs** page or by watching the worker logs.

---

## 7. Next steps

- Explore the `/docs` OpenAPI schema to wire up your own clients.
- Peek at `core/feature_engineering` for custom node examples.
- Join the [GitHub Discussions](https://github.com/flyingriverhorse/skyulf/discussions) to propose new nodes.

Need help? Open an issue or ping the team on Discussions — we respond fast during the alpha phase.
