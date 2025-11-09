# Skyulf

[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue)](LICENSE)
[![Commercial](https://img.shields.io/badge/commercial-license-blueviolet)](COMMERCIAL-LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](#quick-start)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![DCO](https://img.shields.io/badge/DCO-required-green)](DCO.txt)
![](https://img.shields.io/badge/tagline-MLOps%20%3A%20FastAPI%20%2B%20Celery-blue)

> ⚠️ **Early Alpha (v0.0.1)**: This is a passion project in active development. Expect bugs, incomplete features, and visual inconsistencies. Use at your own risk and please report issues on GitHub!

Skyulf is a self‑hosted, privacy‑first MLOps web application. It runs locally by default (SQLite) and scales to PostgreSQL without requiring Kubernetes or cloud services.

What you get out of the box:
- A web UI for data ingestion
- A low‑code Feature Canvas for feature engineering
- Built‑in model training and hyperparameter search (Grid/Random/Halving; Optuna optional)
- Background jobs with Celery/Redis
- LLM helpers that can run locally (Ollama) or via APIs, with dataset‑aware guardrails - if configured (Planned).

## Table of Contents

- [Contributing](#contributing)
- [Quick start](#quick-start)
- [Features](#features)
- [Workflow overview](#workflow-overview)
- [Development](#development)
- [License](#license)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow guidance, and read our [Code of Conduct](CODE_OF_CONDUCT.md). 

**Important:** All commits must include a DCO sign-off (`git commit -s`). See [DCO.txt](DCO.txt) for details.

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

Open:
- API health — http://127.0.0.1:8000/health
- Docs (dev mode) — http://127.0.0.1:8000/docs

Health endpoints
- GET /health — basic health
- GET /health/detailed — DB/cache checks

## Features
- Data ingestion (CSV/Excel/JSON/Parquet/SQL)
- Feature engineering (math, binning, selection)
- Model training and hyperparameter search (Grid/Random/Halving; optional Optuna integration)
- Task queue for background training with Celery
- Authentication scaffolding and admin pages
- LLM helpers (OpenAI, Anthropic, DeepSeek, or local Ollama) - Not integrated yet

## Workflow overview

The high-level flow from dataset to model training inside Skyulf:

<p align="center">
	<img src="static/img/image.png" alt="Dataset → Train/Val/Test split → Celery-driven model trainer" width="520">
	<br />
	<em>Dataset source → train/val/test split → background model training (Celery)</em>
</p>

## Development
- Configuration via `config.py` with sane defaults (SQLite, dev CORS)
- Lifespan hooks initialize the async DB engine automatically
- Tests under `tests/` cover core feature engineering and training helpers
- `docker-compose.yml` to run API + Redis (+ Celery worker)

## License

Open Source: AGPL-3.0-or-later. See `LICENSE` for full terms.
Commercial: A commercial license is available for closed-source integrations that cannot comply with AGPL’s reciprocal and network-use obligations. See `COMMERCIAL-LICENSE.md` for an overview and how to request terms.

If you run Skyulf as a network service and modify it, AGPL requires you to make your modified source available to users interacting with it over the network.

---

If you'd like to contribute, sponsor, or request a commercial license, please star the repo, open a Discussion or issue, or see `.github/FUNDING.yml` for sponsorship options.

---

Thank you for checking out Skyulf. Whether you’re here to experiment, contribute, or run private models for your team, I hope this project makes MLOps more approachable.

> "Not all those who wander are lost." — J.R.R. Tolkien <img src="static/images/lotr-ring.svg" alt="ring" width="20" height="20" style="vertical-align:middle;margin-left:6px;">

---

© 2025 Murat Unsal — Skyulf Project  
SPDX-License-Identifier: AGPL-3.0-or-later

