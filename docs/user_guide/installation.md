# Installation

## Requirements

- **Python 3.9+** (3.10 or 3.11 or newer recommended)
- pip 21 or newer

## Install from PyPI

```bash
pip install skyulf-core
```

## Optional extras

Install only what you need:

| Extra | What it adds | Command |
|---|---|---|
| `viz` | Matplotlib, Rich (visualisation helpers) | `pip install skyulf-core[viz]` |
| `eda` | VADER sentiment, causal-learn | `pip install skyulf-core[eda]` |
| `tuning` | Optuna + Optuna-integration | `pip install skyulf-core[tuning]` |
| `modeling-xgboost` | XGBoost estimators | `pip install skyulf-core[modeling-xgboost]` |
| `preprocessing-imbalanced` | imbalanced-learn (SMOTE, etc.) | `pip install skyulf-core[preprocessing-imbalanced]` |
| `dev` | pytest, twine, build | `pip install skyulf-core[dev]` |

Install everything at once:

```bash
pip install skyulf-core[viz,eda,tuning,modeling-xgboost,preprocessing-imbalanced]
```

## Editable install (contributor workflow)

If you cloned the repository and want live changes reflected:

```bash
git clone https://github.com/flyingriverhorse/Skyulf.git
cd Skyulf
pip install -e ./skyulf-core
```

## Runtime dependencies (auto-installed)

These are pulled in automatically by `pip install skyulf-core`:

- **pandas** >= 2.0
- **numpy** >= 1.24
- **scikit-learn** >= 1.4
- **polars** >= 1.36
- **pyarrow** >= 21.0
- **pydantic** >= 2.0
- **scipy** >= 1.10
- **statsmodels** >= 0.14

## Full platform (Docker)

To run the complete Skyulf platform (backend + frontend + Celery workers):

### 1. Configure environment

```bash
cp .env.example .env
```

Key variables to review:

| Variable | Default | Description |
|---|---|---|
| `DB_TYPE` | `sqlite` | Database backend (`sqlite` or `postgres`) |
| `USE_CELERY` | `true` | Set `false` to skip Redis and run tasks in-process |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Redis URL for Celery |
| `SECRET_KEY` | *(placeholder)* | **Change this** in production |

> **Quick start:** For local development with no Redis, just set `USE_CELERY=false` in `.env`.

### 2. Launch with Docker Compose

```bash
docker compose up --build
```

This starts three containers:

| Service | Port | Purpose |
|---|---|---|
| `api` | `8000` | FastAPI backend + static frontend |
| `redis` | `6379` | Message broker for Celery |
| `worker` | — | Celery worker for background training |

### 3. Verify

- Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Default login: **admin** / **admin123** (change in production)

For the full manual setup (without Docker), see the [Platform Setup guide](../guides/platform_setup.md).

## Import check

Verify the installation:

```python
from skyulf.pipeline import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

print("skyulf-core installed successfully")
```
