# Skyulf

> **The Visual MLOps Builder** — Drag-and-drop machine learning without writing code.

Skyulf is a self-hosted, privacy-first MLOps platform. Bring your data, clean it visually, engineer features with a node-based canvas, train models, and deploy — all in one place.

---

## Why Skyulf?

| Feature | What you get |
|---------|-------------|
| **Visual ML Canvas** | Build end-to-end pipelines by connecting nodes on a React Flow canvas |
| **20+ Models** | Classification & Regression (Random Forest, XGBoost, SVM, KNN, and more) |
| **Advanced Tuning** | Grid, Random, Successive Halving, and Optuna with configurable Samplers & Pruners |
| **Automated EDA** | Profiling, outlier detection, PCA, causal discovery, drift monitoring |
| **Dual Engine** | Polars for fast ingestion/preprocessing, Zero-Copy NumPy/Scikit-Learn via `SklearnBridge` |
| **Leakage Prevention** | Calculator/Applier architecture ensures train-only statistics |
| **One-Click Deploy** | Serve models via REST API with built-in inference testing |

---

## Quick Install

### skyulf-core (standalone Python library)

```bash
pip install skyulf-core
```

### Full platform (backend + frontend)

```bash
git clone https://github.com/flyingriverhorse/Skyulf.git
cd Skyulf
pip install -r requirements-fastapi.txt
python run_skyulf.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the dashboard.

---

## Architecture

Skyulf is built on three components with strict boundaries:

- **Frontend**: React + TypeScript + React Flow (visual ML canvas)
- **Backend**: FastAPI + Celery + Redis (API, jobs, persistence)
- **Core Library**: `skyulf-core` (standalone Python package on [PyPI](https://pypi.org/project/skyulf-core/))
- **Data Engine**: Hybrid Polars (high-performance ingestion) + Pandas (ML ecosystem)

```
frontend  →  (HTTP)  →  backend  →  (import)  →  skyulf-core
```

See [Architecture](architecture.md) and [Data Architecture](data_architecture.md) for deep dives.

---

## Where to Start

### Using the Web Platform

1. Follow the [Platform Setup](guides/platform_setup.md) to get the backend running (Docker or manual).
2. See the [Platform Walkthrough](guides/platform_walkthrough.md) for an 8-step end-to-end guide through the UI.
3. Explore the interactive API docs at [/docs](http://127.0.0.1:8000/docs) (Swagger UI) or [/redoc](http://127.0.0.1:8000/redoc) (ReDoc).

**Backend reference:**
- [Backend API Reference](guides/backend_api.md) — All REST endpoints, request/response shapes, rate limits, WebSocket events.
- [Backend Configuration Reference](guides/backend_configuration.md) — Every `.env` variable grouped by category.

**Frontend development:**
```bash
# Terminal 1 — backend
cd /path/to/Skyulf && .venv/bin/python3 run_fastapi.py

# Terminal 2 — frontend dev server (proxies /api/* to localhost:8000)
cd frontend/ml-canvas && npm run dev
# Opens at http://localhost:5173
```

### Using skyulf-core as a Library

1. [Installation](user_guide/installation.md) — Install from PyPI or editable mode.
2. [Overview](user_guide/overview.md) — Understand the Calculator/Applier pattern.
3. [Pipeline Quickstart](user_guide/pipeline_quickstart.md) — Build your first pipeline in Python.
4. [Step-by-Step (No Config)](user_guide/step_by_step_no_config.md) — Low-level building blocks.
5. [Reference → Preprocessing Nodes](reference/preprocessing_nodes.md) / [Modeling Nodes](reference/modeling_nodes.md) — Full node catalog.

### Going Deeper

- [Automated EDA & Profiling](user_guide/eda_profiling.md) — Statistical analysis powered by Polars.
- [Hyperparameter Tuning](user_guide/hyperparameter_tuning.md) — Grid, Random, Halving, Optuna strategies.
- [SHAP Explainability](user_guide/shap_explainability.md) — Global + per-row model explanations (summary, beeswarm, dependence, waterfall).
- [Segmentation (Clustering)](user_guide/segmentation.md) — Group rows by similarity with K-Means, no target column needed.
- [Drift Monitoring](user_guide/drift_monitoring.md) — Detect data drift in production.
- [Extending Skyulf-Core](user_guide/extending_custom_nodes.md) — Add your own nodes.
- [Validation vs scikit-learn](user_guide/validation_vs_sklearn.md) — Proof that Skyulf avoids leakage.

---

## Links

- **GitHub**: [github.com/flyingriverhorse/Skyulf](https://github.com/flyingriverhorse/Skyulf)
- **PyPI**: [pypi.org/project/skyulf-core](https://pypi.org/project/skyulf-core/)
- **Changelog**: [CHANGELOG.md](https://github.com/flyingriverhorse/Skyulf/blob/master/CHANGELOG.md)
