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
| **Hybrid Engine** | Polars for fast ingestion, Pandas/Scikit-Learn for ML compatibility |
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

1. Follow the [Quickstart](examples/quickstart.md) to get the backend running.
2. Open the ML Canvas in your browser and start building pipelines visually.

### Using skyulf-core as a Library

1. [Installation](user_guide/installation.md) — Install from PyPI or editable mode.
2. [Overview](user_guide/overview.md) — Understand the Calculator/Applier pattern.
3. [Pipeline Quickstart](user_guide/pipeline_quickstart.md) — Build your first pipeline in Python.
4. [Step-by-Step (No Config)](user_guide/step_by_step_no_config.md) — Low-level building blocks.
5. [Reference → Preprocessing Nodes](reference/preprocessing_nodes.md) / [Modeling Nodes](reference/modeling_nodes.md) — Full node catalog.

### Going Deeper

- [Automated EDA & Profiling](user_guide/eda_profiling.md) — Statistical analysis powered by Polars.
- [Hyperparameter Tuning](user_guide/hyperparameter_tuning.md) — Grid, Random, Halving, Optuna strategies.
- [Drift Monitoring](user_guide/drift_monitoring.md) — Detect data drift in production.
- [Extending Skyulf-Core](user_guide/extending_custom_nodes.md) — Add your own nodes.
- [Validation vs scikit-learn](user_guide/validation_vs_sklearn.md) — Proof that Skyulf avoids leakage.

---

## Links

- **GitHub**: [github.com/flyingriverhorse/Skyulf](https://github.com/flyingriverhorse/Skyulf)
- **PyPI**: [pypi.org/project/skyulf-core](https://pypi.org/project/skyulf-core/)
- **Changelog**: [VERSION_UPDATE.md](https://github.com/flyingriverhorse/Skyulf/blob/master/VERSION_UPDATE.md)

