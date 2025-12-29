# Skyulf

Skyulf is a self-hosted MLOps platform with:

- a FastAPI backend
- a React frontend
- a standalone Python ML library: **skyulf-core**

## Architecture

Skyulf is built on a modern stack:

- **Frontend**: React + TypeScript + React Flow
- **Backend**: FastAPI + Celery + Redis
- **Core Library**: `skyulf-core` (Standalone Python package)
- **Data Engine**: Hybrid Polars (Ingestion) + Pandas (ML)

See [Architecture](architecture.md) and [Data Architecture](data_architecture.md) for details.

## If you are here for skyulf-core

Start with:

- **User Guide → Overview**
- **User Guide → Pipeline Quickstart**
- **Reference → Preprocessing Nodes / Modeling Nodes**

## Backend quick start

```bash
pip install -r requirements-fastapi.txt
python run_skyulf.py
```

Open:

- http://127.0.0.1:8000

