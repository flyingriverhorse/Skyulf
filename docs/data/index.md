# Data Ingestion Overview

Skyulf ingests datasets through connectors (file, SQL, API) and runs lightweight profiling so downstream pipelines know row/column counts and inferred types.

## Architecture Note: Polars vs Pandas

Skyulf uses a **two-library architecture** for optimal performance:

| Component | Library | Reason |
|-----------|---------|--------|
| **Data Ingestion** (`core.data_ingestion`) | **Polars** | Fast I/O, lazy evaluation, low memory |
| **ML Pipeline (Library)** (`skyulf` from `skyulf-core`) | **Pandas** | Scikit-learn compatibility, rich ecosystem |

Data flows from Polars (ingestion) → Pandas (preprocessing/modeling) automatically when you run pipelines.

## Quick Demo (file connector + profiler)

```python
import asyncio
import os
import tempfile
from core.data_ingestion.connectors.file import LocalFileConnector
from core.data_ingestion.engine.profiler import DataProfiler

async def demo():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        tmp.write("feature,target\n1,0\n2,1\n3,0\n")
        path = tmp.name

    try:
        connector = LocalFileConnector(path)
        await connector.connect()
        df = await connector.fetch_data(limit=5)
        profile = DataProfiler.profile(df)
        return profile["row_count"], list(profile["columns"].keys())
    finally:
        os.remove(path)

print(asyncio.run(demo()))
```

## Ingestion Flow
- **Upload** a file or register a source (SQL/API) with connection details.
- **Connector** validates and fetches a sample/full dataset.
- **Profiler** computes row/column counts, basic stats, and inferred schema.
- **Metadata** is stored on the `DataSource` record for downstream jobs.

## Key Components
- `core.data_ingestion.connectors` — pluggable connectors for files, SQL databases, and REST APIs.
- `core.data_ingestion.engine.profiler.DataProfiler` — basic statistics used by pipelines and UI.
- `core.data_ingestion.service.DataIngestionService` — orchestrates uploads, sampling, and ingestion tasks.
- `core.data_ingestion.tasks.ingest_data_task` — background ingestion that runs profiling and updates metadata.
