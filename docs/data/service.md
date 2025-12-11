# Data Ingestion Service

`DataIngestionService` orchestrates uploads, sampling, and the background ingestion task that profiles data and persists metadata on `DataSource` records.

## Minimal Response Object
The ingestion API returns an `IngestionJobResponse` so callers can poll status.

```python
from core.data_ingestion.schemas.ingestion import IngestionJobResponse

resp = IngestionJobResponse(job_id="job-123", status="pending", message="Ingestion queued")
print(resp.model_dump())
```

## Responsibilities
- **Uploads**: Accepts files, saves them under `uploads/data/`, and registers a `DataSource` row.
- **Sampling**: Provides `get_sample` for quick previews (uses the appropriate connector).
- **Background ingestion**: Triggers `ingest_data_task`, which loads data, runs `DataProfiler`, and updates metadata (`row_count`, `column_count`, schema, and status).
- **Lifecycle**: Supports listing sources, filtering usable sources (ingestion succeeded), and deleting sources (with file cleanup for `type='file'`).

## Task Flow
1. Persist `DataSource` with pending status.
2. Run connector (`file`/`sql`/`api`) and fetch data.
3. Profile with `DataProfiler` and write schema + stats to `source_metadata`.
4. Mark `test_status` to `success` (or `failed` on exceptions).
