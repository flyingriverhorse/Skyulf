# Backend API Reference

The Skyulf backend exposes a REST API on port `8000` by default.
Interactive docs are available at [`/docs`](http://127.0.0.1:8000/docs) (Swagger UI)
and [`/redoc`](http://127.0.0.1:8000/redoc) (ReDoc) when `DEBUG=true`.

---

## Route Map

| Prefix | Area |
|---|---|
| `/health*` | Health & readiness probes |
| `/data/api/*` | Data source management |
| `/api/ingestion/*` | File upload & ingestion jobs |
| `/api/pipeline/*` | ML pipeline execution, preview, jobs, versions |
| `/api/eda/*` | Automated EDA & profiling |
| `/api/deployment/*` | Model deployment & inference |
| `/api/registry/*` | Model registry |
| `/api/monitoring/*` | Drift monitoring, error tracking, pipeline logs |
| `/ws/jobs` | WebSocket job event stream |

---

## Health

### `GET /health/ping`
Minimal liveness probe — returns `{ "message": "pong" }`. Fastest possible health check with no database involvement.

### `GET /health`
Basic health probe for load balancers. Returns HTTP 200 when the server is running.

```json
{
  "status": "healthy",
  "timestamp": "2025-06-30T12:00:00Z",
  "version": "0.6.3",
  "environment": "development",
  "uptime_seconds": 3600.0
}
```

### `GET /health/detailed`
Detailed diagnostics. Checks database connectivity and Redis (if `USE_CELERY=true`).

```json
{
  "status": "healthy",
  "database_status": "healthy",
  "cache_status": "healthy",
  "external_services": { "snowflake": "not_configured" }
}
```

### `GET /health/ready`
Readiness probe — fits a tiny sklearn pipeline to verify the ML stack is functional.
Returns `503` if the ML dependencies are broken.

---

## Data Sources

### `GET /data/api/sources`
List all data sources. Query params: `limit` (default 50), `skip` (default 0).

### `GET /data/api/sources/usable`
List data sources that have completed ingestion and are ready for pipeline use.

### `GET /data/api/sources/{source_id}`
Get a single data source by ID.

> **Note:** The `config` field in all data source responses has sensitive keys (`storage_options`, AWS credentials, passwords) automatically redacted.

### `GET /data/api/sources/{source_id}/sample`
Get a sample of rows from the data source.

### `DELETE /data/api/sources/{source_id}`
Delete a data source and its associated files.

### `GET /data/api/sources/{source_id}/export`
Export the full dataset as a downloadable file.

---

## File Ingestion

### `POST /api/ingestion/upload` — Rate limited: 10/min
Upload a dataset file. Starts an asynchronous ingestion job.

**Request:** `multipart/form-data` with a `file` field.

**Response:**
```json
{ "job_id": "...", "status": "pending", "message": "File upload started", "file_id": "..." }
```

**Allowed extensions:** `.csv`, `.xlsx`, `.xls`, `.parquet`, `.json`, `.txt`, `.pkl`, `.pickle`, `.feather`, `.h5`, `.hdf5`

**Size limit:** `MAX_UPLOAD_SIZE` (default 10 GB).

### `GET /api/ingestion/{source_id}/status`
Poll the status of an ingestion job. Returns `pending`, `processing`, `completed`, or `failed`.

### `POST /api/ingestion/{source_id}/cancel`
Cancel an in-progress ingestion job.

---

## ML Pipeline

All pipeline endpoints are under `/api/pipeline/*`.

### `POST /api/pipeline/run` — Rate limited: 20/min
Submit a pipeline for asynchronous execution. Accepts a `PipelineConfigModel` JSON body describing the node graph.

**Response:**
```json
{
  "message": "Pipeline execution started",
  "pipeline_id": "...",
  "job_id": "...",
  "job_ids": ["..."]
}
```

Multiple `job_ids` are returned when the pipeline is split into parallel branches.

### `POST /api/pipeline/preview`
Run the pipeline in preview mode (synchronous, limited to 1000 rows). Returns per-node output samples.

### `POST /api/pipeline/schema-preview`
Returns predicted output column schema for a node without executing the pipeline.

### `GET /api/pipeline/jobs`
List all training jobs with status and metrics.

### `GET /api/pipeline/jobs/{job_id}`
Get detailed status, metrics, and logs for a specific job.

### `POST /api/pipeline/jobs/{job_id}/cancel`
Cancel a running or queued job.

### `POST /api/pipeline/jobs/{job_id}/promote`
Mark a completed job as the "promoted winner" — sets the `promoted_at` timestamp.

### `DELETE /api/pipeline/jobs/{job_id}/promote`
Remove the promoted status from a job (unpromote).

### `GET /api/pipeline/jobs/{job_id}/evaluation`
Get full evaluation results for a completed job — model metrics, preprocessing artifacts, and tuning trial history (if applicable).

### `GET /api/pipeline/jobs/node-summaries`
Returns per-node aggregated metrics across all jobs. Useful for comparing model types. Query param: `limit` (default 200).

### Tuning Jobs

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/pipeline/jobs/tuning/latest/{node_id}` | Most recent tuning job for a canvas node |
| `GET` | `/api/pipeline/jobs/tuning/best/{model_type}` | Best-scoring tuning job for a model type |
| `GET` | `/api/pipeline/jobs/tuning/history/{model_type}` | Full tuning history for a model type |

### `GET /api/pipeline/registry`
Returns the full node registry — all available step types with their parameter schemas.

### `GET /api/pipeline/stats`
Aggregate statistics across all jobs (counts by status, model type, etc.).

### `GET /api/pipeline/datasets/list`
List datasets available for pipeline use.

### `GET /api/pipeline/datasets/{dataset_id}/schema`
Get the column schema of a dataset.

### `GET /api/pipeline/hyperparameters/{model_type}`
Get hyperparameter definitions for a model type.

### `GET /api/pipeline/hyperparameters/{model_type}/defaults`
Get default hyperparameter values for a model type.

### Pipeline Save / Load

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/pipeline/save/{dataset_id}` | Save a pipeline configuration for a dataset |
| `GET` | `/api/pipeline/load/{dataset_id}` | Load the current pipeline configuration for a dataset |

### Pipeline Versions

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/pipeline/versions/{dataset_source_id}` | List all saved versions for a dataset |
| `POST` | `/api/pipeline/versions/{dataset_source_id}` | Save a new named version |
| `PATCH` | `/api/pipeline/versions/{dataset_source_id}/{version_id}` | Update version metadata |
| `DELETE` | `/api/pipeline/versions/{dataset_source_id}/{version_id}` | Delete a version |
| `GET` | `/api/pipeline/versions/{dataset_source_id}/audit` | Version change audit log |

### Notebook Export

#### `POST /api/pipeline/pipeline/{dataset_id}/export-notebook`
Generate and download a Jupyter notebook (`.ipynb`) for a pipeline. Supports `full` and `compact` export modes. See [Platform Walkthrough](platform_walkthrough.md#step-5b-export-to-jupyter-notebook) for details.

---

## Deployment & Inference

### `POST /api/deployment/deploy/{job_id}` — Rate limited: 10/min
Deploy a trained model from a completed job. Sets it as the active deployment.

### `GET /api/deployment/active`
Get the currently active deployment.

### `GET /api/deployment/history`
List deployment history. Query params: `limit` (default 50), `skip` (default 0).

### `POST /api/deployment/deactivate`
Deactivate the current active deployment.

### `POST /api/deployment/predict` — Rate limited: 60/min
Run inference using the active model.

**Request:**
```json
{ "data": [{ "feature1": 1.5, "feature2": "value" }] }
```

`data` is an **array** of records — even for a single prediction, wrap it in a list.

**Response:**
```json
{ "predictions": [...], "model_version": "job-id-here" }
```

Returns `404` if no deployment is active.

---

## Model Registry

The model registry is **read-only** — models are auto-registered when jobs complete and promoted via the jobs API.

### `GET /api/registry/stats`
Aggregate registry statistics (model counts by type, total versions, etc.).

### `GET /api/registry/models`
List all registered models.

### `GET /api/registry/models/{model_type}/versions`
List all versions for a specific model type (e.g., `random_forest_classifier`).

### `GET /api/registry/artifacts/{job_id}`
List all stored artifacts (model files, preprocessing pickles) for a job.

---

## EDA & Profiling

### `POST /api/eda/{dataset_id}/analyze` — Rate limited: 20/min
Trigger a full automated EDA analysis for a dataset. Returns a `report_id` immediately; the analysis runs asynchronously.

**Response:**
```json
{ "report_id": 42, "status": "pending", "dataset_id": 7 }
```

### `GET /api/eda/{dataset_id}/latest`
Get the most recently completed EDA report for a dataset (distributions, correlations, outliers, smart alerts).

### `GET /api/eda/{dataset_id}/history`
List all past EDA reports for a dataset.

### `GET /api/eda/reports/{report_id}`
Get a specific EDA report by ID.

### `POST /api/eda/reports/{report_id}/cancel`
Cancel a running EDA analysis.

### `POST /api/eda/{dataset_id}/decomposition`
Run time-series decomposition on a dataset column.

### `GET /api/eda/jobs/all`
List all EDA jobs with status. Query params: `limit` (default 50), `skip` (default 0).

---

## Monitoring

### Drift Detection

#### `GET /api/monitoring/jobs`
List jobs that have reference data available for drift comparison.

#### `POST /api/monitoring/drift/calculate` — Rate limited: 20/min
Calculate data drift between a job's reference data and a new uploaded file.

**Request:** `multipart/form-data`
- `job_id` — the training job whose reference data to compare against
- `file` — current data file (CSV or Parquet)
- `dataset_name` *(optional)* — dataset name to match reference data
- `threshold_psi`, `threshold_ks`, `threshold_wasserstein`, `threshold_kl` *(optional)* — custom thresholds

**Size limit:** `MAX_UPLOAD_SIZE` (default 10 GB).

#### `GET /api/monitoring/drift/history/{job_id}`
List past drift check results for a job.

#### `GET /api/monitoring/drift/status`
Drift health summary across all monitored jobs.

### Performance

#### `GET /api/monitoring/slow-nodes`
Returns the top N step types by cumulative execution time over the last N days.
Query params: `days` (1–90, default 7), `limit` (1–50, default 10).

### Error Tracking

The backend automatically records unhandled 5xx errors to the `error_events` database table.

#### `GET /api/monitoring/errors`
List recorded error events. Supports `since`, `limit`, `show_resolved` (default `false`) query params.

#### `GET /api/monitoring/errors/count`
Count of unresolved errors.

#### `GET /api/monitoring/errors/grouped`
Errors grouped by `error_type`.

#### `GET /api/monitoring/errors/timeline`
Error counts over time (for trend charts).

#### `GET /api/monitoring/errors/{error_id}`
Get a specific error event (includes full traceback — **admin-only in production**).

#### `PATCH /api/monitoring/errors/{error_id}/resolve`
Mark an error as resolved.

#### `PATCH /api/monitoring/errors/{error_id}/unresolve`
Reopen a resolved error.

#### `DELETE /api/monitoring/errors`
Delete all error events. ⚠️ Irreversible.

### Pipeline Run Logs

#### `POST /api/monitoring/pipeline-logs`
Ingest pipeline execution log entries.

#### `GET /api/monitoring/pipeline-logs`
List pipeline run logs. Supports `pipeline_id`, `since`, `limit` (max 500) query params.

#### `PATCH /api/monitoring/jobs/{job_id}/description`
Update the description/notes for a monitored job.

#### `DELETE /api/monitoring/pipeline-logs`
Delete all pipeline run logs. ⚠️ Irreversible.

---

## WebSocket — Job Events

### `WS /ws/jobs`

A WebSocket stream that pushes real-time job status events to connected clients.

**Connection:** `ws://127.0.0.1:8000/ws/jobs`

**Message envelope:**
```json
{ "channel": "jobs", "data": { "event": "progress", "job_id": "abc123", "status": "running", "progress": 42 } }
```

Messages are wrapped in a `channel`/`data` envelope. The `data` object is a `JobEvent`.

**Event types (`data.event`):**

| `event` | When |
|---|---|
| `created` | A new job was submitted |
| `progress` | Job progress updated (0–100, in `data.progress`) |
| `status` | Job status changed (running → completed, etc.) |
| `deleted` | Job was deleted |

**JavaScript example:**
```js
const ws = new WebSocket("ws://127.0.0.1:8000/ws/jobs");
ws.onmessage = (msg) => {
  const { channel, data } = JSON.parse(msg.data);
  if (channel === "jobs") {
    console.log(`Job ${data.job_id}: ${data.event} (${data.status})`);
  }
};
```

> **Important:** WebSocket events are best-effort hints. Maintain a polling fallback (`GET /api/pipeline/jobs/{job_id}`) for critical status checks.

---

## Rate Limiting

The following endpoints are rate-limited by client IP address:

| Endpoint | Limit |
|---|---|
| `POST /api/pipeline/run` | 20/minute |
| `POST /api/eda/{dataset_id}/analyze` | 20/minute |
| `POST /api/ingestion/upload` | 10/minute |
| `POST /api/monitoring/drift/calculate` | 20/minute |
| `POST /api/deployment/deploy/{job_id}` | 10/minute |
| `POST /api/deployment/predict` | 60/minute |

Requests over the limit receive **HTTP 429 Too Many Requests**.

---

## Error Responses

Error responses share a common shape, though some fields may be absent depending on the handler:

```json
{
  "success": false,
  "error": "HTTP 404",
  "message": "The requested resource was not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

The `request_id` is also returned in the `X-Request-ID` response header for log correlation.

Common status codes:

| Code | Meaning |
|---|---|
| `400` | Bad request (invalid parameters or business logic error) |
| `404` | Resource not found |
| `405` | Method not allowed |
| `413` | File too large |
| `422` | Validation error (malformed request body) |
| `429` | Rate limit exceeded |
| `500` | Internal server error |
