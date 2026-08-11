# Scale and Load Readiness Audit

**Date:** 2026-08-11  
**Scope:** Static review of the FastAPI backend, `skyulf-core`, and React canvas
for behaviour under concurrent users/jobs and large datasets. This deliberately
does not restate the separate auth/tenancy/enterprise-gap audit. Severity is
the likely production impact, not a security rating.

## Executive summary

The principal scale boundary is memory, not request handling: production
pipeline loading uses eager pandas readers, EDA does the same, and ingestion
profiles a fully materialised frame. The configured 10 GB upload ceiling is
therefore materially larger than the process can safely execute on ordinary
worker sizes. The default development topology has one Celery `solo` worker,
so 50 submissions queue serially; an alternate default disables Celery and
runs work in the API process. Neither mode has job admission/resource quotas.

The other immediate limits are SQLite's single-writer/one-process deployment
shape, unbounded model-registry/history queries which grow with all completed
jobs, synchronous WebSocket fan-out, and local-disk defaults that make
horizontal API/worker scaling unsafe unless every relevant path is explicitly
moved to shared storage.

## 1. Database queries, indexes, pagination, and database migration

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **High / Medium** | SQLite is the default database (`DATABASE_URL` and `DB_TYPE`), and the SQLite engine uses `StaticPool`; the development Compose API also points at a local `mlops_database.db` file. PostgreSQL is supported by configuration through `get_postgresql_url`; the configuration guide documents connection fields, but not a SQLite-to-PostgreSQL data-migration/deployment procedure. [database.py:11-21](../backend/config/mixins/database.py#L11-L21) [engine.py:54-80](../backend/database/engine.py#L54-L80) [docker-compose.yml:13-28](../docker-compose.yml#L13-L28) [database.py:62-94](../backend/config/mixins/database.py#L62-L94) [backend_configuration.md:54-75](../docs/guides/backend_configuration.md#L54-L75) | Concurrent worker/API writes will contend on SQLite even with the 30-second busy timeout. A file-local DB also cannot be shared safely by scaled instances. Make PostgreSQL the production-only default; supply an Alembic (or equivalent) migration/runbook and load-test write contention. |
| **High / Medium** | `/registry/models` accepts `skip`/`limit`, but its service fetches **all** deployments, data sources, completed fixed jobs, and completed tuned jobs before grouping, sorting, and slicing in Python. Its per-model versions endpoint similarly reads all active deployments and all matching completed versions without a limit. [api.py:26-35](../backend/ml_pipeline/model_registry/api.py#L26-L35) [service.py:127-167](../backend/ml_pipeline/model_registry/service.py#L127-L167) [service.py:277-310](../backend/ml_pipeline/model_registry/service.py#L277-L310) [service.py:313-385](../backend/ml_pipeline/model_registry/service.py#L313-L385) | Pagination does not bound database, network, or Python memory work; registry page loads degrade linearly with historical jobs and can OOM/timeout. Use SQL aggregation/window queries and keyset pagination; fetch versions only on the detail endpoint, with a capped cursor page. |
| **Medium / Small** | `/data/api/sources/usable` has no `limit` or `skip`; the service selects every successful `DataSource`. `test_status`, which is its filter, is not indexed, and `created_at`, used to order the paginated source list, is also not indexed by `TimestampMixin`. [router.py:48-56](../backend/data_ingestion/router.py#L48-L56) [service.py:51-60](../backend/data_ingestion/service.py#L51-L60) [models.py:30-36](../backend/database/models.py#L30-L36) [models.py:87-126](../backend/database/models.py#L87-L126) | Large source catalogs produce full table scans and unbounded response serialization. Add cursor pagination to `usable`, an index matching its access path (at least `test_status, created_at`), and a matching `created_by` prefix when ownership is introduced. |
| **Medium / Small** | Pipeline-version listing calls `list_versions` without an API page parameter. The audit endpoint also reads every version, sorts it, computes a graph diff for every entry, builds facets, and only then limits the response to 200. [pipelines_io.py:182-189](../backend/ml_pipeline/_internal/_routers/pipelines_io.py#L182-L189) [pipelines_io.py:341-435](../backend/ml_pipeline/_internal/_routers/pipelines_io.py#L341-L435) | Autosaves/history on a long-lived dataset turn a routine UI request into O(all versions × graph size) CPU and memory. Add server-side cursor pagination; calculate the previous graph only for the requested page plus its predecessor; index `(dataset_source_id, version_int)` (the model currently indexes only `dataset_source_id`). [models.py:209-227](../backend/database/models.py#L209-L227) |
| **Low / Small** | `/jobs` is paginated and caps deep cross-table offsets, and EDA's global jobs list is bounded to 200. However, job summaries fetch up to 200 from each mode and merge/sort in Python, while tuning history has no request pagination (its service default is 20). [jobs.py:287-296](../backend/ml_pipeline/_internal/_routers/jobs.py#L287-L296) [jobs.py:178-220](../backend/ml_pipeline/_execution/jobs.py#L178-L220) [jobs.py:304-351](../backend/ml_pipeline/_execution/jobs.py#L304-L351) [eda/router.py:60-91](../backend/eda/router.py#L60-L91) [jobs.py:313-318](../backend/ml_pipeline/_internal/_routers/jobs.py#L313-L318) | This is bounded today, not an N+1 query. Replace cross-table offset merging with a unified job view/table or cursor query before history volume makes the per-canvas polling endpoint costly. |
| **Informational** | No material N+1 was found in the principal jobs/datasets/pipeline list paths reviewed: source listing is one select, EDA's list uses one join, and registry builds lookup maps from bulk selects rather than querying per row. [service.py:39-49](../backend/data_ingestion/service.py#L39-L49) [eda/router.py:69-90](../backend/eda/router.py#L69-L90) [service.py:127-147](../backend/ml_pipeline/model_registry/service.py#L127-L147) | The problem is bulk over-fetching, not N+1. Preserve bulk lookup behaviour when refactoring the registry. |

## 2. In-memory datasets and profiling

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **Critical / Large** | The runtime `FileSystemCatalog` used by the pipeline eagerly calls `pandas.read_*`; CSV alone applies `nrows` at read time, while Parquet/JSON/Excel read everything before `head(limit)`. The node runner asks the catalog to load the data loader's dataset, so pipeline execution receives an in-memory frame. [catalog.py:63-106](../backend/data/catalog.py#L63-L106) [engine/_node_runners.py:109-120](../backend/ml_pipeline/_execution/engine/_node_runners.py#L109-L120) | There is no pipeline streaming/chunking mode. Feature transforms, splits, encoders, and model fitting can add copies/arrays beyond the loaded frame. Introduce a size-aware execution contract: lazy Polars/DuckDB for compatible transform-only stages, chunked readers for supported algorithms, and explicit rejection/remote distributed execution for algorithms requiring a full matrix. |
| **High / Medium** | The generic `DataService.load_file` also eagerly uses `pl.read_*` or `pd.read_*`, and EDA invokes it with `force_type="polars"` before analysis. Ingestion calls `connector.fetch_data()` with no limit to obtain an accurate profile, then calculates duplicate rows, per-column cardinality, and value counts over the materialised frame. [data_service.py:33-112](../backend/services/data_service.py#L33-L112) [eda/tasks.py:146-216](../backend/eda/tasks.py#L146-L216) [tasks.py:81-99](../backend/data_ingestion/tasks.py#L81-L99) [profiler.py:19-55](../backend/data_ingestion/engine/profiler.py#L19-L55) | EDA and ingestion compete with training for the same worker RAM/CPU; repeated user analyses independently reload data. Make profiling/EDA sampling-first with an explicit full-scan tier, add memory estimates/admission control, and persist/cache profiles keyed by dataset content hash and configuration. |
| **High / Small** | The upload default permits 10 GB, but no execution-memory limit is configured. [files.py:13-18](../backend/config/mixins/files.py#L13-L18) | A 10 GB CSV commonly expands substantially as parsed strings/objects; pipeline/model copies make the peak multiple of the file. As a practical **planning** limit—not a guaranteed product limit—a 16 GB worker should only accept low-single-digit-GB compressed/on-disk inputs, and even a 64 GB worker should not promise arbitrary 10 GB CSV execution. Enforce an initial conservative per-job input/RSS budget, measure peak RSS by format/workload, then publish tested size tiers. |
| **Low / Small** | CSV/Parquet preview paths are lazy and bounded: `LocalFileConnector` uses `scan_*` for schema/sample, and `DataService.get_sample` does the same. Excel/JSON preview falls back to eager reads. [file.py:13-24](../backend/data_ingestion/connectors/file.py#L13-L24) [file.py:151-183](../backend/data_ingestion/connectors/file.py#L151-L183) [data_service.py:114-144](../backend/services/data_service.py#L114-L144) | This protects ordinary previews but not execution, ingestion, EDA, JSON, or Excel. Retain this pattern and make its limits/type fallbacks visible to users. |

## 3. Job concurrency and queue behaviour

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **Critical / Medium** | `USE_CELERY` defaults to `False`. In that mode a single branch is a FastAPI `BackgroundTasks` task; multiple branches are a local `ThreadPoolExecutor` capped at eight threads. [celery.py:4-10](../backend/config/mixins/celery.py#L4-L10) [run_pipeline.py:254-291](../backend/ml_pipeline/_internal/_routers/run_pipeline.py#L254-L291) | With 50 submitted jobs, FastAPI can retain/running-background-task work in API processes rather than a durable, capacity-managed worker tier. Eight is only a **within-one-pipeline branch** cap, not a global job cap. Require Celery/Kubernetes workers in production; expose queue depth and reject/defer submissions once concurrency/RAM quotas are reached. |
| **High / Medium** | Celery mode does enqueue work (`run_pipeline_batch_task.delay`), and configuration uses `task_acks_late`, `task_reject_on_worker_lost`, and prefetch 1. However no `worker_concurrency`, task time limit, memory limit, task recycling, or queue-specific worker pool is configured in the application configuration. [run_pipeline.py:263-285](../backend/ml_pipeline/_internal/_routers/run_pipeline.py#L263-L285) [celery_app.py:7-28](../backend/celery_app.py#L7-L28) | A production operator may get Celery's default process concurrency (often CPU count), multiplying each full-dataframe job's RAM; no app-level control keeps 50 submissions from overwhelming workers. Set explicit `--concurrency`, per-queue worker deployments, soft/hard limits, `max-tasks-per-child`/RSS recycling, and autoscaling tied to queue depth and memory headroom. |
| **High / Small** | The checked-in Compose worker is explicitly one `--pool=solo` worker. [docker-compose.yml:35-48](../docker-compose.yml#L35-L48) | In that supplied topology, 50 independent jobs queue and execute one at a time (rather than all 50 at once). Latency becomes approximately the sum of job durations; a long EDA/tuning run starves all later work on the shared queue. Define separate queues/workers for ingestion, EDA, training, and tuning, and document a production worker topology. |
| **Medium / Medium** | A batch Celery task runs parallel branches via threads, capped to `MAX_PARALLEL_BRANCH_WORKERS=8`; tuning defaults to one joblib worker. [tasks.py:101-140](../backend/ml_pipeline/tasks.py#L101-L140) [celery.py:15-32](../backend/config/mixins/celery.py#L15-L32) | Per-job branch concurrency is capped, but threads still load/process independent full datasets in one worker. Couple this cap to a RAM budget and disable branch parallelism when the data estimate cannot support it. |

## 4. WebSocket real-time delivery

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **High / Medium** | Every `/ws/jobs` client is added to an in-process `set`; each process has one Redis subscriber task. There is no connection count, authentication/topic subscription, or per-client queue limit. [router.py:19-37](../backend/realtime/router.py#L19-L37) [manager.py:31-44](../backend/realtime/manager.py#L31-L44) [manager.py:64-76](../backend/realtime/manager.py#L64-L76) | Memory and open file descriptors grow per client. Every client receives every job event, even if it only displays one job. Add authenticated scoped channels, a connection limit/backpressure policy, observability (connections, send latency, dropped clients), and a dedicated realtime tier. |
| **High / Medium** | Broadcast snapshots clients then `await`s `send_text` sequentially for every socket. The subscriber awaits that broadcast before consuming the next Redis message. [manager.py:51-62](../backend/realtime/manager.py#L51-L62) [manager.py:95-105](../backend/realtime/manager.py#L95-L105) | Fan-out is O(connections) per event and one slow network client extends delivery latency for all clients/events. Use per-socket bounded outbound queues and independent sender tasks (dropping/coalescing progress updates), then fan out only to a job/org room. |
| **Medium / Small** | The no-Celery fallback uses an unbounded `asyncio.Queue`; publishers schedule `put_nowait` without a maximum size. [local_bus.py:21-32](../backend/realtime/local_bus.py#L21-L32) [local_bus.py:38-56](../backend/realtime/local_bus.py#L38-L56) | A burst of job logs/progress events during slow fan-out can consume unbounded API-process memory. Bound and coalesce it, or make Redis streams/pubsub the required production transport. |

## 5. File and artifact storage

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **High / Medium** | Uploads default to `uploads/data`; training artifacts default to `uploads/models`; the local artifact store writes with `joblib` to its configured filesystem path. [files.py:10-18](../backend/config/mixins/files.py#L10-L18) [files.py:36-39](../backend/config/mixins/files.py#L36-L39) [local.py:13-44](../backend/ml_pipeline/artifacts/local.py#L13-L44) | With multiple API/worker pods, a job can run where its upload/artifact is absent. Local disks also make failover and cleanup capacity problems. Make object storage/shared POSIX storage mandatory in production and pass durable object URIs—not node-local paths—through job messages. |
| **Medium / Medium** | S3 artifact storage is optional: it is selected only with `S3_ARTIFACT_BUCKET` plus S3-source or an explicit upload flag; artifact discovery remains explicitly “currently local-only.” [factory.py:62-108](../backend/ml_pipeline/artifacts/factory.py#L62-L108) [factory.py:42-59](../backend/ml_pipeline/artifacts/factory.py#L42-L59) | Merely configuring a bucket does not guarantee that local-input artifacts or discovery work across replicas. Complete object-store support for discovery, uploads, exported data, and cleanup; test a worker/API on different nodes. |

## 6. Caching

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **High / Medium** | The cache configuration only declares settings; reviewed EDA/ingestion execution paths load and recompute on every request/job. The only local caching discovered is `S3Catalog`'s filesystem cache, which validates S3 freshness and returns a cached pandas read; it is not a shared cache and is unrelated to EDA profile reuse. [cache.py:1-17](../backend/config/mixins/cache.py#L1-L17) [eda/tasks.py:201-224](../backend/eda/tasks.py#L201-L224) [tasks.py:81-94](../backend/data_ingestion/tasks.py#L81-L94) [catalog.py:151-173](../backend/data/catalog.py#L151-L173) [catalog.py:260-280](../backend/data/catalog.py#L260-L280) | Re-opening the same large dataset for EDA/profiling multiplies I/O, CPU, and memory under concurrent users. Store profile/EDA result metadata by immutable dataset version + analysis config; use Redis/object storage for shared data/result cache with TTL/invalidation. |
| **Low / Small** | The frontend uses React Query keys for EDA report/history invalidation, which is browser-local request caching, not a shared backend compute cache. [useEdaJobs.ts:1-28](../frontend/ml-canvas/src/core/hooks/useEdaJobs.ts#L1-L28) | Helpful for a single browser, but it does not protect the service from cross-user recomputation. |

## 7. Frontend scale behaviour

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **Medium / Medium** | The canvas uses React Flow with the complete `nodes` and mapped `edges` arrays. It has useful memoization to avoid a branch-color traversal on position-only drag updates, but no node/edge virtualization or large-graph mode is configured here. [FlowCanvas.tsx:40-63](../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L40-L63) [FlowCanvas.tsx:149-198](../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L149-L198) [FlowCanvas.tsx:307-355](../frontend/ml-canvas/src/components/canvas/FlowCanvas.tsx#L307-L355) | 50 nodes is likely usable after the existing optimization, but all node cards/edges remain mounted and graph-wide mapping runs when topology changes. Measure 50/100/250-node interaction FPS; enable React Flow only-render-visible options or simplify/hide detail/animated edges at a tested threshold. |
| **High / Small** | Results tables render every returned row and every cell with `map`, with no virtualizer. Dataset preview does the same and lets the user repeatedly increase the sample by 500 rows; its column-profile table also maps every column. [ResultsTable.tsx:10-40](../frontend/ml-canvas/src/components/layout/resultsPanel/ResultsTable.tsx#L10-L40) [DatasetPreviewModal.tsx:66-105](../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx#L66-L105) [DatasetPreviewModal.tsx:199-234](../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx#L199-L234) [DatasetPreviewModal.tsx:254-293](../frontend/ml-canvas/src/components/data/DatasetPreviewModal.tsx#L254-L293) | A 10,000-row table produces 10,000 `<tr>`s times column count and will cause DOM/layout stalls. Use the existing shared `VirtualList` approach (which uses `@tanstack/react-virtual`) or a virtualized grid, server-side cursor paging, and hard display/export limits. [VirtualList.tsx:1-12](../frontend/ml-canvas/src/components/shared/VirtualList.tsx#L1-L12) |
| **Low / Small** | The backend export endpoint caps data rows at 50,000, but it materialises the sample and response bytes in memory. [router.py:95-133](../backend/data_ingestion/router.py#L95-L133) | This is a useful cap, not streaming. Stream exports or hand off large exports to an object-storage-backed job. |

## 8. Rate limits and resource quotas

### Findings

| Severity / effort | Finding and evidence | Load consequence / remediation |
|---|---|---|
| **Critical / Medium** | SlowAPI limits are keyed only by client IP, with a 200/minute default; selected submission routes are 10–20/minute. There are no settings or checks for per-user/org dataset bytes, stored bytes, queued/running jobs, CPU/GPU time, or memory. [rate_limiter.py:1-19](../backend/middleware/rate_limiter.py#L1-L19) [security.py:24-36](../backend/config/mixins/security.py#L24-L36) [run_pipeline.py:368-381](../backend/ml_pipeline/_internal/_routers/run_pipeline.py#L368-L381) [router.py:137-173](../backend/data_ingestion/router.py#L137-L173) | A single NATed group can be throttled together, while a client can submit jobs up to a request-rate limit with no capacity reservation and exhaust the shared worker tier/disk. Add durable tenant/user quotas at submission and upload time: bytes, retained objects, queued/running jobs, per-job RAM/CPU wall time, and concurrency; use a Redis/distributed limiter keyed to the authenticated principal. |
| **Medium / Small** | Idempotency only deduplicates the same dataset/node/branch within a 30-second window; it is not a general concurrency quota. [celery.py:42-52](../backend/config/mixins/celery.py#L42-L52) [jobs.py:80-120](../backend/ml_pipeline/_execution/jobs.py#L80-L120) | It prevents double-click duplicates but permits a user to submit many distinct expensive jobs. Keep it, but add atomic quota counters/reservations and queue priority/fairness. |

## Prioritized: five failures most likely to cause an incident at real scale

1. **Full-frame pipeline/EDA/ingestion processing against a 10 GB upload limit**
   (**Critical; Large**): workers OOM, then retry/queue pressure cascades. Start
   with enforced input/RSS budgets and full-frame isolation; build streaming or
   distributed execution for supported workloads.
2. **No global admission control or per-principal resource quotas**
   (**Critical; Medium**): one workload can fill all workers, memory, disk, and
   queue capacity. Add durable queue/concurrency/byte/time quotas before opening
   multi-user production use.
3. **Default in-process jobs / undefined production Celery capacity**
   (**Critical; Medium**): API stability depends on job volume, or the supplied
   single `solo` worker creates severe queue latency. Require documented,
   explicitly-concurrent worker deployments with separate queues and limits.
4. **SQLite/local disk defaults in multi-instance deployment**
   (**High; Medium**): write locks, missing uploads/artifacts on a different
   pod, and non-durable node-local state. Move production to PostgreSQL plus
   object storage and provide the migration/runbook.
5. **Unbounded historical registry/version work and non-virtualized result
   tables** (**High; Small–Medium**): ordinary UI navigation becomes slow/OOM as
   job history or displayed rows grows. Push paging/aggregation to SQL and
   virtualize/paginate every large table.

## Recommended validation before launch

Establish a production-sized test environment and record p95 API latency,
queue wait, worker RSS, disk/object-store throughput, database lock/wait time,
WebSocket send latency, and browser interaction FPS. Exercise at least:
50 mixed submissions; concurrent EDA/ingestion/training against representative
CSV, Parquet, JSON, and Excel inputs; 1,000 WebSocket clients with a slow-client
cohort; 100-node canvases; and 10,000-row, wide-table views. Capacity limits
should be derived from those measurements and enforced, rather than inferred
from the current upload limit.
