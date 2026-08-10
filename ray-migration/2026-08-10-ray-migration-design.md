# Celery to Ray Migration Design

**Date:** 2026-08-10  
**Status:** Approved design, pending written-spec review  
**Scope:** Replace Celery as Skyulf's background execution backend while
preserving the existing API, job history, progress reporting, cancellation
guards, pipeline behavior, and artifact contracts.

## 1. Executive Summary

Skyulf currently uses FastAPI for request handling, Celery and Redis for
background dispatch, SQLAlchemy job records for user-visible state, local or S3
artifact stores, and thread/joblib parallelism inside workers. The Celery worker
uses a solo pool, while independent pipeline branches are parallelized through
local thread pools. Hyperparameter search defaults to one job for safety.

The target design replaces Celery dispatch with Ray while retaining Skyulf's
database-driven job model:

```text
Frontend
   |
FastAPI control plane
   |-- PostgreSQL job and attempt records
   |-- Ray Jobs submission, cancellation, and reconciliation
   `-- Redis or local event bus for WebSocket updates
              |
         Ray head node
              |
      Ray worker processes
      |-- Pipeline branches
      |-- Tuning trials
      |-- Training
      `-- CPU-heavy EDA
              |
      Shared S3-compatible storage
```

Ray Jobs owns the lifecycle of each submitted pipeline process. Ray Core tasks
and, where appropriate, Ray's joblib integration provide internal parallelism.
PostgreSQL remains the source of truth for the frontend. Ray Dashboard is an
operational view, not an application database.

## 2. Goals

1. Remove Celery as the execution backend without changing the public pipeline
   and experiment APIs.
2. Distribute independent pipeline branches and tuning work across CPU or GPU
   workers.
3. Preserve reliable job status, logs, progress, cancellation, retry, and
   duplicate-submission protection.
4. Keep datasets and artifacts accessible from every worker.
5. Support local development without requiring Ray while making production
   failures explicit rather than silently falling back.
6. Provide a reversible migration until Ray behavior reaches parity.

## 3. Non-Goals

1. Rewriting every pipeline node as an individual Ray task.
2. Replacing the Polars and pandas execution engines with Ray Data.
3. Adopting Ray Train for all estimators in the first release.
4. Replacing the current FastAPI prediction API with Ray Serve.
5. Allowing users to submit arbitrary Python code or runtime dependencies.
6. Making SQLite a supported multi-node production database.

## 4. Current Architecture and Constraints

### 4.1 Dispatch

FastAPI creates database job records and dispatches pipeline work through
Celery when enabled. Without Celery, FastAPI BackgroundTasks and a local thread
pool run the same work in-process. The Celery worker uses a solo pool, and a
second thread pool runs independent branches inside a batch task.

### 4.2 Job State

Job state is already database-centric. Celery result data is not the user-facing
source of truth. Cancellation first changes database state and then performs a
best-effort Celery revoke. Guards prevent late worker writes from changing a
cancelled job to completed.

### 4.3 Tuning

Skyulf Core supports grid, random, halving, and Optuna-based search. Parallelism
uses sklearn/joblib settings. The default is intentionally conservative because
process-based parallelism can conflict with FastAPI, platform-specific process
spawning, and nested estimator threads.

### 4.4 Storage

Artifact storage already has local and S3 implementations. Local paths are not
safe as durable cross-node references. Distributed execution therefore requires
shared object storage for production datasets and artifacts.

### 4.5 Database

SQLite is appropriate for single-process development but not for concurrent
writers across Ray workers. PostgreSQL is required for production Ray mode.

## 5. Target Components

### 5.1 FastAPI Control Plane

FastAPI is responsible for:

- request validation and authorization;
- job creation and duplicate-submission protection;
- submission to Ray Jobs;
- storing the external Ray submission identifier;
- cancellation and retry APIs;
- reconciliation between Ray and database state;
- publishing user-facing progress and status events.

FastAPI does not pass dataframes to Ray and does not execute production jobs
when the Ray backend is selected.

### 5.2 PostgreSQL State Plane

PostgreSQL stores:

- logical job identity and requested pipeline graph;
- the active execution backend;
- attempt records and Ray submission IDs;
- queued, running, cancellation, and terminal states;
- progress, current step, sanitized logs, and error classifications;
- dataset and artifact URIs;
- retry lineage and final promoted attempt.

Frontend APIs continue to read this state rather than Ray's internal status.

### 5.3 Ray Compute Plane

Ray is used at two levels:

1. **Ray Jobs** runs a complete Skyulf pipeline entrypoint independently of the
   FastAPI connection.
2. **Ray Core and joblib integration** distribute sufficiently large,
   independent units such as branches and tuning trials.

The first tuning implementation preserves the existing sklearn and Optuna
behavior and uses Ray as a joblib backend. A later, separately approved phase
may adopt Ray Tune for schedulers such as ASHA and cluster-wide trial control.

### 5.4 Shared Data Plane

Production Ray workers use S3-compatible storage for:

- source datasets or materialized intermediates;
- trained models;
- evaluation output;
- charts and reports;
- attempt-scoped temporary artifacts;
- final promoted artifacts.

Workers receive URIs and identifiers, not large data objects from FastAPI.

### 5.5 Event Plane

Execution and realtime transport settings are separated:

```text
EXECUTION_BACKEND=local|celery|ray
EVENT_BUS=local|redis
```

Removing Celery does not require removing Redis. Redis can remain for
multi-process WebSocket pub/sub and caching. Local event delivery remains
available for single-process development.

## 6. Submission Flow

1. FastAPI validates and resolves the submitted graph.
2. FastAPI acquires the existing per-key submission lock.
3. It checks for a matching active logical job.
4. It creates the logical job and first attempt in PostgreSQL.
5. It submits an entrypoint similar to:

   ```text
   python -m backend.ray_jobs.run_pipeline --job-id <job-id> --attempt-id <attempt-id>
   ```

6. The entrypoint receives no dataset contents or secrets on the command line.
7. FastAPI stores the returned Ray submission ID on the attempt.
8. The Ray driver loads the graph, dataset URI, storage configuration, and
   permitted credentials from controlled application services.
9. The driver marks the attempt running and executes the pipeline.
10. On success, it promotes attempt artifacts and commits terminal state.

If Ray submission fails, FastAPI records a submission failure and returns an
explicit error. Production never silently runs the job inside the API process.

## 7. Pipeline Execution Model

### 7.1 Task Boundaries

Ray tasks represent meaningful compute units, not every visual node. Sequential
nodes within one branch run in one branch execution context. Independent
branches may run in parallel.

This avoids scheduling overhead and repeated serialization for small operations.

### 7.2 Shared Inputs

For branches on the same Ray node, an immutable NumPy or Arrow representation
may be placed in the object store once with `ray.put()`. Repeatedly passing a
large Polars or pandas dataframe by value is prohibited.

For large or multi-node workloads, workers load partitioned Parquet or other
shared-storage formats directly. Object spilling directories and capacity are
configured explicitly.

### 7.3 Tuning

The initial implementation:

- keeps Skyulf's existing search strategies, metrics, cross-validation,
  time-series ordering, threshold tuning, wrappers, and result schemas;
- registers Ray's joblib backend;
- allocates explicit resources per trial;
- preserves one code path for local and distributed execution.

Ray Tune is deferred until parity and performance measurements justify the
larger behavioral change.

### 7.4 Resource Safety

Every distributed task declares logical CPU, GPU, and memory needs. Ray's CPU
resources are scheduling reservations, not physical enforcement. The worker
environment and estimators therefore align:

- `OMP_NUM_THREADS`;
- `MKL_NUM_THREADS`;
- OpenBLAS thread settings;
- sklearn estimator `n_jobs`;
- XGBoost and LightGBM thread counts;
- Ray `num_cpus`.

This prevents nested parallelism from oversubscribing a host.

## 8. Job State Machine

The user-visible logical job and its physical attempts are separate.

```text
queued
  |-- submission_failed
  `-- running
        |-- cancel_requested --> cancelled
        |-- failed -----------> retrying --> queued
        `-- completed
```

An attempt stores one Ray submission ID and one terminal outcome. A retry creates
a new attempt instead of overwriting the failed execution record.

### 8.1 Cancellation

1. Atomically change the logical job to `cancel_requested`.
2. Call `JobSubmissionClient.stop_job()` for the active submission.
3. Mark the attempt and job cancelled when termination is confirmed.
4. Preserve the existing late-write guard so cancelled work cannot become
   completed.

Long-running synchronous estimator calls may not stop immediately. The UI must
distinguish cancellation requested from cancellation completed.

### 8.2 Retry

Automatic retry is limited by error classification:

- worker or node loss: eligible for a small bounded retry count;
- transient storage or network failure: eligible when the operation is
  idempotent;
- invalid config or deterministic application error: no automatic retry;
- out-of-memory: no retry with unchanged resources;
- user cancellation: no retry.

Each retry receives a new attempt ID and Ray submission ID.

### 8.3 Idempotent Finalization

Attempts write to:

```text
s3://<bucket>/jobs/<logical-job-id>/attempts/<attempt-id>/
```

Only a successful attempt is promoted as the logical job's final artifact.
Promotion and model-registry updates are idempotent. Failed and cancelled
attempt data is cleaned according to retention policy.

## 9. Reconciliation

A reconciliation process periodically compares active database attempts with
Ray Jobs:

| Ray state | Database state | Action |
|---|---|---|
| RUNNING | queued | update to running |
| SUCCEEDED | running | verify finalized result; otherwise mark inconsistent |
| FAILED | queued/running | fetch sanitized failure and mark failed |
| STOPPED | cancel_requested | mark cancelled |
| Missing | active | mark orphaned and apply retry policy |

Cluster restart or head-node loss must not leave jobs permanently running.

## 10. Periodic Work

Ray Jobs is not used as a general cron scheduler.

- Docker Compose runs a small scheduler service for maintenance commands.
- Kubernetes uses CronJobs.
- PostgreSQL advisory locks or unique execution rows prevent duplicate runs
  across scheduler replicas.

Maintenance work may submit Ray compute jobs when it is genuinely compute-heavy.

## 11. Deployment

### 11.1 Local Development

Local development supports:

- `EXECUTION_BACKEND=local` for the current lightweight BackgroundTasks path;
- `EXECUTION_BACKEND=ray` for integration and distributed behavior.

The Ray mode includes API, PostgreSQL, optional Redis, Ray head, Ray worker, and
scheduler services. Ray Dashboard is available only on localhost or an internal
network.

### 11.2 Production

The initial production deployment can use a fixed Ray head and worker group.
When Kubernetes is adopted, KubeRay manages the Ray cluster and autoscaling.
Ray Service is introduced only if Ray Serve is separately approved.

The API and Ray workers use the same versioned application image. Runtime
environments cannot install arbitrary packages from the internet.

### 11.3 High Availability

Ray head-node loss can terminate a cluster. The baseline response is durable
database state, reconciliation, bounded retries, and idempotent artifacts. If
the service-level objective requires head-node survival, Ray GCS fault tolerance
and its external persistence dependency are evaluated separately.

## 12. Observability

### 12.1 User View

Skyulf exposes:

- queued/running/cancellation/terminal state;
- progress and current node;
- sanitized logs;
- node timings;
- metrics and artifact links;
- retry attempt history.

### 12.2 Operator View

Ray Dashboard exposes internal jobs, tasks, actors, workers, logs, and resource
usage. Production metrics use Prometheus and Grafana. Existing application
logging and Sentry reporting remain.

Each Ray submission includes metadata linking it to:

- Skyulf logical job ID;
- attempt ID;
- pipeline ID;
- workspace or tenant identifier.

## 13. Security

1. Ray Dashboard and Jobs API are never public internet endpoints.
2. FastAPI reaches Ray over a private network.
3. Workers use IAM roles or workload identity instead of embedded user cloud
   credentials.
4. Secrets are excluded from command lines, job metadata, and logs.
5. Only registered Skyulf nodes and models may execute.
6. Runtime package installation is disabled in production.
7. Dataset and artifact access remains scoped to the authorized workspace.

## 14. Migration Strategy

### Phase 1: Execution Backend Interface

Introduce a backend contract:

```text
submit(job_id, attempt_id) -> external_execution_id
cancel(external_execution_id)
status(external_execution_id)
logs(external_execution_id)
```

Wrap current Celery behavior first, then add Ray. Routes and job services stop
calling Celery directly.

### Phase 2: Pipeline and Tuning Pilot

Run selected pipeline and tuning jobs on Ray behind a feature flag. Preserve
Celery as rollback. Compare results and operational behavior.

### Phase 3: EDA and Ingestion Classification

Move CPU-heavy EDA to Ray. Keep small orchestration or metadata operations out
of Ray unless distribution provides measurable value.

### Phase 4: Default Ray Backend

Make Ray the default only after parity, cancellation, retry, reconciliation,
and performance criteria pass in production-like tests.

### Phase 5: Celery Removal

Drain active Celery work, retain compatibility for historical metadata, remove
Celery dispatch and Beat configuration, and decide independently whether Redis
is still required for events or caching.

## 15. Testing Strategy

### 15.1 Unit Tests

- execution backend adapters;
- Ray status mapping;
- submission failure handling;
- retry classification;
- cancellation and late-write guards;
- reconciliation decisions;
- attempt and artifact promotion idempotency;
- resource configuration.

### 15.2 Integration Tests

Using a local Ray cluster, PostgreSQL, and S3-compatible test storage:

- successful submission and completion;
- application failure;
- cancellation before and during execution;
- worker crash;
- cluster restart and orphan reconciliation;
- transient storage failure;
- duplicate submission;
- retry with a new attempt;
- incomplete artifact cleanup.

### 15.3 Backend Parity Tests

Run identical graphs through Celery and Ray and compare:

- output schema;
- model family and type;
- selected best parameters;
- metrics within floating-point tolerance;
- artifact contents and metadata;
- progress and final status behavior.

### 15.4 Load and Performance Tests

Measure:

- queue wait and total runtime;
- branch and tuning throughput;
- CPU and memory utilization;
- nested thread oversubscription;
- object store use and spilling;
- PostgreSQL contention;
- S3 throughput;
- cancellation latency.

## 16. Acceptance Criteria

1. Existing public API and frontend job behavior remain compatible.
2. A cancelled job cannot later become completed.
3. Worker or cluster loss cannot leave a job indefinitely running.
4. Duplicate submissions cannot create duplicate final models or artifacts.
5. Failed attempts cannot overwrite promoted artifacts.
6. Ray and Celery produce equivalent supported pipeline results.
7. Production Ray mode requires PostgreSQL and shared object storage.
8. Ray provides a measured benefit for the selected workloads before Celery is
   removed.
9. Rollback to Celery remains possible until the final removal phase.

## 17. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Ray control-plane outage | Durable DB state, reconciliation, bounded retries |
| CPU oversubscription | Align Ray resources, estimator jobs, and BLAS threads |
| Large dataframe serialization | Pass URIs or one object reference; use Arrow/Parquet |
| SQLite write contention | Require PostgreSQL for production Ray mode |
| Local artifact invisibility | Require shared S3-compatible storage |
| Cancellation delay | Expose `cancel_requested`; preserve late-write guard |
| Duplicate execution | DB idempotency and attempt-scoped artifact paths |
| Dependency drift | One versioned image for API and workers |
| Operational complexity | Incremental backend migration and feature flags |

## 18. Deferred Decisions

The following require separate benchmarks or designs:

- direct Ray Tune adoption;
- Ray Data for out-of-core processing;
- Ray Train for distributed XGBoost, LightGBM, or deep learning;
- Ray Serve for online inference;
- KubeRay production manifests;
- GCS high availability;
- complete Redis removal.

## 19. Official Ray References

- [Ray overview](https://docs.ray.io/en/latest/ray-overview/index.html)
- [Ray Core walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)
- [Ray objects and object store](https://docs.ray.io/en/latest/ray-core/objects.html)
- [Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Ray task fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)
- [Ray Jobs](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html)
- [Ray joblib backend](https://docs.ray.io/en/latest/ray-more-libs/joblib.html)
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
- [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
