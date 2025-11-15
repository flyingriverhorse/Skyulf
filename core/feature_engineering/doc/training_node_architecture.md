# Model Training Node Architecture Blueprint

This document captures the proposed architecture for the upcoming model-training node that operates on train/test split outputs, runs long-lived training jobs through Celery/Redis, and preserves the canvas state while jobs are active. It should serve as the north star before implementation.

## 1. Goals & Constraints
- **Train on split data**: the node must consume the latest `train` partition emitted by upstream split nodes (`train_test_split`, resampling, feature transforms) and use configured hyperparameters to fit a model.
- **Reusable defaults**: when users leave hyperparameters blank, the node should fall back to sensible defaults per model family (e.g., `RandomForestClassifier` with 100 trees, `XGBoost` learning_rate=0.1, etc.).
- **Asynchronous execution**: training can take minutes, so pipeline execution must delegate to a Celery worker backed by Redis. The API responds immediately with a job identifier.
- **Status visibility**: users monitor the job in their dashboard/profile and inside the canvas (node badge / sidebar). Status transitions include `queued → running → succeeded/failed/cancelled`.
- **Canvas continuity**: reopening the canvas must keep all nodes, configs, and in-flight job metadata intact—no resets even if the browser refreshes mid-training.
- **Future validation handoff**: the training node outputs artifacts/metadata that a downstream validation node can consume (model registry entry, metrics reference, etc.).

## 2. Frontend Canvas Experience
1. **Node configuration panel**
   - Model family selector (e.g., Logistic Regression, Random Forest, XGBoost, LightGBM).
   - Hyperparameter form with `Advanced` drawer; empty fields use backend defaults.
   - Optional training metadata: experiment tag, description, max runtime.
   - Checkbox to trigger immediate training (`Run training now`) or leave node configured only.
2. **Execution trigger**
   - When the user clicks `Run training` a POST hits `POST /ml-workflow/api/training-jobs` with:
     ```json
     {
       "dataset_source_id": "...",
       "pipeline_id": "{dataset}_{hash}",
       "node_id": "train-model-42",
       "graph": {"nodes": [...], "edges": [...]},
       "config": {...}
     }
     ```
   - The canvas keeps the draft in local state; saving the pipeline is optional but recommended. We automatically enqueue a save when a training job is spawned to avoid desync.
3. **Status surfaces**
   - Node badge displays `Queued`, `Running`, `Succeeded`, `Failed`.
   - Hover tooltip links to the user dashboard entry -> `/user/training-jobs/{id}`.
   - The preview pane polls `/ml-workflow/api/training-jobs/{job_id}` until terminal state.
4. **Canvas reload**
   - On hydrate we fetch both `FeaturePipelineResponse` and active training jobs keyed by `(pipeline_id, node_id)` to decorate nodes with their current status.

## 3. Backend API & Persistence
### 3.1 New database tables (SQLAlchemy models)
- `training_jobs`
  - `id` (UUID), `pipeline_id`, `node_id`, `dataset_source_id`, `user_id`.
  - `status` (`queued`, `running`, `succeeded`, `failed`, `cancelled`).
  - `model_type`, `hyperparameters` (JSON), `metrics` (JSON), `error_message`.
  - `model_artifact_uri` (link to storage), timestamps.
- `training_job_events` (optional for audit trail).

We can reuse Alembic migrations; ensure Celery workers have DB access.

### 3.2 API surface (FastAPI)
- `POST /ml-workflow/api/training-jobs`
  - Validates payload (`TrainingJobCreate` schema), ensures `train_test_split` exists upstream.
  - Persists a row with `status=queued` and enqueues Celery task `train_model.delay(job_id)`.
  - Returns `TrainingJobResponse` containing job id and initial status.
- `GET /ml-workflow/api/training-jobs/{job_id}`
  - Returns live status, metrics, errors, timestamps, and artifact references.
- `GET /ml-workflow/api/users/me/training-jobs`
  - Powers the user dashboard.

### 3.3 Canvas hydration hook
- Extend existing pipeline GET to include `active_training_jobs`:
  ```json
  {
    "pipeline": {...},
    "jobs": [{"job_id": "...", "node_id": "train-model-42", "status": "running", ...}]
  }
  ```
- Alternatively the frontend can call a dedicated endpoint filtered by `pipeline_id`.

## 4. Celery Orchestration
1. **Task signature**: `train_model(job_id: str)`.
2. **Task flow**
   - Fetch job config and graph from DB.
   - Reconstruct pipeline execution order up to the training node using `_execution_order` from `routes.py`.
   - Load dataset via `_load_dataset_frame(..., execution_mode="full")` if size permits; otherwise degrade to sample with warning.
   - Apply nodes up to training node using `_run_pipeline_execution` with `collect_signals=False` (but retaining split metadata).
   - Extract train/test partitions from the resulting frame; raise if train split missing.
   - Instantiate model using hyperparameters (fallback to defaults defined in a registry `MODEL_DEFAULTS`).
   - Fit model, evaluate quick metrics (accuracy, F1, ROC-AUC for classification; RMSE/MAE for regression).
   - Persist artifact (joblib file on disk/S3/minio) and metrics JSON.
   - Update `training_jobs` row incrementally: `running` at start, `succeeded` with metrics/artifact, `failed` with traceback.
   - Emit socket or polling signal (optional) through Redis pub/sub for realtime UI updates.

3. **Resource management**
   - Use environment-configured Celery concurrency; ensure tasks respect timeouts and support graceful cancellation.
   - Consider chunking if dataset is very large (future work).

## 5. Artifact Management & Downstream Usage
- **Storage**: store models under `uploads/models/{pipeline_id}/{job_id}.joblib` (local) or object storage bucket. Record URI in the job row.
- **Metadata**: metrics JSON includes train/test scores, timestamp, dataset row counts, feature list, and the pipeline hash used for reproducibility.
- **Versioning**: persist a monotonically increasing model version per `(pipeline_id, node_id)`. Each successful training run increments the version, stores the hyperparameter payload and metrics snapshot, and exposes version labels in the API. The user dashboard and canvas badge should reference the latest version while allowing rollbacks or comparisons (e.g., show top-N best runs by metric).
- **Validation node hand-off**: training node output object should expose `job_id`, `model_version`, and `model_artifact_uri`. The future validation node can reference these to run evaluation against validation/holdout sets or compare multiple versions.

## 6. Canvas State Preservation Strategy
- **Autosave on job create**: when a training job is requested, force a pipeline save (`upsert_pipeline`) so the back-end holds the latest graph.
- **Persistent job association**: training jobs tie to `pipeline_id` (dataset+hash). If the graph changes post-save, a new `pipeline_id` is generated; warn the user that running jobs refer to the old graph.
- **Hydration merge**: upon reload, we hydrate the canvas graph first, then overlay job statuses by matching node IDs. If there are jobs for unknown nodes, show a banner linking to history.
- **Soft-lock warning**: if the user edits the graph significantly while a job runs, display a caution that rerunning may produce different results because the `pipeline_id` will change.

## 7. Security & Multi-Tenancy
- Only the user who launched the job (or admins) can access job status/artifacts.
- Ensure Celery workers authenticate DB connections with least privilege.
- Sanitize hyperparameters before persisting to avoid code injection (no free-form Python allowed).

## 8. Implementation Phases
1. **Scaffolding**
   - Add SQLAlchemy models + alembic migration for `training_jobs`.
   - Define Pydantic schemas & API endpoints (POST/GET).
   - Stub Celery task that just updates status to succeeded.
2. **Pipeline integration**
   - Build model registry with defaults, implement task logic to execute upstream nodes and fit baseline models.
   - Persist artifacts locally; expose metrics in API.
3. **Frontend integration**
   - Create node settings UI, mutation hooks, polling UI components.
   - Add hydration query for active jobs and job list in user dashboard.
4. **Hardening**
   - Handle cancellation, retries, timeouts.
   - Add audit logging & notifications.
   - Extend to multiple model families with custom hyperparameter forms.
5. **Validation node (future)**
   - Consume `job_id` + artifact to perform evaluation & produce validation reports.

## 9. Open Decisions / Questions
- Where to store artifacts long-term (local FS vs. S3/minio). Need answer before production.
- Should training jobs capture the entire dataset snapshot for reproducibility (versioned data) or rely on source immutability?
- Do we allow multiple concurrent jobs per node/pipeline or enforce one-at-a-time with an explicit rerun button?
- Notification strategy: polling vs. websockets? Minimal viable approach is polling.

Keeping this document in `core/feature_engineering/doc/training_node_architecture.md` lets revisit and refine as design discussions proceed.
