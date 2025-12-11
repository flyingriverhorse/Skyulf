# Experiments

Experiments in Skyulf provide a unified view of all your training and tuning jobs. The Experiments page aggregates `TrainingJob` and `HyperparameterTuningJob` records from the database, allowing you to compare runs, view metrics, and track model versions.

## How It Works

When you train a model (via the UI or API), Skyulf creates a job record in the database:

| Job Type | Database Table | What It Produces |
|----------|----------------|------------------|
| **Training** | `TrainingJob` | Single model with fixed hyperparameters |
| **Tuning** | `HyperparameterTuningJob` | Best model from hyperparameter search |

Each job automatically tracks:

*   **Metrics**: Accuracy, F1, RMSE, ROC-AUC, etc.
*   **Hyperparameters**: The parameters used (or best params found for tuning).
*   **Artifacts**: Trained model, confusion matrices, feature importance.
*   **Lineage**: Dataset, pipeline, and node that produced the model.
*   **Version**: Auto-incremented per `(dataset_id, model_type)` pair.

## Listing Jobs (Python API)

Use `JobManager.list_jobs()` to retrieve all training and tuning jobs:

```python
from core.ml_pipeline.execution.jobs import JobManager

async def list_experiments(session):
    jobs = await JobManager.list_jobs(session, limit=50)
    
    for job in jobs:
        print(f"[{job.job_type.upper()}] {job.model_type} v{job.version}")
        print(f"  Dataset: {job.dataset_name}")
        print(f"  Status: {job.status.value}")
        print(f"  Metrics: {job.metrics}")
        if job.job_type == "tuning":
            print(f"  Strategy: {job.search_strategy}")
        print()
```

## Getting a Single Job

```python
async def get_job_details(session, job_id: str):
    job = await JobManager.get_job(session, job_id)
    if job:
        print(f"Job: {job.job_id}")
        print(f"Model: {job.model_type}")
        print(f"Hyperparameters: {job.hyperparameters}")
        print(f"Metrics: {job.metrics}")
```

## Filtering by Job Type

```python
# Only training jobs
training_jobs = await JobManager.list_jobs(session, job_type="training")

# Only tuning jobs
tuning_jobs = await JobManager.list_jobs(session, job_type="tuning")
```

## Job Fields Reference

| Field | Description |
|-------|-------------|
| `job_id` | Unique identifier (UUID) |
| `pipeline_id` | Pipeline that created this job |
| `node_id` | The specific node (model) in the pipeline |
| `dataset_id` | Source dataset identifier |
| `dataset_name` | Human-readable dataset name |
| `job_type` | `"training"` or `"tuning"` |
| `status` | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `model_type` | Algorithm name (e.g., `RandomForest`, `LogisticRegression`) |
| `hyperparameters` | Parameters used (or best params for tuning) |
| `metrics` | Evaluation metrics (accuracy, f1, etc.) |
| `version` | Model version for this dataset/model_type |
| `search_strategy` | *(Tuning only)* `random`, `grid`, `bayesian`, `halving` |
| `start_time` / `end_time` | Job execution timestamps |
| `error` | Error message if job failed |

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/jobs` | GET | List all jobs with optional filters |
| `/api/v1/jobs/{job_id}` | GET | Get details for a specific job |
| `/api/v1/jobs/{job_id}/cancel` | POST | Cancel a running or queued job |

## Comparison in the UI

In the Experiments page, you can:

1.  **Filter** by job type, status, model type, or dataset.
2.  **Sort** by date, metrics, or version.
3.  **Select** multiple runs to compare metrics side-by-side.
4.  **Deploy** a completed job directly to the inference endpoint.
