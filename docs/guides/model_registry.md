# Model Registry

The Model Registry is a centralized system for managing the lifecycle of your ML models. It tracks model versions, lineage (which training job created each model), and deployment status.

## How It Works

The registry is **not a separate store**—it is a unified view that aggregates completed `TrainingJob` and `HyperparameterTuningJob` records from the database. When you train a model through the UI or API, the resulting artifact is automatically tracked here.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Model Registry                           │
│  (Aggregates TrainingJob + HyperparameterTuningJob tables)      │
├─────────────────────────────────────────────────────────────────┤
│  RandomForest (Dataset: customers.csv)                          │
│    └── v3 [DEPLOYED] - training - accuracy: 0.92                │
│    └── v2            - tuning   - best_score: 0.91              │
│    └── v1            - training - accuracy: 0.85                │
│                                                                  │
│  LogisticRegression (Dataset: sales.csv)                        │
│    └── v1            - training - accuracy: 0.78                │
└─────────────────────────────────────────────────────────────────┘
```

## Model Versioning

Every time a training or tuning job completes successfully, a new version is created. The `ModelRegistryService` automatically calculates the next version number for each `(dataset_id, model_type)` pair.

*   **Training Jobs**: Produce a single model version.
*   **Advanced training with tuning Jobs**: Produce a model version for the best hyperparameter configuration found.

## Using the Registry (Python API)

### List All Models

```python
from core.ml_pipeline.model_registry.service import ModelRegistryService

async def list_models(session):
    entries = await ModelRegistryService.list_models(session)

    for entry in entries:
        print(f"{entry.model_type} ({entry.dataset_name})")
        for v in entry.versions:
            deployed = "[DEPLOYED]" if v.is_deployed else ""
            print(f"  v{v.version} - {v.source} - {v.metrics} {deployed}")
```

### Get Versions for a Specific Model Type

```python
async def get_versions(session, model_type: str):
    versions = await ModelRegistryService.get_model_versions(session, model_type)
    return versions
```

### Get Next Version Number

```python
async def get_next_version(session):
    next_ver = await ModelRegistryService.get_next_version(
        session, 
        dataset_id="ds_123", 
        model_type="LogisticRegression", 
        job_type="training"
    )
    print(f"Next version will be: {next_ver}")
```

---

## Deploying a Model for Inference

The `DeploymentService` takes a model from the registry and makes it the **active deployment**. Only one model can be active at a time.

### Deploy a Model

```python
from core.ml_pipeline.deployment.service import DeploymentService

async def deploy(session, job_id: str):
    # This deactivates any previous deployment and sets the new one as active
    deployment = await DeploymentService.deploy_model(session, job_id)
    print(f"Deployed {deployment.model_type} from job {deployment.job_id}")
```

### Make Predictions with the Active Deployment

Once a model is deployed, you can send data to it for inference. The service automatically:

1.  Loads the model artifact (including any fitted transformers from the pipeline).
2.  Applies the same preprocessing steps used during training.
3.  Runs the model's `.predict()` method.

```python
async def predict(session, input_data: list[dict]):
    # Example: [{"age": 30, "income": 50000}, {"age": 25, "income": 60000}]
    predictions = await DeploymentService.predict(session, input_data)
    return predictions  # e.g., [0, 1]
```

### Deployment History

```python
async def list_deployments(session):
    deployments = await DeploymentService.list_deployments(session)
    for d in deployments:
        status = "ACTIVE" if d.is_active else "inactive"
        print(f"{d.model_type} ({d.created_at}) - {status}")
```

---

## REST API Endpoints

You can also interact with the registry and deployments via the FastAPI endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/registry/stats` | GET | Get registry statistics |
| `/api/registry/models` | GET | List all models and versions |
| `/api/registry/models/{model_type}/versions` | GET | Get versions for a model type |
| `/api/deployments/deploy` | POST | Deploy a model by `job_id` |
| `/api/deployments/predict` | POST | Make predictions with active model |
| `/api/deployments/` | GET | List deployment history |

