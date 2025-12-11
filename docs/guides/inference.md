# Inference

Skyulf provides a real-time inference API for deployed models. The inference engine reconstructs the exact preprocessing pipeline used during training to ensure consistency.

## How It Works

When a model is deployed, the `DeploymentService`:
1.  **Loads Artifacts**: Retrieves the trained model and all fitted preprocessing transformers (e.g., scalers, encoders) from the `ArtifactStore`.
2.  **Reconstructs Pipeline**: Uses the `APPLIER_MAP` to instantiate the correct `Applier` classes for each preprocessing step.
3.  **Executes**: Incoming data flows through the reconstructed pipeline (transformations) and finally into the model for prediction.

## Deployment Service

The `DeploymentService` manages the lifecycle of deployments using async database sessions.

```python
from sqlalchemy.ext.asyncio import AsyncSession
from core.ml_pipeline.deployment.service import DeploymentService

async def deploy_and_predict(session: AsyncSession):
    # 1. Deploy a trained model from a completed job
    deployment = await DeploymentService.deploy_model(
        session=session,
        job_id="training_job_abc123",
        user_id=1  # Optional: track who deployed
    )
    print(f"Deployed: {deployment.model_type} (ID: {deployment.id})")

    # 2. Get the currently active deployment
    active = await DeploymentService.get_active_deployment(session)
    print(f"Active deployment: {active.id}")

    # 3. Make predictions (uses active deployment)
    input_data = [
        {"age": 25, "income": 50000, "city": "New York"},
        {"age": 35, "income": 75000, "city": "Boston"}
    ]
    predictions = await DeploymentService.predict(session, data=input_data)
    print(f"Predictions: {predictions}")

    # 4. List deployment history
    history = await DeploymentService.list_deployments(session, limit=10)
    for dep in history:
        print(f"  {dep.created_at}: {dep.model_type} (active={dep.is_active})")
```

## REST API

Deployments are exposed via the REST API.

### Deploy a Model
`POST /api/v1/deployments`

```json
{
  "job_id": "training_job_abc123"
}
```

### Make Predictions
`POST /api/v1/inference/predict`

#### Request Body
```json
{
  "data": [
    {"age": 25, "income": 50000, "city": "New York"},
    {"age": 35, "income": 75000, "city": "Boston"}
  ]
}
```

#### Response
```json
{
  "predictions": [1, 0],
  "probabilities": [[0.2, 0.8], [0.7, 0.3]]
}
```

## Supported Transformers

The inference engine supports all preprocessing transformers used during training. The `APPLIER_MAP` includes:

| Category | Transformers |
|----------|-------------|
| **Encoding** | OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder, HashEncoder, DummyEncoder |
| **Scaling** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| **Imputation** | SimpleImputer, KNNImputer, IterativeImputer |
| **Outliers** | IQR, ZScore, Winsorize, ManualBounds |
| **Transformations** | PowerTransformer, SimpleTransformation, GeneralTransformation |
| **Bucketing** | GeneralBinning |
| **Feature Engineering** | PolynomialFeatures, FeatureGeneration |
| **Feature Selection** | VarianceThreshold, CorrelationThreshold, UnivariateSelection, ModelBasedSelection |
| **Cleaning** | TextCleaning, ValueReplacement, AliasReplacement, InvalidValueReplacement, Deduplicate |
| **Missing Data** | DropMissingColumns, DropMissingRows, MissingIndicator |
| **Type Casting** | Casting |

