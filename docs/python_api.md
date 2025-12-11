# Python API Guide

This guide provides a comprehensive overview of using Skyulf as a standalone Python library. While the platform includes a web interface, the core logic is decoupled and can be used programmatically for research, automation, or integration into other systems.

## 1. Environment Setup

Ensure you have the necessary dependencies installed and your environment configured.

```bash
pip install -r requirements-fastapi.txt
```

```python
import sys
import os

# Ensure the project root is in your PYTHONPATH
sys.path.append(os.getcwd())
```

## 2. Data Ingestion

The first step is loading your data. Skyulf provides a `DataLoader` that handles various formats efficiently.

```python
from core.ml_pipeline.data.loader import DataLoader
from core.ml_pipeline.data.container import SplitDataset
import pandas as pd

# Initialize Loader
loader = DataLoader()

# Load Data (Supports CSV and Parquet)
# For this example, we assume data.csv exists
df = loader.load_full("data.csv")

# Inspect Data
print(f"Loaded {len(df)} rows")
print(df.head())
```

## 3. Defining the Pipeline

Pipelines are defined using a configuration object (`PipelineConfig`). This configuration describes the sequence of steps (nodes) to execute.

```python
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig

pipeline_config = PipelineConfig(
    pipeline_id="my_pipeline_01",
    nodes=[
        # Node 1: Data Loader
        NodeConfig(
            node_id="data_loader",
            step_type="data_loader",
            params={
                "path": "data.csv",
                "type": "csv"
            }
        ),
        # Node 2: Feature Engineering
        NodeConfig(
            node_id="feature_engineering",
            step_type="feature_engineering",
            inputs=["data_loader"],
            params={
                "steps": [
                    {
                        "name": "impute_age",
                        "transformer": "SimpleImputer",
                        "params": {
                            "columns": ["age"],
                            "strategy": "mean"
                        }
                    },
                    {
                        "name": "encode_city",
                        "transformer": "OneHotEncoder",
                        "params": {
                            "columns": ["city"]
                        }
                    }
                ]
            }
        ),
        # Node 3: Model Training
        NodeConfig(
            node_id="model_training",
            step_type="model_training",
            inputs=["feature_engineering"],
            params={
                "algorithm": "random_forest_classifier",
                "target_column": "target",
                "params": {
                    "n_estimators": 50
                }
            }
        )
    ]
)
```

## 4. Executing the Pipeline

The `PipelineEngine` orchestrates the execution. It takes the config, runs each step, and stores the results.

```python
from core.ml_pipeline.execution.engine import PipelineEngine
from core.ml_pipeline.artifacts.local import LocalArtifactStore

# 1. Setup Artifact Store
# This is where trained models and preprocessors are saved
artifact_store = LocalArtifactStore("./my_artifacts")

# 2. Initialize Engine
engine = PipelineEngine(artifact_store=artifact_store)

# 3. Run Pipeline
result = engine.run(pipeline_config, job_id="manual_run_001")

print(f"Status: {result.status}")
# Check node results
for node_id, res in result.node_results.items():
    print(f"Node {node_id}: {res.status}")
```

## 5. Inference (Making Predictions)

Once a pipeline is trained, you can use the saved artifacts to make predictions on new data.
Note that Skyulf uses a **Stateless Applier** pattern. You load the *parameters* (artifacts) and pass them to an *Applier* instance.

```python
from core.ml_pipeline.preprocessing.imputation import SimpleImputerApplier
from core.ml_pipeline.preprocessing.encoding import OneHotEncoderApplier
from core.ml_pipeline.modeling.classification import RandomForestClassifierApplier

# 1. Load Artifacts (Parameters)
# The engine saves artifacts using the node ID.
# For Feature Engineering, it saves steps as "{node_id}_{step_name}".

impute_params = artifact_store.load("feature_engineering_impute_age")
encode_params = artifact_store.load("feature_engineering_encode_city")
model_params = artifact_store.load("model_training")

# 2. Instantiate Appliers
imputer = SimpleImputerApplier()
encoder = OneHotEncoderApplier()
model_applier = RandomForestClassifierApplier()

# 3. New Data
new_data = pd.DataFrame({
    "age": [22, None],
    "income": [45000, 85000],
    "city": ["NY", "SF"]
})

# 4. Apply Transformations
# Appliers return (df, metadata) tuple, we take the first element
step1_out = imputer.apply(new_data, impute_params)
step2_out = encoder.apply(step1_out, encode_params)

# 5. Predict
predictions = model_applier.predict(step2_out, model_params)

print("Predictions:", predictions)
```

## 6. Advanced: Hyperparameter Tuning

You can replace the `model_training` node with a `model_tuning` node to automatically find the best hyperparameters. The engine will tune the model and then retrain the final model with the best parameters found.

```python
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig

tuning_pipeline_config = PipelineConfig(
    pipeline_id="tuning_pipeline_01",
    nodes=[
        # ... Data Loader and Feature Engineering nodes (same as above) ...
        NodeConfig(
            node_id="data_loader",
            step_type="data_loader",
            params={"path": "data.csv", "type": "csv"}
        ),
        NodeConfig(
            node_id="feature_engineering",
            step_type="feature_engineering",
            inputs=["data_loader"],
            params={
                "steps": [
                    {"name": "impute", "transformer": "SimpleImputer", "params": {"strategy": "mean"}}
                ]
            }
        ),
        # Tuning Node
        NodeConfig(
            node_id="model_tuning",
            step_type="model_tuning",
            inputs=["feature_engineering"],
            params={
                "algorithm": "random_forest_classifier",
                "target_column": "target",
                "tuning_config": {
                    "strategy": "random",  # grid, random, optuna
                    "metric": "accuracy",
                    "n_trials": 5,
                    "cv_folds": 3,
                    "search_space": {
                        "n_estimators": [10, 50],
                        "max_depth": [5, 10, None]
                    }
                }
            }
        )
    ]
)

# Run Tuning
result = engine.run(tuning_pipeline_config, job_id="tuning_run_001")

# The result contains the best parameters and score
tuning_metrics = result.node_results["model_tuning"].metrics
print("Best Score:", tuning_metrics["best_score"])
print("Best Params:", tuning_metrics["best_params"])

# The artifact at 'model_tuning' is the fully trained model with best params
best_model = artifact_store.load("model_tuning")
```
