# Getting Started

This tutorial walks you through a complete ML workflow using Skyulf's Python API: loading data, preprocessing, training a model, and making predictions.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements-fastapi.txt

# Start the server (optional, for REST API)
python run_skyulf.py
```

## Tutorial: Iris Classification

We'll build a classifier for the classic Iris dataset, covering the full pipeline.

### Step 1: Load and Prepare Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Rename columns for simplicity
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

print(f"Dataset shape: {df.shape}")
print(df.head())
```

### Step 2: Create Train/Test Split

```python
from core.ml_pipeline.data.container import SplitDataset
from sklearn.model_selection import train_test_split

# Split 80/20
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# Create SplitDataset container
dataset = SplitDataset(
    train=train_df.reset_index(drop=True),
    test=test_df.reset_index(drop=True),
    validation=None  # Optional
)

print(f"Train: {len(dataset.train)}, Test: {len(dataset.test)}")
```

### Step 3: Define Preprocessing Steps

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer

# Define preprocessing steps
preprocessing_steps = [
    {
        "name": "scale_features",
        "transformer": "StandardScaler",
        "params": {
            "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        }
    }
]

# Initialize FeatureEngineer
engineer = FeatureEngineer(preprocessing_steps)

# Fit on training data
train_processed, train_metrics = engineer.fit_transform(dataset.train.drop('target', axis=1))

# Apply to test data (using fitted parameters)
test_processed = engineer.transform(dataset.test.drop('target', axis=1))

print(f"Scaling metrics: {train_metrics}")
print(f"Processed train shape: {train_processed.shape}")
```

### Step 4: Train a Model

```python
from core.ml_pipeline.modeling.classification import RandomForestClassifierCalculator, RandomForestClassifierApplier
from core.ml_pipeline.modeling.base import StatefulEstimator
from core.ml_pipeline.artifacts.local import LocalArtifactStore

# Setup artifact store (where models are saved)
artifact_store = LocalArtifactStore("./tutorial_artifacts")

# Create estimator
estimator = StatefulEstimator(
    calculator=RandomForestClassifierCalculator(),
    applier=RandomForestClassifierApplier(),
    artifact_store=artifact_store,
    node_id="iris_classifier"
)

# Prepare dataset with processed features
processed_dataset = SplitDataset(
    train=pd.concat([train_processed, dataset.train['target'].reset_index(drop=True)], axis=1),
    test=pd.concat([test_processed, dataset.test['target'].reset_index(drop=True)], axis=1)
)

# Train and predict
predictions = estimator.fit_predict(
    dataset=processed_dataset,
    target_column="target",
    config={
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
)

print(f"Train predictions: {predictions['train'][:10]}")
```

### Step 5: Evaluate the Model

```python
from sklearn.metrics import accuracy_score, classification_report

# Get predictions
train_preds = predictions['train']
test_preds = predictions['test']

# Calculate metrics
train_acc = accuracy_score(dataset.train['target'], train_preds)
test_acc = accuracy_score(dataset.test['target'], test_preds)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report (Test):")
print(classification_report(dataset.test['target'], test_preds, target_names=iris.target_names))
```

### Step 6: Make Predictions on New Data

```python
# Load saved model
model_artifact = artifact_store.load("iris_classifier")

# New sample data
new_samples = pd.DataFrame({
    'sepal_length': [5.1, 6.3],
    'sepal_width': [3.5, 2.5],
    'petal_length': [1.4, 4.9],
    'petal_width': [0.2, 1.5]
})

# Apply same preprocessing
new_processed = engineer.transform(new_samples)

# Get applier and predict
applier = RandomForestClassifierApplier()
new_predictions = applier.predict(new_processed, model_artifact)

print(f"Predictions: {new_predictions}")
print(f"Species: {[iris.target_names[p] for p in new_predictions]}")
```

---

## Using the Pipeline Engine

For more complex workflows, use the `PipelineEngine` with a configuration-based approach:

```python
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig
from core.ml_pipeline.execution.engine import PipelineEngine
from core.ml_pipeline.artifacts.local import LocalArtifactStore

# Save dataset to file first
df.to_csv("iris_data.csv", index=False)

# Define pipeline config
config = PipelineConfig(
    pipeline_id="iris_pipeline",
    nodes=[
        NodeConfig(
            node_id="load_data",
            step_type="data_loader",
            params={"path": "iris_data.csv", "type": "csv"}
        ),
        NodeConfig(
            node_id="preprocess",
            step_type="feature_engineering",
            inputs=["load_data"],
            params={
                "steps": [
                    {
                        "name": "scale",
                        "transformer": "StandardScaler",
                        "params": {}
                    }
                ]
            }
        ),
        NodeConfig(
            node_id="train_model",
            step_type="model_training",
            inputs=["preprocess"],
            params={
                "algorithm": "random_forest_classifier",
                "target_column": "target",
                "params": {"n_estimators": 100}
            }
        )
    ]
)

# Run pipeline
artifact_store = LocalArtifactStore("./pipeline_artifacts")
engine = PipelineEngine(artifact_store=artifact_store)
result = engine.run(config, job_id="iris_job_001")

print(f"Pipeline Status: {result.status}")
for node_id, node_result in result.node_results.items():
    print(f"  {node_id}: {node_result.status} ({node_result.execution_time:.2f}s)")
```

---

## Next Steps

- **[Preprocessing Reference](../preprocessing/imputation.md)**: Learn about all available transformers
- **[Hyperparameter Tuning](../modeling/tuning.md)**: Optimize your model with Grid/Random/Optuna search
- **[Model Registry](model_registry.md)**: Version and track your models
- **[Inference API](inference.md)**: Deploy models to a REST endpoint
