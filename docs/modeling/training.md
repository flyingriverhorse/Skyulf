# Model Training

This guide details how to train machine learning models using the `StatefulEstimator` interface.

## Basic Training

The primary method for training is `fit_predict`. This method:
1.  Trains the model on the training split of the `SplitDataset`.
2.  Saves the trained model artifact to the `ArtifactStore`.
3.  Generates predictions for all available splits (train, test, validation).
4.  Returns a dictionary of predictions.

### Example

```python
from core.ml_pipeline.modeling.base import StatefulEstimator
from core.ml_pipeline.modeling.classification import RandomForestClassifierCalculator, RandomForestClassifierApplier
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.data.container import SplitDataset
import pandas as pd

# 1. Prepare Data
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5] * 20,
    "feature2": [5, 4, 3, 2, 1] * 20,
    "target": [0, 1, 0, 1, 0] * 20
})
# In a real scenario, use a Splitter to create this
dataset = SplitDataset(train=df, test=df, validation=None)

# 2. Setup Estimator
store = LocalArtifactStore("./artifacts")
estimator = StatefulEstimator(
    calculator=RandomForestClassifierCalculator(),
    applier=RandomForestClassifierApplier(),
    artifact_store=store,
    node_id="rf_model"
)

# 3. Train
# config passes hyperparameters to the underlying sklearn model
predictions = estimator.fit_predict(
    dataset=dataset,
    target_column="target",
    config={"n_estimators": 100, "max_depth": 5}
)

print("Training completed.")
print(f"Train Predictions shape: {predictions['train'].shape}")
```

## Cross-Validation

You can perform cross-validation on the training set using the `cross_validate` method. This is useful for assessing model stability and performance without touching the test set.

```python
# Perform 5-fold Cross-Validation
cv_results = estimator.cross_validate(
    dataset=dataset,
    target_column="target",
    config={"n_estimators": 100},
    n_folds=5,
    cv_type="k_fold" # or 'stratified_k_fold'
)

print("CV Results:", cv_results)
# cv_results contains 'metrics' (list of dicts) and 'predictions' (list of Series)
```

## Supported Algorithms

The system supports various algorithms via their respective Calculators:

*   **Classification**:
    *   `LogisticRegressionCalculator`
    *   `RandomForestClassifierCalculator`
*   **Regression**:
    *   `RidgeRegressionCalculator`
    *   `RandomForestRegressorCalculator`

Each calculator accepts standard hyperparameters in the `config` dictionary (e.g., `n_estimators`, `C`, `alpha`).
