# Modeling Overview

The modeling module in Skyulf provides a unified interface for training and applying machine learning models. It is built around the **StatefulEstimator** pattern, which separates the logic of *training* a model from *applying* it for inference.

## Architecture

### StatefulEstimator
The `StatefulEstimator` is the main entry point. It manages:
1.  **Calculator**: The component responsible for training the model (e.g., `LogisticRegressionCalculator`).
2.  **Applier**: The component responsible for generating predictions using a trained model (e.g., `LogisticRegressionApplier`).
3.  **Artifact Store**: Where the trained model binary (e.g., a pickled scikit-learn object) is saved and loaded from.

### Calculator vs. Applier
*   **Calculator**: Takes training data (`SplitDataset`), fits the model, and returns a `ModelArtifact`. It does *not* store the model in memory itself; it hands it off to the artifact store.
*   **Applier**: Loads a `ModelArtifact` from the store and uses it to predict on new data.

## Usage Example

```python
from core.ml_pipeline.modeling.base import StatefulEstimator
from core.ml_pipeline.modeling.classification import LogisticRegressionCalculator, LogisticRegressionApplier
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.data.container import SplitDataset

# 1. Setup Artifact Store
store = LocalArtifactStore("./artifacts")

# 2. Initialize Estimator
# We combine a Calculator and an Applier
estimator = StatefulEstimator(
    calculator=LogisticRegressionCalculator(),
    applier=LogisticRegressionApplier(),
    artifact_store=store,
    node_id="my_model_node"
)

# 3. Train and Predict
# Create dummy data
import pandas as pd
df = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5],
    "target": [0, 1, 0, 1, 0]
})
my_split_dataset = SplitDataset(train=df, test=df, validation=df)

# 'dataset' is a SplitDataset containing train/test splits
# 'target_column' is the name of the label column
predictions = estimator.fit_predict(
    dataset=my_split_dataset,
    target_column="target",
    config={"max_iter": 500}
)

# The trained model is now saved in ./artifacts/my_model_node
```
