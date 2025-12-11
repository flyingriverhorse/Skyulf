# Hyperparameter Tuning

The `core.ml_pipeline.modeling.tuning` module allows you to optimize model hyperparameters using various search strategies.

## TunerCalculator

The `TunerCalculator` wraps another calculator (like `LogisticRegressionCalculator`) and performs a search over a parameter grid.

### Supported Strategies
*   `grid`: Exhaustive search over specified parameter values (`GridSearchCV`).
*   `random`: Randomized search over parameters (`RandomizedSearchCV`).
*   `halving_grid`: Successive halving grid search (`HalvingGridSearchCV`).
*   `halving_random`: Successive halving random search (`HalvingRandomSearchCV`).
*   `optuna`: Bayesian optimization using Optuna (`OptunaSearchCV`).

## Usage Example

```python
from core.ml_pipeline.modeling.tuning import TunerCalculator, TuningConfig
from core.ml_pipeline.modeling.classification import RandomForestClassifierCalculator

# 1. Define the base model
base_calc = RandomForestClassifierCalculator()

# 2. Define the tuning configuration
tuning_config = TuningConfig(
    strategy="random",
    metric="accuracy",
    cv_folds=5,
    n_trials=20,  # Number of random combinations to try
    search_space={
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
)

# 3. Initialize the Tuner
# The tuner acts as a Calculator that returns the best model found
tuner = TunerCalculator(
    model_calculator=base_calc
)

# 4. Use in a StatefulEstimator
# We use TunerApplier because tuning only produces parameters, not a predictive model
from core.ml_pipeline.modeling.base import StatefulEstimator
from core.ml_pipeline.modeling.tuning import TunerApplier
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.data.container import SplitDataset
import pandas as pd

# Create dummy data for the example
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5] * 10,
    "feature2": [5, 4, 3, 2, 1] * 10,
    "target": [0, 1, 0, 1, 0] * 10
})
dataset = SplitDataset(train=df, test=df, validation=df)

estimator = StatefulEstimator(
    calculator=tuner,
    applier=TunerApplier(),
    artifact_store=LocalArtifactStore("./artifacts"),
    node_id="tuned_rf_node"
)

# 5. Train
# Note: fit_predict will return placeholder predictions as this is a tuning job.
# The best hyperparameters are stored in the artifact store for later use.
estimator.fit_predict(
    dataset=dataset,
    target_column="target",
    config=tuning_config
)
```
