# Hyperparameter Tuning

The `skyulf.modeling.tuning` module allows you to optimize model hyperparameters using various search strategies.

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
from skyulf.modeling.tuning.tuner import TunerCalculator, TunerApplier
from skyulf.modeling.tuning.schemas import TuningConfig
from skyulf.modeling.classification import RandomForestClassifierCalculator

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

from skyulf.modeling.base import StatefulEstimator
from skyulf.data.dataset import SplitDataset
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
    node_id="tuned_rf_node"
)

# 5. Train
# Note: fit_predict will return placeholder predictions as this is a tuning job.
# The best hyperparameters are available on the estimator model (a TuningResult).
estimator.fit_predict(
    dataset=dataset,
    target_column="target",
    config=tuning_config
)

print("Best params:", estimator.model.best_params)
print("Best score:", estimator.model.best_score)
```
