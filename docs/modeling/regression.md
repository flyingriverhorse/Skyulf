# Regression Models

The `core.ml_pipeline.modeling.regression` module provides calculators and appliers for regression tasks.

## Ridge Regression

Linear least squares with l2 regularization. Wraps `sklearn.linear_model.Ridge`.

### Usage
```python
from core.ml_pipeline.modeling.regression import RidgeRegressionCalculator

calc = RidgeRegressionCalculator()
# Override default parameters
params = {
    "alpha": 1.0,
    "solver": "auto"
}
```

### Default Parameters
*   `alpha`: 1.0
*   `random_state`: 42

## Random Forest Regressor

A random forest regressor. Wraps `sklearn.ensemble.RandomForestRegressor`.

### Usage
```python
from core.ml_pipeline.modeling.regression import RandomForestRegressorCalculator

calc = RandomForestRegressorCalculator()
# Override default parameters
params = {
    "n_estimators": 100,
    "criterion": "squared_error"
}
```

### Default Parameters
*   `n_estimators`: 50
*   `max_depth`: 10
*   `min_samples_split`: 5
*   `min_samples_leaf`: 2
*   `n_jobs`: -1
*   `random_state`: 42
