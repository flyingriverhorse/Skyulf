# Classification Models

The `core.ml_pipeline.modeling.classification` module provides calculators and appliers for classification tasks.

## Logistic Regression

A linear model for classification. Wraps `sklearn.linear_model.LogisticRegression`.

### Usage
```python
from core.ml_pipeline.modeling.classification import LogisticRegressionCalculator

calc = LogisticRegressionCalculator()
# Override default parameters
params = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs"
}
```

### Default Parameters
*   `max_iter`: 1000
*   `solver`: "lbfgs"
*   `random_state`: 42

## Random Forest Classifier

An ensemble learning method using multiple decision trees. Wraps `sklearn.ensemble.RandomForestClassifier`.

### Usage
```python
from core.ml_pipeline.modeling.classification import RandomForestClassifierCalculator

calc = RandomForestClassifierCalculator()
# Override default parameters
params = {
    "n_estimators": 100,
    "max_depth": 20
}
```

### Default Parameters
*   `n_estimators`: 50
*   `max_depth`: 10
*   `min_samples_split`: 5
*   `min_samples_leaf`: 2
*   `n_jobs`: -1 (uses all processors)
*   `random_state`: 42
