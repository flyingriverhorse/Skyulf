# Feature Selection

The `feature_selection` module reduces dimensionality by selecting the most relevant features. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer
import pandas as pd
import numpy as np

# Sample Data
df = pd.DataFrame({
    'low_var': [1, 1, 1, 1, 1, 1, 1, 0],
    'feature1': np.random.rand(8),
    'feature2': np.random.rand(8),
    'target': [0, 1, 0, 1, 0, 1, 0, 1]
})

# Define Steps
steps = [
    {
        "name": "remove_low_var",
        "transformer": "VarianceThreshold",
        "params": {
            "threshold": 0.1 # Remove features with variance < 0.1
        }
    },
    {
        "name": "split_xy",
        "transformer": "feature_target_split", # Required before supervised selection
        "params": {"target_column": "target"}
    },
    {
        "name": "select_k_best",
        "transformer": "UnivariateSelection",
        "params": {
            "k": 2,
            "score_func": "f_classif"
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
selected_data, metrics = engineer.fit_transform(df)
# Note: selected_data will be a tuple (X, y) after split
```

## Available Transformers

### VarianceThreshold
Removes features with low variance.

**Parameters:**
- `threshold` (float): Features with variance lower than this are removed.

### CorrelationThreshold
Removes features highly correlated with others.

**Parameters:**
- `threshold` (float): Correlation coefficient threshold (0 to 1).

**Python Config:**
```python
{
    "name": "corr_filter",
    "transformer": "CorrelationThreshold",
    "params": {
        "threshold": 0.95
    }
}
```

### UnivariateSelection
Selects features based on univariate statistical tests.

**Parameters:**
- `k` (int): Number of top features to select.
- `score_func` (str): 'f_classif', 'f_regression', 'chi2', etc.

### ModelBasedSelection
Selects features using an estimator (e.g., Random Forest importance).

**Parameters:**
- `estimator` (str): 'RandomForest', 'LogisticRegression', etc.
- `threshold` (str or float): 'mean', 'median', or float value.

**Python Config:**
```python
{
    "name": "model_select",
    "transformer": "ModelBasedSelection",
    "params": {
        "estimator": "RandomForest",
        "threshold": "mean"
    }
}
```
