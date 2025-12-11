# Imputation

The `imputation` module provides strategies for filling missing values. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd
import numpy as np

# Sample Data
df = pd.DataFrame({
    'age': [25, np.nan, 30],
    'salary': [50000, 60000, np.nan]
})

# Define Steps
steps = [
    {
        "name": "impute_age",
        "transformer": "SimpleImputer",
        "params": {
            "strategy": "mean",
            "columns": ["age"]
        }
    },
    {
        "name": "impute_salary",
        "transformer": "KNNImputer",
        "params": {
            "n_neighbors": 2,
            "columns": ["salary"]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
imputed_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### SimpleImputer
Imputes missing values using univariate statistics.

**Parameters:**
- `strategy` (str): 'mean', 'median', 'most_frequent', or 'constant'.
- `fill_value` (Any): Value to use when strategy is 'constant'.
- `columns` (List[str]): Columns to impute.

### KNNImputer
Imputes missing values using k-Nearest Neighbors.

**Parameters:**
- `n_neighbors` (int): Number of neighbors. Default 5.
- `weights` (str): 'uniform' or 'distance'. Default 'uniform'.
- `columns` (List[str]): Columns to impute.

### IterativeImputer
Multivariate imputation by chained equations (MICE).

**Parameters:**
- `max_iter` (int): Maximum number of imputation rounds.
- `random_state` (int): Seed for reproducibility.
- `columns` (List[str]): Columns to impute.

**Python Config:**
```python
{
    "name": "impute_mice",
    "transformer": "IterativeImputer",
    "params": {
        "max_iter": 10,
        "columns": ["age", "salary"]
    }
}
```
