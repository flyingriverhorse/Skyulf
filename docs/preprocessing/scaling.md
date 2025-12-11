# Scaling

The `scaling` module normalizes numeric features. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'age': [20, 30, 40],
    'salary': [30000, 50000, 90000]
})

# Define Steps
steps = [
    {
        "name": "scale_age",
        "transformer": "StandardScaler",
        "params": {
            "columns": ["age"] # Optional: specify columns
        }
    },
    {
        "name": "scale_salary",
        "transformer": "MinMaxScaler",
        "params": {
            "feature_range": (0, 1),
            "columns": ["salary"]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
scaled_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### StandardScaler
Standardizes features by removing the mean and scaling to unit variance.

**Python Config:**
```python
{
    "name": "standard_scale",
    "transformer": "StandardScaler",
    "params": {}
}
```

### MinMaxScaler
Scales features to a given range (usually 0-1).

**Parameters:**
- `feature_range` (Tuple[float, float]): (min, max). Default `(0, 1)`.

**Python Config:**
```python
{
    "name": "minmax_scale",
    "transformer": "MinMaxScaler",
    "params": {
        "feature_range": (0, 1)
    }
}
```

### RobustScaler
Scales features using statistics that are robust to outliers (IQR).

**Parameters:**
- `quantile_range` (Tuple[float, float]): (q_min, q_max). Default `(25.0, 75.0)`.

**Python Config:**
```python
{
    "name": "robust_scale",
    "transformer": "RobustScaler",
    "params": {
        "quantile_range": (25.0, 75.0)
    }
}
```

### MaxAbsScaler
Scales each feature by its maximum absolute value.

**Python Config:**
```python
{
    "name": "maxabs_scale",
    "transformer": "MaxAbsScaler",
    "params": {}
}
```
