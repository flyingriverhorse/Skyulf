# Outlier Handling

The `outliers` module provides methods for detecting and handling outliers. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'price': [10, 12, 11, 1000, 13], # 1000 is an outlier
    'age': [20, 21, 22, 150, 23]     # 150 is an outlier
})

# Define Steps
steps = [
    {
        "name": "remove_price_outliers",
        "transformer": "IQR",
        "params": {
            "multiplier": 1.5,
            "columns": ["price"]
        }
    },
    {
        "name": "clip_age",
        "transformer": "ManualBounds",
        "params": {
            "bounds": {"age": {"lower": 0, "upper": 100}} # Clip age to 0-100
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
processed_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### IQR (Interquartile Range)
Detects outliers using the IQR method (Q1 - k*IQR, Q3 + k*IQR). Rows with values outside bounds are **removed**.

**Parameters:**
- `multiplier` (float): Multiplier for IQR (k). Default 1.5.
- `columns` (List[str]): Columns to check.

### ZScore
Detects outliers using Z-Score (mean +/- k*std). Rows with Z-score beyond threshold are **removed**.

**Parameters:**
- `threshold` (float): Z-score threshold. Default 3.0.
- `columns` (List[str]): Columns to check.

**Python Config:**
```python
{
    "name": "zscore_filter",
    "transformer": "ZScore",
    "params": {
        "threshold": 3.0,
        "columns": ["feature1"]
    }
}
```

### Winsorize
Limits extreme values to specified percentiles (clips values to bounds instead of removing rows).

**Parameters:**
- `lower_percentile` (float): Lower percentile (0-100). Default `5.0`.
- `upper_percentile` (float): Upper percentile (0-100). Default `95.0`.
- `columns` (List[str]): Columns to winsorize.

**Python Config:**
```python
{
    "name": "winsorize_income",
    "transformer": "Winsorize",
    "params": {
        "lower_percentile": 5.0,
        "upper_percentile": 95.0,
        "columns": ["income"]
    }
}
```

### ManualBounds
Clips values to manually specified bounds. Rows outside bounds are **removed**.

**Parameters:**
- `bounds` (Dict): Dictionary mapping columns to bounds. Each bound has `lower` and/or `upper`.

**Python Config:**
```python
{
    "name": "clip_age",
    "transformer": "ManualBounds",
    "params": {
        "bounds": {
            "age": {"lower": 0, "upper": 100},
            "score": {"lower": 0}  # Only lower bound
        }
    }
}
```

### EllipticEnvelope
Detects outliers using robust covariance estimation (multivariate). Outlier rows are **removed**.

**Parameters:**
- `contamination` (float): Expected proportion of outliers (0-0.5). Default `0.01`.
- `columns` (List[str]): Columns to check. Default: all numeric.

**Python Config:**
```python
{
    "name": "elliptic_envelope",
    "transformer": "EllipticEnvelope",
    "params": {
        "contamination": 0.05,
        "columns": ["feature1", "feature2"]
    }
}
```
