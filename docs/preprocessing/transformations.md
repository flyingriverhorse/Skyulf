# Transformations

The `transformations` module provides mathematical transformations. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd
import numpy as np

# Sample Data
df = pd.DataFrame({
    'skewed_data': np.random.exponential(size=100),
    'large_values': np.random.randint(1, 10000, size=100)
})

# Define Steps
steps = [
    {
        "name": "power_transform",
        "transformer": "PowerTransformer",
        "params": {
            "method": "yeo-johnson",
            "columns": ["skewed_data"]
        }
    },
    {
        "name": "log_transform",
        "transformer": "SimpleTransformation",
        "params": {
            "transformations": [
                {"column": "large_values", "method": "log"}
            ]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
transformed_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### PowerTransformer
Applies a power transform (Yeo-Johnson or Box-Cox) to make data more Gaussian-like.

**Parameters:**
- `method` (str): 'yeo-johnson' or 'box-cox'.
- `standardize` (bool): Apply zero-mean, unit-variance normalization.

### SimpleTransformation
Applies simple mathematical functions to columns.

**Parameters:**
- `transformations` (List[Dict]): List of transformations. Each has:
  - `column` (str): Column to transform.
  - `method` (str): One of:
    - `'log'` - Natural log (uses log1p for safety with zeros)
    - `'square_root'` - Square root
    - `'cube_root'` - Cube root
    - `'square'` - Square
    - `'reciprocal'` - 1/x
    - `'exponential'` - e^x (clipped to prevent overflow)

**Python Config:**
```python
{
    "name": "log_transform",
    "transformer": "SimpleTransformation",
    "params": {
        "transformations": [
            {"column": "income", "method": "log"},
            {"column": "count", "method": "square_root"}
        ]
    }
}
```

### GeneralTransformation
Applies a custom lambda function (use with caution, requires code execution).

**Parameters:**
- `func` (str): Python code string for the function.

**Python Config:**
```python
{
    "name": "custom_lambda",
    "transformer": "GeneralTransformation",
    "params": {
        "func": "lambda x: x + 1"
    }
}
```
