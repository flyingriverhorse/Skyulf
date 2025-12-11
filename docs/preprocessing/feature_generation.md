# Feature Generation

The `feature_generation` module creates new features from existing ones. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Define Steps
steps = [
    {
        "name": "poly_features",
        "transformer": "PolynomialFeatures",
        "params": {
            "degree": 2,
            "interaction_only": True
        }
    },
    {
        "name": "custom_features",
        "transformer": "FeatureGeneration",
        "params": {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a", "b"],
                    "output_column": "sum_ab"
                }
            ]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
generated_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### PolynomialFeatures
Generates polynomial and interaction features.

**Parameters:**
- `degree` (int): The degree of the polynomial features. Default `2`.
- `interaction_only` (bool): If true, only interaction features are produced. Default `False`.
- `include_bias` (bool): If true, include a bias column. Default `False`.
- `columns` (List[str]): Columns to generate features from. Default: all numeric.

### FeatureGeneration
Creates new features using arithmetic or statistical operations.

**Parameters:**
- `operations` (List[Dict]): List of operations to apply. Each operation has:
  - `operation_type` (str): 'arithmetic', 'statistical', 'datetime', 'text'.
  - `method` (str): The method to use (e.g., 'add', 'subtract', 'multiply', 'divide', 'log', 'sqrt').
  - `input_columns` (List[str]): Input columns.
  - `output_column` (str): Name of output column.
  - `constants` (List[float]): Optional constants for operations.
- `epsilon` (float): Small value added to prevent division by zero. Default `1e-8`.
- `allow_overwrite` (bool): Allow overwriting existing columns. Default `False`.

**Python Config:**
```python
{
    "name": "create_ratio",
    "transformer": "FeatureGeneration",
    "params": {
        "operations": [
            {
                "operation_type": "arithmetic",
                "method": "divide",
                "input_columns": ["price", "quantity"],
                "output_column": "unit_price"
            },
            {
                "operation_type": "arithmetic",
                "method": "log",
                "input_columns": ["income"],
                "output_column": "log_income"
            }
        ]
    }
}
```
