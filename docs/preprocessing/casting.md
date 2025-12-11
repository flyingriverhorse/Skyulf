# Type Casting

The `casting` module handles data type conversions. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'id': [1, 2, 3],
    'status': ['0', '1', '0']
})

# Define Steps
steps = [
    {
        "name": "cast_id",
        "transformer": "Casting",
        "params": {
            "target_type": "string",
            "columns": ["id"]
        }
    },
    {
        "name": "cast_status",
        "transformer": "Casting",
        "params": {
            "target_type": "int",
            "columns": ["status"]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
casted_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### Casting
Casts columns to specific data types.

**Parameters:**
- `target_type` (str): 'int', 'float', 'string', 'category', 'datetime'.
- `columns` (List[str]): Columns to cast.
