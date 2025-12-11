# Inspection

The `inspection` module provides read-only nodes that analyze data without modifying it. These are useful for debugging pipelines and capturing intermediate statistics.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'age': [25, 30, 35, None],
    'salary': [50000, 60000, 70000, 80000]
})

# Define Steps
steps = [
    {
        "name": "profile_data",
        "transformer": "DatasetProfile",
        "params": {}
    },
    {
        "name": "snapshot",
        "transformer": "DataSnapshot",
        "params": {"n_rows": 3}
    }
]

# Execute
engineer = FeatureEngineer(steps)
result_df, metrics = engineer.fit_transform(df)

# The data is unchanged, but metrics contain profiling info
print(metrics)
```

## Available Transformers

### DatasetProfile
Computes basic statistics about the dataset without modifying it.

**Parameters:** None required.

**Output Metrics:**
- `rows` (int): Number of rows.
- `columns` (int): Number of columns.
- `dtypes` (Dict): Column data types.
- `missing` (Dict): Missing value counts per column.
- `numeric_stats` (Dict): Descriptive statistics for numeric columns.

**Python Config:**
```python
{
    "name": "profile",
    "transformer": "DatasetProfile",
    "params": {}
}
```

### DataSnapshot
Captures a snapshot of the first N rows for debugging.

**Parameters:**
- `n_rows` (int): Number of rows to capture. Default `5`.

**Python Config:**
```python
{
    "name": "preview",
    "transformer": "DataSnapshot",
    "params": {"n_rows": 10}
}
```

> **Note:** Inspection nodes are pass-through—they do not modify the DataFrame. Their output is captured in the pipeline metrics.
