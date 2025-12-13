# Drop and Missing Handling

The `drop_and_missing` module handles dropping rows/columns and creating missing indicators. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer
import pandas as pd
import numpy as np

# Sample Data
df = pd.DataFrame({
    'id': [1, 1, 2],
    'val': [10, np.nan, 20],
    'empty_col': [np.nan, np.nan, np.nan]
})

# Define Steps
steps = [
    {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["id"], "keep": "first"}
    },
    {
        "name": "drop_cols",
        "transformer": "DropMissingColumns",
        "params": {"threshold": 0.9} # Drop if >90% missing
    },
    {
        "name": "missing_ind",
        "transformer": "MissingIndicator",
        "params": {"columns": ["val"]}
    }
]

# Execute
engineer = FeatureEngineer(steps)
cleaned_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### Deduplicate
Removes duplicate rows.

**Parameters:**
- `subset` (List[str]): Columns to consider for identifying duplicates. If None, uses all columns.
- `keep` (str): 'first', 'last', or False (drop all). Default 'first'.

### DropMissingColumns
Drops columns with too many missing values.

**Parameters:**
- `threshold` (float): Drop columns with missing fraction > threshold. Default 1.0 (drop only if all missing).

### DropMissingRows
Drops rows with too many missing values.

**Parameters:**
- `missing_threshold` (float): Drop rows with missing % >= this value.
- `drop_if_any_missing` (bool): If True, drop rows with any missing values. Default False.

**Python Config:**
```python
{
    "name": "drop_missing_rows",
    "transformer": "DropMissingRows",
    "params": {
        "missing_threshold": 50.0  # Drop if >=50% missing
    }
}
```

Or drop any row with missing values:
```python
{
    "name": "drop_any_missing",
    "transformer": "DropMissingRows",
    "params": {
        "drop_if_any_missing": True
    }
}
```

### MissingIndicator
Creates binary indicators for missing values.

**Parameters:**
- `columns` (List[str]): Columns to create indicators for.
- `suffix` (str): Suffix for new columns. Default '_missing'.
