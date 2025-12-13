# Encoding

The `encoding` module handles categorical variable encoding. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green'],
    'size': ['S', 'M', 'L']
})

# Define Steps
steps = [
    {
        "name": "encode_color",
        "transformer": "OneHotEncoder",
        "params": {
            "columns": ["color"],
            "handle_unknown": "ignore"
        }
    },
    {
        "name": "encode_size",
        "transformer": "OrdinalEncoder",
        "params": {
            "columns": ["size"]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
encoded_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### OneHotEncoder
Creates binary columns for each category.

**Parameters:**
- `handle_unknown` (str): 'error' or 'ignore'. Default 'ignore'.
- `drop` (str): 'first' or None.
- `columns` (List[str]): Columns to encode.

### OrdinalEncoder
Encodes categories as integers.

**Parameters:**
- `columns` (List[str]): Columns to encode.

### LabelEncoder
Encodes target labels as integers (0 to n_classes-1).

**Python Config:**
```python
{
    "name": "label_encode",
    "transformer": "LabelEncoder",
    "params": {}
}
```

### TargetEncoder
Encodes categories based on the mean of the target variable.

**Parameters:**
- `smoothing` (float): Smoothing factor.
- `columns` (List[str]): Columns to encode.

**Python Config:**
```python
{
    "name": "target_encode",
    "transformer": "TargetEncoder",
    "params": {
        "columns": ["zipcode"],
        "smoothing": 10.0
    }
}
```

### HashEncoder
Encodes categories using the hashing trick.

**Parameters:**
- `n_components` (int): Number of hash buckets.
- `columns` (List[str]): Columns to encode.

**Python Config:**
```python
{
    "name": "hash_encode",
    "transformer": "HashEncoder",
    "params": {
        "n_components": 8,
        "columns": ["category"]
    }
}
```
