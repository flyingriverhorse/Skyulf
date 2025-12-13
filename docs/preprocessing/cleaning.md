# Data Cleaning

The `cleaning` module provides transformers for cleaning text and replacing values. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'description': ['Hello World!', '  Test  ', 'Foo'],
    'status': ['active', 'inactive', '?']
})

# Define Steps
steps = [
    {
        "name": "clean_text",
        "transformer": "TextCleaning",
        "params": {
            "columns": ["description"],
            "lowercase": True,
            "remove_punctuation": True,
            "strip_whitespace": True
        }
    },
    {
        "name": "replace_values",
        "transformer": "ValueReplacement",
        "params": {
            "columns": ["status"],
            "mapping": {"?": None}
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
cleaned_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### TextCleaning
Performs basic text cleaning operations on string columns.

**Parameters:**
- `columns` (List[str]): Columns to clean.
- `lowercase` (bool): Convert to lowercase. Default `True`.
- `remove_punctuation` (bool): Remove punctuation. Default `True`.
- `remove_numbers` (bool): Remove digits. Default `False`.
- `strip_whitespace` (bool): Strip leading/trailing whitespace. Default `True`.

### ValueReplacement
Replaces specific values in columns.

**Parameters:**
- `mapping` (Dict): Dictionary mapping old values to new values (e.g., `{"?": None, "N/A": None}`).
- `columns` (List[str]): Columns to apply replacement to.

### AliasReplacement
Replaces aliases with a canonical value.

**Parameters:**
- `replacements` (Dict): Dictionary mapping canonical values to lists of aliases.
- `columns` (List[str]): Columns to apply replacement to.

**Python Config:**
```python
{
    "name": "fix_usa",
    "transformer": "AliasReplacement",
    "params": {
        "replacements": {
            "USA": ["United States", "U.S.A.", "US"]
        },
        "columns": ["country"]
    }
}
```

### InvalidValueReplacement
Replaces invalid values (e.g., negative ages) with NaN.

**Parameters:**
- `rules` (List[Dict]): List of rules. Each rule has `column`, `operator` (<, >, <=, >=, ==, !=), and `value`.

**Python Config:**
```python
{
    "name": "fix_invalid_age",
    "transformer": "InvalidValueReplacement",
    "params": {
        "rules": [
            {"column": "age", "operator": "<", "value": 0}
        ]
    }
}
```
