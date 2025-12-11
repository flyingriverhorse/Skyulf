# Bucketing (Binning)

The `bucketing` module discretizes continuous variables into bins. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'age': [10, 20, 30, 40, 50, 60, 70, 80],
    'score': [0.1, 0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
})

# Define Steps
steps = [
    {
        "name": "bin_age",
        "transformer": "GeneralBinning",
        "params": {
            "n_bins": 4,
            "strategy": "equal_width",
            "columns": ["age"]
        }
    },
    {
        "name": "bin_score",
        "transformer": "CustomBinning",
        "params": {
            "bins": [0, 0.3, 0.7, 1.0],
            "columns": ["score"]
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
binned_df, metrics = engineer.fit_transform(df)
```

## Available Transformers

### GeneralBinning
Bins data into intervals using various strategies.

**Parameters:**
- `n_bins` (int): Number of bins. Default 5.
- `strategy` (str): Binning strategy. Options:
    - `'equal_width'`: Uniform width bins (default)
    - `'equal_frequency'`: Equal number of samples per bin
    - `'kmeans'`: Bin edges determined by k-means clustering
    - `'custom'`: Use manually specified bin edges
- `columns` (List[str]): Columns to bin.
- `column_strategies` (Dict): Per-column strategy overrides.
- `custom_bins` (Dict): Custom bin edges for 'custom' strategy. Format: `{'column': [edge1, edge2, ...]}`
- `duplicates` (str): How to handle duplicate bin edges. Default 'drop'.

### CustomBinning
Bins data using manually specified edges.

**Parameters:**
- `bins` (List[float]): List of bin edges.
- `columns` (List[str]): Columns to bin.

### KBinsDiscretizer
Wrapper around sklearn's KBinsDiscretizer.

**Parameters:**
- `n_bins` (int): Number of bins.
- `encode` (str): 'ordinal', 'onehot', or 'onehot-dense'.
- `strategy` (str): 'uniform', 'quantile', or 'kmeans'.

**Python Config:**
```python
{
    "name": "kbins_discretizer",
    "transformer": "KBinsDiscretizer",
    "params": {
        "n_bins": 5,
        "strategy": "kmeans",
        "encode": "ordinal"
    }
}
```
