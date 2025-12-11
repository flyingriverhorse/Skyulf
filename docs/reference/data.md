# Data Ingestion

The `core.ml_pipeline.data` module handles loading datasets from various file formats.

## DataLoader

The `DataLoader` class provides methods to load full datasets or samples for previewing.

### Supported Formats
*   **CSV** (`.csv`)
*   **Parquet** (`.parquet`)

### Usage

```python
import os
import tempfile
from core.ml_pipeline.data.loader import DataLoader

loader = DataLoader()

# Create a tiny CSV for the example
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
    tmp.write("feature,target\n1,0\n2,1\n")
    path = tmp.name

try:
    df = loader.load_full(path)
    df_sample = loader.load_sample(path, n=1)
    print(len(df), len(df_sample))
finally:
    os.remove(path)
```

## SplitDataset

The `SplitDataset` container holds the training, testing, and optional validation splits of your data. This is the standard data object passed through the ML pipeline.

```python
from core.ml_pipeline.data.container import SplitDataset
import pandas as pd

train_df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
test_df = pd.DataFrame({"feature": [4, 5], "target": [1, 0]})
val_df = pd.DataFrame({"feature": [6], "target": [1]})

dataset = SplitDataset(
    train=train_df,
    test=test_df,
    validation=val_df  # Optional
)

# Accessing splits
print(dataset.train.shape)       # (3, 2)
print(dataset.test.shape)        # (2, 2)
print(dataset.validation.shape)  # (1, 2)

# Create a deep copy
dataset_copy = dataset.copy()
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `train` | `DataFrame` or `Tuple[DataFrame, Series]` | Training data (required) |
| `test` | `DataFrame` or `Tuple[DataFrame, Series]` | Test data (required) |
| `validation` | `DataFrame` or `Tuple[DataFrame, Series]` | Validation data (optional) |
