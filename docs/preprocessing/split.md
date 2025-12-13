# Splitting

The `split` module handles dataset splitting. These transformers are typically used within a `FeatureEngineer` pipeline.

## Usage Example

```python
from skyulf.preprocessing.pipeline import FeatureEngineer
import pandas as pd

# Sample Data
df = pd.DataFrame({
    'feature': range(20),
    'target': [0, 1] * 10
})

# Define Steps
steps = [
    {
        "name": "split_xy",
        "transformer": "feature_target_split",
        "params": {"target_column": "target"}
    },
    {
        "name": "split_train_test",
        "transformer": "TrainTestSplitter",
        "params": {
            "test_size": 0.2,
            "stratify": "target" # Stratify by y (target)
        }
    }
]

# Execute
engineer = FeatureEngineer(steps)
split_data, metrics = engineer.fit_transform(df)
# split_data will be a SplitDataset object (train, test, validation)
```

## Available Transformers

### TrainTestSplitter
Splits the dataset into training, testing, and optionally validation sets.

**Parameters:**
- `test_size` (float): Proportion of the dataset for test split. Default 0.2.
- `validation_size` (float): Proportion of the dataset for validation split. Default 0.0 (no validation set).
- `random_state` (int): Seed for reproducibility. Default 42.
- `shuffle` (bool): Whether to shuffle data before splitting. Default True.
- `stratify` (bool): Whether to stratify the split by target. Default False.
- `target_column` (str): Column to use for stratification (required if stratify=True).

### feature_target_split
Separates features (X) from the target variable (y).

**Parameters:**
- `target_column` (str): Name of the target column.
