# SplitDataset & Leakage

## Why `SplitDataset` exists

Many preprocessing nodes learn statistics from data (means, categories, bin edges, …).
If those statistics are computed on the full dataset and then evaluated on test data, you leak information.

`SplitDataset` is a container for:

- `train`
- `test`
- optional `validation`

Each split can be either:

- a `pd.DataFrame` (with the target column inside), or
- a tuple `(X: pd.DataFrame, y: pd.Series)`.

## Recommended patterns

### Pattern A: split in preprocessing

Use the `TrainTestSplitter` transformer early.

```python
{
  "name": "split",
  "transformer": "TrainTestSplitter",
  "params": {"test_size": 0.2, "random_state": 42, "target_column": "target"}
}
```

### Pattern B: create `SplitDataset` yourself

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline

df = pd.DataFrame(
  {
    "age": [10, 20, None, 40, 50, 60, None, 80],
    "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "target": [0, 1, 0, 1, 1, 0, 1, 0],
  }
)

config = {
  "preprocessing": [
    {
      "name": "impute",
      "transformer": "SimpleImputer",
      "params": {"strategy": "mean", "columns": ["age"]},
    },
    {
      "name": "encode",
      "transformer": "OneHotEncoder",
      "params": {"columns": ["city"], "drop_original": True},
    },
  ],
  "modeling": {
    "type": "random_forest_classifier",
    "params": {"n_estimators": 50, "random_state": 42},
  },
}
train, test = train_test_split(df, test_size=0.2, random_state=42)

dataset = SplitDataset(train=train, test=test, validation=None)

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(dataset, target_column="target")

print(metrics)
```

## Notes on inference

At inference time, `FeatureEngineer.transform()` skips splitters and resampling steps.
That ensures your `predict()` path remains deterministic.
