# Pipeline Quickstart

This guide shows a production-style workflow: split → fit → evaluate → predict → save/load.

## What data shape should I pass?

`SkyulfPipeline.fit(...)` supports two common inputs:

1. A single `pd.DataFrame` that **includes the target column**.
    - You pass `target_column="..."` and Skyulf splits features/target internally.
    - This is what the quickstart uses, because it’s the simplest onboarding path.

2. A `SplitDataset` (recommended when you already have train/test splits).
    - Each split can be either a DataFrame (with the target column) **or** a tuple `(X, y)`.

You do not need to manually build `X` and `y` for the pipeline unless you want to.

If you prefer a scikit-learn-style workflow (`X`/`y` + `train_test_split`), see:

- [Validation vs scikit-learn (Proof)](validation_vs_sklearn.md)

## Complete runnable example

This single snippet is intentionally end-to-end (no duplicated setup across steps).

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

# 1) Define a pipeline config
# Each preprocessing step is:
#   {"name": "...", "transformer": "TransformerType", "params": {...}}
config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.25,
                "validation_size": 0.0,
                "random_state": 42,
                "shuffle": True,
                "stratify": True,
                "target_column": "target",
            },
        },
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": ["age"]},
        },
        {
            "name": "encode",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 50, "random_state": 42},
    },
}

# 2) Training data (includes the target column)
df = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, 60, None, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

# 3) Fit (learn params on train split, apply to test split)
pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(df, target_column="target")
print("Metrics keys:", list(metrics.keys()))

# 4) Predict (feature-only dataframe)
incoming = pd.DataFrame({"age": [25, None], "city": ["A", "C"]})
preds = pipeline.predict(incoming)
print("Preds:")
print(preds)

# 5) Save / load
with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "model.pkl"
    pipeline.save(model_path)
    loaded = SkyulfPipeline.load(model_path)
    preds2 = loaded.predict(incoming)

print("Preds after reload:")
print(preds2)
```
