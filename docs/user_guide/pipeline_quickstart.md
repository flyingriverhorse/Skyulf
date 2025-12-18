# Pipeline Quickstart

This guide shows a production-style workflow: split → fit → evaluate → predict.

## 1) Define a config

Preprocessing steps are executed by `FeatureEngineer`.

Each step has:

- `name`: human-readable identifier
- `transformer`: the node type (string)
- `params`: node configuration passed to the Calculator

```python
config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.2,
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
            "params": {"columns": ["city"], "drop_original": True},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 200, "random_state": 42},
    },
}
```

## 2) Fit

```python
from __future__ import annotations

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.2,
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
            "params": {"columns": ["city"], "drop_original": True},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 200, "random_state": 42},
    },
}

df = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, 60, None, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(df, target_column="target")
print(metrics)
```

## 3) Predict

`predict()` expects a feature-only dataframe.

```python
from __future__ import annotations

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.2,
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
            "params": {"columns": ["city"], "drop_original": True},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 200, "random_state": 42},
    },
}

train_df = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, 60, None, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

pipeline = SkyulfPipeline(config)
_ = pipeline.fit(train_df, target_column="target")

incoming = pd.DataFrame({"age": [25, None], "city": ["A", "C"]})
preds = pipeline.predict(incoming)

print(preds)
```

## 4) Save / load

SkyulfPipeline supports pickling.

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.2,
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
            "params": {"columns": ["city"], "drop_original": True},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 200, "random_state": 42},
    },
}

df = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, 60, None, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.pkl"

    pipeline = SkyulfPipeline(config)
    _ = pipeline.fit(df, target_column="target")
    pipeline.save(path)

    loaded = SkyulfPipeline.load(path)
    preds = loaded.predict(pd.DataFrame({"age": [15, 65], "city": ["A", "B"]}))

print(preds)
```
