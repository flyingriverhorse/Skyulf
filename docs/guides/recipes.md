# Recipes

This page contains practical patterns for common workflows.

## Recipe: split, preprocess, train

If you want evaluation metrics, use a split step (or pass a `SplitDataset`).

```python
from __future__ import annotations

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

# In real usage you'd likely load a file:
# df = pd.read_csv("your_data.csv")

df = pd.DataFrame(
    {
        "free_text": [
            "  Hello   World  ",
            "Skyulf is GREAT! ",
            "  hello\tworld ",
            "  ML pipelines   ",
            "Encode + scale",
            " text cleaning  ",
        ],
        "country": ["TR", "TR", "DE", "DE", "TR", "DE"],
        "age": [10, 20, 30, 40, 50, 60],
        "target": [0, 1, 0, 1, 1, 0],
    }
)

config = {
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "test_size": 0.34,
                "validation_size": 0.0,
                "random_state": 42,
                "shuffle": True,
                "stratify": True,
                "target_column": "target",
            },
        },
        {
            "name": "text_clean",
            "transformer": "TextCleaning",
            "params": {
                "columns": ["free_text"],
                "operations": [
                    {"op": "trim", "mode": "both"},
                    {"op": "case", "mode": "lower"},
                    {"op": "regex", "mode": "collapse_whitespace"},
                ],
            },
        },
        {
            "name": "encode",
            "transformer": "OneHotEncoder",
            "params": {
                "columns": ["country", "free_text"],
                "drop_original": True,
                "handle_unknown": "ignore",
            },
        },
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": ["age"]},
        },
        {
            "name": "scale",
            "transformer": "StandardScaler",
            "params": {"auto_detect": True},
        },
    ],
    "modeling": {"type": "random_forest_classifier", "params": {"n_estimators": 200}},
}

pipeline = SkyulfPipeline(config)
report = pipeline.fit(df, target_column="target")
print(report.get("modeling"))
```

## Recipe: safe inference

At inference time you typically:

1. load a persisted pipeline
2. call `predict(df)` on a dataframe without the target column

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from skyulf.pipeline import SkyulfPipeline

df = pd.DataFrame(
    {
        "age": [10, 20, 30, 40, 50, 60],
        "city": ["A", "B", "A", "C", "B", "A"],
        "target": [0, 1, 0, 1, 1, 0],
    }
)

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
        "params": {"n_estimators": 50, "random_state": 42},
    },
}

with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / "model.pkl"

    pipeline = SkyulfPipeline(config)
    _ = pipeline.fit(df, target_column="target")
    pipeline.save(model_path)

    loaded = SkyulfPipeline.load(model_path)

    new_df = pd.DataFrame({"age": [25, 55], "city": ["A", "B"]})
    preds = loaded.predict(new_df)

print(preds)
```

## Recipe: use a single component (debug)

For debugging, you can run one node directly.

```python
import pandas as pd

from skyulf.preprocessing.imputation import SimpleImputerApplier, SimpleImputerCalculator

df = pd.DataFrame({"A": [1, 2, None, 4]})
config = {"columns": ["A"], "strategy": "mean"}

params = SimpleImputerCalculator().fit(df, config)
out = SimpleImputerApplier().apply(df, params)

print(params)
print(out)
```
