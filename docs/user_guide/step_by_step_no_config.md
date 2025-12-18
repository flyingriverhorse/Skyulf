# Step-by-Step (No Config)

This tutorial runs an end-to-end workflow **without** building a pipeline `config` dictionary.
Instead, it uses Skyulf’s low-level building blocks directly:

- preprocessing **Calculators** and **Appliers**
- `SplitDataset` to avoid data leakage
- modeling via `StatefulEstimator`

Use this approach when you want maximum transparency, debugging control, or custom orchestration.

## What you will build

1. Split a DataFrame into train/test using `TrainTestSplitter`
2. Apply a few preprocessing steps (imputation + encoding)
3. Train a model and generate predictions

## Runnable example

This single snippet is fully self-contained and can be pasted into a Python session as-is.

```python
from __future__ import annotations

import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from skyulf.preprocessing.encoding import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.imputation import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.split import (
    FeatureTargetSplitApplier,
    FeatureTargetSplitCalculator,
    SplitApplier,
    SplitCalculator,
)

# 0) Setup data
df = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, None, 70, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

# 1) Split the dataset (no leakage)
split_params = {
    "test_size": 0.25,
    "validation_size": 0.0,
    "random_state": 42,
    "shuffle": True,
    "stratify": True,
    "target_column": "target",
}

dataset = SplitApplier().apply(df, SplitCalculator().fit(df, split_params))

# 2) Convert DataFrames to (X, y) tuples
fts_params = {"target_column": "target"}
dataset_xy = FeatureTargetSplitApplier().apply(
    dataset, FeatureTargetSplitCalculator().fit(dataset, fts_params)
)

X_train, y_train = dataset_xy.train
X_test, y_test = dataset_xy.test

# 3) Fit preprocessing on train, apply to test
imp_cfg = {"columns": ["age"], "strategy": "mean"}
imp_params = SimpleImputerCalculator().fit((X_train, y_train), imp_cfg)
X_train_imp, y_train = SimpleImputerApplier().apply((X_train, y_train), imp_params)
X_test_imp, y_test = SimpleImputerApplier().apply((X_test, y_test), imp_params)

ohe_cfg = {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"}
ohe_params = OneHotEncoderCalculator().fit((X_train_imp, y_train), ohe_cfg)
X_train_fe, y_train = OneHotEncoderApplier().apply((X_train_imp, y_train), ohe_params)
X_test_fe, y_test = OneHotEncoderApplier().apply((X_test_imp, y_test), ohe_params)

# 4) Train a model and predict
estimator = StatefulEstimator(
    calculator=RandomForestClassifierCalculator(),
    applier=RandomForestClassifierApplier(),
    node_id="rf_model",
)

dataset_for_model = SplitDataset(
    train=(X_train_fe, y_train),
    test=(X_test_fe, y_test),
    validation=None,
)

model_cfg = {"params": {"n_estimators": 50, "random_state": 42}}
preds = estimator.fit_predict(dataset=dataset_for_model, target_column="target", config=model_cfg)

print("Train preds:")
print(preds["train"].head())
print("Test preds:")
print(preds.get("test", pd.Series(dtype=float)).head())
```

## 5) Summary: what to remember

- Split first (or provide a `SplitDataset`) to prevent leakage.
- Fit preprocessing on train, reuse the learned `params` for test/inference.
- Modeling can be driven directly through `StatefulEstimator` when you don’t want a pipeline config.
