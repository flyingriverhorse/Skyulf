# Getting Started

The fastest path from zero to a working Skyulf pipeline.

## 1. Install

```bash
pip install skyulf-core
```

Or install with all optional extras:

```bash
pip install skyulf-core[viz,eda,tuning,modeling-xgboost,preprocessing-imbalanced]
```

> For editable installs and Docker, see [Installation](../user_guide/installation.md).

## 2. Minimal example

```python
import pandas as pd
from skyulf.pipeline import SkyulfPipeline

df = pd.DataFrame({
    "age": [10, 20, None, 40, 50, 60, None, 80],
    "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "target": [0, 1, 0, 1, 1, 0, 1, 0],
})

config = {
    "preprocessing": [
        {"name": "split", "transformer": "TrainTestSplitter",
         "params": {"test_size": 0.25, "random_state": 42,
                    "stratify": True, "target_column": "target"}},
        {"name": "impute", "transformer": "SimpleImputer",
         "params": {"columns": ["age"], "strategy": "mean"}},
        {"name": "encode", "transformer": "OneHotEncoder",
         "params": {"columns": ["city"], "drop_original": True}},
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 50, "random_state": 42},
    },
}

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(df, target_column="target")
print(metrics)

preds = pipeline.predict(df.drop(columns=["target"]))
print(preds.head())
```

## 3. What just happened?

1. **TrainTestSplitter** separated data into train/test sets (no leakage).
2. **SimpleImputer** learned the mean of `age` from training data only.
3. **OneHotEncoder** created dummy columns for `city`.
4. **RandomForestClassifier** trained on the processed training split.

## 4. Next steps

| Goal | Page |
|---|---|
| Full train / evaluate / save / load workflow | [Pipeline Quickstart](../user_guide/pipeline_quickstart.md) |
| All 20 supported models and config keys | [Configuration](../user_guide/configuration.md) |
| Hyperparameter tuning (grid, random, Optuna) | [Hyperparameter Tuning](../user_guide/hyperparameter_tuning.md) |
| Add your own custom nodes | [Extending Skyulf-Core](../user_guide/extending_custom_nodes.md) |
| Full platform setup (backend + UI) | [Architecture](../architecture.md) |