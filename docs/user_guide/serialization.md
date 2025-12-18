# Serialization

## What is persisted

`SkyulfPipeline.save()` uses Python `pickle` to serialize the entire pipeline object.

That includes:

- preprocessing fitted artifacts (per-step `params`)
- the trained model (sklearn estimator object)

## Practical guidance

- Prefer saving in environments where the same library versions are available.
- Some preprocessing nodes store sklearn objects inside `params` (e.g., KNN/Iterative imputers, OneHotEncoder).
  Those are not JSON-serializable and require pickling.

## Load and use

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

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
  new_df = pd.DataFrame({"age": [25, None], "city": ["A", "C"]})
  preds = loaded.predict(new_df)

print(preds)
```
