# Getting Started (skyulf-core)

This page is the fastest path to running `skyulf-core` locally.

## Install

From the repository root:

```bash
pip install -e ./skyulf-core
```

## Minimal end-to-end example

`SkyulfPipeline` expects a configuration with:

- `preprocessing`: a list of steps
- `modeling`: a single model config

```python
import pandas as pd

from skyulf.pipeline import SkyulfPipeline

df = pd.DataFrame(
    {
        "age": [10, 20, None, 40],
        "city": ["A", "B", "A", "C"],
        "target": [0, 1, 0, 1],
    }
)

config = {
    "preprocessing": [
        {
            "name": "impute_age",
            "transformer": "SimpleImputer",
            "params": {"columns": ["age"], "strategy": "mean"},
        },
        {
            "name": "encode_city",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["city"], "drop_original": True},
        },
    ],
    "modeling": {
        "type": "logistic_regression",
        "params": {"max_iter": 1000},
    },
}

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(df, target_column="target")
preds = pipeline.predict(df.drop(columns=["target"]))

print(metrics)
print(preds.head())
```

## Next steps

- Read the User Guide section “Pipeline Quickstart” for train/test splits.
- Use the Reference section for supported preprocessing and modeling nodes.
