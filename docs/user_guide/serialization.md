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
import tempfile
from pathlib import Path

import pandas as pd

from skyulf import SkyulfPipeline

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

## Reproducibility fingerprint

`pipeline.fingerprint()` returns a deterministic SHA-256 over the pipeline's
topology **and** its fitted artifacts:

```python
print(pipeline.fingerprint())  # 64-char hex, e.g. "9f2c4e..."
```

- Two pipelines with the same fingerprint produce the same predictions —
  callers can prove "this prediction came from exactly this pipeline".
- The digest is **semantic** (`skyulf.pipeline.seal.artifact_digest` walks
  hyperparameters + fitted weights, tree structures, tuned-model tuples,
  numpy arrays and RNG state), not pickle bytes — so it is stable across
  library, platform, and pickle-protocol versions.
- Objects without a canonical representation raise `TypeError` (fail-loud)
  instead of silently hashing a `repr`.

The fingerprint is also part of `export_model_card()`, alongside the
preprocessing lineage, model params, fit metrics, and a Mermaid `diagram`.

## Security note

`pickle.load` (and `joblib.load`, which is pickle under the hood) can execute
arbitrary code from a malicious file. Treat pipeline files like executables:

- Only load artifacts you produced yourself or received from a trusted store.
- The platform's artifact store (local/S3) only ever loads artifacts it wrote
  for its own jobs.
- Prefer `fingerprint()` comparisons over re-loading when you only need to
  verify identity.
