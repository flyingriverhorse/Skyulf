# SplitDataset & Leakage

## Why `SplitDataset` exists

Many preprocessing nodes learn statistics from data (means, categories, bin edges, …).
If those statistics are computed on the full dataset and then evaluated on test data, you leak information.

`SplitDataset` is a container for:

- `train`
- `test`
- optional `validation`

Each split can be either:

- a `pd.DataFrame` (with the target column inside),
- a `pl.DataFrame` (Polars; same shape), or
- a tuple `(X: frame, y: Series)` on either engine.

A `SplitDataset` handed to `fit()` is treated as the trust boundary: the gate assumes the
caller already split safely, so it does not require a `TrainTestSplitter` in the config.

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

### Branches in backend pipelines

When the backend executes a graph with multiple training branches, each leaf
`training` or `tuning` node must have a `TrainTestSplitter` on its own input
path. A splitter on a different branch does not protect it.

For example:

```text
DataLoader → TrainTestSplitter → StandardScaler → training_A  (protected)
           └→ StandardScaler ─────────────────→ training_B  (warning/error)
```

`StandardScaler` learns its mean and standard deviation from the rows it
receives. In the second branch it sees the full dataset, so the backend leakage
gate reports the branch before training. With the default `on_leakage="raise"`
mode, execution stops before any node fits. `on_leakage="warn"` records the
warning and continues.

The submission API checks the graph before partitioning it into jobs. A
violation returns HTTP 400 before any job is created. When a single training
node is selected, only its upstream path is checked, while the full graph's
split/no-split context is retained.

If the training node explicitly uses internal validation and its unsplit
preprocessing path is linear, the branch is also accepted: the engine can
refit that chain inside each fold. Set `cv_enabled: true` for a fixed training
run or use `run_mode: "tuned"` for a tuning run. CV alone does not protect
unsplit merged preprocessing paths; add a splitter before the learning nodes
or use a supported linear path. A graph with no train/test splitter at all
receives a `no_split` advisory because the backend cannot establish a
train/test guarantee from that graph.

### Standalone core: fail before fitting unsafe preprocessing

`SkyulfPipeline.fit()` and `get_fitted_split()` reject learned preprocessing
before a configured train/test splitter by default, before fitting any step.
This protection does not require the backend or frontend:

```python
import pandas as pd

from skyulf import SkyulfPipeline

data = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
leaking_pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        {"name": "split", "transformer": "TrainTestSplitter", "params": {"test_size": 0.2}},
    ],
    "modeling": {},
})
try:
    leaking_pipeline.fit(data, target_column="target")
except ValueError as error:
    print(error)  # Expected: scaler learns from test rows before the split.

# Correct fix: put the splitter before the scaler.
# Explicit compatibility escape hatch, not a leakage-safe fix:
leaking_pipeline.fit(data, target_column="target", on_leakage="warn")
```

`leaking_pipeline` is an ordinary variable containing a `SkyulfPipeline`
instance. Its name highlights this example's intentionally wrong step order;
it is not a special API, class, or execution mode.

| Mode | Config-only diagnostic for a definite violation | `fit()` / `get_fitted_split()` |
|------|--------------------------------------------------|--------------------------------|
| `raise` (default) | Raises `ValueError` | Stops before any preprocessing fit |
| `warn` | Returns warning strings without fitting | Logs warnings and continues |
| `ignore` | Returns an empty list | Continues without leakage warnings |

Only these three mode strings are accepted; other values raise `ValueError`.
`warn` and `ignore` do not correct an unsafe order. Registered stateless nodes
are not definite violations: fixed per-row rules do not fit population
statistics. For example, constant imputation uses the configured value and is
allowed before splitting, whereas mean imputation learns a value from the rows
and must follow the split.

For a self-contained, numbered walkthrough of all modes, safe training,
prediction, an externally supplied split, and a fresh pipeline per CV fold,
open `skyulf-core/examples/09_leakage_safety.ipynb` in the repository.
The companion `skyulf-core/examples/09_leakage_safety.py` offers a shorter
terminal-only walkthrough.

`on_leakage="ignore"` also permits the unsafe fit without a warning. Pipelines
with no configured splitter still receive an advisory rather than an error;
the diagnostic cannot guarantee leakage safety for an unsplit dataset.
Passing an existing `SplitDataset` supplies the boundary externally, so each
transformer learns from its training partition without a false no-split warning.

Registered stateless nodes are allowed before splitting. Parameter-dependent
exceptions are shared with the backend, including constant imputation, explicit
column dropping/hashing/missingness indicators, and target-only label encoding.
Feature encoders and unknown transformers remain subject to the guard. Pass
`target_column` to `validate_leakage_safety()` when the config does not identify
the target through a splitter.

The shared JSON fixture at
`skyulf-core/tests/test_cases/leakage/registry_nodes.json` drives the core and
API integration matrices. Registry coverage assertions require new node types
to be classified explicitly. These cases test admission/rejection for every
registered node and model, not a full training run of every algorithm. A separate
numerical case checks that held-out rows do not influence fitted scaler statistics.

### Pattern B: create `SplitDataset` yourself

```python
import pandas as pd
from sklearn.model_selection import train_test_split

from skyulf.data.dataset import SplitDataset
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

## Detailed execution and operation guides

See [Leakage Safety: Core & Backend](leakage_core_backend.md) for runnable core
examples, native tuning, supported backend graphs, and editable Mermaid diagrams.
The [preprocessing leakage audit](preprocessing_leakage_audit.md) documents all
62 preprocessing registrations and their parameter-dependent behavior.
