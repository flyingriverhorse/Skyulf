# Pipeline quickstart

Train a classifier, read its held-out accuracy, predict new raw rows and verify
that saving/reloading preserves the predictions. Everything runs in one Python
process; no API server, Canvas, Spark session or MLflow service is needed.

## Install and prepare the input

Use Python 3.12 or newer in an activated virtual environment:

```bash
uv venv --python 3.12
# Activate .venv using your shell, then install into that environment.
uv pip install skyulf-core
```

For a repository checkout, replace the last command with
`uv pip install -e ./skyulf-core`. The example uses pandas, NumPy and
scikit-learn supplied by the core installation. Preserve their versions when
moving a saved artifact between environments.

| Column | Training type | Meaning | New prediction rows |
| --- | --- | --- | --- |
| `age` | Floating point, nullable | Numeric feature | Include, even if missing |
| `city` | String | Categorical feature | Include; unseen values are allowed by this recipe |
| `target` | Integer 0 or 1 | Class label | Omit |

Split before learning the imputation mean or category vocabulary. Otherwise
holdout observations can influence training features and make evaluation
optimistic. The explicit first split below learns subsequent steps from six
training rows, then applies the saved state to two held-out rows.

## What data shape should I pass?

`SkyulfPipeline.fit(...)` supports two common inputs:

1. A single `pd.DataFrame` that **includes the target column**.
    - You pass `target_column="..."` and Skyulf splits features/target internally.
    - This is what the quickstart uses, because it's the simplest onboarding path.

2. A `SplitDataset` (recommended when you already have train/test splits).
    - Each split can be either a DataFrame (with the target column) **or** a tuple `(X, y)`.

You do not need to manually build `X` and `y` for the pipeline unless you want to.

If you prefer a scikit-learn-style workflow (`X`/`y` + `train_test_split`), see:

- [Validation vs scikit-learn (Proof)](validation_vs_sklearn.md)

## Complete runnable example

This single snippet is intentionally end-to-end (no duplicated setup across steps).

<!-- skyulf-example: native quickstart -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from skyulf import SkyulfPipeline

config = {
    "preprocessing": [
        {"name": "split", "transformer": "TrainTestSplitter", "params": {
            "test_size": 0.25, "validation_size": 0.0,
            "random_state": 42, "shuffle": True, "stratify": True,
            "target_column": "target",
        }},
        {"name": "impute", "transformer": "SimpleImputer",
         "params": {"strategy": "mean", "columns": ["age"]}},
        {"name": "encode", "transformer": "OneHotEncoder", "params": {
            "columns": ["city"], "drop_original": True,
            "handle_unknown": "ignore", "max_categories": None,
        }},
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 50, "random_state": 42},
    },
}
training = pd.DataFrame({
    "age": [10.0, 20.0, None, 40.0, 50.0, 60.0, None, 80.0],
    "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "target": [0, 1, 0, 1, 1, 0, 1, 0],
})
pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(training, target_column="target")

# This is the held-out report, rather than training accuracy or metric names.
test_report = metrics["modeling"]["splits"]["test"]
accuracy = test_report.metrics["accuracy"]
assert 0.0 <= accuracy <= 1.0
fill_values = metrics["preprocessing"]["steps"]["1:impute"]["details"]["fill_values"]
assert fill_values == {"age": 37.5}
print("Held-out accuracy:", accuracy)
print("Training fill values:", fill_values)

# Pass raw features: predict applies the fitted imputer and encoder once.
incoming = pd.DataFrame({"age": [25.0, None], "city": ["A", "NEW"]})
predictions = pipeline.predict(incoming)
assert len(predictions) == len(incoming)
assert set(predictions).issubset({0, 1})
print("Predictions:", predictions.tolist())

with TemporaryDirectory() as temporary:
    model_path = Path(temporary) / "model.pkl"
    pipeline.save(model_path)
    restored = SkyulfPipeline.load(model_path)
    np.testing.assert_array_equal(predictions, restored.predict(incoming))
print("Reload predictions match")
```

The example produces a held-out accuracy of `0.5`, fill values
`{'age': 37.5}`, predictions `[0, 0]` and `Reload predictions match` with the
shown recipe. A two-row holdout demonstrates the API; it does not establish a
useful model. Evaluate representative data before using predictions for decisions.

The missing age uses the saved training mean, 37.5. The unseen city `NEW` receives
zeros in all learned city indicators because `handle_unknown="ignore"` is set;
it does not create a new feature or refit the vocabulary. Increasing the training
sample or changing package versions can change predictions and metrics.

## Change the recipe deliberately

| Setting | Effect |
| --- | --- |
| Split `test_size`, `validation_size`, `random_state` | Choose evaluation populations and reproducibility; stratification needs enough rows per class |
| Imputer `columns`, `strategy` | Choose which missing values are filled and what is learned on training data |
| Encoder `handle_unknown`, `max_categories` | Control new-category behavior and grouping; `None` disables category grouping here |
| Model `type`, `params` | Select the registered estimator and its parameters |

If you provide an existing `SplitDataset`, omit the splitter from this recipe.
Keep original feature names, order and compatible types for prediction. Do not
pass already encoded/scaled features back into `pipeline.predict`: they would
be transformed again.

A `modeling_error` entry means evaluation failed even if fitting completed;
inspect it before treating the run as evaluated. Missing input columns, a
non-numeric unencoded feature or insufficient rows for stratification need input
or configuration correction. A split with no usable held-out rows has no usable
held-out accuracy. Only load pickle artifacts from trusted producers; they can
execute Python when loaded. The temporary file above is removed after the check.

## Inspecting a pipeline

### Evaluation population after filtering

Explicit row filters, such as dropping missing rows, select the eligible
population for both ordinary evaluation and tuning/CV. Fitted cleaning bounds
come from training data. Serving preserves the requested row alignment.

Held-out reports record `input_rows`, `scored_rows`, and `excluded_rows`.
The denominator begins when that test or validation split exists. For example,
filtering 12 rows out of a 120-row test split reports 120 input, 108 scored,
and 12 excluded. Filtering before splitting changes the source population;
those earlier exclusions are not counted again as test exclusions. Training
coverage describes the final training frame, not all rows loaded upstream.
Consequently, the displayed metrics describe eligible observations and should
not be interpreted as accuracy on the original unfiltered dataset.

A split with no eligible rows has no evaluation metrics and an explicit
unavailable reason. Missing original counts remain unknown rather than zero.
In Canvas, merging nonempty filtered evaluation branches without shared row
lineage is rejected: equal counts do not establish that both branches contain
the same rows. Merge branches before applying row filters. All-empty branches
retain their coverage and remain visible in the report.
This also rejects branches that inherit the same earlier filter: the recorded
counts do not establish shared row identity, even when both branches happen
to contain identical rows.

### Pipeline description and diagram

A fitted (or unfitted) pipeline is fully introspectable — useful for logs,
docs, PR descriptions, and model registries:

<!-- skyulf-example: native quickstart -->
```python
print(pipeline.describe())
# Human-readable summary: each preprocessing step with its params, then the model.

print(pipeline.to_mermaid())
# Mermaid flowchart of the topology (data -> steps -> model).
# Paste into GitHub docs/PRs, or render with any mermaid viewer.

card = pipeline.export_model_card()
# JSON-friendly dict: schema_version, fitted, fingerprint, preprocessing
# lineage, model + params, fit metrics, and `diagram` (the same mermaid
# string as to_mermaid()). Intended for audit logs and registries.
```

The same surfaces appear across the platform:

- **Experiments page** — every completed run persists its mermaid topology
  (`metrics.pipeline_diagram`); select runs and open the **Pipeline Diagram** tab.
- **Notebook export** — exported notebooks embed the topology as a rendered
  "Pipeline topology" markdown cell (compact and full modes).

Diagram labels are human-readable — internal node ids are never shown. Each
node gets the step's display name, and a second line with the most relevant
detail: a digest of the step params for config diagrams (`strategy: mean`),
or the run's one-line summary for completed jobs (`acc 0.87 · f1 0.84`).
Model nodes show the algorithm in readable form (`Random Forest Classifier`).

Mermaid labels are always emitted double-quoted, so node names containing
parentheses, brackets, or quotes stay parseable.

## Continue from here

Use [step-by-step without config](step_by_step_no_config.md) to inspect each
learned state, [preprocessing recipes](preprocessing_recipes.md) to select other
registered transformations, and [inference artifacts](inference_bundles.md) for
explicit saved schemas and portable execution. [Preprocessing placement](preprocessing_placement.md)
explains leakage and filter placement.

This example is the standalone library. Canvas runs through its backend job and
artifact services; its completed-run reports and inference page use those
services rather than this temporary local pickle.
