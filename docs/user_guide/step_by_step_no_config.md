# Step-by-step without a pipeline config

Use Calculator/Applier pairs to see exactly what training learns and what
inference reuses. This example splits eight labeled rows, fills missing ages,
encodes cities, fits a classifier, evaluates two held-out rows and predicts two
new raw rows using the same saved state. No server, Canvas or cloud is required.

A **Calculator** fits on training input and returns learned state. An **Applier**
uses that state on training, test and new input. Do not call a Calculator on new
prediction rows: their statistics and vocabulary must not replace training state.

## Install

Use Python 3.12 or newer and an activated virtual environment:

```bash
uv venv --python 3.12
# Activate .venv using your shell.
uv pip install skyulf-core
```

When working in a repository checkout, use `uv pip install -e ./skyulf-core`
instead. The core package supplies pandas, NumPy and scikit-learn dependencies.
The [pipeline quickstart](pipeline_quickstart.md) implements the same data flow
with a configuration dictionary and whole-pipeline save/load.

## 1. Create data and separate training from test

Run the following blocks in order in the same Python session. `age` is a nullable
floating-point feature, `city` a string feature, and `target` a 0/1 class label.
New prediction rows will contain only `age` and `city`.

<!-- skyulf-example: native manual -->
```python
import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import (
    RandomForestClassifierApplier,
    RandomForestClassifierCalculator,
)
from skyulf.preprocessing.encoding import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.imputation import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.split import SplitApplier, SplitCalculator

training = pd.DataFrame({
    "age": [10.0, 20.0, None, 40.0, 50.0, 60.0, None, 80.0],
    "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
    "target": [0, 1, 0, 1, 1, 0, 1, 0],
})
split_state = SplitCalculator().fit(training, {
    "test_size": 0.25, "validation_size": 0.0,
    "random_state": 42, "shuffle": True, "stratify": True,
    "target_column": "target",
})
dataset = SplitApplier().apply(training, split_state)
# With target_column configured, the splitter already returns (X, y) partitions.
X_train, y_train = dataset.train
X_test, y_test = dataset.test
assert (len(X_train), len(X_test)) == (6, 2)
assert X_train.columns.tolist() == ["age", "city"]
assert "target" not in X_train
print("Train/test rows:", len(X_train), len(X_test))
```

The split state records splitting settings, not an imputation mean or model.
The result is a `SplitDataset`; training has six feature rows plus six labels,
and test has two of each. `validation_size=0.0` leaves no validation population.
`FeatureTargetSplitCalculator`/`FeatureTargetSplitApplier` are available when you
need to separate a target from other unsplit frames; already separated tuples
pass through that helper unchanged.

## 2. Learn one fill value and reuse it

<!-- skyulf-example: native manual -->
```python
imputer_state = SimpleImputerCalculator().fit(
    X_train, {"columns": ["age"], "strategy": "mean"},
)
train_imputed = SimpleImputerApplier().apply(X_train, imputer_state)
test_imputed = SimpleImputerApplier().apply(X_test, imputer_state)
assert imputer_state["fill_values"] == {"age": 37.5}
assert not train_imputed["age"].isna().any()
assert not test_imputed["age"].isna().any()
print("Saved fill value:", imputer_state["fill_values"])
```

The mean, 37.5, uses only observed training ages. `train_imputed` and
`test_imputed` keep their rows, labels and city values aligned. Neither apply
call recalculates the mean. Other local strategies are available, but their
support for a distributed adapter is a separate question.

## 3. Learn the training vocabulary and create model features

<!-- skyulf-example: native manual -->
```python
encoder_state = OneHotEncoderCalculator().fit(train_imputed, {
    "columns": ["city"], "drop_original": True,
    "handle_unknown": "ignore", "max_categories": None,
})
train_features = OneHotEncoderApplier().apply(train_imputed, encoder_state)
test_features = OneHotEncoderApplier().apply(test_imputed, encoder_state)
feature_names = ["age", "city_A", "city_B", "city_C"]
assert train_features.columns.tolist() == feature_names
assert test_features.columns.tolist() == feature_names
print("Model columns:", train_features.columns.tolist())
```

The saved encoder contains the learned category vocabulary and generated feature
names. The original string column is removed; each city becomes a numeric
indicator. `max_categories=None` disables infrequent-category grouping for this
example. `handle_unknown="ignore"` will encode an unseen city with zero in all
three learned city indicators.

## 4. Fit the model and inspect held-out accuracy

<!-- skyulf-example: native manual -->
```python
estimator = StatefulEstimator(
    calculator=RandomForestClassifierCalculator(),
    applier=RandomForestClassifierApplier(),
    node_id="example_forest",
)
model_data = SplitDataset(
    train=(train_features, y_train),
    test=(test_features, y_test),
    validation=None,
)
predictions = estimator.fit_predict(
    model_data, target_column="target",
    config={"params": {"n_estimators": 50, "random_state": 42}},
)
report = estimator.evaluate(model_data, target_column="target")
accuracy = report["splits"]["test"].metrics["accuracy"]
assert len(predictions["train"]) == 6
assert len(predictions["test"]) == 2
assert 0.0 <= accuracy <= 1.0
print("Held-out accuracy:", accuracy)
```

`estimator.model` is the fitted model artifact. `fit_predict` returns a dictionary
of predictions by populated split; it does not return the preprocessing state.
The example's held-out accuracy is 0.5. Two test rows illustrate access to the
report, and provide little evidence about accuracy on a real population.

## 5. Predict fresh rows with those same artifacts

<!-- skyulf-example: native manual -->
```python
incoming = pd.DataFrame({"age": [25.0, None], "city": ["A", "NEW"]})
new_imputed = SimpleImputerApplier().apply(incoming, imputer_state)
new_features = OneHotEncoderApplier().apply(new_imputed, encoder_state)
new_predictions = estimator.applier.predict(new_features, estimator.model)

assert new_imputed["age"].tolist() == [25.0, 37.5]
assert new_features.columns.tolist() == train_features.columns.tolist()
assert new_features.loc[1, ["city_A", "city_B", "city_C"]].tolist() == [0, 0, 0]
assert len(new_predictions) == len(incoming)
assert set(new_predictions).issubset({0, 1})
np.testing.assert_array_equal(new_predictions, [0, 0])
print(new_features.to_string(index=False))
print("New predictions:", new_predictions.tolist())
```

Expected prepared values:

| age | city_A | city_B | city_C |
| ---: | ---: | ---: | ---: |
| 25.0 | 1 | 0 | 0 |
| 37.5 | 0 | 0 | 0 |

Both predictions are 0 for this tiny deterministic recipe. Missing age uses the
training mean, and `NEW` creates no new column. The label column is unnecessary
for prediction. Calculator calls ended before evaluation and new-row prediction;
all those calls share `imputer_state`, `encoder_state` and `estimator.model`.

## Options, errors and persistence

Change the split settings before fitting; stratification requires enough rows
for every class in each split. When reordering or filtering data manually, keep
features and labels aligned. Select explicit preprocessing columns so the target
or record identifiers do not accidentally become model features.

Missing required columns, incompatible values or unencoded string model features
raise errors in the corresponding step. Inspect the saved state and prepared
frame at that boundary. A recipe that filters rows also needs an explicit policy
for matching predictions to requests; this example preserves every row.

These separate objects are an in-memory example, not a new serialization format.
For whole-artifact persistence, build the equivalent `SkyulfPipeline` and use
[its save/load example](pipeline_quickstart.md#complete-runnable-example), or the
[versioned fitted-pipeline format](inference_bundles.md#fitted-pipeline-or-portable-bundle).
Do not load pickle artifacts from an untrusted producer.

Use the [preprocessing recipe catalog](preprocessing_recipes.md) to choose other
steps, [placement guide](preprocessing_placement.md) to preserve the training
boundary, and [context diagnostics](preprocessing_context.md) when a saved step
may depend on other incoming rows. This manual pandas example grants no Spark
worker or endpoint admission; those paths have their own fitted-recipe checks.
