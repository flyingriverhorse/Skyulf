# Standalone inference bundles

The 0.9.0 development API packages a fitted feature pipeline and a Python
estimator with an explicit inference contract. It supports local pandas/Polars
prediction and native Spark FE followed by Python regression on Spark workers.
Importing `skyulf.inference` requires neither PySpark nor MLflow.

Start with [How inference works](inference_flow.md) for training and inference
diagrams and a worked example of reusing the same fitted FE and model.

## Raw data and prepared features

Choose the input stage when building a bundle:

| `input_stage` | Expected input | Execution |
| --- | --- | --- |
| `raw` | Original feature columns | Apply saved FE once, then predict |
| `features` | Already transformed model features | Predict directly |

Both bundles contain the same frozen FE state and estimator. Their declared
stages differ. A stage is an explicit caller contract: numeric values alone
cannot reveal whether a column has already been scaled. Keep raw and prepared
data on the corresponding path.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference import build_bundle, load_bundle, predict_local, save_bundle
from skyulf.pipeline import SkyulfPipeline

training = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "target": [10.0, 20.0, 30.0]})
pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "fill", "transformer": "SimpleImputer",
         "params": {"columns": ["amount"], "strategy": "mean"}},
        {"name": "scale", "transformer": "StandardScaler",
         "params": {"columns": ["amount"]}},
    ],
    "modeling": {"type": "linear_regression"},
})
pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")

raw = pd.DataFrame({"amount": [4.0, np.nan]})
bundle = build_bundle(pipeline, input_stage="raw", feature_order=("amount",))
with TemporaryDirectory() as temporary:
    destination = Path(temporary) / "model"
    save_bundle(bundle, destination)
    restored = load_bundle(destination)
    output = predict_local(raw, restored)
    np.testing.assert_allclose(output["prediction"], [40.0, 20.0], atol=1e-12)

prepared = pipeline.feature_engineer.transform(raw)
features_bundle = build_bundle(pipeline, input_stage="features", feature_order=("amount",))
np.testing.assert_allclose(predict_local(prepared, features_bundle)["prediction"],
                           output["prediction"], atol=1e-12)
```

The repository also includes `skyulf-core/examples/inference_bundle.py`. Run
it from the root with `.venv/Scripts/python.exe skyulf-core/examples/inference_bundle.py`.
Add `--bundle-dir path/to/new-model` to retain the artifact. An existing
destination is rejected, so an earlier package cannot be overwritten accidentally.

## Input and output contracts

A successful `SkyulfPipeline.fit` records two schemas: original training
features with the target excluded, and the actual features supplied to model
training. They contain names and dtypes only, never sample rows. A failed
replacement fit clears this metadata along with the fitted model.

`feature_order` must exactly match the recorded model order. `predict_local`
rejects missing, extra, duplicate or reordered columns and incompatible dtypes.
It does not sort columns or silently cast them. Equivalent pandas/Polars numeric
and boolean dtype labels are normalized; an integer feature is still distinct
from a floating-point feature. Select/cast columns explicitly before calling
when the source schema differs intentionally.

Supply feature columns only. Extra targets or identity columns are rejected.
The returned pandas DataFrame preserves a pandas input index; Polars input
receives a RangeIndex. The Spark runner carries explicit `FrameSpec` keys
separately from model features. Calling `predict_local` with a Spark frame
fails before any collection or local conversion.

Regression returns a float64 `prediction` column. Supported classifiers return
`prediction` and `probability_0`, `probability_1`, etc. `bundle.classes` records
the original labels in exactly that probability order; labels are not converted
into column names. Binary `positive_label` is the second estimator class, matching
the existing probability/threshold convention. Empty inputs return the declared
empty output schema.

Tuning's saved decision thresholds remain active by default, as in the existing
tuning applier. Thresholds saved by `pipeline.optimize_thresholds(...)` remain
opt-in: pass `use_tuned_thresholds=True` to `build_bundle`. The manifest retains
both saved sets and the selected source; opting in without saved pipeline
thresholds fails. Probabilities remain unchanged by threshold decisions.

## Package contents and loading

Each new directory contains three fixed-name files:

| File | Contents |
| --- | --- |
| `manifest.json` | Version, stage, ordered schemas, class/threshold metadata, runtime versions and digests |
| `features.json` | The supported fitted FeatureEngineer state |
| `model.pkl` | The frozen fitted sklearn estimator |

The bundle holds immutable metadata and bytes. Later mutation of the training
pipeline does not alter it. Building or predicting never fits a transformer or
estimator. The model payload excludes the pipeline, training reports, runtime
sessions and registry clients. Runtime requirements are a fixed list of package
versions, not a copy of environment variables or connection configuration.
An estimator can retain training observations as part of its own fitted state;
K-neighbors models are one example. Bundling preserves that state and its size
is subject to the model byte limit.

**Only load bundles from trusted producers.** The estimator uses pickle, which
can execute Python during deserialization. Checksums detect corruption; they
are not signatures and do not establish trust. Size, structural, payload-checksum
and runtime checks run before the pickle loader. Model class, feature width,
classes and fitted semantic digest are checked after deserialization.

The semantic digest covers the inference contract and learned state. Exact
payload checksums also protect transported bytes, but changing only the pickle
protocol does not change semantic identity. Neither digest promises portability
across arbitrary library versions.

Loading requires the same Python major/minor and exact recorded `skyulf-core`,
scikit-learn, NumPy and SciPy versions. pandas/Polars versions are recorded for
provenance; the input schema remains the execution check. There is no automatic
dependency installation. Prepare a matching environment before loading.

The optional `options=ExecutionOptions(...)` argument to build/save/load/predict
sets the existing budgets: by default 8 MiB for manifest plus FE bytes and 256 MiB
for estimator bytes. Oversized file reads and pickle serialization are bounded.
These are wire-size checks, not total process-memory guarantees. A wider input
batch or a loaded estimator can use more memory than its serialized form.

## Native Spark FE and worker model inference

`predict_spark(..., mode="native_features")` accepts a raw-input regression
bundle and a Spark DataFrame. The fitted FE runs as native Spark expressions.
Only the prepared feature columns and row keys reach the Python workers;
the estimator receives the features in the saved training order. Pandas/NumPy
batches exist inside those workers, without collecting the dataset to the driver.
Local training describes where the fit runs; it does not waive the matching
runtime requirements above. Train/package and consume in compatible environments,
even when switching from pandas or Polars inputs to Spark inputs.

This example continues the local training example above. `spark` is an existing,
caller-owned SparkSession; the runner neither creates nor closes it:

```python
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.inference import predict_spark

incoming = spark.createDataFrame(
    [(101, 4.0, "unused"), (102, None, "unused")],
    "id long, amount double, extra string",
).repartition(2)

predictions = predict_spark(
    incoming,
    bundle,
    frame_spec=FrameSpec(row_keys=("id",)),
    options=ExecutionOptions("spark", python_batch_rows=2),
    mode="native_features",
)
# This small example materializes two output rows; the runner does not collect them.
rows = predictions.orderBy("id").collect()
assert [row.id for row in rows] == [101, 102]
np.testing.assert_allclose([row.prediction for row in rows], [40.0, 20.0], atol=1e-12)
```

Production consumers receive a Spark DataFrame and can select their own action
or output sink. Output rows have no guaranteed physical order; match them by
the declared unique, non-null keys. Composite integer/string/boolean keys are
supported; floating-point and date/time keys are rejected by this first runner.
Keys must not also be model features. Incoming column names must not collide
with prediction output names. Leave `FrameSpec.target` unset. Unused input
columns are projected away, while the relative order and dtypes of required
feature columns are validated rather than silently corrected.

Native FE output must also match the recorded model-feature dtypes before any
validation action runs. Some local/Spark operations promote numeric types
differently: a local integer column without missing values can remain integer
after mean imputation, while Spark's replacement expression produces double.
That bundle is rejected instead of casting silently. Normalize numeric types
explicitly before training and use the same input types for inference when a
cross-engine pipeline requires floating-point features.

For this first worker runner, integral/boolean **model features** must have a
non-nullable Spark schema. Arrow can convert a nullable integer batch to pandas
float when it contains nulls, losing precision for large integers before the
model sees it. Nullable integral/boolean feature schemas are rejected eagerly,
even if their present rows happen to contain no nulls. Floating-point features
can retain null/NaN behavior. Row keys have a separate distributed non-null
check and can retain nullable schema metadata. Broader nullable feature transport
is part of the later worker validation gate.

The model is loaded once per worker iterator invocation and reused over its
prediction chunks. Task retries or later actions may load it again. The worker
receives model bytes and inference metadata, without a SparkSession, registry
client or tracking connection. Prepare matching dependencies on driver and
workers; the runtime contract is checked in both places.

`python_batch_rows` limits the rows passed to an individual model prediction.
Spark's `spark.sql.execution.arrow.maxRecordsPerBatch` separately controls Arrow
transport batches. Set that Spark configuration on your session when needed;
the runner does not change it. Neither row limit is a byte-memory guarantee.
Input-key and FE preservation checks can execute bounded validation actions
before prediction; prediction itself remains a lazy Spark computation.
Validation does not freeze a changing source. The caller owns any snapshot or
persist policy needed to keep validation and later actions on the same input.

The distributed runners accept `input_stage="raw"` and regression only.
Prepared-feature bundles, classification and streaming remain unsupported.
The Python FE worker mode is available for the portable row-independent path;
local classification remains available. Databricks,
Spark Connect and worker-wheel isolation still require their later validation
gates; local PySpark tests do not establish those deployment guarantees.

## Current support and legacy adapters

The first version accepts a successfully fitted standalone `SkyulfPipeline`
with recorded schemas, primitive numeric/boolean features, supported sklearn
regression/classification behavior and portable FE. Currently that FE is
SimpleImputer mean/constant and StandardScaler, including an empty chain.
Classifiers must expose class probabilities. Custom appliers and unsupported
FE nodes require a separate adapter; they are not serialized as an opaque
whole pipeline. Training-only split nodes in the FE chain are therefore not
portable yet; an explicit `SplitDataset` can supply training/test partitions.

Saving fitted transformations alongside the model already exists in Skyulf.
The new bundle adds an explicit execution contract for local and distributed
consumers; it does not relearn or replace those fitted transformations.

| Existing artifact | Saved information and current consumer | New bundle bridge |
| --- | --- | --- |
| Standalone `SkyulfPipeline.save()` pickle | The pipeline object, fitted FE and model; `SkyulfPipeline.load(...).predict(...)` | Load with the existing API, then call `build_bundle` if recorded schemas and supported FE are present |
| Backend job artifact, normally `.joblib` | Model, fitted FeatureEngineer, target/drop metadata, model feature order/dtypes and training engine; `DeploymentService` | Explicit adapter planned for SM-18; do not pass the dictionary to `build_bundle` |
| Estimator-only pickle | Whatever the producer saved in the estimator | External preprocessing is not recovered automatically; a fitted pipeline and its input contract are needed |

The backend serving path already transforms raw input and aligns the result to
the saved training feature order. Some serving choices, including enabled tuned
thresholds and request overrides, are resolved outside that artifact. Its adapter
must preserve that behavior and target-label decoding as well as the fitted model.

Existing `SkyulfPipeline.save/load` pickle behavior is unchanged. A standalone
pipeline saved after schema capture can be loaded through that API and passed
to `build_bundle`. Historical standalone artifacts without captured schemas
must be fitted successfully again; the bundle does not invent missing column
names or types. Backend artifact dictionaries are rejected. Their explicit
adapter and threshold/label compatibility checks belong to SM-18.

Changing the inference engine does not make every fitted Python transformer a
native Spark operation. For example, a future binning adapter must use the saved
training bin boundaries; it must not discover new boundaries from each inference
batch. The native path requires an implemented Spark applier and a supported
state codec. The Python-worker path requires explicit support for independent
batches and row preservation. Rolling/lag and transformations that
need neighboring rows cannot be made correct merely by putting them inside a
worker batch. Unsupported combinations fail explicitly; there is no automatic
conversion of a whole Spark dataset to local pandas or Polars.

MLflow is a separate packaging/integration layer. Logging only an estimator
does not include external feature engineering; log a fitted pipeline or provide
a model wrapper that runs it. MLflow's
[Spark UDF](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)
executes Python model inference through pandas batches; it does not compile
arbitrary Python transformations into native Spark expressions. Skyulf's MLflow
adapter will expose the selected bundle contract in a later stage.

MLflow/Unity Catalog, Databricks runtime validation, endpoints and template
generation remain later stages of the [Spark work](spark.md).
