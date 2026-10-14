# Package and reload a model with MLflow

Package a fitted model once, load its `runs:/...` URI and verify that predictions
match direct Skyulf inference. Registration selects named model versions;
[registry aliases](mlflow_registry.md) and endpoint deployment are later operations.

## Choose an artifact

| You need | Package and prediction route |
| --- | --- |
| Existing fitted pandas/Polars preprocessing, including recipes outside portable support | `save_pipeline` then `log_pipeline_model`; direct `predict_pipeline` or MLflow pyfunc. Bounded whole-frame execution is the base contract. |
| A supported portable feature state and estimator, with a possible distributed Spark consumer | `build_bundle` then `log_model`; direct `predict_local`, pyfunc or the separately validated Spark routes. |
| Distributed execution of a fitted pandas package | Use only the separately inspected/certified Spark pyfunc route; logging alone does not certify a recipe. |

Install `uv pip install "skyulf-core[mlflow]"` in a Python 3.12+ environment.
Pin a published release and resolved dependencies for repeatable jobs. New
integration contracts shown here require the matching release/build; development
verification against a checkout does not prove the published package contains
them. For an unpublished build, distribute its exact wheel to consumers through
your package delivery process.

## Fitted pipelines (pandas or Polars)

Use a **fitted pipeline artifact** when fitted preprocessing contains nodes that
the portable Spark bundle does not yet support, such as categorical encoding or
binning. This path saves the whole fitted pipeline, its recorded training engine,
raw and model-feature column order and dtypes, dependency versions, and a payload
checksum. It never fits the pipeline again. It is separate from the portable
Spark-capable `InferenceBundle` described below.

<!-- skyulf-example: native models -->
```python
from pathlib import Path

import mlflow
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.integrations.mlflow import TrackingConfig, track_run
from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model
from skyulf.pipeline import SkyulfPipeline

frame = pd.DataFrame({
    "x": [1., 2., 3., 4., 5., 6., 7., 8.],
    "target": [2., 4., 6., 8., 10., 12., 14., 16.],
})
fitted_pipeline = SkyulfPipeline({
    "preprocessing": [], "modeling": {"type": "linear_regression"},
})
fitted_pipeline.fit(
    SplitDataset(train=frame.iloc[:6], test=frame.iloc[6:]), target_column="target"
)
save_pipeline(fitted_pipeline, "artifacts/fitted-pipeline")
tracking_uri = "sqlite:///" + (Path.cwd() / "mlflow.db").as_posix()
config = TrackingConfig(
    enabled=True, tracking_uri=tracking_uri, experiment_name="model-packaging",
)
with track_run(config, run_name="linear-model") as run:
    model_uri = log_pipeline_model(
        "artifacts/fitted-pipeline", run_id=run.run_id,
        artifact_path="model", tracking_uri=tracking_uri,
    )

request_frame = pd.DataFrame({"x": [9., 10.]})
artifact = load_pipeline("artifacts/fitted-pipeline")
direct_predictions = predict_pipeline(request_frame, artifact)
mlflow.set_tracking_uri(tracking_uri)
loaded = mlflow.pyfunc.load_model(model_uri)
pyfunc_predictions = loaded.predict(request_frame)
pd.testing.assert_frame_equal(direct_predictions, pyfunc_predictions)
print(model_uri)
print(pyfunc_predictions)  # prediction: approximately 18.0 and 20.0
```

Run the complete block in an empty working directory. Keep `mlflow.db` and its
artifact storage. Start `mlflow ui --backend-store-uri sqlite:///mlflow.db`, open
**Experiments > model-packaging > linear-model > Artifacts > model**, and inspect
`MLmodel`, the saved environment and packaged artifact. The returned URI pins
this run and artifact path. A successful load/parity assertion demonstrates
replay in this environment; it does not deploy the model.

The example columns must match the fitted pipeline's actual raw schema. Direct
local prediction accepts a pandas or Polars frame, validates column names,
order, and dtypes after converting to the **recorded fit engine**, and returns a
pandas DataFrame. A pandas input retains its index; a Polars input receives a
range index. The MLflow pyfunc boundary takes named pandas input; a pipeline
trained on Polars is explicitly converted back to Polars before applying its
learned preprocessing and model. MLflow may align named columns and cast values
according to its signature before that conversion. Check parity for each input
dtype you intend to serve, especially nullable or categorical dtypes.

### Nullable integers and Boolean inputs

MLflow validates its input signature **before** calling the saved pipeline.
A nullable integer or Boolean column can therefore be rejected before its
fitted imputer runs. Newly logged local pipelines and model sets declare these
columns as strings at the MLflow boundary and restore their recorded dtype
before applying feature engineering. Nulls remain nulls until the saved
preprocessing handles them; the adapter does not fill them or retrain an imputer.
`ColSpec(required=False)` makes a column optional; it does not make nullable
integer or Boolean values pass their primitive signature. Floating-point
columns keep numeric signatures and do not need this string transport.

Use the public helper with a loaded model:

<!-- skyulf-example: native models -->
```python
# Continue the imports/config from the complete example above.
from skyulf.integrations.mlflow.models.pipeline_model import prepare_pyfunc_input

nullable = pd.DataFrame({
    "count": pd.Series([1, None, 3, 4, 5, 6, 7, 8], dtype="Int64"),
    "flag": pd.Series([True, None, False, True, False, True, False, True], dtype="boolean"),
    "x": [1., 2., 3., 4., 5., 6., 7., 8.],
    "target": [2., 4., 6., 8., 10., 12., 14., 16.],
})
nullable_pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "fill_count", "transformer": "SimpleImputer",
         "params": {"columns": ["count"], "strategy": "mean"}},
        {"name": "fill_flag", "transformer": "SimpleImputer",
         "params": {"columns": ["flag"], "strategy": "constant", "fill_value": False}},
    ],
    "modeling": {"type": "linear_regression"},
})
nullable_pipeline.fit(
    SplitDataset(train=nullable.iloc[:6], test=nullable.iloc[6:]), target_column="target"
)
save_pipeline(nullable_pipeline, "artifacts/nullable-pipeline")
with track_run(config, run_name="nullable-model") as run:
    nullable_uri = log_pipeline_model(
        "artifacts/nullable-pipeline", run_id=run.run_id,
        artifact_path="model", tracking_uri=tracking_uri,
    )
nullable_model = mlflow.pyfunc.load_model(nullable_uri)
raw_frame = nullable.iloc[:3].drop(columns="target")
request = prepare_pyfunc_input(raw_frame, nullable_model)
assert request["count"].tolist() == ["1", None, "3"]
assert request["flag"].tolist() == ["true", None, "false"]
expected = predict_pipeline(raw_frame, load_pipeline("artifacts/nullable-pipeline"))
pd.testing.assert_frame_equal(expected, nullable_model.predict(request))
```

For example, an `Int64` column `[25, None, 40]` travels as
`["25", None, "40"]`, becomes `Int64` again inside the adapter, and then enters
the fitted cleaning steps. A pipeline with no suitable missing-value handler
can still fail in its model. The helper preserves input order and index and
does not mutate the caller's frame. It is needed only at the MLflow boundary;
direct `predict_pipeline` calls continue to take native values.
Call the helper once: it accepts native values and deliberately rejects
already encoded strings. REST/JSON clients can send the declared strings and
nulls directly; they do not call this Python helper.

The transport covers pandas extension `Int32`, `Int64`, and `boolean`, and
Polars `Int32`, `Int64`, and `Boolean`. Polars does not record column
nullability separately, so newly logged models use this transport for all
columns of those three Polars types. Ordinary pandas NumPy `int32`, `int64`,
and `bool` signatures stay native. Other unsupported dtypes still fail at
packaging time; portable `InferenceBundle` signatures are unchanged.

The package records `skyulf_input_transport` metadata with codec
`nullable_primitives_v1` and its column-to-dtype mapping. The helper checks
this mapping against the loaded artifact. Integer strings must be canonical
decimal values within the fitted signed integer range; Boolean strings must
be `"true"` or `"false"`. Invalid strings and floating-point integer inputs
are rejected instead of rounded. This preserves integers above `2**53`.
Existing logged packages keep their previous contract; log a new version to
use the nullable transport.

For a separately verified batch-safe pipeline, Spark callers must cast the
listed integer and Boolean columns to strings **in Spark before** passing
them to `mlflow.pyfunc.spark_udf`. Casting after conversion to pandas can be
too late: Arrow may already have represented nullable integers as floats.
Other feature columns keep their declared types. This transport does not
make arbitrary preprocessing safe across Spark batches or change nullable
prediction-output limitations. See
[MLflow signature validation](https://mlflow.org/docs/latest/model/signatures/).

For classifiers, the output is `prediction` followed by
`probability_0`, `probability_1`, and so on in the saved model class order. Pass
`use_tuned_thresholds=True` to `save_pipeline` only when the fitted
pipeline has stored tuned thresholds and those decisions are intended for this
artifact; otherwise predictions use the estimator's default decision rule.
Validate replay for the actual recipe, input dtypes and estimator you package.
Use [preprocessing diagnostics](preprocessing_context.md) to inspect saved-state
and request-context behavior; compare direct and pyfunc predictions with the
same typed features. Diagnostic success does not grant worker eligibility.

This package retains its `whole_frame_local` scope. It supports bounded local
batch inference, including inside a Databricks Python task with the recorded
dependencies installed. It does **not** certify independent HTTP row requests,
Spark worker partitions, or Spark-native feature engineering by itself. Newly
logged, inspected pandas artifacts may additionally carry a partition-safety
certificate for the [Spark pyfunc route](databricks_spark_scoring.md).
That adapter revalidates the loaded artifact, package signature, source hash and
certificate before creating a named-input UDF with all output columns. Workers
revalidate the same evidence on load. This package does not write Delta tables.
The model URI points
to a concrete run artifact; registering or promoting an alias is a separate,
explicit step.

Install `skyulf-core[mlflow]` in the loading environment. The MLflow model
records exact package requirements, including the Skyulf version; provide the
matching Skyulf wheel when the version is unpublished. Certified Spark packages
embed an exact-source wheel for isolated worker installation; ordinary local
packages retain their existing external-wheel requirement. Only load local
pipeline artifacts from trusted producers: pickle
can execute code while loading, and the checksum detects corruption rather than
authenticating the source.

## Portable inference bundles

Skyulf can package an existing `InferenceBundle` as an MLflow `pyfunc` model.
The package contains the frozen feature state, estimator payload, manifest, and
runtime requirements already produced by the bundle workflow. It does not fit
again, capture a notebook session, or serialize a Spark session.

Install the optional dependency only in environments that log or load these
models:

```bash
uv pip install "skyulf-core[mlflow]"
```

`log_model` requires the concrete MLflow run ID. Pass the same tracking URI
used to create a client-bound `track_run` when it is not already configured as
MLflow's process-wide URI:

<!-- skyulf-example: native models -->
```python
# Continue the first complete example: fitted_pipeline, config and request_frame.
from skyulf.inference.bundle import build_bundle, predict_local
from skyulf.integrations.mlflow.models.model import log_model

bundle = build_bundle(
    fitted_pipeline, input_stage="raw", feature_order=("x",),
)
with track_run(config, run_name="portable-model") as run:
    portable_uri = log_model(
        bundle, run_id=run.run_id, artifact_path="portable-model",
        tracking_uri=tracking_uri,
    )
portable_loaded = mlflow.pyfunc.load_model(portable_uri)
pd.testing.assert_frame_equal(
    predict_local(request_frame, bundle), portable_loaded.predict(request_frame)
)
print(portable_uri)
```

The returned URI is `runs:/<run_id>/<artifact_path>`. Uploading uses an
explicit MLflow client and run ID, so an unrelated fluent active run cannot
receive the artifact. The `raw` bundle path applies the frozen FE state before
prediction; the `features` path validates that the caller supplies the saved
feature schema. Regression outputs and classification labels, class order,
probability columns, and saved threshold decisions remain the bundle's
contract.

The model signature uses named tabular columns. MLflow may reorder named input columns and safely
cast compatible values before the adapter receives the `pandas.DataFrame`; undeclared extra columns
are ignored by MLflow. Positional NumPy or list input, and inputs missing a declared column, are
rejected. Bundle dtypes that MLflow cannot represent exactly, such as `int8`, are rejected during
packaging instead of being silently widened. Input examples are one synthetic row derived from the
manifest schema and never a training or production row. This is MLflow's input transport contract;
direct `predict_local` calls still reject different column orders, extra columns and dtype
mismatches. See [MLflow signature enforcement](https://mlflow.org/docs/latest/model/signatures/).

The pyfunc entry point accepts named pandas frames (or named inputs that MLflow
can convert). A pipeline trained with Polars can produce the same bundle;
convert a Polars request to pandas explicitly for pyfunc prediction. Supported
bundle input dtypes are `bool`, `int32`, `int64`, `float32`, and `float64`.
String classification outputs are supported; raw string FE inputs remain
outside the existing bundle contract.

Install the recorded requirements, including the matching Skyulf wheel, before
loading the artifact. The wheel itself is not embedded: for an unpublished
version, distribute the built wheel through your package delivery process.
A worker or subprocess can load it without importing the repository checkout. The
estimator payload is a trusted pickle and must only be loaded from a trusted
producer; checksums detect corruption but do not make pickle deserialization
safe.

Packaging uses run artifacts, not MLflow 3's separate LoggedModel entity.
Choose a fresh artifact path for each model; uploading the same run/path is
not an atomic publish operation and may overwrite files. Logging failures
propagate to the caller even when tracking was configured with `warn`.
Producer project files and the temporary source bundle path are excluded.

Use the [registry guide](mlflow_registry.md) to register a concrete run artifact
and manage candidate/champion versions. The [Databricks Python SDK](databricks_sdk.md)
adds pinned UC source reads and bounded Delta publication; the
[Bundle guide](databricks_bundle.md) describes distributed scoring and deployment
settings. Those integrations keep their own schema, context and admission checks;
logging a model alone does not deploy it or create a prediction table.

## If packaging or loading fails

| Symptom | Action |
| --- | --- |
| Raw columns/order/dtype differ | Use the recorded schema; direct inference is stricter than MLflow named-column alignment. |
| Nullable integer/Boolean rejected | Use native typed values with `prepare_pyfunc_input` once, or send the declared string/null wire values directly. |
| Missing package or version mismatch | Install the recorded requirements and matching published release or unpublished wheel before loading. |
| Portable export rejects a step/dtype | Use a fitted-pipeline package for bounded scoring or change the recipe and verify it; do not assume Spark eligibility. |
| Predictions differ | Compare the same raw rows, fit engine, class order and threshold setting; stop before promotion. |
| Artifact cannot load after an upgrade | Use a compatible build or rebuild/log the trusted producer's model under current imports. |

## Library layout and upgrades

MLflow implementation lives under `skyulf.integrations.mlflow`: `models` owns
pyfunc adapters, `spark` owns distributed prediction and worker environments,
`lifecycle` owns validation and champion/challenger decisions, `registration`
owns registry resolution, and `runs` owns tracking. Shared client, signature and
nullable transport helpers live in `shared`.

Use these grouped paths directly. The former `local_model` and `local_feature_model` modules and
their renamed APIs have been removed. Saved packages that reference removed paths may require
rebuilding and logging with the current imports; see
[upgrading integration code](databricks_project_upgrade.md#upgrade-integration-code). The root
exports `TrackingConfig`, `TrackingRun` and `track_run` without loading the optional MLflow
dependency. Worker wheels and source certificates cover the entire `skyulf` package.
