# MLflow model packaging

## Library layout and compatibility

MLflow implementation lives under `skyulf.integrations.mlflow`: `models` owns
pyfunc adapters, `spark` owns distributed prediction and worker environments,
`lifecycle` owns validation and champion/challenger decisions, `registration`
owns registry resolution, and `runs` owns tracking. Shared client, signature and
nullable transport helpers live in `shared`.

Use these grouped paths in new code. Released flat import paths remain available
through lazy `_compat` aliases, including classes referenced in saved models.
The root still exports `TrackingConfig`, `TrackingRun` and `track_run` without
loading the optional MLflow dependency. Worker wheels and source certificates
continue to cover the entire `skyulf` package.

## Fitted pipelines (pandas or Polars)

Use a **fitted pipeline artifact** when fitted preprocessing contains nodes that
the portable Spark bundle does not yet support, such as categorical encoding or
binning. This path saves the whole fitted pipeline, its recorded training engine,
raw and model-feature column order and dtypes, dependency versions, and a payload
checksum. It never fits the pipeline again. It is separate from the portable
Spark-capable `InferenceBundle` described below.

```python
import mlflow
import pandas as pd

from skyulf.inference.fitted_pipeline import (
    load_pipeline,
    predict_pipeline,
    save_pipeline,
)
from skyulf.integrations.mlflow import TrackingConfig, track_run
from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model

# fitted_pipeline was already trained on pandas or Polars data.
save_pipeline(fitted_pipeline, "artifacts/fitted-pipeline")
config = TrackingConfig(
    enabled=True,
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="customer-risk",
)
with track_run(config, run_name="fit-2026-09") as run:
    model_uri = log_pipeline_model(
        "artifacts/fitted-pipeline",
        run_id=run.run_id,
        artifact_path="model",
        tracking_uri=config.tracking_uri,
    )

request_frame = pd.DataFrame({"age": [42.0], "income": [72000.0]})
artifact = load_pipeline("artifacts/fitted-pipeline")
direct_predictions = predict_pipeline(request_frame, artifact)
mlflow.set_tracking_uri(config.tracking_uri)
pyfunc_predictions = mlflow.pyfunc.load_model(model_uri).predict(request_frame)
```

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

```python
from skyulf.integrations.mlflow.models.pipeline_model import prepare_pyfunc_input

loaded = mlflow.pyfunc.load_model(model_uri)
# raw_frame contains the original, typed values expected by this artifact.
request = prepare_pyfunc_input(raw_frame, loaded)
predictions = loaded.predict(request)
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
certificate for the [Spark pyfunc route](databricks_bundle.md#distributed-inference-settings).
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

```python
import mlflow

from skyulf.inference.bundle import build_bundle
from skyulf.integrations.mlflow import TrackingConfig, track_run
from skyulf.integrations.mlflow.models.model import log_model

bundle = build_bundle(
    fitted_pipeline,
    input_stage="raw",  # use "features" when callers already apply FE
    feature_order=("age", "income"),
)
config = TrackingConfig(
    enabled=True,
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="customer-risk",
)

with track_run(config, run_name="fit-2026-09") as run:
    model_uri = log_model(
        bundle,
        run_id=run.run_id,
        artifact_path="skyulf-model",
        tracking_uri=config.tracking_uri,
    )

mlflow.set_tracking_uri(config.tracking_uri)
loaded = mlflow.pyfunc.load_model(model_uri)
predictions = loaded.predict(request_frame)
```

The returned URI is `runs:/<run_id>/<artifact_path>`. Uploading uses an
explicit MLflow client and run ID, so an unrelated fluent active run cannot
receive the artifact. The `raw` bundle path applies the frozen FE state before
prediction; the `features` path validates that the caller supplies the saved
feature schema. Regression outputs and classification labels, class order,
probability columns, and saved threshold decisions remain the bundle's
contract.

The model signature uses named tabular columns. MLflow may reorder named input
columns and safely cast compatible values before the adapter receives the
`pandas.DataFrame`; undeclared extra columns are ignored by MLflow. Positional
NumPy or list input, and inputs missing a declared column, are rejected.
Bundle dtypes that MLflow cannot represent exactly, such as `int8`, are
rejected during packaging instead of being silently widened. Input examples
are one synthetic row derived from the manifest schema and never a training or
production row. This is MLflow's input transport contract; direct
`predict_local` calls still reject different column orders, extra columns and
dtype mismatches. See [MLflow signature enforcement](https://mlflow.org/docs/latest/model/signatures/).

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
and manage candidate/champion versions. The [Databricks Python SDK](databricks_local_sdk.md)
adds pinned UC source reads and bounded Delta publication; the
[Bundle guide](databricks_bundle.md) describes distributed scoring and deployment
settings. Those integrations keep their own schema, context and admission checks;
logging a model alone does not deploy it or create a prediction table.
