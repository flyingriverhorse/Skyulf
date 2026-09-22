# MLflow model packaging (development)

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
from skyulf.integrations.mlflow.model import log_model

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

This stage provides local MLflow pyfunc packaging. The optional registry adapter
can now publish and resolve concrete local registry versions; live Unity Catalog,
Databricks job delivery, Delta batch sinks, and Spark UDF/endpoint adapters
remain later initiative tasks. Until those platform gates are complete,
`load_model` plus `predict` is the supported pyfunc packaging contract.
