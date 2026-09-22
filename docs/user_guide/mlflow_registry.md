# MLflow model registry (development)

Skyulf keeps model publication and model selection as explicit operations. The
registry adapter is optional: importing the core package does not import MLflow,
open a client, or contact a registry.

Install the integration in the environment that publishes or resolves models:

```bash
uv pip install "skyulf-core[mlflow]"
```

`register_model` publishes a model already logged by `log_model` as a new
registered-model version. It accepts a `runs:/<run_id>/<artifact_path>` URI and
returns MLflow's model-version object. Registration never assigns an alias:
promotion is a separate operation owned by the caller.

```python
from skyulf.integrations.mlflow.model import log_model
from skyulf.integrations.mlflow.registry import register_model

model_uri = log_model(
    bundle,
    run_id=run_id,
    artifact_path="model",
    tracking_uri="sqlite:///tracking.db",
)
version = register_model(
    model_uri,
    "customer_risk",
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
print(version.version)
```

Resolve exactly one selector at the start of a job. An alias is looked up once;
the returned object carries the concrete version URI, so a later alias move does
not change the model used by the running job.

```python
from skyulf.integrations.mlflow.registry import resolve_model

resolved = resolve_model(
    "customer_risk",
    alias="champion",
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
assert resolved.model_uri == f"models:/{resolved.name}/{resolved.version}"

# The versioned URI can be carried into a Spark batch or another runner.
print(resolved.signature, resolved.digest)
```

Passing both `alias` and `version`, or neither, is a configuration error. The
adapter reports missing models, access failures, and missing MLflow as separate
exception types so a runner can choose the right remediation. The `signature`
and `digest` fields are read from the packaged MLflow model metadata; they are
validation data, not a second feature-engineering implementation.

For Unity Catalog, pass a three-part model name (`catalog.schema.model`) and a
`databricks-uc` registry URI. Tracking and registry URIs remain separate. The
local tests cover this configuration validation and separate local stores; a
live Databricks/Unity Catalog connection is deliberately reserved for the SM-16
platform gate.

Load the selected artifact with `load_registered_bundle`. It uses the pinned
name/version and the package's declared bundle path, rejects paths outside the
downloaded package, and checks the package/bundle digest against `resolved.digest`.
The same bundle can then be passed to `predict_local` or either Spark inference
mode, subject to their existing schema and FE capabilities.

```python
from skyulf.integrations.mlflow.registry import load_registered_bundle

bundle = load_registered_bundle(
    resolved,
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
assert bundle.semantic_digest == resolved.digest
```

Only load trusted registry artifacts: bundle loading deserializes pickle, and
digest matching is an identity check, not authentication of an unknown producer.
Missing bundle metadata/artifacts or mismatched digests fail explicitly. A later
alias move does not redirect the pinned reference.

The `databricks_batch_smoke.py` example logs a uniquely named test model, loads
its concrete registry version, and compares known gold predictions through both
Spark inference modes. It accepts pandas or Polars training and emits a partial
validation report. Local SQLite/Spark evidence does not certify live Unity Catalog,
wheel deployment, distributed admission, Delta publication or platform scale.

Registry resolution/loading does not predict, mutate aliases, start Spark or
write a table. HTTP/SQL endpoints and Databricks Bundle generation remain later
initiative tasks; the monthly Spark/Delta runner has its own
[batch contract](databricks_batch.md).
