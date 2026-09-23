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
local tests cover configuration validation and separate stores. A live
Databricks serverless Spark Connect 4.2.0 test registered and loaded a
Polars-trained regression bundle, then verified both Spark inference modes.
A separate restricted service principal loaded an allowed model and received
`RegistryAccessError` for a model without `EXECUTE` permission. This evidence
does not establish compatibility with every compute type or registry backend.

Load the selected artifact with `load_registered_bundle`. It uses the pinned
name/version and the package's declared bundle path, rejects paths outside the
downloaded package, and checks the package/bundle digest against `resolved.digest`.
The same bundle can then be passed to `predict_local` or either Spark inference
mode, subject to their existing schema and FE capabilities.

For fitted pandas/Polars packages created with `log_local_model`, use
`load_registered_local_pipeline(resolved, ...)` instead. It verifies the
declared local artifact path, scope, engine and payload digest against the
pinned registry version. That package supports whole-frame local inference;
it does not become a Spark bundle merely because it is registered.

```python
from skyulf.integrations.mlflow.registry import load_registered_bundle

bundle = load_registered_bundle(
    resolved,
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
assert bundle.semantic_digest == resolved.digest
```

## How the adapters use Skyulf Core

The Databricks local workflow calls Skyulf Core's existing pipeline fit and
prediction methods. Its UC reader and Delta writer handle transport and
publication; they do not reimplement feature engineering or model behavior.
The MLflow adapter packages that same saved pipeline, resolves a registry
alias to a concrete version, and loads the artifact for prediction or
comparison. Held-out evaluation runs the saved pipeline once and passes its
predictions and class-ordered probabilities to Core's model metric calculators.
There is no separate Databricks or MLflow training algorithm.

For example, a training job can log every available held-out metric to
its own MLflow run without implementing a second evaluator:

```python
from skyulf.inference.local_evaluation import evaluate_local_holdout
from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

metrics = evaluate_local_holdout(artifact, heldout, target_column="target")
with track_run(
    TrackingConfig(enabled=True, experiment_name="customer-risk"),
    run_name="candidate-training",
) as run:
    run.log_metrics(metrics)
```

## Compare a local challenger before promotion

For a fitted pandas/Polars pipeline, resolve both registered versions to
concrete identities and evaluate them on the **same held-out labeled rows**.
Pin the source snapshot and split in the caller; `dataset_id` records that
identity in the report but does not verify the underlying table by itself.
The report also records the running Skyulf Core version and both model digests.
The comparison uses the saved FE, model, class order and tuned thresholds.
It accepts a row/memory budget and rejects incompatible tasks or class labels.

```python
from skyulf.integrations.mlflow.registry import resolve_model
from skyulf.integrations.mlflow.validation import compare_registered_local_models

candidate = resolve_model(
    "catalog.schema.customer_risk", version="2", registry_uri="databricks-uc"
)
champion = resolve_model(
    "catalog.schema.customer_risk", alias="champion", registry_uri="databricks-uc"
)
report = compare_registered_local_models(
    candidate,
    champion,
    heldout,  # bounded pandas or Polars frame with raw inputs and true labels
    target_column="target",
    dataset_id="catalog.schema.labels@version=42/split=holdout-v1",
    metric="heldout_rmse",
    min_improvement=0.1,
    quality_threshold=5.0,  # maximum acceptable RMSE
    max_rows=10_000,
    max_bytes=20_000_000,
    registry_uri="databricks-uc",
)
print(report.eligible, report.reason, report.candidate_metrics)
```

The report contains all finite metrics available for the task and labeled
holdout. Regression includes MAE, MSE, RMSE, R2, MAPE and explained variance.
Classification includes accuracy, balanced accuracy, weighted precision/recall/F1,
Matthews correlation, binary precision/recall/F1 and, when defined, log loss,
ROC-AUC and PR-AUC variants. Optional metrics such as geometric-mean score may
be absent when their dependency or the required label distribution is unavailable.
Probability metrics use the saved model's class order; hard-label metrics use
its effective prediction thresholds.

Choose one primary metric before evaluating candidates. Error metrics
(MAE/MSE/RMSE/MAPE/log loss) minimize; the other selectable scores maximize.
`quality_threshold` is an upper limit for error metrics and a lower limit for
scores. A tie does not qualify. If no champion exists, pass `None` and
inspect the candidate report; this never creates a champion. The report is
read-only; the caller explicitly logs the desired numeric metrics to its
MLflow run. Unlabeled production predictions cannot produce supervised
quality metrics until their true labels arrive.
Evaluation does not move aliases, publish predictions or deploy endpoints.
Explicit version-checked promotion and rollback are the next SM-22b task;
monthly candidate training follows in SM-28a before Bundle generation.

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
