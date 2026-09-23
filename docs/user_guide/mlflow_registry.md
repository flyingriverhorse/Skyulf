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
Promotion is a separate, explicit operation. SM-28a monthly candidate training
follows before Bundle generation.

## Stage, promote, and roll back a local pipeline

After a successful comparison, stage_challenger re-evaluates the same pinned
holdout and assigns the concrete candidate version to `@challenger`. It does
not move `@champion`. A replacement challenger requires an explicit
`expected_challenger_version`; staging cannot silently overwrite another
candidate. Merely registering or comparing a model never assigns an alias.

The promote_candidate function re-resolves both concrete model versions and
re-evaluates the same bounded labeled holdout. It rejects a changed comparison
report, an ineligible candidate, a missing champion alias, a stale expected
champion version, or a challenger alias without a matching committed staging
event. The caller must pin the holdout's source snapshot externally;
dataset_id alone is not proof of the underlying data. There is no implicit
first-champion initialization.

Alias writes have no expected-prior-version parameter. Every alias writer
must use the same non-expiring admission authority, and registry write
permissions must be restricted to that controlled path. A local development
store can use LocalAliasAdmission with a shared lock directory:

~~~python
from skyulf.integrations.mlflow.promotion import (
    LocalAliasAdmission,
    promote_candidate,
    rollback_promotion,
    stage_challenger,
)

admission = LocalAliasAdmission("/var/lib/skyulf/alias-locks")
staging = stage_challenger(
    report,
    heldout,
    target_column="target",
    expected_champion_version="1",
    admission=admission,
    max_rows=10_000,
    max_bytes=20_000_000,
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
receipt = promote_candidate(
    report,
    heldout,
    target_column="target",
    expected_champion_version="1",
    admission=admission,
    max_rows=10_000,
    max_bytes=20_000_000,
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
reversal = rollback_promotion(
    receipt,
    expected_current_version=receipt.new_version,
    admission=admission,
    tracking_uri="sqlite:///tracking.db",
    registry_uri="sqlite:///registry.db",
)
~~~

For Unity Catalog, use DeltaAliasAdmission(spark, control_table) with a
separate, preprovisioned Delta control table containing exactly one row:
target_id = alias_resource_id(model_name, "champion") and nullable owner = NULL.
Provision it once with a plain CREATE TABLE statement so an existing authority
cannot be overwritten:

~~~python
from skyulf.integrations.mlflow.promotion import DeltaAliasAdmission, alias_resource_id

key = alias_resource_id("catalog.schema.customer_risk", "champion")
spark.sql(
    "CREATE TABLE catalog.schema.customer_risk_alias_admission USING DELTA AS "
    f"SELECT '{key}' AS target_id, CAST(NULL AS STRING) AS owner"
)
admission = DeltaAliasAdmission(
    spark, "catalog.schema.customer_risk_alias_admission"
)
~~~

All participating staging, promotion and rollback jobs must point to that same table.
Do not use the predictions-table admission row for aliases. Local file locks
are rejected when the registry URI is databricks-uc.

Staging writes a prepared model-version tag, assigns `@challenger`, verifies
the alias, and commits a staging event. Promotion assigns
`@previous_champion` to the old champion, moves `@champion` to the
challenger, then removes `@challenger`. The returned receipt records the
concrete model versions, event ID, and the previous value of
`@previous_champion`. Rollback verifies the committed promotion event,
restores `@champion` and the previous rollback pointer, and does not
reinstate the rolled-back model as challenger. A prior release's single-alias
promotion receipt remains reversible. A staged newer challenger blocks
rollback until the operator reconciles it.

MLflow performs these alias changes as separate calls, not one atomic
transaction. If a later call fails after an earlier alias moved, the operation
reports an unknown outcome; inspect all three aliases and the prepared event
before retrying. `@previous_champion` identifies only the latest rollback
pointer, not the full history. Concrete versions and promotion/rollback event
tags retain that history; predictions should record the resolved model version.
Rollback refuses a later promotion, even if it selects the same version.
AliasOutcomeUnknownError includes the event ID for inspection. A prepared tag
is not proof that promotion completed. Registry tags are audit records for
controlled writers, not tamper-proof authorization tokens.
The event tag key uses underscores, and its compact value stays within Unity
Catalog's 256-byte tag-value limit.

The lock protects only participating jobs. A principal with direct registry
write permission can bypass it. A crashed Delta admission owner does not
expire; clear it only after proving the original job cannot publish.

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
