# SM-21a: optional Unity Catalog Feature Engineering adapters

Status: **PARTIAL**. This delivery adds a callable Python adapter boundary,
validated offline. It does not switch the generated training/batch lifecycle
to Feature Engineering or establish native Databricks acceptance. The user
explicitly deferred cloud execution. No resources were created or modified.

## Implemented boundary

`skyulf.integrations.databricks.feature_store` provides:

- `FeatureLookupSpec`: explicit catalog/schema/table, ordered lookup keys,
  explicit feature names, optional timestamp lookup, a declared Spark time
  type (`timestamp` or `date`), and optional nonnegative `timedelta` lookback.
- `FeatureTrainingSpec`: frozen lookup contracts, optional label, explicit
  exclusions, duplicate-name and direct label-leakage checks.
- `create_feature_training_set`: validates source columns/time types, creates
  real SDK `FeatureLookup` objects and returns the native SDK `TrainingSet`.
- `log_feature_model`: delegates packaging to `FeatureEngineeringClient.log_model`
  with the caller's original training-set object and unchanged flavor options.
- `score_feature_model`: calls `FeatureEngineeringClient.score_batch`, checking
  configured keys/time types and requiring an explicit Spark result type.

The adapters accept an injected client; training also accepts an injected
lookup factory. Defaults import `databricks.feature_engineering` only when
called. Importing the package requires no Databricks SDK, Spark, or MLflow.

Version floor recommendation: optional `feature-store` extra containing
`databricks-feature-engineering>=0.11,<1.0`. The repository uses MLflow 3;
Databricks documents support beginning in Feature Engineering 0.11.0.
Manifest changes belong to the coordinating root/YAML owner, not this package.
[Databricks release notes](https://docs.databricks.com/aws/en/release-notes/feature-store/databricks-feature-store).

## Important behavior

The table must already have correct Unity Catalog primary-key metadata and a
TIMESERIES key when used for history lookup. Configured key ordering must match
the table. Source time types must match `timestamp_type` during both training
and scoring. Table key/type compatibility and model lineage are checked by the
SDK, not proven by the offline adapter tests. These helpers do not scan source
rows to validate nulls, duplicate entity/time values, or point-in-time results.
[Databricks point-in-time joins](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series).

Unlike direct SDK defaults, these helpers reject feature columns already
present in the input. Set `allow_feature_overrides=True` explicitly to adopt
the SDK behavior in which input values replace table features. Feature names
must be explicit; wildcard/select-all lookups and output renaming are outside
this first contract. Case variants cannot bypass the override guard.
[Feature Engineering client API](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html).

Train from the exact `TrainingSet.load_df()` result and pass the same native
training-set object to model logging. Learned transformations must live inside
the logged model. The adapter cannot establish that the caller obeyed this
data-provenance requirement. `score_feature_model` accepts the same caller-held
configuration; the SDK uses the model's saved feature metadata for the actual
join. Passing an unrelated specification does not replace that metadata.

The required `result_type` avoids silently accepting the SDK's scalar-double
default for models with named prediction/probability outputs. MLflow signatures,
artifact bindings, environment arguments, and score parameters pass unchanged;
passing a signature does not by itself prove SDK packaging compatibility.

## Minimal SDK usage

```python
from datetime import timedelta

from skyulf.integrations.databricks.feature_store import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
    create_feature_training_set,
    log_feature_model,
    score_feature_model,
)

spec = FeatureTrainingSpec(
    lookups=(FeatureLookupSpec(
        table_name="catalog.features.customer_history",
        lookup_key=("customer_id",),
        feature_names=("spend_30d",),
        timestamp_lookup_key="event_time",
        timestamp_type="timestamp",
        lookback_window=timedelta(days=30),
    ),),
    label="target",
    exclude_columns=("customer_id", "event_time"),
)
training_set = create_feature_training_set(labeled_spark_df, spec, client=fe)
training_df = training_set.load_df()
# Fit a compatible model from training_df in an explicit MLflow run.
# Any learned preprocessing must be part of that model.
log_feature_model(
    fitted_model,
    training_set=training_set,
    flavor=mlflow.sklearn,
    artifact_path="model",
    client=fe,
)
predictions = score_feature_model(
    feature_packaged_model_uri,
    unlabeled_spark_df,
    spec,
    result_type="double",  # Use a named struct for a multi-column model.
    client=fe,
)
```

## Existing Skyulf lifecycle and next integration

The current training path still reaches
`integrations/mlflow/models/local_model.py::log_local_model`. That function
constructs `SkyulfLocalPythonModel`, named signatures, nullable transport,
artifact bindings, worker source/environment metadata and optional partition
certificates before uploading its MLflow artifact. It is not a call to
`FeatureEngineeringClient.log_model` and is intentionally unchanged here.

The generic adapter does not convert a `LocalPipelineArtifact` into an MLflow
flavor or certify partition safety. Existing pandas training, fitted artifacts,
registry governance and Spark eligibility gates remain in their existing paths.
An arbitrary pyfunc must not be treated as certified solely because Feature
Engineering can call it from `score_batch`.

Next integration should preserve the native `TrainingSet` and its immutable
lookup specification through fit and registration, extract a shared validated
Skyulf pyfunc packaging builder, and let Feature Engineering package that exact
artifact while preserving named/nullable schemas, requirements, source digest
and partition certificate. Batch routing should select `fe.score_batch` only
for feature-packaged artifacts, preserving controlled model resolution and
output schema checks. Native acceptance must compare historical lookup results,
saved/reloaded predictions and feature-lineage scoring on Databricks before
SM-21a is closed.

## Verification

Focused tests were written first: the initial run failed with the new package
absent. A later case-insensitive feature override probe also failed before the
guard was corrected. Tests use schema-only input doubles and injected SDK
boundaries; they do not execute joins, fit a model, or simulate successful cloud
execution.

Final adapter state, 2026-10-07:

- `.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_databricks_feature_store.py skyulf-core/tests/integration/platforms/test_databricks_import_order.py -q --no-cov --tb=short`: **48 passed** (47 adapter cases and one direct import consumer).
- Ruff check and format check on the new package and test file: **passed**.
- Ty check on the new package and test file: **passed**. Coordinating root owns
  the required full CI Ty scope and repository-wide gates.
- Lizard on the new package with `--CCN 10 -w`: **passed**. The first run exposed
  CCN 11 in input validation; a behavior-preserving helper extraction fixed it,
  followed by a fresh affected test run.
- Native Spark/Feature Engineering/MLflow packaging integration, cloud execution,
  full suite, documentation build, pre-commit hooks and commit/push: **not run**
  by this subtask. Documentation-only evidence does not require a site build.
