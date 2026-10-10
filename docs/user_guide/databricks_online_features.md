# Look up online features before prediction

An online feature model accepts entity keys, lets Databricks fetch current
feature values, and then runs the saved Skyulf preprocessing and predictor. An
explicit saved policy rejects missing or stale feature values before
preprocessing can impute them. This workflow uses the native Databricks Feature
Engineering envelope and Online Feature Store.

The saved guard supports raw pandas and Polars model execution. Endpoint
admission retains the existing certificate requirement: the fitted artifact
must use pandas and supported independent-row preprocessing. The initial online
request contract supports non-temporal scalar source inputs. It does not omit
required source fields to make an incompatible model appear to accept only keys.

## Create a latest-value source and an online store

Use a Unity Catalog Delta feature table containing one current row per entity.
Declare its primary key columns `NOT NULL`, enable Change Data Feed, and include
a `DOUBLE` feature timestamp containing UTC Unix seconds. The timestamp records
when that entity's values were produced; a recent table synchronization is not a
substitute for a recent feature timestamp.

Provision an online store explicitly with the native Feature Engineering client
and wait until it is `AVAILABLE`. Store provisioning is never a side effect of
Skyulf prediction. Install `skyulf-core[feature-store,mlflow]` in a compatible
Databricks environment and use the same Skyulf wheel for packaging and serving.

```python
from databricks.feature_engineering import FeatureEngineeringClient

features = FeatureEngineeringClient()
# Run provisioning once with a new name and an appropriate capacity.
features.create_online_store(name="customer-features", capacity="CU_1")
```

Prepare the destination catalog and schema before publication. Keep one online
publication per source when exact store selection matters: native lookup can
choose the oldest online copy if the same source has several publications. The
model's lookup contract identifies its source features, not a chosen online
table replica.

## Publish and inspect a sync

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.data.admission import SingleWriterAdmission
from skyulf.integrations.databricks.feature_store import (
    OnlinePublicationSpec,
    online_publication_status,
    publish_online_features,
)

source = "features.customer.current_values"
detail = spark.sql(f"DESCRIBE DETAIL {source}").first()
publication = OnlinePublicationSpec(
    source_table=source,
    online_table="features.online.customer_values",
    online_store="customer-features",
    source_table_id=detail["id"],
)
receipt = publish_online_features(
    spark,
    publication,
    feature_client=features,
    admission=SingleWriterAdmission(),
)
print(receipt)
```

This validates the source table identity, CDF, key declarations and actual key
rows before requesting a `TRIGGERED` sync. Source and publication writers must
share the declared exclusive ownership; the adapter does not lock outside
writers or change the source schema for you. It publishes all source columns.
Temporal source tables and continuous streaming are not enabled by this adapter.

Inspect status later without submitting another sync:

```python
workspace = WorkspaceClient()
status = online_publication_status(workspace, receipt)
print(status["status"], status.get("update_id"))
```

`SUBMITTED` and `PENDING` are not completion. `SYNCED` requires a native pipeline
update that started after submission and completed; `FAILED` identifies a failed
or canceled update. Keep the receipt and update ID for inspection. The recorded
source version is the version checked during preflight, not a guarantee that
native latest-value publication copied precisely that snapshot. Verify an actual
endpoint lookup when testing propagation.

## Save the freshness policy with the model

Build the native training set with explicit lookup fields and train from its
`load_df()` result, retaining that same native training-set object for packaging.
The freshness field must already be a fitted raw `float64`/`double` input and a
selected lookup feature. Each lookup table needs its own freshness field.

```python
from skyulf.integrations.databricks.feature_store import OnlineFeaturePolicy
from skyulf.integrations.mlflow.models.feature_model import log_feature_pipeline_model

online_policy = OnlineFeaturePolicy(
    required_features=("total_spend", "feature_updated_at"),
    freshness=(("feature_updated_at", 3600.0),),
)

# fitted_path, training_set, lookup_spec and lookup_binding come from training.
# run_id identifies the explicit MLflow run that owns this package.
model_uri = log_feature_pipeline_model(
    fitted_path,
    training_set=training_set,
    lookup_spec=lookup_spec,
    lookup_binding=lookup_binding,
    run_id=run_id,
    artifact_path="model",
    client=features,
    tracking_uri="databricks",
    online_policy=online_policy,
)
```

`lookup_binding` is the saved binding for that training set: its lookup spec and
`training_snapshot` evidence identify the feature table IDs and Delta versions
used during training. Retain that binding with the fitted artifact; do not
replace it with a new snapshot when packaging an already fitted model.

Choose the maximum age for your domain. The example's one-hour limit is not a
universal default. All fetched features are required and non-null. Every request
uses one real UTC clock reading; nonfinite, future or over-age timestamps fail
before fitted preprocessing. Callers cannot pass another clock to the saved
model. The policy is checked across saved model metadata, configuration and
serialized state when the package is loaded.

The timestamp is an actual fitted input: adding it only at packaging time is
rejected. If it should not affect the estimator, remove it through a supported
fitted preprocessing step and verify that step's serving certificate. Do not
drop or transform inputs outside the saved preprocessing and expect Databricks
lookup to repeat those transformations. `log_feature_model_set` also accepts
`online_policy`; its raw saved guard covers the model-set request before its
components run.

## Admit and query the registered package

Register the native outer package, then inspect a concrete version:

```python
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    create_pinned_endpoint,
    prepare_online_endpoint,
    query_named_records,
    require_pinned_endpoint_ready,
)

spec = PinnedEndpointSpec(
    endpoint_name="customer-risk-online",
    model_name="models.risk.customer_risk_online",
    model_version="3",
    logging_catalog="ops",
    logging_schema="serving",
    logging_table_prefix="customer_risk_online",
)
plan = prepare_online_endpoint(
    spec, tracking_uri="databricks", registry_uri="databricks-uc"
)
print(plan.input_schema)
create_pinned_endpoint(workspace, plan)  # Once, using a new endpoint name.
```

After deployment is ready:

```python
require_pinned_endpoint_ready(workspace, plan)
response = query_named_records(workspace, plan, [{"customer_id": 42}])
print(response)
```

The example request is valid only if the inspected input schema requires exactly
`customer_id`. Native Feature Engineering can retain excluded training-source
columns in its outer signature. Supply every field reported by
`plan.input_schema`, or train from a narrower source. Temporal/date and binary
source inputs are rejected by this bridge. The endpoint identity must have
permission to read the required online features.

The helper's exact schema excludes feature overrides. Direct native REST calls
can still provide feature values, including a freshness value. The saved guard
checks the values it receives; it cannot prove that those values came from the
online store after enrichment. Restrict direct endpoint access or use an
application boundary if your application must prohibit caller overrides.

## Schedule publication and test updates

Generate a Bundle project with `include_online_publication: "true"`. Fill
`online_publication` in `config/serving.yml` with the explicit source table ID,
source and online table names, and store name. Establish exclusive source/sync
writer ownership before setting `exclusive_writer: true` and `enabled: true`.
The hourly job starts `PAUSED`; deploy, inspect a manual run and its pipeline
status, then unpause it. Its result may be `PENDING` while the native pipeline
continues. A later run should not be used to blindly retry an ambiguous native
failure; inspect its pipeline and receipt first.

For acceptance, compare an endpoint prediction with the saved fitted artifact,
change a source feature and its timestamp, publish, and verify that the same
key-only request reflects the new value. Check that an absent entity and an
expired timestamp produce the intended feature error. A successful local model
load or native batch `score_batch` call does not establish online endpoint lookup.

The native Feature Engineering wrapper can return a generic model-evaluation
error to the REST caller. Inspect the served model's logs for the underlying
`Online required feature` or `Online freshness feature` exception. Check that
the traceback belongs to the rejected request; a generic HTTP error alone does
not establish that the freshness policy ran.

The latest-value policy does not weaken historical batch
`training_snapshot` checks. Those checks still reject changed feature bindings.
Reconstructing original heldout data for a later
[automatic champion promotion](databricks_serving_rollout.md) can therefore stop
when the original feature source has advanced; current features are not silently
substituted for the saved training evidence.
