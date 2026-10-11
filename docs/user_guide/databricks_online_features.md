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

## Generate, train and set up an online Bundle

From a Skyulf checkout, save these selections as `online-init.json`, completing
the ordinary project, source and target answers when prompted:

```json
{
  "project_name": "customer_online",
  "engine": "pandas",
  "training_layout": "single_model",
  "task": "regression",
  "input_columns": "total_spend, feature_updated_at",
  "record_key_columns": "customer_id",
  "include_online_serving": "true",
  "include_online_publication": "false",
  "include_daily_rollout": "false",
  "score_handoff": "disabled",
  "scoring_mode": "manual",
  "promotion_policy": "manual_approval"
}
```

```bash
databricks bundle init ./skyulf-core/templates/databricks --config-file online-init.json --output-dir ./generated
cd generated/customer_online
```

The three serving selections are independent and default to `"false"`.
`include_online_serving` adds an unscheduled `online_serving_setup` job and its
notebook. It installs the native Feature Engineering dependency for training
even when another project owns publication. Initialization creates files only.
Selecting `include_online_publication: "true"` asks for
`online_publication_mode`: `TRIGGERED` (default) keeps the paused hourly job;
`CONTINUOUS` generates explicit start/status/stop/restart controls without a
periodic schedule. The mode is independent of online serving and rollout.

Complete the generated `config/training.yml` defaults, retaining the source,
label, split and model settings for your project. For example:

```yaml
feature_lookup:
  lookups:
    - table_name: "{catalog}.{input_schema}.customer_features"
      lookup_key: [customer_id]
      feature_names: [total_spend, feature_updated_at]
online_features:
  required_features: [total_spend, feature_updated_at]
  freshness:
    - column: feature_updated_at
      max_age_seconds: 3600.0
```

All lookup features must be declared in `input_columns`; the required features
must cover them exactly. Choose a maximum age for each lookup's freshness field.
Empty generated placeholders fail validation. The policy is frozen into the
training specification and carried through prepare, fit, log and registration.
The original feature-table snapshot binding and fitted preprocessing remain
enforced. This generated flow supports pandas single-model and competition
training; multi-target policy transport and temporal feature lookups are rejected.
No Spark scoring job or batch prediction table needs to run or exist.

Keep the published Core release pinned in `deployment/requirements.txt`
(`skyulf-core[mlflow]==0.9.2`) and complete your Bundle target and ordinary training
settings. Change only this requirement's version to upgrade Core and redeploy.
The optional online jobs inherit this pin through `serving-requirements.txt`,
which also installs the pinned feature-engineering SDK. Validate saved-model
runtime compatibility before upgrading production.
Provision and publish the feature source as described below, or use an existing
publication maintained externally. Then validate, deploy and train:

```bash
databricks bundle validate -t test
databricks bundle deploy -t test
databricks bundle run train -t test
```

Take the **concrete registered version** from the training result and complete
`online_serving` in `config/serving.yml`:

```yaml
online_serving:
  enabled: true
  exclusive_writer: true
  tracking_uri: databricks
  registry_uri: databricks-uc
  endpoint:
    endpoint_name: customer-online-v3
    model_name: models.risk.customer_online
    model_version: "3"
    logging_catalog: ops
    logging_schema: serving
    logging_table_prefix: customer_online_v3
    logging_mode: telemetry
  publications:
    - source_table: features.customer.customer_features
      source_table_id: "REPLACE_WITH_DESCRIBE_DETAIL_ID"
      online_table: features.online.customer_features
      online_store: customer-features
      publish_mode: TRIGGERED # Match the existing publication; use CONTINUOUS for a stream.
```

Use actual resolved UC names here; the source table must be the model's saved
lookup source with the same Delta identity. Configure exclusive endpoint writer
permissions before setting `exclusive_writer: true`. The setup identity must be
able to inspect the source, publication and store; the serving identity also
needs access to the online features. Grant visibility of all publications for
these sources and prevent concurrent publication changes during setup. UC
enumeration covers the setup identity's visible inventory, not hidden resources.
Setup rejects missing, mismatched or ambiguous accessible publications and requires
evidence of a completed first sync before admitting creation. It reports a
validation error while the first sync is incomplete; it does not poll or republish.

```bash
databricks bundle deploy -t test
databricks bundle run online_serving_setup -t test
```

The setup result reports `SUBMITTED`, `model_uri`, `input_schema` and a typed
`request_example` derived from the actual native signature. Replace its example
values with a real entity and all other required request values. Wait for native
endpoint readiness, then send that request body to
`POST /serving-endpoints/customer-online-v3/invocations` or use
`query_named_records` below. A setup submission does not establish feature
availability. Rerunning setup never overwrites an existing endpoint; inspect an
ambiguous creation result before taking another action.

Save the returned `request_example` object as `request.json`, replace its sample
values, and query after the endpoint reports readiness:

```bash
databricks serving-endpoints get customer-online-v3
databricks serving-endpoints query customer-online-v3 --json @request.json
```

Required source metadata can survive in the native signature, even when excluded
from estimator inputs. Inspect `input_schema`; do not assume every model accepts
keys alone. Direct REST callers can override fetched values. Freshness validation
checks the received values and does not attest their source.

## Create a latest-value source and an online store

Use a Unity Catalog Delta feature table containing one current row per entity.
Declare its primary key columns `NOT NULL`, enable Change Data Feed, and include
a `DOUBLE` feature timestamp containing UTC Unix seconds. The timestamp records
when that entity's values were produced; a recent table synchronization is not a
substitute for a recent feature timestamp.

Provision an online store explicitly with the native Feature Engineering client
and wait until it is `AVAILABLE`. Store provisioning is never a side effect of
Skyulf prediction. Install `deployment/serving-requirements.txt` with `uv pip install`
in a compatible Databricks environment. Packaging and serving must use the same
Core release and source runtime.

Choose the store name and capacity when provisioning it in your selected
workspace. Databricks manages the underlying Lakebase infrastructure. Bundle
initialization does not create that infrastructure: `online_store` in
`config/serving.yml` refers to this existing store. `source_table` identifies
the Delta source, while `online_table` names the destination as
`catalog.schema.table`; prepare that destination catalog and schema first.

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
    publish_mode="TRIGGERED", # Default; choose CONTINUOUS to start a native stream.
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
Temporal source tables remain unsupported. `CONTINUOUS` starts native feature
synchronization and releases the writer admission after submission, allowing
subsequent source writes. No caller process needs to keep running.

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

## Operate continuous feature publication

Select `online_publication_mode: "CONTINUOUS"` together with
`include_online_publication: "true"` in the Bundle initialization answers. Complete
the generated publisher configuration with the actual resources:

```yaml
online_publication:
  enabled: true
  exclusive_writer: true
  publish_mode: CONTINUOUS
  source_table: features.customer.current_values
  source_table_id: "REPLACE_WITH_DESCRIBE_DETAIL_ID"
  online_table: features.online.customer_values
  online_store: customer-features
```

After configuring exclusive source/publication ownership and deploying the Bundle,
run one explicit operation at a time:

```bash
databricks bundle validate -t test
databricks bundle deploy -t test
databricks bundle run online_publication -t test --params publication_action=start
databricks bundle run online_publication -t test --params publication_action=status
databricks bundle run online_publication -t test --params publication_action=stop
databricks bundle run online_publication -t test --params publication_action=status
# Wait until pipeline_state is IDLE or FAILED, then explicitly restart:
databricks bundle run online_publication -t test --params publication_action=restart
```

`RUNNING` describes pipeline activity, including resource provisioning; it is not
first-sync completion or entity freshness. Inspect `pipeline_state`, `update_id`,
`update_state`, `synchronization_state`, `first_sync_completed` and
`last_sync_end_timestamp` separately. A failed update or synchronization is
reported by the inspected `status` command as `FAILED` even if the pipeline still
reports running during recovery.
`STOP_REQUESTED` is a submitted stop; check again for `STOPPING` or `STOPPED`.
Restart requires native `pipeline_state` to be `IDLE` or `FAILED`. Overall
`status=FAILED` can coexist with a `RUNNING` or `STOPPING` pipeline; stop active
compute first and inspect its pipeline state. A restart receipt keeps its new
update ID and reports `PENDING` until that update appears in native metadata.
Earlier update/synchronization failures do not establish failure of that attempt.
Once its acknowledged update appears, that update/pipeline activity decides the
attempt status; synchronization failure labels remain separately visible because
they can lag an intentional cancellation or restart. An intentionally stopped
pipeline can therefore report native `pipeline_state=IDLE` alongside overall
`status=FAILED` and `update_state=CANCELED` from the last operation.
The model's saved missing/stale guard still validates every prediction request.

A repeated continuous `start` inspects and reuses the active pipeline without
submitting another update. Restart uses the same pipeline after a confirmed stop
or failure. Calls that time out or lose a response are not retried automatically;
inspect native status before another action. Changing `publish_mode` on an existing
publication is rejected. Stop and inspect the existing pipeline, then explicitly
plan the new publication and model-routing migration with its resource owner;
this adapter never drops or recreates an online table.

With `TRIGGERED`, the same `start` command requests one sync and the generated
hourly schedule remains `PAUSED` until you deliberately enable it. `status` inspects
the latest native update; `SYNCED` does not attest entity freshness. Continuous
stop/restart controls require a continuous publication. If online serving uses an
external publication, set its `online_serving.publications[].publish_mode` to that
publication's actual mode before setup.

The SDK exposes the same operations without a Bundle:

```python
from skyulf.integrations.databricks.feature_store import (
    publication_receipt,
    restart_online_publication,
    stop_online_publication,
)

# publication is an OnlinePublicationSpec with publish_mode="CONTINUOUS".
receipt = publication_receipt(workspace, publication)
print(online_publication_status(workspace, receipt))
print(stop_online_publication(workspace, receipt, admission=SingleWriterAdmission()))
# Inspect status in a later cell; restart only when pipeline_state is IDLE or FAILED.
receipt = restart_online_publication(
    spark, receipt, client=workspace, feature_client=features,
    admission=SingleWriterAdmission(),
)
```

If the source was dropped, replaced or became inaccessible, the guarded SDK
control fails closed during binding checks, even for stop. The resource owner
must inspect permissions, the saved receipt, online table source/store properties
and native pipeline identity before stopping through the native API. For example,
after that ownership review, verify the remaining publication metadata:

```python
saved = receipt["spec"]
online = workspace.tables.get(saved["online_table"])
synced = workspace.database.get_synced_database_table(name=saved["online_table"])
pipeline_id = receipt["pipeline_id"]
assert online.pipeline_id == synced.data_synchronization_status.pipeline_id == pipeline_id
assert (online.properties or {}).get("source_table") == saved["source_table"]
assert synced.spec.source_table_full_name == saved["source_table"]
assert synced.effective_database_instance_name == saved["online_store"]
print(workspace.pipelines.get(pipeline_id=pipeline_id))
# Resource owner only: after confirming this saved pipeline is the intended resource.
workspace.pipelines.stop(pipeline_id=pipeline_id)
```

This recovery stops pipeline compute and does not alter the source table or
publication. If metadata or ownership cannot be confirmed, resolve that with
the resource owner before any mutation. Inspect native pipeline state after the
stop request. Restart still requires restored, valid source identity/access and
all normal source preflight checks.

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
