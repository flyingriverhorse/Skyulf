# Set up and upgrade the monitoring store

## When you need it

Set up one durable store before enabling production monitoring, or upgrade its owned views before
deploying a newer producer or dashboard. Follow [Enroll models](databricks_monitoring_enrollment.md)
next.

The end-to-end flow from scoring to dashboard is on the
[monitoring overview](databricks_monitoring.md#how-monitoring-runs).

The canonical dashboard definition is `src/monitoring/monitoring.lvdash.json`
inside the existing Databricks project template. This is not a separate Bundle
or another project template. Single, competition and model-set projects write to
one configured store and display their evidence in one shared dashboard.

| Operation | Trigger and effect |
| --- | --- |
| Measure drift and performance | Successful scoring calls the separate Spark monitoring job when monitoring is configured and enabled. A production-mode target is required; development targets bypass shared monitoring. |
| Revisit delayed outcomes | Unpause the monitoring job's schedule after setup. The generated schedule starts paused. |
| Refresh dashboard results | The configured native Dashboard task reruns published datasets after monitoring completes. It does not calculate new monitoring metrics. |
| Request training | Drift or performance loss can qualify independently when its own policy enables retraining. Shared guards must also pass. |

Changing training/scoring compute to a cluster does not enable monitoring, change
its storage destination, or enable automatic retraining. The generated monitoring
job has its own serverless environment. Use Databricks to inspect job run
history, compute usage and billing.

Native AI/BI is the visualization product, distinct from Databricks Data Profiling. This Bundle does
not provision a separate native data profile. The older `quality_monitors` API is deprecated; new
integrations use `data_quality`. An InferenceLog profile can calculate model quality from prediction
and label columns, but it requires its own input table, refresh lifecycle and metric tables. It does
not replace this project's delayed-outcome matching, concrete training holdout references,
consecutive-window policy or guarded training requests. See
[the current Data Profiling API](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/create-monitor-api).

## What to edit

Setup asks `monitoring_catalog` and `monitoring_schema` in both quick and
detailed mode and writes the answers to `deployment/variables.yml`; see
[the setup questions](databricks_project_setup.md#monitoring-destination). To
set or change them per target, put this excerpt under
`targets.<target>.variables` in `deployment/targets.yml`. The values are
examples; use your existing catalog and a schema reserved for the shared store.

```yaml
monitoring_catalog: operations
monitoring_schema: model_monitoring
monitoring_enabled: "true"
```

### If no destination is set

A blank catalog at setup generates `monitoring_enabled: "false"`. Train and
score then run normally without enrollment, monitoring or a store. Nothing is
written to a monitoring schema. To enable monitoring later, set both names and
`monitoring_enabled: "true"`, then validate and redeploy.

| `monitoring_enabled` | Catalog and schema | Result at train/score |
| --- | --- | --- |
| `"false"` | Both blank | Runs without monitoring |
| `"false"` | Both set | Publishes a paused enrollment |
| `"true"` | Both set | Enrolls and monitors |
| `"true"`, or either value set alone | Not both set | Fails with `Set both monitoring_catalog and monitoring_schema.` |

Development targets skip this check and never monitor.

Every project uses the same independent
`monitoring_catalog` and `monitoring_schema` values. These control monitoring
storage only; they do not change model, training, source or prediction locations.
The catalog must exist. The first configured enrollment or enabled scheduled run creates the schema
and missing owned tables/views if its identity has the required permissions.
Later runs validate and reuse existing objects without replacing their data or
view definitions. Explicit `initialize_monitoring_store` remains the upgrade API
for deliberately refreshing view definitions.

When a release adds views, run that owner upgrade **before upgrading producer
jobs**. Bootstrap still needs CREATE permission for any missing object; a writer
with only SELECT/MODIFY fails with the platform's permission error until the owner
creates the missing views. There is no fallback to partially initialized storage.

| Object | First initialization | Later runs |
| --- | --- | --- |
| `model_inventory` (Delta) | Create if absent | Producer MERGE inserts/updates one environment/project/model identity. |
| `monitoring_results` (Delta) | Create if absent | Monitoring job inserts immutable observations; exact retries deduplicate. |
| `complete_observations` (view) | Create before dependent views | Exclude policy-only late-label reports from feature/serving selection; preserve failed and zero-row observations. |
| `current_health` (view) | Create if absent | Reuse; read inventory and latest applicable observation. |
| `monitoring_identities` (view) | Create after `current_health` | Include enrolled and historical model/version/context identities, including unresolved versions; latest measurement time counts full observations only. |
| `serving_windows` (view) | Create after `complete_observations` | Share enabled, current-configuration serving windows, error-class fallback and per-context/version ordering across charts. |
| `metric_history` (view) | Create if absent | Reuse; expand stored report metrics for dashboard queries. |
| `performance_actions` (Delta) | Create if absent | Append request and skip receipts without rewriting observations. |
| `retraining_requests` (Delta) | Create if absent with Serializable isolation | Retain request identities, submission receipts and a shared cooldown gate per training job. |
| `performance_history` (view) | Create if absent | Reuse; join performance evidence with the latest applicable action. |
| `monitoring_issues` (view) | Create if absent | Reuse; read current health and recorded quality, drift and active-policy issues. |
| `model_version_details` (view) | Create if absent | Reuse; read concrete version lineage and bounded metadata snapshots from existing reports. |

One store serves all enrolled models; inventory, observations and action history
are shared rather than recreated per model or score run. Spark reference
preparation additionally creates four immutable Delta tables under the configured
monitoring catalog/schema: `monitor_ref_<identity>_train`, `_source`, `_seen`
and `_receipt`. Their identity binds the enrollment and verified model/training
evidence. They preserve training populations, freshness evidence and holdout
metric metadata. They are prepared once for that identity and reused; a new
model/training identity can require new reference tables. Ordinary monitoring
observations append to the shared results table, not to new per-run tables.
Existing foreign objects or incompatible table schemas are rejected.
Store initialization sets inventory and retraining-request isolation to Serializable. Producer
registrations make at most three attempts (two retries) on optimistic Delta
conflicts; exhausted
conflicts and other errors remain visible. Each project monitoring job serializes
its own observations; independent projects share the same store. An inventory read
never writes its older snapshot back over another repository's configuration update.

## The command

For jobs without schema/table creation privileges, have the store owner run
`initialize_monitoring_store(spark, monitoring_catalog, monitoring_schema)` before
starting those jobs. This also prepares `retraining_requests`, so the first
automatic training request can use the existing table. Grant the runtime identity
the required catalog/schema access and table read/write privileges. Reusing a
compatible store requires no schema, table or view creation SQL. If any owned
table or view is missing, the next enrollment or monitoring run must create it,
so either rerun owner initialization first or give that run's identity CREATE
privileges. Normal runs refuse incompatible schemas or weak isolation.

Run this cloud-only Python example in a store-owner notebook after installing the Core requirement
that will serve the producers and dashboard. Define `monitoring_catalog` and `monitoring_schema`
with the destination selected above.

For an explicit schema/view upgrade, run `initialize_monitoring_store` after installing the updated
Core requirement and **before deploying or publishing the updated dashboard**. The dashboard reads
`complete_observations`, `monitoring_identities` and `serving_windows`; a dashboard refresh does not
create these views. In an owner-run notebook:

```python
from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
    initialize_monitoring_store,
)

namespace = initialize_monitoring_store(spark, monitoring_catalog, monitoring_schema)
for view in ("complete_observations", "monitoring_identities", "serving_windows"):
    spark.table(f"{namespace}.{view}").limit(0).collect()
```

Grant dashboard viewers SELECT on the new views, using the same access policy as
the existing monitoring views, then publish. The initializer checks ownership of
every object before changing any of them. Normal producer bootstrap creates only
missing views and never replaces existing definitions. The inventory and
observation table schemas and saved data stay compatible. Store bootstrap
creates `performance_actions` with the other tables. Any active performance
policy, in `report` or `retrain` mode, writes its actions there, so the
monitoring identity needs SELECT/MODIFY on `performance_actions`.

Projects on different Core versions share these views. Keep existing columns and
their meanings compatible across those versions; coordinate owner upgrades with
all consumers before changing a view contract. Owner-run monitoring can replace
view definitions using its installed Core version. Adding a result-table column
also requires reinitialization to refresh the views' stored column lists.
Chart labels and missing-value display text stay in the dashboard, so wording
changes do not require a store upgrade.

## What you should see

The initialization call returns the namespace, and all three `spark.table(...).limit(0).collect()`
checks finish without an error. Existing observations remain in place. Each later enrollment or
enabled observation reuses compatible objects. Monitoring metrics appear only after an observation
run; initialization does not calculate them.

The producer principal needs access to the central catalog/schema and
SELECT/MODIFY on `model_inventory` for enrollment. The score job's monitoring
task also needs SELECT/MODIFY on `monitoring_results`, read access to the saved
model/training/source/prediction/label data, and access to its optional MLflow
experiment. First bootstrap additionally requires permission to create the missing schema,
Delta tables and views. A complete existing store needs no recurring CREATE/ALTER
permission for bootstrap; normal observation/reference/action writes retain their
own grants. Foreign objects or incompatible schemas fail visibly.
The central monitoring principal needs inventory access, report writes, read access
to enrolled models/artifacts, original source snapshots, predictions and optional
labels, plus the monitoring MLflow experiment. These table grants assume trusted
producer repositories; they do not provide per-row tenant isolation.

Configure [the shared dashboard](databricks_monitoring_dashboard.md) after the store exists. Its
training tables also require `training_summary_json` in the owned `model_version_details` view.

## Common errors and fixes

| Symptom | Fix |
| --- | --- |
| Missing view or CREATE permission error after an upgrade | Run owner initialization before upgrading producer jobs; grant viewers SELECT on the refreshed views. Dashboard refresh does not install schema changes. |
| Foreign object, incompatible schema or weak isolation rejected | Have the store owner inspect the destination and use the explicit upgrade API. Do not overwrite an unrelated object. |
| No monitoring in a development target | Use a shared `mode: production` target; development deliberately bypasses enrollment and monitoring. |
| `Set both monitoring_catalog and monitoring_schema.` | Monitoring is enabled, or one name is set alone. Set both names for the target and redeploy, or set `monitoring_enabled: "false"` and clear both. |
| Train and score succeed but nothing is monitored | Setup left the catalog blank, so `monitoring_enabled` is `"false"`. Set both names and `monitoring_enabled: "true"`, then redeploy. |
| New reports contain NULL counts | Unmeasured values are unavailable. Keep measured zero separate. |

New `monitoring_results` rows store unmeasured counts as SQL `NULL`: this applies
to reference, current, scored and labeled rows, and drifted columns. An explicit
zero remains `0`. Failed or performance-only observations can omit these counts.
Historical rows and retries keep their saved values; upgrading does not backfill
old zeros. For custom queries over older reports, check `report_json` to distinguish
an absent count from a measured zero rather than replacing every `NULL` with `0`.

Each project keeps its own Core pin in `deployment/requirements.txt`. Projects on
different pins can share one store, so coordinate view contracts across all of
them before an owner upgrade.
