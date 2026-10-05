# Shared Databricks model monitoring

Independent model repositories register directly in a shared Delta inventory.
The central job reads that inventory; there is no models.json or export/copy step.
Models and their data retain their own catalog/schema locations.

## Choose the monitoring destination

The central Bundle and every producer Bundle use the same independent
`monitoring_catalog` and `monitoring_schema` values. These control monitoring
storage only; they do not change model, training, source or prediction locations.
The catalog must exist. The central initializer creates the schema if missing.

| Object | First initialization | Later runs |
| --- | --- | --- |
| `model_inventory` (Delta) | Create if absent | Producer MERGE inserts/updates one environment/project/model identity. |
| `monitoring_results` (Delta) | Create if absent | Central job inserts immutable observations; exact retries deduplicate. |
| `current_health` (view) | Create | Refresh definition; read inventory and latest applicable observation. |
| `metric_history` (view) | Create | Refresh definition; expand stored report metrics for dashboard queries. |
| `performance_actions` (Delta) | Create if absent | Append request and skip receipts without rewriting observations. |
| `performance_history` (view) | Create | Refresh definition; join performance evidence with the latest applicable action. |

One store serves all enrolled models; new models do not create new monitoring
tables. Existing foreign objects or incompatible table schemas are rejected.
The central initializer sets inventory isolation to Serializable. Producer
registrations retry optimistic Delta conflicts at most three times; exhausted
conflicts and other errors remain visible. Observation runs are serialized by
the central job. An inventory read never writes its older snapshot back over
another repository's configuration update.

## Deploy the central Bundle once

Build and upload the Skyulf wheel. Set Bundle variables `monitoring_catalog`,
`monitoring_schema`, `warehouse_id`, `wheel_path` and `experiment_name` through
an authenticated target or `--var` flags. No credentials are stored here.

1. From this directory, validate with `databricks bundle validate --strict -t dev`
   and deploy with `databricks bundle deploy -t dev`. Initially only the job is included.
2. Run `databricks bundle run monitoring -t dev`. An empty inventory is valid:
   this creates the store before producer projects register themselves.
3. Execute all dataset SQL queries from `src/monitoring.lvdash.json` on
   the chosen warehouse with its catalog/schema set to the central namespace.
4. Add `resources/monitoring.dashboard.yml` to the same Bundle's include list,
   validate and deploy again. Preserve its state and `monitoring_dashboard` key
   to update the same dashboard. Readers use their own credentials.

## Enable each independent model repository

In the generated model Bundle's target variables, configure:

```yaml
monitoring_catalog: operations
monitoring_schema: model_monitoring
monitoring_expected_interval_hours: "24"
# Optional shared MLflow experiment for the post-scoring task:
monitoring_experiment_name: /Shared/skyulf/monitoring
monitoring_dashboard_url: "https://<workspace-host>/dashboardsv3/<dashboard-id>/published"
# Optional actual results:
monitoring_label_table: outcomes.production.labels
monitoring_result_available_at_column: available_at
monitoring_performance_policies: '{}'
```

Performance policy is disabled by default. Each producer sets an independent
JSON mapping keyed by the exact fully qualified enrolled model name. For example,
the policy below reports a weighted F1 drop of at least 0.05 across three distinct,
completed windows for version 2:

```yaml
monitoring_performance_policies: >-
  {"models.risk.churn":{"mode":"report","metric":"f1_weighted",
  "direction":"higher","baseline":{"kind":"training_holdout",
  "model_version":"2"},"tolerance":0.05,"tolerance_mode":"absolute",
  "window_hours":24,"label_delay_hours":6,"minimum_labeled_rows":20,
  "minimum_label_coverage":0.8,"consecutive_windows":3}}
```

Use `mode: "retrain"` in the JSON policy to request guarded automatic training,
or `{"mode":"off"}` for an explicitly disabled model. An active policy needs
the label table and UTC availability column shown above. Select a
`training_holdout` baseline or pin an existing production observation with
`{"kind":"production_window","model_version":"2","report_id":"<64 hex characters>"}`.
The baseline version must match the observed version; a mismatch is unavailable
and cannot request training. No metric, direction, tolerance or baseline is
inferred. Fixed UTC windows wait `label_delay_hours` after their end, then count
only labels available at the cutoff. A 00:00–24:00 UTC window with a six-hour
delay can first be evaluated at 06:00 UTC the next day. Replays do not advance
the consecutive-window count; missing and immature labels remain unavailable.
Distribution drift remains controlled separately by `on_drift`. Its cooldown
and minimum new labeled training-row settings are shared request guards for
either retraining trigger. Model evaluation and approval rules still apply.

For an existing store, run the central initializer with the updated wheel before
updating the dashboard. This adds `performance_actions` and `performance_history`;
the inventory and observation table schemas stay compatible. Producers using
automatic performance retraining also need CREATE TABLE in the monitoring schema
when the action table is absent, and SELECT/MODIFY on `performance_actions`.

Monitoring is enabled by default in newly generated Bundles. Set the independent
catalog/schema before running them; use monitoring_enabled: "false" to opt out.
Existing projects without any monitoring settings retain their previous behavior.
When a destination is provided without an enabled flag, registration and
post-scoring observation are enabled. A configured false flag publishes a pause.

Keep model/data `catalog`, `input_schema`, `output_schema` and `metadata_schema`
unchanged. After a successful activation, the train job's `register_monitor` task
registers the version from the verified lifecycle result before scoring. Initial
activation, approval and rollback are supported; pending/rejected candidates do
not replace the active enrollment. Merely running `bundle deploy` does not
activate a model. A successful scoring or CDF-recovery task also registers its
actual model version and physical Delta prediction generation. It never resolves
an alias a second time. No-op scoring also repairs registration without writing
duplicate predictions. A registration error fails the task visibly; previously
committed predictions remain intact and their receipt makes scoring retries safe.
A pending recovery request does not register an output as successfully scored.

The producer principal needs access to the central catalog/schema and
SELECT/MODIFY on `model_inventory` for enrollment. The score job's monitoring
task also needs SELECT/MODIFY on `monitoring_results`, read access to the saved
model/training/source/prediction/label data, and access to its optional MLflow
experiment. It does not create central tables or views.
The central monitoring principal needs inventory access, report writes, read access
to enrolled models/artifacts, original source snapshots, predictions and optional
labels, plus the monitoring MLflow experiment. These table grants assume trusted
producer repositories; they do not provide per-row tenant isolation.

The inventory holds the latest enrollment for environment/project/fully qualified
model name. A newly scored model version updates that enrollment and keeps old
version observations in history. It appears never_observed until the central job
measures the new configuration. Removing deployment settings does not delete an
existing enrollment. To pause, retain the destination, set monitoring_enabled to
false and run scoring, or upsert a disabled MonitorConfig through enroll_monitor.
Do not edit only the duplicated config/digest fields by hand.

Automatic registration supports single models, competition winners and model-set
components. Model-set activation enrolls each component's concrete version before
scoring. Each component retains its parent set name/version/branch and reads its
own predictions from the shared physical output table. Use `all` or `separate_views`;
`combined_only` cannot supply individual model performance and is rejected before
scoring when monitoring storage is configured. No JSON model list is required.

## Measurement and operation

The central notebook reads a bounded inventory snapshot, default maximum 10,000
entries. It calls run_monitoring without a model list. The SDK can also enroll
explicit records in memory; it does not require a configuration file.

Only saved predictions are scored. Reference loading replays the original training
membership, and current features use the source snapshots recorded with predictions.
The runner materializes bounded pandas/Polars inputs under each enrollment's limits;
it is not Spark-native. Labels join by saved record keys and only count when their
UTC availability is at or before as_of. Missing labels or fewer than two finite
pairs leave performance unavailable; an unavailable value is not zero accuracy.

Performance reuses skyulf.modeling._evaluation.metrics. Classification includes
G-score, accuracy, balanced accuracy, MCC and precision/recall/F1; probabilities
also enable log-loss, ROC-AUC and PR-AUC variants. Regression includes MAE, MSE,
RMSE, R2, MAPE and explained variance. The job installs Core's optional G-score
dependency and compatibility pin. Feature drift uses Core DriftCalculator.

Generated score jobs run a visible `monitor_model` task after successful scoring
(or after the successful recovery/report branch). It measures only the saved model
version and exact scoring commit, writes to the shared results table, and fails
visibly after persisting a measurement error. A retry reuses that commit and does
not re-enroll an older configuration. No-op scoring retains an existing successful report; if a committed batch has
no successful measurement yet, it completes that missing observation instead.
An exact successful monitor-task repair is also skipped, preserving its evidence. Drift itself is an observed result, not a task failure.

Generated score jobs expose `score -> monitor_model -> drift_report` (with the
existing recovery/report branch before monitoring when enabled). `monitor_model`
calculates drift, quality and available performance; `drift_report` displays the
saved batch's feature values and limits without repeating those calculations.
Both task outputs link to `monitoring_dashboard_url` when configured. Independent
model repositories use the same shared dashboard URL. Existing deployed projects
need updated producer templates/notebooks and the wheel followed by redeployment.

The central inventory-wide example remains manual; schedule it with a suitable
observation window for delayed label arrival and periodic checks. Refreshing the dashboard does not compute
new metrics. Each model failure is saved, other models are attempted, and the
notebook then fails if any model failed. Inspect reports before retrying.

Observation identity includes configuration, cutoff, window and snapshot evidence.
Exact retries do not overwrite history. Current health ranks the newest window,
then its latest measurement; historical backfills cannot replace newer windows.
Keep the Delta retention needed for source replay and audit history.

| Status | Meaning |
| --- | --- |
| healthy | Measured drift/quality checks passed; model accuracy is not an acceptance gate. |
| drift | Distribution or declared feature-schema checks detected drift. |
| stale | The scoring commit is older than expected_interval_hours, regardless of the last monitoring run. |
| never_observed | No observation exists for the current enrollment configuration. |
| degraded | Quality/measurement issues, no usable predicted outputs, or a nonfinite calculated metric. |
| failed | Monitoring raised a saved error. |
| no_data | The selected reference or current population is empty. |
| disabled | Enrollment explicitly paused. |
| unavailable | A result exists but scoring time is unavailable. |

Counts represent enrollments, not prediction rows. measurement_status retains the
underlying measured verdict when freshness changes. No retraining, performance
acceptance thresholds or notifications are enabled by this example.

## Dashboard layout

Catalog, schema, model and version filters select the models shown on each page.
Environment/project remain part of enrollment identity but are not displayed as
dashboard filters or table columns. No monitoring tables change for this layout.

The drift/quality page keeps two tables and adds two charts for one selected
model/version: the latest observation's top ten feature PSI values beside each
feature's own limit, and feature counts with drift or unavailable checks over
measurement time. Each feature is counted once even if multiple checks flag it;
the count chart uses full monitoring date/time and the SQL session timezone,
rather than ambiguous abbreviated hour labels. Monitoring time is when checks
were calculated; scoring time is when predictions were committed.
KS p-value is excluded from decisions. Unavailable checks are shown separately,
and missing values never become zero. PSI is only one drift check; the counts
and table include other distribution and schema checks too.

Drift columns show each
metric's value / its own limit. KS p-value is diagnostic only; it is never divided
by the KS statistic's threshold. Missing features and changed types stay visible.
Quality rows identify model version, scoring time and feature, with separate
missing/nonfinite percentages. Both tables retain observation history.

Performance explains the number of predictions with matched usable true target
values. No configured label source means no measured performance, not zero
accuracy. A target in an arbitrary source is not discovered automatically: use
monitoring_label_table with the saved record keys and target column, and declare
the result-availability timestamp. That table can be the scoring source when it
already satisfies this contract. Fewer than two eligible pairs cannot be scored.

Performance metrics appear as dynamic pivot columns, so selecting a regression
model shows its regression metrics, while classification shows its available
metrics, including G-score and AUC. Same-time reports get deterministic observation
suffixes so the pivot cannot merge distinct reports/monitors. Blank cells remain
unavailable, never zero. The outcome-count table explains which rows were used.

Performance pairs a latest-observation bar chart with a time series for one
selected model, version and metric. Both use the same metric and units; their
legends identify whether higher or lower is better. The bar remains useful
when only one observation exists. A latest unavailable measurement stays blank,
rather than falling back to an older measured value.
Select model/version before viewing the portable example; the live demonstration
has sample selections. A single observation does not establish a trend. The chart
does not combine different monitor identities for the same model/version. Metric
selection does not remove columns from the performance table. Latest-observation
charts choose the latest saved report before applying date/feature/target filters;
those filters can hide its bars but do not choose an older report. The PSI chart
shows the top ten features before feature filtering; use the tables for the rest.
Dashboard refresh
reads stored results and does not trigger new monitoring calculations.

## File responsibilities

Library files live under skyulf/integrations/databricks/:

| File | Responsibility |
| --- | --- |
| monitoring_config.py | Validated model/table identity, version and bounded policy. |
| monitoring_registration.py | Activation/scoring enrollment and immutable scoring handoff. |
| monitoring_tasks.py | Visible activation enrollment and per-commit scoring observation tasks. |
| monitoring_output.py | Saved-batch drift report and safe shared-dashboard navigation in task output. |
| monitoring_reference.py | Registered artifact and original training-reference replay. |
| monitoring_sources.py | Saved predictions, source snapshots, receipts and delayed labels. |
| monitoring_metrics.py | Core metric adapters and finite keyed reports. |
| monitoring_store.py | Owned Delta objects, concurrent registration, inventory reads and report history. |
| monitoring.py | Inventory-driven observations, failure isolation and MLflow reports. |
| job_runtime.py / scoring_recovery.py | Invoke enrollment after successful scoring/recovery. |
| model_set_project.py / monitoring_model_set.py | Enroll pinned components after activation and scoring. |

In this example, databricks.yml selects central storage and compute;
resources/monitoring.job.yml defines the runner and dependencies;
src/monitoring_notebook.py calls the library; resources/monitoring.dashboard.yml
binds storage; src/monitoring.lvdash.json defines SQL, filters and three pages.
Generated model deployment variables and score tasks pass independent monitoring
settings. Their README documents default-enabled registration and pause behavior.

The test_monitoring_config/reference/sources/metrics/store/job/registration/dashboard
files cover the corresponding library contracts. test_monitoring_template runs
real CLI generation across layouts and recovery branches. The monitoring tests
also cover Core metrics in both engines, bounded reads, concurrent registration
retry, no-op repair, original source snapshots and delayed labels.

Activation enrollment is ordered by the original serialized lifecycle run start.
A delayed score cannot replace an activated different version, and repairing an
older activation cannot replace a newer activation. Enrollment-only repairs read
the verified original receipt without repeating training or alias changes.
The private activation-order metadata lives in inventory config_json; table
schemas and model configuration digests are unchanged. Deploy the updated central
monitoring runner together with model jobs so inventory readers understand it.
