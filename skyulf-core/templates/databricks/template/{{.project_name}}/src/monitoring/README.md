# Shared monitoring in the existing project template

The canonical dashboard definition is `src/monitoring/monitoring.lvdash.json`
inside the existing Databricks project template. This is not a separate Bundle
or another project template. Single, competition and model-set projects write to
one configured store and display their evidence in one shared dashboard.

Choose one project/target to own the dashboard resource. All other projects reuse
its published URL. An existing dashboard can remain under its current deployment
ownership; using its URL does not create a copy. Keep its dashboard ID and Bundle
state when migrating ownership; never deploy a second owner against it.

## What is automatic?

| Operation | Trigger and effect |
| --- | --- |
| Measure drift and performance | Successful scoring calls the separate Spark monitoring job when monitoring is configured and enabled. A production-mode target is required; development targets bypass shared monitoring. |
| Revisit delayed outcomes | Unpause the monitoring job's schedule after setup. The generated schedule starts paused. |
| Refresh dashboard results | The configured native Dashboard task reruns published datasets after monitoring completes. It does not calculate new monitoring metrics. |
| Request training | Drift or performance loss can qualify independently when its own policy enables retraining. Shared guards must also pass. |

Changing training/scoring compute to a cluster does not enable monitoring, change
its storage destination, or enable automatic retraining. The generated monitoring
job has its own serverless environment. Job runs and attributed billing appear
after Databricks system telemetry becomes available. CPU/RAM cards and trends use
classic cluster node samples during the selected job; serverless CPU/RAM is not
exposed by these system tables and stays unavailable, never zero.

Native AI/BI is the visualization product, distinct from Databricks Data Profiling.
This Bundle does not provision a separate native data profile. The older
`quality_monitors` API is deprecated; new integrations use `data_quality`.
An InferenceLog profile can calculate model quality from prediction and label
columns, but it requires its own input table, refresh lifecycle and metric tables.
It does not replace this project's delayed-outcome matching, concrete training
holdout references, consecutive-window policy or guarded training requests.
See [the current Data Profiling API](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/create-monitor-api).

The native [Dashboard job task](https://docs.databricks.com/aws/en/jobs/tasks/dashboard)
runs after the existing monitoring job has saved its evidence and evaluated
retraining. Configure it in each project as described below. Development or
disabled monitoring skips refresh. No email subscription is created.

Updating source data does not automatically invalidate the
[dashboard result cache](https://docs.databricks.com/aws/en/dashboards/caching).
The task now performs the explicit refresh after successful monitoring. It runs
as the job identity and preserves `embed_credentials=false`; other viewers keep
their own data permissions and caches. This is automatic result refresh, not a
continuously pushed update of every already-open browser tab. A separate
dashboard schedule can refresh independently of monitoring if desired.

## Choose the monitoring destination

Every project uses the same independent
`monitoring_catalog` and `monitoring_schema` values. These control monitoring
storage only; they do not change model, training, source or prediction locations.
The catalog must exist. The first configured enrollment or enabled scheduled run creates the schema
and missing owned tables/views if its identity has the required permissions.
Later runs validate and reuse existing objects without replacing their data or
view definitions. Explicit `initialize_monitoring_store` remains the upgrade API
for deliberately refreshing view definitions.

| Object | First initialization | Later runs |
| --- | --- | --- |
| `model_inventory` (Delta) | Create if absent | Producer MERGE inserts/updates one environment/project/model identity. |
| `monitoring_results` (Delta) | Create if absent | Monitoring job inserts immutable observations; exact retries deduplicate. |
| `current_health` (view) | Create if absent | Reuse; read inventory and latest applicable observation. |
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
registrations retry optimistic Delta conflicts at most three times; exhausted
conflicts and other errors remain visible. Each project monitoring job serializes
its own observations; independent projects share the same store. An inventory read
never writes its older snapshot back over another repository's configuration update.

For jobs without schema/table creation privileges, have the store owner run
`initialize_monitoring_store(spark, monitoring_catalog, monitoring_schema)` before
starting those jobs. This also prepares `retraining_requests`, so the first
automatic training request can use the existing table. Grant the runtime identity
the required catalog/schema access and table read/write privileges. Reusing a
compatible store requires no schema, table or view creation SQL. For older stores,
rerun owner initialization to add the missing request table and upgrade its
isolation if necessary; normal runs refuse incompatible schemas or weak isolation.

## Configure the one shared dashboard

Set the same `monitoring_catalog` / `monitoring_schema` in all production targets.
No separate monitoring template or setup Bundle is required. The first activation,
scoring registration or scheduled monitoring run bootstraps missing objects.
Initialize before validating the dashboard SQL, because its queries need the views.

For an **existing shared dashboard**, use its URL in each project:

```powershell
python src/tools/configure_monitoring_dashboard.py --target test --dashboard-url "https://<workspace-host>/dashboardsv3/<dashboard-id>/published" --warehouse-id <warehouse-id>
databricks bundle validate --strict -t test --profile <profile>
databricks bundle deploy -t test --profile <profile>
```

For a **new shared dashboard**, run this in only the one owning project/target:

```powershell
python src/tools/configure_monitoring_dashboard.py --target test --create-dashboard --warehouse-id <warehouse-id>
# Validate every dashboard dataset against the chosen monitoring store first.
databricks bundle validate --strict -t test --profile <profile>
databricks bundle deploy -t test --profile <profile>
```

The helper generates `resources/monitoring.dashboard.yml` and
`resources/monitoring_dashboard.job.yml`. The owner URL and refresh task reference
the same managed `monitoring_dashboard` resource. It accepts one owner target and
refuses to overwrite operator-managed files or silently move ownership. Other
projects use the published URL from this deployment, not `--create-dashboard`.
`--disable` removes only the refresh overlay and preserves the dashboard resource.
No monitoring-engine question is added to initialization; measurement uses Spark.

Dashboard deployment is configuration, not metric calculation. Run monitoring to
save new measurements, then the native task refreshes the published datasets.
The dashboard retains the selected store across all project layouts.

## Investigate a model or version

Start on **Overview → Issues to investigate**. The table uses native pagination, with failures first, then recorded
quality/drift/performance issues, then missing or stale evidence. The counter gives the full filtered issue count, including rows on later pages. One model can have multiple issues; this is an issue count,
not a model count. Use shared Model, Monitored version and Context filters to narrow
the list. Disabled monitoring and disabled performance policies do not create
new failures. Never-observed, no-data, failed, stale and insufficient-label
results keep their distinct recorded reasons. Missing metric values remain empty.

Open a concrete **Model version** link for the registry entry, or **Evidence run**
for the monitoring run that recorded that issue. The latter is separate from the
owning training run. A missing link means its exact run/experiment identity was
not recorded or is unavailable; it does not mean the issue is healthy.

Use the **shared filter panel** for Model, Monitored version and Context.
These choices persist across Overview, Training and lineage, Drift and quality,
Model performance and Online serving. Model names include catalog and schema.
Drift and Performance include every matching identity, selecting its latest
evidence independently. Execution and compute has a separate Workspace/Job scope.
The full page scope table and optional controls are described under Dashboard layout below.

**Training and lineage** shows every identity matching the shared filters.
Choosing All models / All versions / All contexts shows all available identities; an obsolete or incompatible
selection leaves empty tables and never substitutes the latest different model.
Each table includes its model, version and context. Historical observed versions
retain their concrete identities, while NULL versions explicitly say unresolved.
The page shows creation time, owning training run, recorded algorithm, artifact
digest and available code evidence. Open **Training run** to inspect that run's
native Inputs and artifacts. A source commit is shown only when the owning run
already records `mlflow.source.git.commit`; a project Python source digest is
shown only when the saved artifact records one.

**Aliases at capture** describes the concrete version's alias metadata when the
monitoring reference was loaded, alongside **Captured at**. It does not describe
historical deployment identity or guarantee today's registry alias assignment.
New observations refresh the bounded snapshot in existing `evidence_json`.
Retrying the same observation preserves the first saved snapshot and does not
invalidate prepared Spark references. Dashboard refresh only rereads saved
evidence; it makes no per-row registry calls and does not move aliases.

Older reports can show **not_recorded**, never-observed enrollments have no
training metadata, and optional metadata permission failures show **unavailable**
with a reason. Missing creation/code/alias fields remain unavailable rather than
inferred. Existing store owners can run `initialize_monitoring_store` explicitly
to refresh owned view definitions; ordinary runs create missing compatible views
without replacing existing ones. Viewers retain their own data permissions.

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
monitoring_label_table: labels.production.outcomes
monitoring_result_available_at_column: available_at
monitoring_performance_policies: >-
  {"models.risk.churn":{
    "mode":"report",
    "metric":"f1_weighted",
    "direction":"higher",
    "baseline":{"kind":"training_holdout","model_version":"2"},
    "tolerance":0.05,
    "tolerance_mode":"absolute",
    "window_hours":24,
    "label_delay_hours":6,
    "minimum_labeled_rows":20,
    "minimum_label_coverage":0.8,
    "consecutive_windows":3}}
```

These are example values, not production defaults. Put them under the producer
Bundle's `targets.<target>.variables`, alongside the monitoring destination.
Replace the model and label table names with your own fully qualified names.
The label table must expose the saved record keys, the model's target column
and the UTC timestamp when the actual outcome became available. `available_at`
is outcome availability, not the prediction time or the job execution time.

| Setting | Meaning in this example |
| --- | --- |
| `monitoring_label_table` | Actual outcomes joined to original predictions by stable record keys. Predictions are never used as actual outcomes. |
| `monitoring_result_available_at_column` | Outcome availability timestamp used to exclude labels unavailable at the observation cutoff. |
| `models.risk.churn` | Exact enrolled model name; a component model name for a model-set. |
| `mode: report` | Persist and display the performance decision without requesting training. |
| `metric: f1_weighted` | Class-support-weighted F1 over eligible prediction/label pairs; this does not enable training sample weights. |
| `direction: higher` | A decrease is worse. Regression MAE uses `lower`, where an increase is worse. |
| `baseline` | Saved training holdout for concrete model version `2`; production-window references are also supported. |
| `tolerance: 0.05`, `absolute` | A baseline of 0.84 is degraded at 0.79 or below: five percentage points, not a five-percent relative drop. |
| `window_hours: 24` | Measure each completed, fixed UTC day independently. |
| `label_delay_hours: 6` | Wait six hours after the window ends before assessing it. This is independent of the job schedule. |
| `minimum_labeled_rows: 20` | Require at least 20 eligible labeled predictions in that window. |
| `minimum_label_coverage: 0.8` | Require eligible labels for at least 80 percent of the window's prediction population; both minimums must pass. |
| `consecutive_windows: 3` | Require three adjacent eligible failing windows for trigger eligibility; replaying one window does not count again. |

An individual window can already be reported as degraded before the required
streak is reached. `report` never requests training, even after three failures.
With `retrain`, reaching the streak only makes the performance trigger eligible;
fresh training data, cooldown, active-run and duplicate-request guards still apply.

Policy reports retain the required window count even when labels, a baseline or
a source snapshot are unavailable. For example, a three-window policy reports
`0 / 3` after a failed observation; that failure does not count as measured
degradation. Older reports without a saved required count show `Unavailable` in
the notebook's failure-streak denominator instead of an invented zero.

To enable performance-driven retraining, change only the relevant model's policy
from `"mode":"report"` to `"mode":"retrain"` in the producer target variables,
retain its metric, matching baseline and evidence requirements, then validate and
deploy that producer Bundle. Run activation registration or successful scoring to update enrollment, then
run monitoring to evaluate the next eligible window. This is a job configuration change; there is
no dashboard switch that starts training. A policy change does not turn an old
report into a new eligible request or bypass consecutive-window requirements.

| Project layout | Policy scope |
| --- | --- |
| Single | One policy for the active concrete model version. |
| Competition | Policy for the activated winner; losing candidates are not production monitors. |
| Model-set / multi-target | A separate mapping entry and matching baseline for each component model; each keeps its own target, metric and version. |

The producer's label table must contain the target columns needed by its enrolled
components. A retraining request runs the configured project train job; it does
not introduce selective retraining of just one model-set branch.

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

For an explicit schema/view upgrade, run `initialize_monitoring_store` after installing the updated Core requirement
and before updating the dashboard. This adds `performance_actions` and `performance_history`;
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
experiment. First bootstrap additionally requires permission to create the missing schema,
Delta tables and views. A complete existing store needs no recurring CREATE/ALTER
permission for bootstrap; normal observation/reference/action writes retain their
own grants. Foreign objects or incompatible schemas fail visibly.
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

### Spark execution in generated projects

Generated producer Bundles always use Spark monitoring; no engine selection is
required. Scoring validates its exact receipt, then a native Run Job task calls
the independent third `monitoring` job and waits for its result. Training can call
scoring through the same native job hierarchy. Monitoring uses its own serverless
environment, timeout and retry settings, and a UTC schedule that starts `PAUSED`. Prepare and
verify references before unpausing it. The scheduled run observes enrolled models
and delayed labels even if no new scoring batch arrives. Its tasks run
`monitor_model -> monitoring_report -> evaluate_retraining`; a guarded policy may submit
the producer's complete train job.

With monitoring configured and enabled, every successful scoring receipt reaches
the dedicated monitoring job. Failed scoring and missing saved batches do not
create a successful observation. Dashboard refresh reads saved results; it does
not run monitoring and an already open page is not a push stream. Use the
dashboard's refresh control to load new observations; reloading the browser can
still reuse cached query results. Scheduled monitoring covers late labels even
when scoring has not run again.

`monitoring_report` displays feature drift, observed prediction/outcome metrics
and the separate performance-loss policy. A measured RMSE or F1 value alone does
not establish degradation: the policy also requires a valid baseline, completed
window, label coverage and configured tolerance. The policy population can differ
from the exact scoring batch shown under observed performance.
`healthy` means the selected metric has not degraded beyond its configured
tolerance; it does not establish that the model meets an absolute quality goal.

| Eligible signal | Retraining decision |
| --- | --- |
| Neither drift nor performance loss | No training request. |
| Drift only, with `on_drift: retrain` | One request, subject to shared guards. |
| Performance loss only, with policy `mode: retrain` | One request, subject to shared guards. |
| Both signals eligible | One combined request, not two training jobs. |
| Missing labels, unavailable metrics or policy `mode: report` | No performance-driven training request; drift remains independently eligible. |

Shared guards still require usable input, fresh training data, cooldown and no
active or duplicate request. Older generated projects may still call the
compatible `drift_report` and `retrain_on_drift` notebook entrypoints.
New templates omit those unused wrappers; their library functions remain
available to already deployed projects. Historical run graphs retain their old
task names after a job is redeployed. Open the current job or a new run to see
`monitoring_report` and `evaluate_retraining`.

The two retraining opt-ins form an OR decision, not an AND requirement:
`on_drift: retrain` enables drift; a model's performance policy `mode: retrain`
enables performance loss. Setting only one leaves the other disabled. For
performance, eligible labels, a matching baseline, tolerance and the configured
consecutive-window count must pass first. Missing labels, an unavailable metric,
`mode: report`, or a historical backfill cannot independently start training.
Both eligible signals are combined into one idempotent project training request.
The resulting candidate still goes through the existing evaluation, promotion
and approval rules; detecting a change does not itself approve a new model.

`monitoring_revisit_windows` controls how many completed performance windows are
checked for late labels. It defaults to `3`, includes the latest window, and
accepts 1 to 100. Historical windows can update saved evidence, but do not
independently submit training. Spark joins saved predictions, pinned feature
snapshots and eligible actual outcomes, then writes durable Delta observations.
Ongoing Spark measurements do not use the local one-million-row monitoring cap.

Spark computes regression and confusion-matrix metrics from full-population
aggregates, and probability AUC/AP from exact tied-score counts. Numeric drift
uses exact empirical-CDF KS/Wasserstein effects and reference percentile bins;
rows are not sampled. Ordered CDF and probability curves use an ordered Spark
window over distinct values, which can concentrate work on one executor for
high-cardinality features. Removing the local row cap does not remove compute,
shuffle or executor-memory requirements.

Statistical evidence is bounded separately: KS uses the exact lattice calculation
when the two population sizes multiply to at most 1,000,000, the Core asymptotic
method when either size exceeds 10,000, and explicitly named
`ks_dkw_union_bound` evidence in the remaining range. An inconclusive bound is
unavailable rather than evidence of no drift. Categorical summaries allow at
most 1,024 current categories and use Core's 50-category reference limit;
classification allows at most 256 saved classes. Temporal/nested feature drift
and unsupported cardinalities are explicit failures or unavailable checks.
These limits bound summaries and supported algorithms, not observation rows.

Reference preparation is a separate, one-time activation or explicit migration
step. It replays the original training source under the existing bounded training
row and byte budget, validates the original split and writes pinned reference
populations. Models without `monitoring_source_evidence.json` need retraining or
verified original source proof; the job never silently reconstructs provenance.
Spark freshness checks fail explicitly for unsupported fixed normalizers, custom
pre-split filters and custom weight generators. Supported source weight columns
still obey the training contract. Spark monitoring does not change the training
or scoring engine.

Databricks AI/BI presents the durable Skyulf inventory and result tables. It is
not a second metric calculator. Databricks native profiling and its generated
dashboard are separate from these Skyulf observations; see
[Databricks dashboards](https://docs.databricks.com/aws/en/dashboards/).

### Library API and legacy local execution

The library can read a bounded inventory snapshot, default maximum 10,000
entries. The SDK can enroll explicit records and run inventory-wide observations;
it does not require a configuration file or a second template.

Only saved predictions are scored. Reference loading replays the original training
membership, and current features use the source snapshots recorded with predictions.
Legacy local enrollments materialize bounded pandas/Polars inputs under their
limits; Spark enrollments use distributed readers. Labels join by saved record
keys and only count when their UTC availability is at or before as_of.
Missing labels or fewer than two finite
pairs leave performance unavailable; an unavailable value is not zero accuracy.

Local performance reuses skyulf.modeling._evaluation.metrics. Classification includes
G-score, accuracy, balanced accuracy, MCC and precision/recall/F1; probabilities
also enable log-loss, ROC-AUC and PR-AUC variants. Regression includes MAE, MSE,
RMSE, R2, MAPE and explained variance. The job installs Core's optional G-score
dependency and compatibility pin. Local feature drift uses Core DriftCalculator;
Spark enrollments use distributed metric and drift aggregations.

The public local API and older deployed inline notebooks remain compatible.
`MonitorConfig.max_rows=10000` is their default observation budget;
`MAX_MONITOR_ROWS=1_000_000` is the local safety ceiling. Neither limits Spark
observation rows. These are not two separate Spark limits.

New generated score jobs use `prepare_monitoring -> monitoring_ready -> monitor_model`,
where the final node is the native child-job call. Calculation, saved drift reports
and guarded retraining belong to that child job. All projects can link to the same
`monitoring_dashboard_url`. Existing projects need reviewed template/notebook and
Core requirement updates followed by redeployment. The older asynchronous API's
`monitoring_invocation_id` identifies a producer run for duplicate suppression;
the new native job graph does not expose or require that setting.

For custom inventory-wide SDK runs, select a suitable observation window.
The inventory limit bounds enrolled
models, not prediction rows. Refreshing the dashboard does not compute new
metrics. Each model failure is saved, other models are attempted, and the
notebook then fails if any model failed. Inspect reports before retrying.

Observation identity includes configuration, cutoff, window and snapshot evidence.
Exact retries do not overwrite history. Current health ranks the newest window,
then its latest measurement; historical backfills cannot replace newer windows.
Keep the Delta retention needed for source replay and audit history.

| Status | Meaning |
| --- | --- |
| healthy | Measured drift/quality checks passed; model accuracy is not an acceptance gate. |
| drift | Distribution or declared feature-schema checks detected drift. |
| stale | The latest scoring commit (batch) or captured request (online) is older than expected_interval_hours, regardless of the last monitoring run. |
| never_observed | No observation exists for the current enrollment configuration. |
| degraded | Quality/measurement issues, no usable predicted outputs, or a nonfinite calculated metric. |
| failed | Monitoring raised a saved error. |
| no_data | The selected reference or current population is empty. |
| disabled | Enrollment explicitly paused. |
| unavailable | A result exists but the scoring or captured-request time is unavailable. |

Counts represent enrollments, not prediction rows. measurement_status retains the
underlying measured verdict when freshness changes. No retraining, performance
acceptance thresholds or notifications are enabled by the dashboard.

## Dashboard layout

Use the **shared filters** on the left: **Model**, **Monitored version** and
**Context**. Model names include catalog and schema, so there are no separate
catalog/schema controls. Context includes environment, project and a short monitor
ID. The selection persists between pages; there are no competing model controls
inside the pages. Date, feature, metric, target and job controls refine only the
page where they appear.

**All models**, **All versions** and **All contexts** are the initial selections.
Drift and Model performance include every matching model/version/context, with
the latest result selected independently for each identity. Disabled policies and
identities without measurements remain visible. Choose Model, Version and Context
to narrow the portfolio. Online serving lists matching endpoint contexts separately.
Incompatible explicit selections stay empty; clear Version and Context when
switching models. An explicitly selected model
never falls back to another model.

| Page | Scope | What to use it for |
| --- | --- | --- |
| Overview | All models with the initial All selections; otherwise all matching registrations. | Four summary counts, prioritized issues, current status distributions and daily trends. |
| Training and lineage | All matching model/version/context identities. | Compare concrete versions, owning training runs and recorded provenance. |
| Drift and data quality | Every matching model/version/context, with its own latest observation. | Separate feature PSI comparisons and drift histories, with detailed quality measurements. |
| Model performance | Every matching model/version/context, including disabled and unmeasured policies. | Per-identity prediction/label coverage, metric history, policy comparisons and classification evidence. |
| Online serving | All matching serving enrollments, separated by model/version/context; choose Model to narrow them. | Distinguish monitored and served versions; inspect captured counts and latency per observation in tables and three window charts. |
| Execution and compute | The explicitly selected workspace and job, independently of model filters. | Job runs, estimated attributed cost and available compute telemetry. |

Overview puts the issue list before trends. Four counters show distinct registered
models, monitoring registrations, issue rows and distinct scoring-input tables.
The issue count can exceed the registration count because one observation may
have several issues. The model/table mapping includes prediction and optional
label tables; shared tables count once. Two compact status distributions show
feature/data health and performance-policy status separately, including missing
evidence and disabled registrations. These are current registration counts, not
prediction rows or daily totals.

Two line charts show daily drift and performance-policy checks. Each counts one
enrollment per UTC window day using the latest saved window, then measurement and
report ID to break ties. They include only the current configuration/version;
missing or failed evidence stays unavailable. Unrecorded days are gaps, not zero
checks or carried-forward healthy models. These are daily recorded checks, not
historical snapshots of fleet health.

Healthy feature/data status does not mean accuracy met its baseline. The
performance distribution compares each enrollment's current configuration and
concrete version against its latest saved policy window. Within tolerance and
Performance loss require usable, sufficiently recent evidence. Insufficient
evidence includes missing checks, unavailable labels/baselines and stale windows.
Policy disabled and Monitoring disabled remain distinct states.

Freshness uses the latest completed UTC policy window after `label_delay_hours`:
a window becomes stale when it trails that boundary by more than
`expected_interval_hours`. Future or immature windows cannot count as success.
The dashboard does not initiate training. Retraining retains its independent
opt-in and request guards; historical losses remain visible even when the latest
window has insufficient evidence.

The drift/quality page keeps two history tables and separates charts by
model/version/context. Each identity shows its latest observation's top ten
feature PSI values beside their own limits, plus feature counts with drift or
unavailable checks over time. Each feature is counted once within its observation
even if multiple checks flag it. The count axis uses whole numbers; compact UTC date/time labels keep
the chart readable. Hover to see the full window and calculation timestamp.
For example, data from 09:00 to 10:00 appears at 10:00 even if the checks ran at
11:30. Drift and Performance date filters use the same window end as their
time charts. Calculation time remains available separately in the details.
The matching-identity table shows which contexts the shared filters include.
The full model/table inventory remains on Overview.

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
unavailable, never zero. **Latest prediction coverage by model** replaces the
three shared counters with one row per model/version/context. Each row shows
predictions, matched outcomes and label coverage from that identity's latest
full observation. A failed latest measurement clears its counts rather than
reviving an older success. Counts and coverage are never aggregated across
models, contexts or observation windows.

The performance page starts with matching identities, then policy results and
policy history. Prediction coverage and observed metrics follow below.
**Latest policy result for every model** gives
each identity its own row with metric, baseline, current score, allowed limit,
improvement direction, result, explanation and training action. Accuracy, F1 and
RMSE policies can coexist, with separate model/metric chart panels and independent
scales. The policy charts use each identity's latest saved policy metric;
selecting MAE in **Chart metric** does not hide an RMSE policy. Current is the
observed metric, Baseline is its reference value, and Tolerance boundary is the
permitted limit.
An accuracy baseline of 0.96 with an absolute tolerance of 0.10 gives a lower
limit of 0.86. For an error metric, exceeding the upper limit indicates loss.
Missing scores are unavailable and must not be interpreted as zero.
The latest policy summary ranks by
window end before measurement time for each identity: a late historical degradation
cannot replace a newer unavailable window. Disabled policies are explicitly
labeled disabled, and identities without a saved check show No policy evidence.
The Performance checks table is the history of those per-window comparisons.
Performance result means Within tolerance, Performance loss, Insufficient evidence
or Policy disabled. Training action is separate: loss can be report-only, awaiting
further checks, skipped or actually requested. Eligible is not already submitted.
Full reasons, source references and request IDs remain in `performance_history`
and the job's saved reports. Classification
matrix availability appears below the confusion matrix; the full metric matrix
follows below.

Performance pairs a latest-observation metric table with time series in separate
model/version/context and metric panels. Each panel keeps its own units and
scale; legends identify whether higher or lower is better. The table identifies
each metric, value, target and window even when only one observation exists.
A latest unavailable measurement stays blank,
rather than falling back to an older measured value.
The metric selector defaults to an available MAE, weighted F1, accuracy or other
saved metric independently for each matching context. Selecting a metric narrows
the observed table and time chart to identities with that metric. A single observation does not
establish a trend. The charts never combine different monitor identities or windows.
Metric selection does not remove columns from the full metric-history table.
Latest-observation views choose the latest saved report before applying
date/feature/target filters; those filters can hide its values but do not choose
an older report. The PSI chart
shows at most ten features per identity before feature filtering; use the tables
for the rest.
Dashboard refresh
reads stored results and does not trigger new monitoring calculations.

The PSI matrix includes numeric and categorical features. Each cell divides a
recorded PSI by its own recorded limit; values above 1 receive a fixed highlight.
An absent or nonpositive limit leaves the cell blank. Each identity retains its
latest 12 full observations and the 10 features with the largest ratios within
that history. Feature and date filters narrow this retained set; they do not
retrieve older observations. Observation labels include the year and a sequence
number, so separate reports for the same window remain separate cells.

The missing-value matrix uses the same observation and feature bounds, ranking
features by their largest missing fraction. Its reference rate is the reference
dataset's missing fraction, **not an alert limit**. A recorded quality issue can
depend on a statistical test or unscored inputs. The tooltip identifies that saved
decision. Nonfinite fractions remain available in the quality table.

The scoring-volume chart uses one stacked bar per full observation: labeled
rows and rows not labeled at measurement add up to scored rows. Gray identifies
observations without measured label coverage. The tooltip shows coverage as a
percentage. Counts absent from the original report stay blank even if an older
store row defaulted them to zero. Failed and policy-only reports never invent
scoring counts; bars from different or overlapping windows must not be added.

The policy-margin comparison uses each identity's latest saved policy window,
independently of the chart metric and history date filters. Remaining tolerance is
the direction-aware distance from the saved limit divided by the absolute
baseline-to-limit distance: 1 is at baseline, 0 is at the limit and negative values
are beyond it. Bar color follows the saved policy verdict, including boundary
equality rules. Unavailable, disabled or zero-tolerance policies stay in the
policy table without a bar. Accuracy and error metrics can therefore share this
normalized comparison without mixing their original units.
The **Policy status** legend distinguishes blue **Within tolerance** from red
**Performance loss**. Rows without a calculable margin remain in the policy
table and are excluded from this chart and its legend.

Latest PSI bars and the observed-metric table use the newest saved report for each context,
ranked by observation window and measurement time. A newer empty or failed report
does not resurrect an older measured value. Historical trend charts retain history.
The drift context summary exposes the newest report status even when charts are empty.
Mature performance-window-only reports stay in policy history; they do not replace
the full feature observation, current model health or confusion matrix.

The confusion-matrix pivot separates model/version/context first, then shows
actual classes on rows and predictions on columns from each identity's latest
report. Counts from different identities are never combined. Numeric class indices
preserve identity when display labels collide. Measured zeros are real counts; unavailable
classification, regression and historical reports without a stored matrix show an
explicit status. The matrix is independent of history date and target filters.

### Execution and compute

This page has its own Workspace and Job selection. Shared model filters do not
attribute a job's spend to one model; a job may process several models.

This page requires SELECT access to `system.lakeflow.jobs`,
`system.lakeflow.job_run_timeline`, `system.lakeflow.job_task_run_timeline`,
`system.compute.node_timeline`, `system.billing.usage` and
`system.billing.list_prices`. Dashboard readers need these permissions themselves;
a permission error is not evidence of zero usage. The portable dashboard leaves
workspace and job selections empty. Deployment-specific defaults may select exact
owned IDs after verification. **Recent accessible jobs** lists jobs with runs in the last 30 days, including
one-off submissions and jobs whose names do not follow the Bundle convention.
Copy Workspace and Job ID from that table into the two input controls. Both
inputs accept exact IDs even when system metadata has not arrived. The list is
independent of model enrollment and does not establish model cost attribution.

Both workspace ID and job ID must match before execution or billing rows are
shown. Execution details summarize the last 30 days of run timeline records.
Billing includes only usage with an explicit matching job ID. It preserves usage
units and correction quantities, joins the applicable USD list price by SKU,
cloud, unit and effective time, and reports unpriced records separately. Priced
cost is an estimate, not an invoice or full project cost. Shared compute without
job attribution is excluded. System telemetry and billing can arrive late; empty
results do not prove that a job did not run or incur cost.

CPU combines user and system utilization; memory uses `mem_used_percent`.
The queries join task compute IDs to node samples using account, workspace and
cluster identity, bounded to overlapping task time in the last 30 days. Overlapping
tasks/retries are merged before aggregation so shared nodes are not counted twice.
Average cards weight each node sample by its observed duration; peak cards show
the highest individual node sample, not a cluster-wide capacity percentage.
Each metric excludes missing samples from its own denominator. Shared-cluster
usage can include other workloads during the job; it is not exclusive job usage.

The [compute system tables](https://docs.databricks.com/aws/en/admin/system-tables/compute)
cover classic compute; serverless usage and very short-lived nodes may have no
node samples. Availability explains the absence; the cards remain NULL rather
than displaying zero CPU or memory. CPU and memory trends show recorded averages
and peaks against sample time. The daily cost chart separates billing SKUs and
uses up to four USD decimal places so small amounts do not disappear into zero.

### Online serving

All matching contexts are visible initially; choose **Model** to narrow them.
**Monitored version** filters the
component model's concrete version; **Context** narrows its monitoring registration.
The identity table separately shows Endpoint, Served model, Served version and
Branch. A model-set parent can have a different name/version from the monitored
component. Explicit mismatched selections stay empty. Only enabled serving
enrollments are included.

| Table | Scope |
| --- | --- |
| Selected model and serving scope | Selection guidance and number of matching current contexts. |
| Endpoint and model identity | Monitored model/version versus served model/version and branch. |
| Latest observation per context | Latest full observation per matching enrollment and current configuration, ordered by window end, measurement time and report ID. |
| Observation history | All matching saved full observations, with native pagination and an optional history window-end filter; there is no global 100-row cutoff. |

Five charts above the detailed tables show captured requests, HTTP error classes,
captured error rate, mean/p95 execution latency, and the latency distribution.
Each model/version/context has a separate panel; **Model**, **Version** and
**Context** apply to all five. The history window-end control also filters the
latency matrix, while the latest snapshot stays unchanged. The latency bar chart
and matrix retain only the latest 12 observations per identity before the date
selection; use the
history table to inspect older windows.

| Chart | Reading the result |
| --- | --- |
| Captured requests per window | Number of logged requests, including failed responses. One request may contain several prediction rows. |
| How many errors of each kind? | Grouped bars with an **HTTP error class** legend: **4xx client** (blue), **5xx server** (red), and **Other / class not recorded** (gray). Value labels show captured counts. Zero means none captured; a missing class count stays blank. If an older report lacks complete class counts, its entire error total appears in gray. A date plus observation number keeps repeated windows separate. |
| Captured error rate per window | Captured errors divided by captured requests in that same observation. Missing counts, failed observations and zero requests leave it unmeasured. |
| How long does the model take to answer? | Two grouped bars per observation: blue is the average execution time and orange is approximate p95, in milliseconds per request (lower is faster). The last 12 observations per identity are shown separately, including repeated windows. Hover for p50, p99, maximum, timed sample count and exact window timestamps. Small samples can make p95 and p99 equal to the maximum; this is not a separate anomaly. |
| Where do captured execution times fall? | Nine duration ranges across the latest 12 observations per identity. Each valid column totals 100% of non-null timings. A shared 0-100% blue scale allows comparisons; hover for exact counts and window times. Old, failed, empty or invalid summaries produce no cells. |

For example, a report with ten captured requests, two 4xx responses and one 5xx
response has a 30% captured error rate. If only eight requests have a recorded
execution duration, latency percentiles and distribution shares use those eight,
including any failed responses with durations. A p95 of 250 ms estimates the
95th percentile of those captured execution durations; it is not the slowest
request or an end-to-end service guarantee.

Timings use Spark `percentile_approx` with accuracy 10,000. The fixed distribution
boundaries are 5, 10, 25, 50, 100, 250, 500 and 1,000 ms: the first range is below
5 ms, each middle range includes its lower bound and excludes its upper bound,
and the final range is at least 1,000 ms. Null timings are excluded from both the
percentiles and distribution. No request payloads are read by dashboard queries;
these fields come from saved aggregate reports.

Latest/history tables also show class counts, p50/p95/p99 and timed sample count.
Older reports retain the fields originally recorded; missing percentiles are
blank in numeric tables and produce no bar. The latency chart's detail tooltip
shows **Not recorded** for missing p50/p95/p99 and timed sample counts; a recorded
zero remains zero. Running monitoring with the updated Core creates these new fields; simply
refreshing or redeploying the dashboard does not backfill old reports. The
monitoring environment must match the runtime recorded in the registered model
artifact. Updating the Core requirement alone can fail that runtime check;
prepare a compatible model artifact before upgrading its monitoring jobs. Because
serving evidence participates in observation identity, recomputing an old window
with these added fields creates a new report identity. History can therefore
contain both calculations of the same window. The rollout controller continues
to use its existing maximum-latency and error-rate checks; percentiles do not
replace its admission or promotion gates.

Observation history also shows prediction rows per captured request when saved.
Its denominator includes failed captured requests. Request counts and latency
stay per observation: windows may overlap, and model-set branches may describe
the same parent requests. Do not sum these rows as unique traffic or average their
latency means into an endpoint SLA. **Observation status** describes the saved
monitoring result, not live endpoint availability.

Logging is asynchronous and custom-model HTTP 4xx/5xx errors may be absent, so
captured counts do not measure all endpoint traffic. Latency describes model
execution durations recorded by the platform, not client/network round-trip time.

| What you see | Next check |
| --- | --- |
| `awaiting_logs` | No applicable saved report exists. Check enrollment, the monitoring job and asynchronous log arrival; this is not a healthy measurement. |
| `failed` | Inspect the saved report and monitoring task error. A failed latest observation clears earlier counts instead of showing an older success. |
| Blank counts or latency | Check the report status and log coverage. Unavailable measurements are not zero requests or zero latency. |
| `no_data` | Inspect the selected request window and captured logs. An empty window does not prove endpoint health or absence of HTTP traffic. |
| Captured failures or unexpected latency | Inspect endpoint telemetry and serving logs for that endpoint/version. A healthy input-drift status does not certify request success or latency. |

Window end describes the request-time observation window; `measured_at` in the
saved dataset is when monitoring calculated the report. Shared health freshness
uses the latest captured request time for online serving and the scoring commit
time for batch predictions. Neither proves that the feature values themselves
are recent: a new request or scoring run can use old source data. Refreshing the
dashboard rereads saved results and does not obtain new measurements.

For accuracy, open Model performance and select the matching model, version and
monitoring context. Missing or immature actual outcomes remain insufficient
evidence; successful HTTP requests alone do not establish prediction quality.

## File responsibilities

Import monitoring helpers from the domain modules below. Paths are relative to
`skyulf/integrations/databricks/`.

| File | Responsibility |
| --- | --- |
| `observability/monitoring/monitoring_config.py` | Validated model/table identity, version and bounded policy. |
| `observability/monitoring/monitoring_registration.py` | Activation/scoring enrollment and immutable scoring handoff. |
| `jobs/monitoring/monitoring_tasks.py` | Visible activation enrollment and per-commit scoring observation tasks. |
| `observability/monitoring/monitoring_output.py` | Saved-batch drift report and safe shared-dashboard navigation in task output. |
| `observability/monitoring/monitoring_reference.py` | Registered artifact and original training-reference replay. |
| `observability/monitoring/monitoring_sources.py` | Saved predictions, source snapshots, receipts and delayed labels. |
| `observability/monitoring/monitoring_metrics.py` | Core metric adapters and finite keyed reports for bounded frames. |
| `observability/monitoring/monitoring_performance.py` | Mature performance windows measured against a pinned model baseline. |
| `observability/monitoring/spark/` | Distributed monitoring measurements, references and source handling. |
| `observability/monitoring/monitoring_store.py` | Owned Delta objects, concurrent registration, inventory reads and report history. |
| `observability/monitoring/monitoring.py` | Inventory observations, engine dispatch, failure isolation and MLflow reports. |
| `jobs/shared/job_runtime.py`, `scoring/incremental/scoring_recovery.py` | Invoke enrollment after successful scoring/recovery. |
| `model_sets/model_set_project.py`, `model_sets/monitoring_model_set.py` | Enroll pinned components after activation and scoring. |

The existing template's `src/monitoring/monitoring.lvdash.json` defines SQL,
shared investigation filters and six content pages.
`src/tools/configure_monitoring_dashboard.py` connects one
shared dashboard and its native refresh task to the existing project. The normal
project variables select monitoring storage, and `resources/monitoring.job.yml`
contains its separate Spark measurement/report/retraining job. No separate
monitoring Bundle is required.

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
schemas and model configuration digests are unchanged. Redeploy the updated Core
requirement with model and monitoring jobs so all inventory readers understand it.
