# Configure and investigate the shared dashboard

## When you need it

Connect a project to the existing AI/BI dashboard, create its single owner if needed, or investigate
saved health and lineage. Metric-specific chart reading belongs to [drift](databricks_drift.md),
[performance loss](databricks_performance.md) and
[serving observations](databricks_serving_monitoring.md).

Choose one project/target to own the dashboard resource. All other projects reuse
its published URL. An existing dashboard can remain under its current deployment
ownership; using its URL does not create a copy. Keep its dashboard ID and Bundle
state when migrating ownership; never deploy a second owner against it.

The native [Dashboard job task](https://docs.databricks.com/aws/en/jobs/tasks/dashboard)
runs after the existing monitoring job has saved its evidence and evaluated
retraining. Configure it in each project as described below. Development or
disabled monitoring skips refresh. No email subscription is created.

Updating source data does not automatically invalidate the
[dashboard result cache](https://docs.databricks.com/aws/en/dashboards/caching).
The task performs the explicit refresh after successful monitoring. It runs
as the job identity and preserves `embed_credentials=false`; other viewers keep
their own data permissions and caches. This is automatic result refresh, not a
continuously pushed update of every already-open browser tab. A separate
dashboard schedule can refresh independently of monitoring if desired.

## What to edit

Set the same `monitoring_catalog` / `monitoring_schema` in all production targets.
No separate monitoring template or setup Bundle is required. The first activation,
scoring registration or scheduled monitoring run bootstraps missing objects.
Initialize before validating the dashboard SQL, because its queries need the views.

Dashboards are set per shared target in `deployment/dashboards.yml`. Each workspace
has its own dashboard and SQL warehouse, so `test`, `syst` and `prod` are configured
separately; leave a target out for no dashboard. A configured target adds two tasks
after `evaluate_retraining` in the monitoring job: `dashboard_refresh_enabled` checks
that `monitoring_enabled` is `"true"`, then `refresh_monitoring_dashboard` refreshes.
Existing schedules, policies and compute are unchanged, and no email subscription is
created. Development targets have no dashboard block, so they never refresh. For a dashboard published without shared
data permissions, the job's Run as identity needs access to its data and warehouse.
See the [Databricks dashboard task documentation](https://docs.databricks.com/aws/en/jobs/tasks/dashboard).

For each target choose one:

| Choice | What to write under the target | Result |
|---|---|---|
| `create` | the dashboard (with `warehouse_id`), the link variable and the refresh task | `bundle deploy` creates the dashboard; monitoring refreshes it |
| `connect` | the published URL as `monitoring_dashboard_url` and its ID in the refresh task | monitoring refreshes the existing dashboard and links it in its output |
| nothing | omit the target | monitoring still saves its results; no dashboard |

### How the refresh task is added

`deployment/dashboards.yml` does not define a second job. `resources/monitoring.job.yml`
defines the `monitoring` job for every target, and a target block repeats only its
name with two extra tasks:

```yaml
# resources/monitoring.job.yml (every target)
resources:
  jobs:
    monitoring:
      tasks:
        - task_key: monitor_model
        - task_key: monitoring_report
        - task_key: evaluate_retraining

# deployment/dashboards.yml (only the test target)
targets:
  test:
    resources:
      jobs:
        monitoring:            # same job name: merged, not a new job
          tasks:
            - task_key: dashboard_refresh_enabled   # skip when monitoring is off
              depends_on:
                - task_key: evaluate_retraining
              condition_task:
                op: EQUAL_TO
                left: "${var.monitoring_enabled}"
                right: "true"
            - task_key: refresh_monitoring_dashboard
              depends_on:
                - task_key: dashboard_refresh_enabled
                  outcome: "true"
              dashboard_task:
                dashboard_id: ${resources.dashboards.monitoring_dashboard.id}
```

On `bundle deploy -t test`, Bundle merges every included file into one
configuration. Jobs with the same name become one job, and their tasks are merged
by `task_key`: a new key is added, an existing key is updated. The deployed `test`
job therefore runs `monitor_model -> monitoring_report -> evaluate_retraining ->
dashboard_refresh_enabled -> refresh_monitoring_dashboard`. A target without a block keeps the first three
tasks. `deployment/targets.yml` uses the same merge to add `run_as` and
`permissions` to jobs defined in `resources/`. Run
`databricks bundle validate -t <target> -o json` to see the merged job.

Choose `create` only in the first project that uses a target's monitoring store;
every other project uses `connect`. The dashboard ID is the 32 characters after
`/dashboardsv3/` in its URL. The warehouse ID is the 16-character value under
**SQL Warehouses > your warehouse > Connection details**, also visible in the
warehouse page URL; syst and prod need IDs from their own workspaces. The file
contains both blocks as commented examples to copy.

Setup fills the `test` target for you. With a monitoring store, answer
`monitoring_dashboard`:

| Answer | Also asks | Result |
|---|---|---|
| `none` (default) | nothing | no target is configured |
| `create` | `monitoring_dashboard_warehouse_id` (required) | `test` gets the create block |
| `connect` | `monitoring_dashboard_url`, optional warehouse ID | `test` gets the connect block, with the ID taken from the URL |

`syst` and `prod` are always set by hand in the same file. `monitoring_catalog` /
`monitoring_schema` stay in `deployment/variables.yml`; the monitoring job creates
its tables there on its first run, and the dashboard reads from them.

The template's `src/monitoring/monitoring.lvdash.json` defines SQL, shared
investigation filters and five content pages, plus the global filter page.
`resources/monitoring.job.yml` contains the separate Spark measurement, report and
retraining job. No separate monitoring Bundle is required.

## The command

After editing `deployment/dashboards.yml` for a target, validate and deploy it:

```powershell
databricks bundle validate --strict -t test --profile <profile>
databricks bundle deploy -t test --profile <profile>
```

For a new dashboard, validate every dashboard dataset against the chosen monitoring
store first. Other projects connect with the published URL from this deployment.
To stop refreshing, remove the target's two refresh tasks. Removing a `create` block
deletes that dashboard on the next deploy, so move ownership deliberately.

Initialize or upgrade the store using
[the owner notebook](databricks_monitoring_setup.md#the-command) before validating every dataset SQL
query and publishing the dashboard. The dashboard needs SELECT on `complete_observations`,
`monitoring_identities`, `serving_windows` and the other monitoring views. For training summaries,
refresh `model_version_details` to expose `training_summary_json`.

## What you should see

Dashboard deployment is configuration, not metric calculation. Run monitoring to
save new measurements, then the native task refreshes the published datasets.
The dashboard retains the selected store across all project layouts.

Use the **shared filters** on the left: **Model (choose one for details)**,
**Monitored version** and **Context**. Model names include catalog and schema,
so there are no separate catalog/schema controls. Context includes environment,
project and a short monitor ID. The selection persists between pages; there are
no competing model controls inside the pages. Date, feature, metric, target and
job controls refine only the page where they appear.

**All models**, **All versions** and **All contexts** are the initial selections.
Drift and Model performance include every matching model/version/context, with
the latest result selected independently for each identity. Disabled policies and
identities without measurements remain visible. Choose Model, Monitored version and Context
to narrow the portfolio. Online serving lists matching endpoint contexts separately.
Incompatible explicit selections stay empty; clear Monitored version and Context
when switching models. An explicitly selected model
never falls back to another model.

| Page | Scope | What to use it for |
| --- | --- | --- |
| Overview | All models with the initial All selections; otherwise all matching registrations. | Four summary counts, prioritized issues, current status distributions and daily trends. |
| Training and lineage | All matching model/version/context identities. | Compare concrete versions, owning training runs and recorded provenance. |
| Drift and data quality | Every matching model/version/context, with its own latest observation. | Separate feature PSI comparisons and drift histories, with detailed quality measurements. |
| Model performance | Every matching model/version/context, including disabled and unmeasured policies. | Per-identity prediction/label coverage, metric history, policy comparisons and classification evidence. |
| Online serving | All matching serving enrollments, separated by model/version/context; choose Model to narrow them. | Distinguish monitored and served versions; inspect captured counts, errors and latency in tables and five window charts. |

### Overview

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

The Drift and Performance charts show up to two columns and two rows of panels at once; scroll
inside a chart for additional identities. Fewer panels use the available width, within the fixed
compact card height. Choose Monitored version and Context as well as Model to reduce panels.

### Latest observations and page filters

Latest-observation views pick each identity's newest saved report first, ranked
by window end and then measurement time. Page filters for date, feature, target
or metric apply afterwards. A filter can hide that report's values, but it never
selects an older report instead. A newer empty or failed report likewise hides
an older measured value. History tables and trend charts keep every saved
observation.

### Training and lineage

**Training and lineage** shows every identity matching the shared filters. Choosing All models / All
versions / All contexts shows all available identities; an obsolete or incompatible selection leaves
empty tables and never substitutes the latest different model. Each table includes its model,
version and context. Historical observed versions retain their concrete identities, while NULL
versions explicitly say unresolved. The page shows creation time, owning training run, recorded
algorithm, artifact digest and available code evidence. Open **Training run** to inspect that run's
native Inputs and artifacts. A source commit is shown only when the owning run already records
`mlflow.source.git.commit`; a project Python source digest is shown only when the saved artifact
records one.

Three additional tables explain how each monitored version was produced:

- **Training candidates and recorded selection** shows saved candidate scores, metric direction, CV
  evaluation mode and selection reason. Only the selected, registered candidate
  has a registered version; losing candidates retain their own training run.
  A single configured model is identified explicitly, without an invented
  competition score. Run duration is elapsed run time, not isolated fitting time.
  Post-selection CV is identified separately from an unbiased estimate.
- **Chosen model parameters** lists the fitted estimator's saved settings,
  including defaults, ensemble and split settings, by parameter name and source.
  Structured values remain JSON; these are recorded settings, not a new search.
- **Training datasets and provenance** shows the saved training/holdout
  populations, source table and version, membership evidence and feature binding.
  Delta table identity and Unity Catalog identity are different fields. Version
  zero is valid. **Dataset Inputs run** opens the verified run containing native
  Inputs; in a task-based competition this may be the winning child rather than
  the registered model's parent run.

All three tables use the same **Model (choose one for details)**, **Monitored
version** and **Context** selection.
They summarize allowlisted metadata only; raw training rows, executable source
and complete run configs are not copied into the dashboard. Missing, inaccessible,
invalid or oversized evidence is reported independently for each section. Empty
evidence never becomes a zero score or a different version's training history.
These optional summaries cannot change monitoring or promotion decisions.

Each new observation reads up to three known JSON artifacts and the saved
competition's child-run metadata. Competition summaries admit at most 32
candidates; each section admits at most 256 rows and 64 KiB of stored JSON.
Oversized sections show `too_large` instead of truncating or partially ranking
candidates. Feature-binding details appear once, on the training-population row.

The summary is copied into the monitoring store. Readers with access to its
tables/views can see candidate scores, model parameter values and source
metadata even if they cannot open the original MLflow experiment. Grant store
access to the intended audience; opening the linked run still requires its own
permissions. Lookup details contain table/column names, time-window settings,
table identities and versions, not individual lookup-key values or row filters.

In the **Alias metadata snapshot** table, **Aliases at capture** describes the
concrete version's alias metadata when the
monitoring reference was loaded, alongside **Captured at**. It does not describe
historical deployment identity or guarantee today's registry alias assignment.
New observations refresh the bounded snapshot in existing `evidence_json`.
Retrying the same observation preserves the first saved snapshot and does not
invalidate prepared Spark references. Dashboard refresh only rereads saved
evidence; it makes no per-row registry calls and does not move aliases.

## Common errors and fixes

Start on **Overview → Issues to investigate**. The table uses native pagination, with failures
first, then recorded quality/drift/performance issues, then missing or stale evidence. The counter
gives the full filtered issue count, including rows on later pages. One model can have multiple
issues; this is an issue count, not a model count. Use the shared **Model (choose one for
details)**, **Monitored version** and **Context** filters to narrow the list. Disabled monitoring
and disabled performance policies do not create new failures. Never-observed, no-data, failed, stale
and insufficient-label results keep their distinct recorded reasons. Missing metric values remain
empty.

Open a concrete **Model version** link for the registry entry, or **Evidence run**
for the monitoring run that recorded that issue. The latter is separate from the
owning training run. A missing link means its exact run/experiment identity was
not recorded or is unavailable; it does not mean the issue is healthy.

Older reports can show **not_recorded**, never-observed enrollments have no
training metadata, and optional metadata permission failures show **unavailable**
with a reason. Missing creation/code/alias fields remain unavailable rather than
inferred. Existing store owners can run `initialize_monitoring_store` explicitly
to refresh owned view definitions; ordinary runs create missing compatible views
without replacing existing ones. Viewers retain their own data permissions.
For the training tables, refresh the owned `model_version_details` view to add
`training_summary_json` before publishing the dashboard. The next new observation
captures the summary. Refreshing the dashboard or retrying an already saved
observation does not backfill historical reports; their section status remains
`not_recorded`.

A refresh only queries the saved summary; it neither rereads training rows nor fits a model. New
observations read the saved artifacts again. Incompatible explicit model/version/context choices
stay empty; clear Monitored version and Context when changing models.

If a browser still shows old results, use the dashboard refresh control. A browser reload may reuse
cached queries. The native job task refreshes published datasets after observation but does not push
updates into every open tab. Store/view upgrades and producer runtime upgrades remain separate from
dashboard refresh.
