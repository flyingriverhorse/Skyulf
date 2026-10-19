# taxi_fare_robust

New here? Read [START_HERE.md](START_HERE.md) first: it lists what to edit, the
first training and prediction delivery, and which guide answers which question. This page maps
each setting to the file that owns it.

If your source tables already exist, start with the read-only source check in
[SOURCES.md](SOURCES.md). It explains this project's input/outcome arrangement
and column contract. If tables do not exist yet, choose `include_source_sql: "true"`
during setup to include an editable `config/sources.sql` starter. The default is
`"false"`; initialization and deployment never create your source tables.

Skyulf 0.9.5 Bundle: **regression** with **pandas** training and **batch** prediction delivery, using **serverless**
compute. Training and preprocessing fit in one bounded Python process on this compute.
Batch scoring runs a partition-safe MLflow pyfunc on Spark workers and keeps keyed predictions distributed.

## Where to make a change

| Change | File |
| --- | --- |
| Sources, record keys, model inputs, targets and training limits | `config/training.yml` defaults/model entries |
| Model, tuning, CV, quality gates and recipe selection | `config/training.yml` |
| Fixed cleanup and training eligibility | `config/pre_split.yml` |
| Fitted preprocessing | `config/preprocessing.yml` |
| Saved model selection, score source/output and change policy | `config/inference.yml` |
| Upstream feature groups and their checked merge | `config/features.yml`, `src/features/groups/` |
| Scoring eligibility, outputs and combined rules | `config/scoring.yml` |
| Custom feature/scoring callbacks and packaged assets | `src/features/` |
| Online publication, endpoint setup and rollout | `config/serving.yml`, when generated |
| Host, namespaces, run identities, compute and schedules | `deployment/targets.yml`, `deployment/variables.yml` |
| Monitoring store and on/off switch | `monitoring_catalog`, `monitoring_schema`, `monitoring_enabled` in `deployment/variables.yml` |
| Installed Core and optional dependencies | `deployment/*requirements.txt` |

The shared runtime pin is `skyulf-core[mlflow]==0.9.5` in
`deployment/requirements.txt`. Use the Core release matching this template
([runtime packaging](https://www.skyulf.com/manual/user_guide/databricks_deployment.html#runtime-packaging)). Changing the pin
does not migrate saved models or rewrite already generated project files.

Quick setup accepts editable defaults. Detailed setup asks applicable model,
branch, competition and deployment choices. Both ask for the monitoring store;
a blank catalog starts without monitoring. When a catalog is set, an omitted
schema uses `skyulf_monitoring`; choose another schema to reuse an existing store.
Neither mode inspects source rows.
Empty `config/features.yml` groups create no feature job or merged table;
training and scoring read their configured sources directly.

Training defaults to at most **10,000 rows and 64 MiB per Python input read**.
For a larger source, set `training_sample_rows` in `config/training.yml` to
sample eligible rows before splitting, or raise `max_rows` and `max_input_mb`
within your compute's memory budget. Sampling never samples scoring rows.
Detailed setup asks for the sample limit and optional `save_preprocessing_diagnostics`.
The probe records diagnostics on at most 256 holdout rows and 8 MiB; Spark and
endpoint admission checks are a separate requirement.

With `initial_scoring: full_snapshot` (default) in `config/inference.yml`, the
first scoring run includes existing rows. If training uses the same table,
this includes training history; its measured performance is not independent.
Choose `initial_scoring: from_now` before the first score to commit a zero-row
checkpoint and score only later inserts. Rebuilds preserve that boundary using
saved prediction keys and recent CDF. CDF-expiry recovery additionally needs the
original starting snapshot to remain readable.
The choice cannot be changed on an existing prediction target.
`python src/tools/preview.py` explains the choice.
Scheduled training with a
`fixed_window` repeatedly reads the same date interval; use `rolling_days` or
`rolling_calendar` when the interval should advance. Preview also warns about
this combination; intentional repeated experiments are still allowed.

Monitoring job failures can notify through `monitoring_email_notifications`,
`monitoring_webhook_notifications` and `monitoring_health_rules`. Measured drift
does not fail the job; use a separate SQL alert over `monitoring_issues` for it.
See [notifications](https://www.skyulf.com/manual/user_guide/databricks_drift.html#notifications).
The separate `monitoring_cleanup` job runs Sundays at 04:00 UTC with
`cleanup_confirm=true` and a 90-day minimum reference age. Configure
`monitoring_cleanup_cron`, `monitoring_cleanup_confirm`,
`monitoring_cleanup_pause_status` and `monitoring_reference_retention_days` per
target. Personal development targets keep this schedule paused.
Review [reference retention](https://www.skyulf.com/manual/user_guide/databricks_monitoring_setup.html#retain-or-clean-old-spark-references)
and its lifecycle-writer requirements before enabling deletion.

These short guides ship with this project revision and link to detailed topic
pages. Public documentation can describe a newer release; use the release
upgrade notes before changing deployed jobs. Fitting saves preprocessing and
model state, so editing a recipe afterwards needs a new trained version.
Promotion, score selection and online traffic are separate decisions. All public
guides are listed in the [topic index](https://www.skyulf.com/manual/user_guide/databricks_bundle.html).

## Report WAPE performance changes

`deployment/targets.yml` enables a report-only WAPE policy for this model.
It compares completed daily windows with version 1's saved training holdout,
using a 20% relative tolerance and two consecutive windows. It never requests
training. Before deployment, replace the baseline version with the concrete
version you will monitor; older Spark reference caches without WAPE remain unavailable.
See [SERVE_EXAMPLE.md](SERVE_EXAMPLE.md#report-performance-loss-without-retraining)
for label coverage, timing, baseline changes and disabling the policy.
