# Air quality: train once, append one day, score the new rows

Use the existing `air_quality_no2` project with one shared training/scoring
table. This walkthrough uses a separate `workspace.air_quality_no2_append`
schema so the original winter comparison stays available. Its three tables are
`observations`, `labels` and, after scoring, `predictions`. Both projects use
`workspace.skyulf_monitoring` for monitoring.

The model, custom preprocessing, exclusions, SHAP and charts are the same as in
[the original example](README.md#air-quality-real-seasonal-drift-and-every-exclusion-kind).
Training still fits in one Python process; scoring applies the saved model
through a Spark pandas UDF. There is no serving endpoint in this example.

## 1. Seed the historical training rows

Import [load_data.py](load_data.py) as a Databricks notebook and set:

| Widget | Value |
| --- | --- |
| `catalog` | `workspace` |
| `datasets` | `air_quality_no2` |
| `air_quality_layout` | `single_table` |
| `air_quality_day` | Leave empty for initialization. |

Run it once. It verifies the original UCI archive, creates `observations` with
5,646 March–October 2004 records, and creates `labels` for their measured NO2
outcomes. Missing outcomes marked `-200` do not become labels. The source has
Change Data Feed enabled before any winter rows arrive.

Initialization refuses to overwrite either table. If initialization fails after
one table was created, inspect that partial setup before retrying; initialization
is not a transaction across both tables. Daily appends below can recover partial
writes independently.

## 2. Make a deployment copy of the existing project

From this `real_data` directory, copy the project into a **new** directory:

```powershell
if (Test-Path -LiteralPath air_quality_no2_append) { throw "Choose a new deployment directory." }
Copy-Item -LiteralPath air_quality_no2 -Destination air_quality_no2_append -Recurse
```

In that copy, change these settings. Keep the supplied feature code and recipes.

| File | Setting | Value |
| --- | --- | --- |
| `databricks.yml` | `bundle.name` | `air_quality_no2_append` |
| `config/training.yml` | `defaults.training_table` | `"{catalog}.{input_schema}.observations"` |
| `config/inference.yml` | `score_source_table` | `"{catalog}.{input_schema}.observations"` |
| `deployment/targets.yml` | `targets.test.workspace.host` | Your workspace URL. |
| `deployment/targets.yml` | `targets.test.workspace.root_path` | `/Workspace/Projects/air_quality_no2_append/test` |
| `deployment/targets.yml` | `targets.test.variables.input_schema`, `output_schema`, `metadata_schema` | Set each to `air_quality_no2_append`. |

Keep `catalog: workspace` for this walkthrough. `monitoring_schema` is already
`skyulf_monitoring`. The existing label and model-policy names use schema
variables, so they resolve to the new project's `labels` and
`workspace.air_quality_no2_append.air_quality_no2_model`. Set the report-only
policy's baseline to the actual concrete version after training; version `1`
is the initial value for a new model.

Use a released Core version containing these features, or replace the Core
requirement with this checkout's wheel in your development deployment copy as
explained in [the examples guide](README.md). From the new project directory:

```powershell
uv run python src/tools/smoke.py
uv run python src/tools/preview.py --action train
uv run python src/tools/preview.py --action score
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run train -t test --profile YOUR_PROFILE
```

The training cutoff remains November 1, 2004; September–October is the holdout.
Automatic promotion uses the existing quality and coverage gates, and a
successful first activation starts scoring through the configured handoff.
Wait for that scoring job before appending more data.

**The initial score includes every existing summer row**, including training
rows. It establishes the incremental checkpoint. Its performance is not an
independent production evaluation. The policy cannot filter out that initial
batch: ignore its daily performance result. Winter rows scored on that same UTC
day share a window with thousands of summer rows, which can hide degradation.
Shared-window drift also describes that mixed population. For a seasonal policy
demonstration, start winter scoring on a later UTC day.

## 3. Append a winter day and score it

Run the loader again with the same settings and `air_quality_day=2004-11-01`.
It appends that archive day's inputs to `observations`. The new rows' `no2_gt`
is NULL: the model receives no answer. Measured outcomes go into `labels` with
the time this load made them available. It never updates source rows with labels.
This archive already contains the outcomes, so the demo publishes them before
scoring; it does not simulate delayed arrival. In a live feed, publish the real
outcome only when it becomes known and retain its actual availability time.

Then, from the deployment copy:

```powershell
databricks bundle run score -t test --profile YOUR_PROFILE
databricks bundle run monitoring -t test --profile YOUR_PROFILE --params monitoring_action=observe
```

Scoring uses the saved champion and handles only the new source inserts. Its
normal handoff also requests a batch monitoring observation; the explicit
monitoring command can revisit outcomes once the policy window becomes ready.

Retrying the same loader day inserts only keys missing from each destination.
If input insertion succeeded but label insertion failed, retrying completes the
labels without duplicating the inputs. Run loader invocations **one at a time**;
the notebook is a serial example, not a concurrent ingestion service. Its
ownership marker and CDF setting are checked before appending.

Repeat the loader and scoring commands with `2004-11-02`, then later dates from
the archive. A day without a measured outcome still produces a request; it does
not acquire an invented label. Unsupported dates fail before writing.

To inspect the two first winter batches:

```sql
SELECT o.observed_at, p.record_id, p.scoring_status, p.exclusion_reason,
       p.prediction, l.no2_gt, l.result_available_at
FROM workspace.air_quality_no2_append.predictions p
JOIN workspace.air_quality_no2_append.observations o USING (record_id)
LEFT JOIN workspace.air_quality_no2_append.labels l USING (record_id)
WHERE o.observed_at >= TIMESTAMP '2004-11-01 00:00:00'
ORDER BY o.observed_at;
```

Excluded rows remain visible with a null prediction and a reason. Running score
again without a new append must report no new predictions. Changing or deleting
an existing source row requires a rescore policy; do not use UPDATE to add labels.

## 4. Read the performance policy at the right time

The shipped policy measures **completed UTC daily prediction windows**, with
a six-hour label delay, at least 20 labelled predictions and 80% label coverage.
It reports degradation in the first eligible window whose WAPE is at least 20%
above the saved holdout WAPE. It never starts training. It also counts adjacent
failing windows: `consecutive_windows: 2` would gate trigger eligibility in
`retrain` mode; it does not delay reports in this example's `report` mode.

`observed_at` is the historical sensor time. The performance window uses the
time predictions were committed. Loading November 1 and November 2 immediately
one after another today therefore creates two data batches in the **same**
policy day. Rerunning monitoring on that day does not increase the consecutive
window count.

For a daily demonstration, score one archive day per actual UTC day, after the
initial summer-scoring day. Run monitoring after 06:00 UTC on the following day.
Two eligible adjacent loss windows make the consecutive count reach two;
the first loss was already reportable on its own.
An empty day, insufficient labels, or low coverage remains unavailable evidence.
Keep the schedule paused until you deliberately want recurring runs.

For a shorter acceptance exercise, a separate test deployment can explicitly
use shorter policy windows and label delay. That tests the same policy machinery;
it is not evidence that two production days have elapsed. Do not backdate saved
predictions or change the original example's evidence to make a policy pass.

## 5. Training again is a separate decision

Appending data and scoring never refit the model. The existing fixed training
window still selects only summer rows. To learn from new outcomes, prepare a
labelled training snapshot by joining the inputs and outcomes, review the new
training/holdout boundaries, then start training explicitly. Do not update the
append-only scoring source to make that join permanent.

Changing the window to `rolling_days` does not join the label table or fill
`no2_gt`. If the selected training window contains these unlabelled winter rows,
the supplied recipe rejects their null targets. An explicit `DropMissingRows`
could exclude them, but would not teach the model from their outcomes; use the
labelled snapshot when the goal is to learn from those new rows.
