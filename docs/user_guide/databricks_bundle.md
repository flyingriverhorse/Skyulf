# Local-engine Databricks Bundle

The custom Skyulf template generates one editable Bundle with `dev`, `test`,
`syst` and `prod` targets. It fits and predicts with pandas or Polars. Spark
reads bounded Unity Catalog Delta rows and publishes predictions; local
feature engineering and model prediction are not distributed Spark work.

Initialize a project from a Skyulf checkout:

```powershell
databricks bundle init skyulf-core/templates/databricks --output-dir ./generated
```

The short path asks for project name, engine, one existing source row-key
column, model-change and model-selection modes, optional retraining mode and cron, serverless
or policy-backed job compute and the existing `dev`
catalog/schema. Serverless is the default. Reviewable
noninteractive examples are in `skyulf-core/templates/databricks/examples/`. The generated
project has its own `databricks.yml`; Skyulf's root has no Bundle config.

`dev` uses the selected CLI profile's workspace host. `test`, `syst` and
`prod` each have a different placeholder host and catalog in the generated
file. Edit each target's host, catalog and input/output/metadata schema
bindings, then validate with its designated profile. No Danske, Danica or
personal-workspace value is built into those three targets. Policy-backed
compute uses the policy name, runtime, node type and cost tag supplied at
initialization. Serverless compute needs none of those fields.

## What is created

`bundle deploy` creates only two jobs and uploads their code:

| Job | Purpose | UC objects created when run |
| --- | --- | --- |
| `train` | Fit one candidate and log held-out metrics | One registered model/version |
| `score` | Score initial and later CDF rows | Prediction output and rows |

The default project has no schedule. Choosing `monthly_paused` at initialization
adds a paused schedule to the existing `train` job, without adding a third
job or running it at deployment. The Quartz cron and timezone are selected
at initialization and remain editable Bundle variables. No endpoint or Unity Catalog table is
created by deployment alone.
`pinned_version` keeps manual model selection. `auto_champion` uses the same
train job to compare candidate and champion on a pinned temporal holdout. The
selected `metric`, `min_improvement`, and absolute `quality_threshold` stay
editable in `config/workflow.json`. A first champion requires a numeric
absolute threshold because no prior version exists for comparison. Later
versions must pass that threshold and improve on champion by the chosen
minimum. A passing candidate is staged and promoted through checked registry
receipts; an ineligible candidate leaves champion unchanged. The train job
then calls the existing score job. No third job or control table is added.

In automatic mode, score resolves `@champion` once to a concrete version per
run. Only the serialized train job identity may write this model's aliases;
other alias-write grants must be removed before enabling this table-free mode.
Alias promotion and Delta scoring are separate transactions. If scoring fails
after promotion, the last successful prediction output remains and the score
job must be retried. Unknown alias outcomes need receipt reconciliation.
Skyulf marks an alias transition as pending before writing it; automatic
training and scoring stop until that pending event is reconciled. An existing
champion alias set outside this controlled lifecycle also needs reconciliation
before automatic mode can use it.

For example, a regression project can select its gate in the generated config:

```json
{
  "engine": "polars",
  "model_selection_mode": "auto_champion",
  "metric": "heldout_rmse",
  "min_improvement": 0.1,
  "quality_threshold": 5.0,
  "model_change_mode": "incremental_append"
}
```

These example values mean RMSE must be at most `5.0`; a later candidate must
reduce champion's RMSE by at least `0.1` on the same holdout. Improvement is
an absolute metric difference, not a percentage. Skyulf derives the direction
from the metric: RMSE/MAE/log loss are minimized, while R²/accuracy/F1 are
maximized. Use a metric supported by the configured model task. The first
comparison reports `reason=no_champion` and `eligible=false` because no
baseline exists; successful bootstrap is recorded separately as
`alias_change.kind=initial` after the absolute gate passes.

If promotion succeeds but scoring fails, the train job reports a failed
dependent score task. Correct the scoring problem and run `score` again;
retraining is unnecessary. Full-rebuild output continues to expose the last
successfully activated generation during this recovery.

The four relevant Unity Catalog names have different roles:

| Name | Meaning |
| --- | --- |
| `training_table` | Existing labeled input table, pinned to a Delta version for training |
| `score_source_table` | Existing CDF-enabled source of rows to predict |
| `prediction_table` | The output table in append mode, or the stable active view in full-rebuild mode |
| `model_name` | Registered Unity Catalog model, not a table |

The first two references point to **the same existing table by default**.
Neither reference creates a table. Separate them only when labeled training
data and new scoring data have different lifecycles. If they stay together,
the first `score` run predicts all existing rows, including historical labeled
rows; review whether that is intended for your use case.

In `full_rebuild`, a version-specific physical table records the model name,
version and artifact digest. Scoring rejects an existing table at that name
when its model provenance differs. A new model with the same numeric version
needs a different logical prediction name or a new model version.

The initialization question `row_key` defaults to `entity_id`, but you can
choose an existing `customer_id` column. It becomes `row_keys` in the generated
configuration, and the same column appears in the prediction table. It must
be non-null, `STRING` or `BIGINT`, and unique across the initial data and all
later inserts. For multiple predictions per customer, edit the generated
configuration to a composite key such as
`"row_keys": ["customer_id", "observation_id"]` before deployment. The Bundle
does not create an ID in the source. Keep keys out of `input_columns`. With
`customer_id` in the output, a query
can find a customer's predictions:

```sql
SELECT customer_id, prediction, model_version
FROM catalog.schema.predictions
WHERE customer_id = 'C123';
```

## First run

Build and place the matching Skyulf wheel in the generated project's `dist/`,
then edit `config/workflow.json` for real source columns, preprocessing, model,
temporal split and size limits. The JSON values are an example, not a dataset.
Enable Change Data Feed on the scoring source before later inserts arrive.

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks bundle run train -t dev --profile <profile>
```

In manual mode, inspect the registered model version, put that concrete value
in `model_version`, redeploy the changed JSON, then run `score`. In automatic
mode, inspect the train and dependent score task results. The first score
rejects a missing source, disabled CDF, unsuitable row keys, a model output
mismatch or an existing target schema mismatch before creating prediction
output. It checks initial row count against `max_rows`; each score
also checks decoded transfer bytes against `max_bytes`. Existing tables are
never overwritten. The first score processes the current source
snapshot; later runs process only new inserts since the committed Delta
receipt. A repeat without new rows is a no-op. No monthly date or source
version is entered for each run.

In manual mode, the first candidate does not become champion automatically;
scoring stays pinned to its configured version. At initialization, choose
`incremental_append` to keep
v1 predictions and score only later source inserts with pinned v2. Choose
`full_rebuild` to write a new physical `<prediction_table>_v2` generation,
switch the stable `prediction_table` view after a successful complete score,
and append future inserts to v2. The previous generation remains available;
a failed rebuild leaves the active view unchanged. Full mode needs view
creation/ownership privileges and cannot silently turn an existing append-mode
physical table into a view. `max_concurrent_runs: 1` serializes the generated score job, but
does not coordinate other jobs. Grant prediction-table writes only to this
job's identity. For multiple publishers, use Skyulf Core's shared admission
provider. Online endpoints and Spark-native FE/model execution remain
separate work.

The optional monthly `train` schedule defaults to 03:00 UTC on day three and
starts paused. Edit its Bundle cron and timezone variables for the desired
monthly run time. After configuring real labeled data and verifying a manual
run, unpause it deliberately. Each run pins the source's latest Delta version and uses the
first day of the current UTC month as the label cutoff. The preceding month is
holdout; `monthly_lookback_months` (default four) controls the full window.
Only labels available by the cutoff are eligible. `@champion` is resolved to a
concrete version for comparison if present. Manual mode leaves selection
unchanged; automatic mode applies its metric gates and invokes score. The source version is
pinned at run start, so `label_at` must faithfully record availability.

The older SM-20a personal serverless rehearsal passed, but its jobs and test
schemas were removed at the user's request. The subsequent clean generic
`dev` rehearsal trained a Polars model from 600 real taxi rows, wrote 600
initial and 50 later predictions to one table, and replayed without another
commit. The later two-job design passed a separate personal serverless
rehearsal: 650 existing source rows, one later insert, and a no-op replay
left one prediction table with 651 rows. The `test`, `syst` and `prod`
placeholders have not been deployed in a company workspace.
