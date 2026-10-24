# Project guide

Use [README.md](README.md) for the first run and learning checkpoints.
This reference covers settings, saved model behavior and operational results.

- [File map](#file-map)
- [Source data](#source-data)
- [Preprocessing and custom code](#preprocessing-and-custom-code)
- [Training and model decisions](#training-and-model-decisions)
- [Batch predictions](#batch-predictions)
- [Serving requests](#serving-requests)
- [Optional feature tables](#optional-feature-tables)
- [Monitoring](#monitoring)
- [Deployment and schedules](#deployment-and-schedules)
- [Local tools](#local-tools)
- [Failures and upgrades](#failures-and-upgrades)

## How do I

| I want to | Go to |
| --- | --- |
| Check my tables before the first run | [Source data](#source-data) |
| Fill missing values, encode categories or scale columns | [Preprocessing](#preprocessing-and-custom-code) |
| Drop invalid rows before training | [Preprocessing](#preprocessing-and-custom-code) (`pre_split.yml`) |
| Change the model or its parameters | [Training](#training-and-model-decisions) |
| Set the quality bar a version must pass | [Training](#training-and-model-decisions) |
| Choose which version scoring uses | [Batch predictions](#batch-predictions) |
| Skip rows or add output columns when scoring | [Batch predictions](#batch-predictions) |
| Call the model over REST or SQL | [Serving requests](#serving-requests) |
| Turn on drift and performance monitoring | [Monitoring](#monitoring) |
| Change hosts, schedules or compute | [Deployment and schedules](#deployment-and-schedules) |
| Recover from a failed run | [Failures and upgrades](#failures-and-upgrades) |

## Words used here

- **Candidate**: a newly trained model version in the registry. **Champion**: the
  version approved for use. `champion` is an *alias*, a movable label on one version.
- **Holdout**: rows kept out of fitting and used only to measure the final model.
- **Pre-split** steps run before the train/holdout split and never learn from data.
  **Preprocessing** steps run after it and learn (means, categories) from training rows only.
- **Receipt**: the saved record of what a run read and wrote (versions, row counts),
  used to check results and repeat runs safely.
- **Admission**: the training-time check that a saved recipe can also run on Spark
  workers or a serving endpoint.
- **CDF** (Change Data Feed): Delta's log of row changes; incremental scoring uses it
  to score only new rows.

More terms: [glossary](https://www.skyulf.com/manual/user_guide/databricks_glossary.html).

## File map

| Change | Owner |
| --- | --- |
| Training source (`training_table`), keys (`record_key_columns`), inputs (`input_columns`), target (`target_column`), split, model, tuning, quality | `config/training.yml` |
| Fixed cleanup and training eligibility | `config/pre_split.yml` |
| Transformations fitted on training rows | `config/preprocessing.yml` |
| Saved-model selection, prediction source/output and change policy | `config/inference.yml` |
| Saved scoring eligibility, output columns and business rules | `config/scoring.yml` |
| Custom factories, callbacks, declared assets and exact dependency pins | `src/features/` |
| Optional upstream Spark groups and joins | `config/features.yml`, `src/features/groups/` |
| Endpoint, SQL function, caller permissions and optional online delivery | `config/serving.yml`, when generated |
| Hosts, namespaces, run identities, compute, schedules and monitoring | `deployment/targets.yml`, `deployment/variables.yml` |
| Installed Core and optional dependencies | `deployment/*requirements.txt` |
| Generated tasks and local graph refresh tools | `resources/*.job.yml`, `src/tools/` |

### After you change a file

| You changed | Then |
| --- | --- |
| `config/training.yml`, `pre_split.yml`, `preprocessing.yml`, `scoring.yml` or `src/features/` | `preview.py --action train`, validate and deploy, then train. Only the new version uses the change; trained versions keep what they were trained with. |
| `config/inference.yml`, `config/serving.yml` or `deployment/` | Validate and deploy. The next run uses it. |

Change models/recipes without editing job notebooks. Training captures selected
preprocessing, scoring rules and code; later edits do not alter saved versions.

Do not edit `src/jobs/*.py`: they are thin entry points that a template upgrade
replaces, and the job logic ships in the Core wheel. Model code belongs in
`src/features/`, job settings in `deployment/`, and a private package index or
bundled wheels for logged models in the `SKYULF_MODEL_PIP_*` job variables; see
[serving packages](https://www.skyulf.com/manual/user_guide/databricks_serving.html).

## Source data

The selected arrangement is **single_table**. Setup and deployment do not
create or fill input tables; they must exist.

- **Record keys** identify one observation (for example `customer_id, month`), not
  just one person. They must be unique, never empty and never change; use STRING or
  BIGINT columns, not dates.
- **Model inputs** exclude the keys and the target.
- Keys in a separate outcomes table need the same types as in the source.
- Setup already wrote your answers into `config/training.yml` under `defaults:`
  (`training_table`, `record_key_columns`, `input_columns`, `target_column`); edit
  them there.
- Scoring reads new rows through Change Data Feed. Turn it on once per source table:

  ```sql
  ALTER TABLE your_catalog.your_schema.your_table
  SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
  ```

Training expects targets in its training table. A monitoring label-table
binding does not join outcomes into training. To train from external outcomes,
explicitly configure `training_label_table`, its version and availability policy.
An actual outcome is a measured result, never a copied prediction. Missing or
late results remain unavailable. Event time and outcome availability time can
differ; configure their parsing/timezone and cutoff from the source's meaning.

For current table metadata, types and key checks, use the source action after
installing the local dependencies from README:

```powershell
python src/tools/preview.py --action sources --profile YOUR_PROFILE --warehouse-id YOUR_EXISTING_WAREHOUSE --catalog YOUR_CATALOG --input-schema YOUR_INPUT_SCHEMA --output-schema YOUR_OUTPUT_SCHEMA --metadata-schema YOUR_METADATA_SCHEMA
```

`--warehouse-id` is a SQL warehouse ID (SQL Warehouses > your warehouse > Connection
details). This is read-only, but uniqueness checks can scan tables. It checks the supplied
profile's access, not another run identity or future data. Add the target's
`--resource-suffix` when needed.

No DDL starter was selected. If needed, generate a separate project with
`include_source_sql: "true"` and adapt its SQL without overwriting this project.
See [source contracts and time windows](https://www.skyulf.com/manual/user_guide/databricks_training_data.html).

## Preprocessing and custom code

The order is source selection → fixed cleanup/eligibility → final split →
fold-local preprocessing/model fit. Keep learned means, quantiles and categories
after splitting. Both phase YAML files have `version: 1`, `recipes`, and a
`default` recipe. The `default` recipe runs unless `config/training.yml` picks
another under `defaults:` with `pre_split_recipe: <name>` or
`preprocessing_recipe: <name>`; other recipes run only when picked. An empty
recipe does nothing.

A typical recipe, with your own column names:

```yaml
# config/preprocessing.yml: learned on training rows after the split
version: 1
recipes:
  default:
    - name: fill_income
      transformer: SimpleImputer
      params: {columns: [income], strategy: median}
    - name: city_onehot
      transformer: OneHotEncoder
      params: {columns: [city], handle_unknown: ignore}
```

```yaml
# config/pre_split.yml: fixed rules before the split; nothing is learned
version: 1
recipes:
  default:
    - name: known_income
      transformer: DropMissingRows
      params: {subset: [income]}
```

Run `python src/tools/preview.py --action train` to see the resolved steps in order.
Every built-in step and its parameters are in the
[preprocessing catalog](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html).

### Your own step in Python

Use a built-in step when one fits. When none does, write a small factory. For example,
`src/features/my_rules.py` adds `log_income = log(1 + income)`:

```python
import numpy as np

from skyulf.preprocessing import column_step


def log1p_value(df, params):
    return np.log1p(df[params["column"]].clip(lower=0))


def log_feature(column):
    return column_step(
        f"log_{column}", log1p_value, output=f"log_{column}", params={"column": column}
    )
```

Select it in a recipe; `params` become the factory's keyword arguments:

```yaml
    - custom: my_rules.log_feature
      params: {column: income}
```

The new `log_income` column goes to the model automatically: the model receives every
column left after preprocessing except the target.

A step that learns from training rows (a mean, a mapping) uses `fitted_step` with a
`learn(df, y, params)` and an `apply(df, state, params)` function;
`src/features/preprocessing.py` shows both kinds with comments.

Rules for your own code:
- A factory is a top-level function in a module under `src/features/`; it returns a
  Core step, not a DataFrame.
- Import `filter_step` for fixed row eligibility, `column_step` for fixed columns,
  or `fitted_step` for learn/apply state from `skyulf.preprocessing`.
- Learned JSON-compatible state is fitted per training fold and reused on
  holdout/prediction.
- Keep recipes in the YAML files. Do not also define the same recipe in Python, and do
  not write pipeline steps inside `training.yml`.

Keep imports/factories free of I/O. Preserve keys and protected time/group/weight
metadata. Use relative imports for packaged helpers; never import `groups/`
from model callbacks. Declare small data files in `src/features/assets.json`,
read them with `read_project_asset`, and declare exact package pins in
`src/features/requirements.txt`. New code, assets or dependencies require a new fit.

Spark workers and endpoints require admission for the exact saved recipe.
Training checks every step and rule that Spark workers or an endpoint will run; a
step is admitted only when that check passes, not because its code says it is row-local. Avoid index-dependent behavior
and mutable global caches. Group/window/global context cannot be treated as
independent prediction rows. `probe_pipeline_for_serving: true` requests the
training admission checks; Spark/online-feature routes already require them.

`save_preprocessing_diagnostics: true` separately checks at most 256 holdout rows
and 8 MiB, logging `preprocessing_probe.json` in the fitting run. Preview does not
run it; passing does not prove quality or admission. History-dependent local
prediction needs the explicit history API and coordinated state writes.
See [recipes](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html),
[custom factories and assets](https://www.skyulf.com/manual/user_guide/databricks_custom_steps.html),
[diagnostics/history](https://www.skyulf.com/manual/user_guide/preprocessing_context.html),
and [Spark admission](https://www.skyulf.com/manual/user_guide/databricks_spark_scoring.html).

## Training and model decisions

Edit `models.main.model` and sibling `tuning`; this layout has one named model.
Model entries replace complete nested mappings from defaults, rather than
deep-merging them. Remove `tuning` for fixed model parameters. Inspect the
resolved recipe with `preview.py --action train` before deploying changes.

For example, a fixed model with a quality bar in `config/training.yml`:

```yaml
defaults:
  quality_threshold: 5.0   # holdout error must stay at or below this
models:
  main:
    model:
      type: ridge_regression
      params: {alpha: 1.0}
    # no tuning block: these values are used as given
```

The threshold is compared with the `metric` in the same `defaults` block (setup wrote
this project's metric there; for example
`heldout_accuracy`): error metrics such as RMSE must stay at or below it, scores such as
accuracy must reach it. Choose it from what the business needs, before training. The
[training field reference](https://www.skyulf.com/manual/user_guide/databricks_training_config.html)
lists every model, its parameters and the tuning options.

Training fits **pandas** in one process. `max_rows`/`max_input_mb` bound input,
not total RAM or preprocessing/model expansion. `training_sample_rows` selects
eligible rows in Spark before transfer, including holdout; scoring is never sampled.
`training_table_version: null` pins the latest Delta snapshot once per run.
For a controlled comparison, reuse the recorded source/label versions and
split/window settings; a rolling window otherwise advances on the next run.

Choose random versus temporal splitting from the problem, not the column name.
Final holdout rows remain separate from fitting/search. Configure a meaningful
numeric `quality_threshold` before training if first-champion activation or
automatic promotion is intended. Coverage gates prevent apparent improvement
by excluding too many rows. A registered candidate is not automatically champion.

This project has `promotion_policy=automatic`,
`score_model_selection=champion`,
`follow_champion: true`, and
`rescore_when_history_expires: false`.
Read **training_report > Output > Where to look** for version, fitting run,
holdout evidence and decision. Inspect the outcome: a report can finalize failure.
Copy **Available action** parameters into a train job run for
approve/reject/rollback; first approval expects champion `none`. Keep receipts.
Child score/serving completion is separate. See [training fields](https://www.skyulf.com/manual/user_guide/databricks_training_config.html),
[reports](https://www.skyulf.com/manual/user_guide/databricks_training_reports.html),
and [model lifecycle](https://www.skyulf.com/manual/user_guide/databricks_lifecycle.html).

## Batch predictions

This project uses **spark** batch compute.
`pinned_version` uses the saved `model_version`; `champion` resolves the
controlled alias once per run. `score_model_version=N` overrides either for one
run without editing config or aliases. Use the report's actual version. Set
the persistent selector/version and redeploy for future scheduled runs.
Before the first score, choose `initial_scoring`: `full_snapshot` includes
existing rows; `from_now` commits a zero-row boundary and waits for later inserts.
An enabled training handoff may trigger that first score. The choice is fixed
once recorded; use a new destination for a separate scoring history. Enable
Delta CDF before scoring, keep keys immutable and deliver late labels to their
outcomes table instead of inserting duplicate prediction keys.

After each score run:

- **Check the result** in the run output: the model and source versions used, rows
  in and out, excluded rows, and the Delta version written.
- **Retrying is safe.** The same model on an unchanged source writes nothing new.
- **New source rows** are scored on the next run. Changing or deleting existing source
  rows makes ordinary single-model scoring fail.
- **A new model version** follows `on_model_change`: `score_new_rows` keeps old
  predictions, `rescore_all` scores every row again into a new prediction table generation.
- **Old history:** if the source's change history (CDF) has expired, rescoring is an
  explicit option (`rescore_when_history_expires`). Never reset the scoring position
  by hand to force a retry.

Read **score > Output**; this graph has no `scoring_report` task.
Workers run admitted pandas-fitted MLflow pyfunc batches with matching dependencies.
Polars whole-frame behavior and arbitrary history/context do not become worker-safe.

In `config/scoring.yml`, `mode: pre_split` reuses saved eligibility; `custom`
uses its declared lists; `combined` applies pre-split first and retains its first
exclusion reason. Target-reading filters require explicit
`skip_target_pre_split_steps: true` when appropriate; the whole step is skipped,
so separate target and feature filters when feature checks must remain.
Built-ins include `required_values`, `value_range` and `prediction_band`.
Custom rules use `custom: scoring.<function>` and declared output columns/dtypes.
`mode: combined` runs the `pre_split.yml` filters first, then these rules. (A model
set's separate `combined:` section computes one value from several models; it is not
this mode.) Do not
add Python builders beside YAML; legacy builders remain valid without YAML.
For example, to score only rows with a known, adult age:

```yaml
# config/scoring.yml
version: 1
mode: custom
skip_target_pre_split_steps: false
eligibility:
  - name: observed_age
    check: required_values
    params: {columns: [age]}
  - name: adult_age
    check: value_range
    params: {column: age, minimum: 18.0, maximum: 120.0}
```

`value_range` bounds are inclusive. A failing row is not deleted: it stays in the output with `scoring_status: excluded`,
its `exclusion_reason` and an empty prediction. See [scoring and publication](https://www.skyulf.com/manual/user_guide/databricks_scoring.html).

## Serving requests

This batch project has no `serving_endpoint` job. To add an endpoint:

1. Generate a new project in another folder with the same answers and
   `prediction_delivery: "both"`. Get the template with
   `git clone https://github.com/flyingriverhorse/Skyulf`, then from that folder:

   ```powershell
   databricks bundle init skyulf-core/templates/databricks --output-dir ../serving_copy
   ```

   The wizard asks the setup questions again. To skip them, pass
   `--config-file answers.json`, a JSON file of answers such as
   `{"project_name": "taxi_fare_robust", "prediction_delivery": "both"}`; unlisted
   questions keep their defaults, so list every answer you changed at setup.
2. Copy its new files: `config/serving.yml`, `resources/serving_endpoint.job.yml`,
   `src/jobs/serving_endpoint.py`, `src/jobs/dispatch_serving.py` and
   `deployment/serving-requirements.txt`. Copying generated files is fine; only do
   not hand-edit `src/jobs/*.py`.
3. Compare and merge its `config/training.yml`, `config/inference.yml`,
   `deployment/variables.yml` and `resources/train.job.yml` into yours.
4. Train a new version: models trained before the change have no serving checks.
5. Follow that project's README **Serve the model** steps: publish, check what is
   live, call it, monitor it.

Inspect serving verification and generated REST/SQL/SDK examples. Test all
required inputs; approval/submission does not prove which version is serving.

`serving_endpoint.follow_champion` requests automatic updates after completed
champion changes in production Bundle mode; personal development targets run
manually. REST `CAN_QUERY` and SQL-function `EXECUTE` grants are separate;
`function_permissions` declares SQL callers, who also need namespace access.
SQL is optional. Logging is optional and separately configured; endpoint and
SQL compute costs remain even when logging is disabled.

For serving monitoring, enable the target's monitoring, choose `telemetry` or
`ai_gateway` logging, and configure its store before endpoint changes. Successful
verification enrolls the version; actual requests and delivered logs are still
needed for observations, and real matched outcomes for performance. Use the
separate monitoring job after requests arrive. Learn rollout only after one
version works. See [serving and caller examples](https://www.skyulf.com/manual/user_guide/databricks_serving.html),
[serving monitoring](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html),
and [traffic rollout](https://www.skyulf.com/manual/user_guide/databricks_serving_rollout.html).

## Optional feature tables

Empty `groups: {}` creates no feature job. The comments in `config/features.yml` list
the steps and hold a complete two-group example to uncomment.
In `config/features.yml`, table names
are three-part and may use the `training.yml` placeholders (`{catalog}`,
`{input_schema}`, `{output_schema}`, `{metadata_schema}`, `{resource_suffix}`);
each target's feature job binds them. Each selected
`src/features/groups/<file>.py:<function>` receives its pinned Spark frame and
returns unique keys/time plus exactly the declared features; no YAML `params`.
Keep reads/collection/writes outside transforms and learned statistics in preprocessing.

Run `refresh_feature_graph.py`, validate/deploy, then the `features` job.
`features_start_training: "true"` (a Bundle variable) makes it start the train job
after the merge, so one scheduled features job drives features, training and the
training's score handoff.
Inspect `initialize_features`, `feature_<group>` and `merge_features` receipts.
Point training/scoring at the merged table explicitly. `selected_groups=*`
recomputes all; an empty selection reuses existing outputs and only merges.
`asof` selects latest available values, without cumulative sums or an implicit
freshness bound. Upserts do not propagate deletes or evolve schemas. Repair only
with unchanged code/config and original pinned inputs; otherwise start anew.
Groups are excluded from model code. Native UC lookup/online publication is separate.
See [feature production, joins and repair](https://www.skyulf.com/manual/user_guide/databricks_features.html)
and [online features](https://www.skyulf.com/manual/user_guide/databricks_online_features.html).

## Monitoring

To set it up:

1. **Turn it on** for the target in `deployment/targets.yml`:

   ```yaml
   targets:
     test:
       variables:
         monitoring_enabled: "true"
   ```

2. **Pick the store.** Results go to `monitoring_catalog`.`monitoring_schema` in
   `deployment/variables.yml`. If setup had monitoring off, both are empty: set them,
   for example to your catalog and `skyulf_monitoring`. Its tables are created on first
   use if the job identity may create a schema. Use one store for all projects so they
   share a dashboard.
3. **Dashboard (optional).** One project creates it, the others connect to it; the
   comments in `deployment/targets.yml` show both.
4. **First pinned version only:** prepare its reference once, then score again:

   ```powershell
   databricks bundle run -t test --profile YOUR_PROFILE monitoring --params monitoring_action=prepare_references
   ```

5. **Run scoring.** Each committed score run starts the monitoring job, which writes
   drift for that run.
6. **Performance needs real outcomes:** set `monitoring_label_table` and
   `monitoring_result_available_at_column` in `deployment/variables.yml`.

Personal development targets use their own monitoring schema and never retrain.
After preparing references, inspect the enrolled model and configured selectors. A one-run score override does not change monitoring's
persistent selector. For model sets, component monitoring requires `all` or
`separate_views`, not `combined_only`. Enable schedules only after checking references.

Drift is distribution change; performance metrics need real matched labels.
Performance policies additionally need mature windows and enough eligible labels.
Missing outcomes are not zero accuracy. Retraining opts in via `on_drift: retrain`
or policy `mode: retrain`, retaining freshness/cooldown/request guards. Dashboard
refresh reads results; reference cleanup is separate. See [monitoring setup](https://www.skyulf.com/manual/user_guide/databricks_monitoring_setup.html),
[enrollment](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html),
[labels](https://www.skyulf.com/manual/user_guide/databricks_monitoring_labels.html),
[performance policies](https://www.skyulf.com/manual/user_guide/databricks_performance.html),
and [retraining](https://www.skyulf.com/manual/user_guide/databricks_retraining.html).

## Deployment and schedules

Replace `REPLACE_*` hosts/namespaces and use the chosen target/profile. Deployment
credentials differ from job `run_as`; configure UC/model grants and writer ownership.
Shared `test`/`syst`/`prod` use production mode; personal targets isolate resources.
Training/scoring use the configured serverless Python environment.
Monitoring has separate serverless environment and budget controls.
Inspect resolved hosts, roots, suffixes and permissions with strict Bundle validation.

Deployment updates definitions; it does not run jobs or create source data.
Train/score/monitoring clocks are independent of data windows. `PAUSED` stops
schedule triggers, not manual/already queued runs. A scheduled `fixed_window`
keeps selecting the same interval; use rolling selection when it should advance.
Keep Core, generated graph and entrypoints aligned; preserve stable Bundle/root
and job identities on upgrade. See [deployment and operational controls](https://www.skyulf.com/manual/user_guide/databricks_deployment.html).

## Local tools

| Tool | Purpose and boundary |
| --- | --- |
| `smoke.py` | YAML, syntax and package checks; no editable hooks or cloud access. JSON passed/failed, exit 1 on a checked failure. |
| `preview.py --action train` | Resolve trusted factories and describe/validate training; no cloud rows or fit. |
| `preview.py --action score` | Inspect batch configuration; not applicable to endpoint-only delivery. |
| `preview.py --action approve` | Inspect saved-model approval configuration, not proof of an eligible cloud version. |
| `preview.py --action sources` | Explicit profile/warehouse source checks, as above. |
| `preview.py --list-models` / `--list-preprocessors` | Installed node metadata, not universal runtime availability. |
| `smoke.py --probe sample.csv` | Single-model local fit/admission probe on labeled CSV/Parquet; runs project code. |
| `refresh_training_graph.py` | Refresh changed model names/count/layout; parameter-only edits keep the graph shape. |
| `refresh_feature_graph.py` | Refresh feature graph and inherited execution controls before redeploying. |

Run beside `databricks.yml` with matching dependencies. Smoke/preview accept
`--catalog`, `--input-schema`, `--output-schema`, `--metadata-schema`,
`--resource-suffix`; they do not resolve the target file. Preview errors exit 2.
Local probe samples need actual inputs/target and adequate rows; real holdout
and remote checks remain necessary. See [tool details](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html#local-tools-and-preview).

## Failures and upgrades

Start from the first failed task and record its run ID.

| What failed | What to do |
| --- | --- |
| Training (any train-job task) | Fix the cause and start a **new** train run. Repairing or retrying the failed run is rejected. |
| Scoring, after an approval or promotion succeeded | Run the score job again; do not retrain or approve again. |
| Monitoring, after predictions were written (for example `Spark reference is not prepared`) | Predictions stay. Prepare the reference ([Monitoring](#monitoring) step 4) if that is the error, then run the **same** score request again; it writes nothing new and finishes the measurement. |
| An alias change whose outcome is unknown | Stop jobs that change aliases and follow alias recovery below; do not roll back instead. |

**Upgrading Core** (`deployment/requirements.txt`): a model loads only with the
library versions that trained it. After raising the pin, train a new version; to
use an older one again you need its old environment. Keep the old working project
and its versions until the new one is checked. Follow [ordered upgrades](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html),
[alias recovery](https://www.skyulf.com/manual/user_guide/databricks_lifecycle_recovery.html),
and [troubleshooting](https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html).
These local instructions match the generated revision; public guides may be newer.

## Score raw trips with Spark, REST and SQL

This project trains a pandas random forest on up to 4,096 sampled January trips
from `samples.nyctaxi.trips`. The train job fits and saves the filters and five
preprocessing steps. Its admission checks run automatically because
`batch_compute: spark`. The separate `save_preprocessing_diagnostics: true` adds diagnostics.

### Train and score

Use a Skyulf build that includes the recipe admission, excluded-row outputs and
endpoint compute settings shown here. When testing unreleased template changes,
replace the release pin in the deployment copy of `deployment/requirements.txt`
with the matching wheel and use that same build in the notebook. Updating the
template alone does not update a package already published on PyPI.

Follow the parent NYC examples README to load data, edit the workspace target
and deploy. Run `train`; automatic promotion requires RMSE <= 6 and at least
80% holdout coverage, below the standard 90% floor. This is deliberate: the
1.5-IQR rule on right-skewed trip distances can exclude valid long trips as
statistical outliers. Inspect their exclusion rate
and business impact before using this example's policy in production.
After promotion, the `score` job runs `mlflow.pyfunc.spark_udf` on Spark workers
and writes `trips_predictions`. It does not call a serving endpoint.

The fare range only selects training rows. The distance range is reused at
prediction time. A request needs no `fare_amount`. Out-of-range distances and
fitted IQR outliers retain their rows with `scoring_status = excluded`, an
`exclusion_reason` and an empty prediction. Missing numeric inputs pass through
Yeo-Johnson to the saved mean imputer. A zero distance is valid for Yeo-Johnson;
the preceding eligibility and IQR checks still apply.

SHAP explains sampled rows accepted by the saved preprocessing. Its JSON and
HTML report disclose sampled, accepted and excluded counts, including exclusion
reasons. Fewer than two accepted rows give `too_few_rows_after_exclusions`;
the sample is not topped up. Evaluation charts retain the original holdout
sample and plot its predicted rows, with excluded counts in captions and output.

Batch monitoring measures input drift, exclusion rates and observed performance.
This example's `deployment/targets.yml` binds `monitoring_label_table` to
`${var.catalog}.${var.input_schema}.trips_labels` and
`monitoring_result_available_at_column` to `result_available_at`. The training
configuration supplies the join key `trip_id` and target `fare_amount`. Run the
loader first: it creates outcomes with the trip's end time as availability time.
Only predictions with status `predicted` and an available, finite outcome enter
performance metrics; excluded rows are counted separately. This prepared project
also enables the report-only WAPE policy described below; it never requests training.
Some taxi outcomes have a zero fare, which can make MAPE extremely large.
Inspect MAE and RMSE alongside coverage; do not treat MAPE alone as a useful
relative-error summary for this data.
WAPE is also reported: total absolute error divided by total absolute actual
fare. It remains useful with individual zero fares but is unavailable when the
entire scored-and-labeled population has zero actual fares. A value of 0.1 means
10%. Outside this prepared project, performance policies remain opt-in.
Scoring labels keep their observed fare amounts; the training target-range filter
does not remove unusual outcomes from monitoring. A much larger RMSE than MAE
indicates that a small number of large errors dominate the squared error.

The loader initially scores the same January population used for training, so
that first observation is a wiring check. To observe later data without retraining,
run only its commented February inserts for `trips_score` and `trips_labels`,
then run the score job again. Leave `trips_train` unchanged. The incremental batch
receipt identifies the new prediction window; inspect its row counts and source
versions instead of treating the entire prediction table as one observation.

`pickup_zip` is a string, not a numeric measurement. By default, repeated categorical domains
with more than 50 reference values now compare the 50 most frequent reference
labels plus one pooled tail. The reference uniqueness ratio must be at most 0.5,
and neither side may exceed 10,000 categories. Inspect its `psi_categorical`,
statistical evidence and diagnostic `new_category_rate`; pooling can hide shifts
within rare ZIP codes. A supported ZIP comparison no longer degrades the report
solely because it exceeds 50 categories. Other unavailable checks can still do
so. Do not cast ZIP codes to numbers. See [Monitoring](#monitoring) for monitoring
setup and dashboard publication.

### Report performance loss without retraining

Each target in `deployment/targets.yml` includes `monitoring_performance_policies`
for `${var.catalog}.${var.metadata_schema}.taxi_fare_robust_model${var.resource_suffix}`.
The example uses `mode: report`, `metric: wape`, `direction: lower` and a
`training_holdout` baseline pinned to model version `"1"`. This initial version
assumes a fresh registered model. Before activation/scoring enrollment, replace
it with the concrete version you will observe, including when using an existing
model name or moving to a newly promoted version. Aliases such as `champion` are
not valid baseline versions.

The tolerance is 0.2 relative: a holdout WAPE of 0.05 permits values below 0.06.
A value at or above 0.06 indicates loss for that window. Each window spans 24
hours, waits six hours for outcomes, and needs at least 20 labeled predictions
and 80% label coverage. Two adjacent eligible loss windows meet the streak rule.
Even then, `mode: report` never requests training or changes serving traffic.

Deploy the producer, run activation/scoring to update enrollment, then run the
monitoring job. Its performance section reports the current value, saved baseline,
limit, reason and streak. These daily windows differ from a whole-month batch
metric; do not use February's aggregate WAPE as if it were a daily policy result.
An older prepared Spark reference without cached WAPE, a version mismatch, zero-total actuals
or insufficient labels gives unavailable/insufficient evidence. Historical
training-run metrics are not rewritten. Use a compatible prepared reference or select an
explicit supported production-window baseline before expecting a verdict.

To disable the policy, set that model's entry to `{"mode":"off"}`. Use the
[performance policy guide](https://www.skyulf.com/manual/user_guide/databricks_performance.html)
for the full configuration and other baseline choices.

### Create the endpoint explicitly

No endpoint or online feature store is created by `bundle deploy` or `train`.
This example's `config/serving.yml` is read by the ordinary Python file
`src/serve.py`; it is not the
online-feature setup configuration. It uses the same public pinned-endpoint
SDK described in [Serving requests](#serving-requests).

1. Read the approved **concrete model version** from the training report. Fill
   `endpoint.model_version` in `config/serving.yml`. Check its three-part
   `model_name`, endpoint name and `sql_function`, especially on another target.
2. Set the four compute/readiness fields in that file if needed. The default is
   CPU / Small / scale to zero / 3,600 seconds. Workspaces can restrict these
   choices. With `logging_mode: disabled`, there is no endpoint payload table
   or serving monitoring; batch prediction monitoring remains separate.
3. Deploy the edited files. Open a Databricks Python notebook with the **same
   requirements as training** and run the following cells deliberately. Replace
   both paths with the deployed files' paths (Bundle root, then `files/`).
   Names must be new; creation never overwrites an existing endpoint or function.

```python
import runpy

helpers = runpy.run_path("/Workspace/<deployed project>/files/src/serve.py")
config_path = "/Workspace/<deployed project>/files/config/serving.yml"
function = helpers["prepare"](config_path)  # Read-only inspection.
print(function.create_sql)
```

In a second cell, create the endpoint, wait for readiness and create the SQL function:

```python
function = helpers["create"](spark, config_path)
```

A readiness timeout leaves the endpoint intact. Inspect its build logs, then
resume the existing endpoint using the `function` prepared in the first cell:

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import (
    create_serving_sql_function,
    wait_for_pinned_endpoint_ready,
)

client = WorkspaceClient()
wait_for_pinned_endpoint_ready(client, function.endpoint)
# A timeout during create() occurs before SQL function creation. If another
# run already created the function, omit this line after checking its identity.
create_serving_sql_function(spark, client, function)
```

Do not rerun endpoint creation blindly. Delete the endpoint when finished so
idle or active compute cannot keep charging; drop the example function after
its callers are stopped. [Serving requests](#serving-requests) explains the API details.

### Compare REST and SQL

In that notebook, using the prepared `function` from the previous cells:

```python
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import query_named_records

records = [
    {
        "pickup_hour": 10.0,
        "pickup_dow": 3.0,
        "trip_distance": 2.0,
        "duration_min": 12.0,
        "pickup_zip": "10001",
    },
    {
        "pickup_hour": 10.0,
        "pickup_dow": 3.0,
        "trip_distance": 99.0,
        "duration_min": 12.0,
        "pickup_zip": "10001",
    },
    {
        "pickup_hour": 10.0,
        "pickup_dow": 3.0,
        "trip_distance": None,
        "duration_min": None,
        "pickup_zip": "10001",
    },
]
print(query_named_records(WorkspaceClient(), function.endpoint, records).predictions)
```

The second row demonstrates a distance that passes the fixed 0-100 range but
is expected to exceed the **learned** IQR bound; inspect its actual reason.
The third demonstrates missing values. The fitted model, not the example code,
determines each result. SQL uses the typed function to preserve numeric NULLs
for imputation; a bare `ai_query` call can coerce a numeric NULL to zero.

```sql
SELECT s.trip_id,
  workspace.taxi_fare_robust.score_fare_v1(
    pickup_hour => s.pickup_hour, pickup_dow => s.pickup_dow,
    trip_distance => s.trip_distance, duration_min => s.duration_min,
    pickup_zip => s.pickup_zip) AS result
FROM VALUES
  (1, 10D, 3D, 2D, 12D, '10001'),
  (2, 10D, 3D, 99D, 12D, '10001'),
  (3, 10D, 3D, CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE), '10001')
AS s(trip_id, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip);
```

The function wraps `ai_query`; it returns a struct with prediction and status.
Your SQL can store that result with `CREATE TABLE ... AS SELECT ...`.
Never interpret an excluded row's empty prediction as a zero fare.
