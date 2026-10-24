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
    return column_step(f"log_{column}", log1p_value, output=f"log_{column}", params={"column": column})
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

This project has `promotion_policy=automatic`.
Read **training_report > Output > Where to look** for version, fitting run,
holdout evidence and decision. Inspect the outcome: a report can finalize failure.
Copy **Available action** parameters into a train job run for
approve/reject/rollback; first approval expects champion `none`. Keep receipts.
Child score/serving completion is separate. See [training fields](https://www.skyulf.com/manual/user_guide/databricks_training_config.html),
[reports](https://www.skyulf.com/manual/user_guide/databricks_training_reports.html),
and [model lifecycle](https://www.skyulf.com/manual/user_guide/databricks_lifecycle.html).

## Batch predictions

This project has endpoint-only delivery: no batch score job, watermark or
Delta prediction destination. `inference.yml` retains the endpoint smoke source.
Use [Serving requests](#serving-requests); the saved scoring-rule contract below
also applies to endpoint predictions.

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

Configure `serving_endpoint` in `config/serving.yml` and the target's
`cost_tag_value` before running `serving_endpoint`. It resolves champion once;
`--params model_version=N` selects a concrete registered version for that run.
Use the actual version, not another project's sample number.
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

## Call the taxi endpoint and monitor its requests

First complete [deployment and training](README.md#run-the-example). Successful
champion promotion submits `serving_endpoint`, which creates or updates the
endpoint, verifies REST/SQL predictions against the saved model, and enrolls
the served version for monitoring. Check that child run, including its
`verification` and `monitoring` results, before making requests.

### 1. Call the generated SQL function

With the example's unchanged `test` bindings, the endpoint is
`taxi-fare-endpoint-test-endpoint` and its SQL function is
`workspace.taxi_fare_endpoint.taxi_fare_endpoint_score`. If you change bindings,
use the exact function name and query from the serving job's `examples.sql`.
Callers need function `EXECUTE`, endpoint `CAN_QUERY` and catalog/schema usage.

Run this query on a Databricks SQL warehouse supporting `ai_query`:

```sql
SELECT s.trip_id,
       workspace.taxi_fare_endpoint.taxi_fare_endpoint_score(
         pickup_hour => s.pickup_hour,
         pickup_dow => s.pickup_dow,
         trip_distance => s.trip_distance,
         duration_min => s.duration_min,
         pickup_zip => s.pickup_zip
       ) AS result
FROM (
  SELECT trip_id, pickup_hour, pickup_dow, trip_distance, duration_min, pickup_zip
  FROM workspace.taxi_fare_endpoint.trips_score
  ORDER BY trip_id
  LIMIT 100
) AS s;
```

The function calls the endpoint through `ai_query`; `result` contains the model's
prediction and scoring fields. This `SELECT` displays results without creating
a batch prediction table. REST/SDK request examples are also in the serving run's
output. Requests use raw feature values; the endpoint applies saved preprocessing.

### 2. Read the captured traffic

`logging_mode: telemetry` captures REST and SQL requests. By default Databricks
creates `workspace.taxi_fare_endpoint.taxi_fare_endpoint_serving_payload` and its
backing `taxi_fare_endpoint_serving_otel_logs` table. Treat them as Databricks-managed
logging objects. Monitoring reads them; it does not create a batch prediction sink.

Log delivery is asynchronous. Once the payload view has received requests, run:

```bash
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

The job saves captured request counts, errors, latency and available feature-drift
measurements in `workspace.taxi_monitoring`. An empty observation immediately after
a request can mean logs have not arrived yet; rerun after delivery. Enrollment by
itself creates no traffic measurements. Monitoring is enabled on shared targets;
the generated hourly schedule starts paused. After the first successful manual
observation, set `monitoring_pause_status: UNPAUSED` in your target's variables
and redeploy to collect observations automatically.

### 3. Open Online serving

Reuse the dashboard for `workspace.taxi_monitoring` if one exists: copy the
commented `connect` example in `deployment/monitoring.yml` under `test` and paste
its published URL and ID. If this is the first project owning that store's
dashboard, copy the `create` example instead and set your SQL warehouse ID; the file
explains where to find it. Then deploy and run monitoring:

```bash
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

Use the published link from the monitoring run and choose **Online
serving**, model `workspace.taxi_fare_endpoint.taxi_fare_endpoint_model`, then the
served version/context. Training charts and SHAP are available in the MLflow run.

**Performance needs actual outcomes.** This example leaves online label binding
and performance-loss policies off. The shared loader's `trips_labels` uses
`trip_id`; online monitoring needs `databricks_request_id` and zero-based
`request_row_index` to match each prediction. Do not bind that table directly.
See [serving monitoring](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html)
to supply matched labels and configure a performance policy.

Automatic champion following replaces the endpoint's single served version; it
does not enable gradual traffic rollout. To retire the example, disable
`serving_endpoint.follow_champion`, redeploy, then remove its endpoint and SQL
function explicitly. Destroying Bundle jobs alone does not remove those resources.
