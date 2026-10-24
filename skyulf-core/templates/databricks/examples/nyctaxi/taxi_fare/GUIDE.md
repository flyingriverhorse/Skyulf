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

## File map

| Change | Owner |
| --- | --- |
| Training source, keys, inputs/target, split, model, tuning, weights, quality | `config/training.yml` |
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

Change models/recipes without editing job notebooks. Training captures selected
preprocessing, scoring rules and code; later edits do not alter saved versions.

Do not edit `src/jobs/*.py`: they are thin entry points that a template upgrade
replaces, and the job logic ships in the Core wheel. Model code belongs in
`src/features/`, job settings in `deployment/`, and a private package index or
bundled wheels for logged models in the `SKYULF_MODEL_PIP_*` job variables; see
[serving packages](https://www.skyulf.com/manual/user_guide/databricks_serving.html).

## Source data

The selected arrangement is **single_table**. Initialization/deployment do
not create or populate input tables. Bind existing namespaces and choose immutable,
nonnull, unique supported record keys identifying observations, not just people.
Keep keys/targets out of model inputs; matching outcome keys need identical types.

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

This is read-only, but uniqueness checks can scan tables. It checks the supplied
profile's access, not another run identity or future data. Add the target's
`--resource-suffix` when needed. No DDL starter was selected. If needed, generate a separate project with
`include_source_sql: "true"` and adapt its SQL without overwriting this project.
See [source contracts and time windows](https://www.skyulf.com/manual/user_guide/databricks_training_data.html).

## Preprocessing and custom code

The order is source selection → fixed cleanup/eligibility → final split →
fold-local preprocessing/model fit. Keep learned means, quantiles and categories
after splitting. Both phase YAML files have `version: 1`, `recipes`, and a
`default` recipe; empty recipes do nothing. Example recipes run only when selected.
Select `pre_split_recipe` and `preprocessing_recipe` in training settings.

Prefer a registered step's `name`, `transformer` and `params` when suitable.
`custom: my_rules.make_feature` selects a top-level factory in
`src/features/my_rules.py`; YAML `params` become keyword arguments.
It returns a Core step, not a DataFrame. Import `filter_step` for fixed row
eligibility, `column_step` for fixed columns, or `fitted_step` for learn/apply
state from `skyulf.preprocessing`. Learned JSON-compatible state is fitted per
training fold and reused on holdout/prediction. Do not combine YAML ownership
with Python builders or add inline pipeline steps to `training.yml`.

Keep imports/factories free of I/O. Preserve keys and protected time/group/weight
metadata. Use relative imports for packaged helpers; never import `groups/`
from model callbacks. Declare small data files in `src/features/assets.json`,
read them with `read_project_asset`, and declare exact package pins in
`src/features/requirements.txt`. New code, assets or dependencies require a new fit.

Spark workers and endpoints require admission for the exact saved recipe.
Eligible built-ins and custom steps/rules undergo training checks; declaring
row-local context alone is not certification. Avoid index-dependent behavior
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
Copy **Available action** parameters to the serialized train job for
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

Inspect concrete model/source versions, input/output counts, exclusions and
the Delta receipt. Incremental replay with the same model and unchanged source
should be a no-op. `on_model_change` governs model replacement; ordinary single
model insert scoring rejects source UPDATE/DELETE. Model sets have their own
`on_source_change` policy. Rebuilds and CDF-expiry recovery are explicit options;
recovery needs the matching generated graph and retained evidence. Do not reset
watermarks to retry. Read **score > Output**; this graph has no `scoring_report` task.
Workers run admitted pandas-fitted MLflow pyfunc batches with matching dependencies.
Polars whole-frame behavior and arbitrary history/context do not become worker-safe.

In `config/scoring.yml`, `mode: pre_split` reuses saved eligibility; `custom`
uses its declared lists; `combined` applies pre-split first and retains its first
exclusion reason. Target-reading filters require explicit
`skip_target_pre_split_steps: true` when appropriate; the whole step is skipped,
so separate target and feature filters when feature checks must remain.
Built-ins include `required_values`, `value_range` and `prediction_band`.
Custom rules use `custom: scoring.<function>` and declared output columns/dtypes.
The `combined` list is cross-model arithmetic, not the eligibility mode. Do not
add Python builders beside YAML; legacy builders remain valid without YAML.
See [scoring and publication](https://www.skyulf.com/manual/user_guide/databricks_scoring.html).

## Serving requests

This batch project has no `serving_endpoint` job. Generate/review the documented
endpoint or both-delivery configuration before using this route. Existing saved
models need serving admission evidence; changing delivery does not certify them.
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

Empty `groups: {}` creates no feature job. In `config/features.yml`, table names
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

Use the target's `monitoring_enabled` and catalog/schema bindings; storage is
independent of training/model/output destinations. Share one intended store and
dashboard owner; initialize/upgrade it before producers without creation grants.
Personal development monitoring explicitly opts into a separate schema and never
enables development retraining. Inspect the actual target block.

Prepare missing references with the monitoring job's
`monitoring_action=prepare_references`; then inspect concrete model enrollment
and configured selectors. A one-run score override does not change monitoring's
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

Identify the first failure and last confirmed result. A failed training phase
does not support generic repair/resume; inspect evidence before a new run. A
completed alias change and failed scoring need downstream recovery, not another
fit. If scoring committed but its monitoring failed, retry the **same score
request**, preparing references if necessary, so the existing receipt can be
reused. An uncertain alias outcome requires reconciliation, not guessed rollback.

Before changing `deployment/requirements.txt`, preserve the working project,
artifact/version, dependencies and receipts. Compare a fresh template separately;
align graph/controller/runtime and upgrade metadata owners in the documented order.
A changed pin does not migrate models. Prove actual load/predict in compatible
environments; restored aliases alone are insufficient. Follow [ordered upgrades](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html),
[alias recovery](https://www.skyulf.com/manual/user_guide/databricks_lifecycle_recovery.html),
and [troubleshooting](https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html).
These local instructions match the generated revision; public guides may be newer.
