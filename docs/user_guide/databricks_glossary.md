# Databricks project glossary

Use these meanings when reading the [project walkthrough](databricks_bundle_walkthrough.md).
The file/action column tells you where to make a decision or inspect its evidence
in a generated project. Configuration names describe this checkout; use the
matching Core release/build for your project revision.

## Data and preprocessing

| Term | Plain meaning | File or action |
| --- | --- | --- |
| Record key | Stable columns identifying a source record and its published result. These are data identity, not a DataFrame's temporary row index. | Set `record_key_columns` in `config/training.yml`; verify uniqueness in the source. |
| Target / label | The actual outcome the estimator learns to predict, such as observed revenue or whether a customer churned. A prediction is not an actual label. | Set `target_column` in `config/training.yml`; maintain real outcomes upstream. |
| Pre-split | Fixed edits and eligibility filters applied before dividing labeled rows into training and holdout populations. They must not learn values from the complete dataset. | Select `pre_split_recipe` from `config/pre_split.yml`; decide target-filter reuse in `config/scoring.yml` before preview. |
| Fitted preprocessing | Transformations whose learned state, such as imputation means or scaling statistics, is fitted on training rows and reused for evaluation and prediction. | Define recipes in `config/preprocessing.yml`, select `preprocessing_recipe` in `training.yml`, then train a new version after edits. |
| Holdout | Rows reserved from model fitting for the final evaluation. They are distinct from the validation folds used during tuning. | Set split/window fields in `config/training.yml`; inspect the saved split membership and held-out metrics in `training_report`. |
| Cross-validation (CV) | Repeated train/validation splits within the training population, used to estimate or select model settings. It does not replace the final holdout. | Set `cv_enabled`, `cv_type` and related fields in `config/training.yml`; inspect saved fold reports. |
| Training window | The source time range eligible for a training run, before the final holdout boundary and other eligibility rules are applied. | Set `training_window_mode`, event/date parsing and window fields in `config/training.yml`. |
| Observation window | The prediction or request-time interval measured by monitoring. A completed performance window may wait for actual outcomes before assessment. | Set `window_hours` and `label_delay_hours` in `monitoring_performance_policies`; inspect monitoring reports. This differs from the training window and the job schedule. |
| Recipe | A named, ordered list of steps in `config/pre_split.yml` or `config/preprocessing.yml`. Only the selected recipe runs. | Select `pre_split_recipe` and `preprocessing_recipe` in `config/training.yml`. See the [recipe catalog](preprocessing_recipes.md). |
| Preprocessing probe | An optional check that replays the saved preprocessing on up to 256 holdout rows, whole, repeated, in chunks and reversed. It does not refit or block promotion. | Set `save_preprocessing_diagnostics: true`; read [preprocessing diagnostics](preprocessing_context.md#how-to-read-the-report). |
| Row, group, window and global context | What a step needs at prediction time: one row and saved state (`row`), all rows of a group (`group`), ordered earlier rows (`window`) or the whole batch (`global`). | Read [what context means](preprocessing_context.md#what-does-context-mean). |
| Outcome availability | When a real label became available for use. This can be later than the underlying event or prediction. | Configure `result_available_at_column` and cutoff rules for training; configure `monitoring_result_available_at_column` for monitoring labels. |

Read the [recipe catalog](preprocessing_recipes.md),
[training field reference](databricks_training_config.md) and
[context-dependent preprocessing guide](preprocessing_context.md). For the target-reading filter
error, see [troubleshooting](databricks_troubleshooting.md#a-pre-split-step-reads-the-target).

## Model lifecycle and scoring

| Term | Plain meaning | File or action |
| --- | --- | --- |
| Candidate | A fitted, registered model version with saved evaluation evidence. Registration alone does not make it active. | Run the train job, then inspect `training_report` and its concrete version. |
| Champion | The active model version selected through the lifecycle's `champion` alias. The alias can move; the concrete version identifies one saved artifact. | Use the approved activation action and inspect its receipt. `score_model_selection: champion` resolves the active version for scoring. |
| Challenger | A candidate staged for comparison or an approval decision. Its existence does not route production requests or replace the champion. | Inspect the `challenger` alias and saved candidate report; use approve/reject according to policy. |
| Alias | A named pointer such as `champion` or `challenger` to a registered version. It is mutable, unlike a concrete version number. | Inspect registry aliases and lifecycle receipts. Use `score_model_selection: pinned_version` with `model_version` when a fixed score version is required. |
| Promotion / activation | A guarded decision to make a candidate active. A passing training task or deployment alone is insufficient. | Set `promotion_policy`, metric and quality thresholds in `config/training.yml`; inspect the activation receipt. |
| Competition | Several estimators compete for one target under shared data, split and selection rules. Only the activated winner becomes the production model. | Use `training_layout: model_competition` and candidate entries in `config/training.yml`. |
| Model set / multi-target | One coordinated release containing independent models for several targets, such as revenue and cost. It is not a competition for one winner. | Use `training_layout: multi_target`; define branches in `config/training.yml`, and set activation/publication policy in `config/inference.yml`. See [model sets](databricks_model_sets.md) and its [replacement rule](databricks_model_sets.md#quality-policy-and-replacement-rule). |
| Scoring eligibility | Rules determining whether an input receives a prediction. Excluded rows remain in the result with their status and reason. | Configure `mode`, target-step skipping and rules in `config/scoring.yml` before fitting the version. |
| Publication | Committing predictions and their metadata to the output Delta table, optionally exposing selected model-set views. This does not promote a registry version. | Set output and publication policy in `config/inference.yml`; inspect the score task and committed output. |
| Change Data Feed (CDF) | Delta's record of source inserts, updates and deletes over retained versions. Incremental scoring uses it to discover source changes after a saved checkpoint. | Enable CDF on the configured source upstream. Ordinary single/competition scoring accepts inserts; only model sets expose `model_set.on_source_change` in `config/inference.yml` for readable corrections. Use the documented recovery action if required history expires. |
| Full rebuild | Recompute the configured prediction population under the selected model rather than only processing changes since a checkpoint. | Review model/source change policy and the explicit recovery action in `config/inference.yml`; inspect the resulting output generation. |
| Expected champion and `none` | The champion version an operator action expects to replace. The action fails if the alias moved. `none` means no champion exists yet. | Copy it from the report's next actions. See [operator actions](databricks_lifecycle.md#operator-actions-through-run-now). |
| `previous_champion` / `previous_challenger` | History aliases. Promotion moves the old champion to `previous_champion`; a new candidate moves the displaced challenger to `previous_challenger`. | Inspect registry aliases. See [challenger history](databricks_lifecycle.md#recent-challenger-history). |
| `next_actions` | Ready-to-copy parameters in the training report for the next approve, reject or rollback run. | Open **training_report > Output**. See [operator actions](databricks_lifecycle.md#operator-actions-through-run-now). |
| `lifecycle_action` | The train job's Run now parameter: `train`, `approve`, `reject` or `rollback`. Only `train` fits a model. | Set it with **Run now with different parameters**. See [operator actions](databricks_lifecycle.md#operator-actions-through-run-now). |
| Comparison digest | A SHA-256 hash of the reviewed candidate, data, metric policy and expected champion. Approval checks it instead of reevaluating. | Usually resolved from the candidate's saved evidence. See [approve without training](databricks_lifecycle.md#approve-an-existing-candidate-without-training). |
| Promotion receipt | The saved record of a completed alias change. Rollback needs it to restore the prior champion. | Copy `promotion_receipt_json` from the report. See [reject or roll back](databricks_lifecycle.md#reject-a-candidate-or-roll-back-a-promotion). |
| Prediction receipt | The saved record of a committed scoring write: source identity, versions and output. Recovery reads it before any retry. | Inspect the score task output. See [scoring](databricks_scoring.md#how-new-rows-are-discovered). |
| Watermark | The last committed source Delta version. The next score run reads changes after it. | Never edit it by hand. See [how new rows are discovered](databricks_scoring.md#how-new-rows-are-discovered). |
| `follow_champion` | Whether a successful champion change starts the score job (`true`) or not (`false`). It is fixed at deployment. | Set it in `config/inference.yml`, then redeploy. See [operator actions](databricks_lifecycle.md#operator-actions-through-run-now). |
| Handoff | A saved request from one job to another, such as activation requesting scoring or scoring requesting monitoring. It carries concrete evidence rather than leaving the child to guess the current alias. | Set `follow_champion` in `config/inference.yml`; follow the native child job in the run graph. |

Read [registry actions](mlflow_registry.md), [scoring and recovery](databricks_scoring.md)
and [training policy fields](databricks_training_config.md#quality-weights-and-bounded-reports).

## Features and online requests

| Term | Plain meaning | File or action |
| --- | --- | --- |
| Feature group | An optional upstream job unit that reads one source, runs the configured transform, and writes its own feature output table. | Add a group in `config/features.yml` and its transform under `src/features/groups/`; refresh the feature graph, deploy and run the feature job. |
| Merged table | The optional output obtained by joining the configured base table with feature-group outputs using declared keys and timestamp rules. It is a source for training/scoring only when you select it. | Configure the merge in `config/features.yml`, then point `training_table` and `score_source_table` to its output. Empty `groups: {}` creates neither a feature job nor a merged table. |
| Feature lookup | Retrieval of explicitly named features from a Unity Catalog feature table by entity keys and optional time rules. It is separate from the template's feature-group merge job. | Configure `feature_lookup` in `config/training.yml`; verify key/feature metadata and input columns. |
| Point-in-time lookup | A lookup that selects historical features according to each observation's timestamp and configured lookback, avoiding the use of later values. | Configure timestamp lookup settings in `feature_lookup`; inspect saved training lookup/snapshot evidence. |
| Online publication / synchronization | Copying eligible feature data into the configured online feature service so an endpoint can retrieve it. Publishing features does not train a model or deploy an endpoint. | Configure and run the optional publication/sync steps in the online feature guide and generated `SERVING.md`. |
| Online lookup | A live entity-key lookup made while handling a prediction request. Missing or stale required features can reject the request. | Configure the model's `online_features` required columns/freshness rules and the serving feature lookup contract. Inspect the endpoint response and logs. |
| Pinned endpoint | A serving endpoint tied to a concrete registered model version and inspected input/output contract. | Use the pinned serving helpers and generated `SERVING.md`; keep the endpoint, version and inference-log identity together. |
| Rollout | A separate, guarded process controlling serving traffic or stage progression. A champion alias update alone does not establish that traffic changed. | Configure the optional `config/serving.yml` rollout and inspect saved health/decision evidence. |

Read [feature groups and merged tables](databricks_features.md),
[online features](databricks_online_features.md) and
[serving rollout](databricks_serving_rollout.md).

## Runtime, engine and operations

| Term | Plain meaning | File or action |
| --- | --- | --- |
| Engine | The DataFrame execution implementation selected for fitted Core preprocessing, such as pandas or Polars. It is not a package version or cluster size. | Set `engine` in `config/training.yml`; follow the supported engine/route combinations in the inference reference. |
| Runtime | The installed Python, Skyulf and dependency environment required to load and execute a saved artifact. A different engine or compute target does not make incompatible dependencies safe. | Inspect `deployment/*requirements.txt`, the saved artifact manifest and job environment. Install the matching release/build. |
| Inference mode | The route that applies a saved model, such as local scoring or the supported Spark UDF route. Spark inference does not turn training into Spark-native fitting. | Set `batch_compute` and Spark UDF options in `config/inference.yml`; verify the selected artifact's support boundary. |
| Admission / certified package | Spark workers score only a saved MLflow package with a safety certificate. Admission rechecks it and rejects unsupported steps. | Log a new certified version if admission fails. See [distributed inference](databricks_spark_scoring.md). |
| `workflow_contract` | A version marker on the generated job tasks. The runtime rejects a deployed graph and project that disagree. | Deploy the matching files together. See [project upgrade](databricks_project_upgrade.md). |
| Production-mode target | The shared `test`, `syst` and `prod` targets. Only they enroll models in monitoring; personal development targets skip it. | Select the target with `-t`. See [deployment](databricks_deployment.md). |
| Monitoring enrollment | The shared inventory entry binding a project/model/version to its prediction evidence, labels and policy. It is separate from a dashboard display. | Inspect activation `register_monitor` and the score-to-monitor handoff; configure monitoring target variables. |
| Dashboard refresh | Rerunning the dashboard's published datasets over saved monitoring evidence. It neither computes a new observation nor retrains the model. | Configure the native refresh task with `src/tools/configure_monitoring_dashboard.py`; inspect the monitoring job separately. |
| Drift | A measured change in feature distributions or declared schema checks. It is not automatically a loss in predictive quality. | Inspect the saved monitoring report and drift thresholds; enable guarded retraining separately if desired. |
| Performance degradation | A measured worsening against a matching baseline under an explicit metric, tolerance, window and label-coverage policy. Missing labels are unavailable evidence. | Configure `monitoring_performance_policies`; read the saved window and baseline evidence before interpreting its verdict. |

Read [inference artifacts and runtime checks](inference_bundles.md),
[Spark execution boundaries](spark.md) and the [monitoring overview](databricks_monitoring.md).
