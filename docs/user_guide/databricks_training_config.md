# Training YAML: fields and complete examples

Use `config/training.yml` to choose the data, target, estimator, evaluation and
promotion policy for each model. This reference explains every accepted training,
model and tuning field, then supplies a complete file for each layout. The result
of training is a fitted registered candidate with source/split/quality evidence;
activation and scoring remain separate decisions.

Use a Core package matching your generated project; see
[runtime packaging](databricks_deployment.md#runtime-packaging). These local
configuration examples do not create Delta tables, contact Databricks or prove
cloud permissions. For first deployment and operator actions, follow the
[Bundle walkthrough](databricks_bundle_walkthrough.md).

## File ownership and inheritance

| File or section | What belongs here |
| --- | --- |
| `training.yml` > `defaults` | Shared data, engine, split/CV, budgets, gates and optional model settings |
| `training.yml` > `models.<name>` | One model or candidate's explicit overrides |
| `pre_split.yml` | Fixed training eligibility and normalization before splitting |
| `preprocessing.yml` | Ordered fitted transformations; selected recipes fit inside training folds |
| `inference.yml` | Scoring source, version selector, publication/change policy and score handoff; model-set activation/publication |
| `deployment/variables.yml` and target overrides | Job compute, identities, schedules and environment bindings |
| `src/features/` | Trusted custom functions and scoring policy selected before training |

The tables below apply to shared defaults and individual model entries unless
their conditions say otherwise. A model entry **replaces the whole value** of a
shared field; nested mappings are not deep-merged. For example, overriding
`model` replaces its type and entire params mapping. Repeat the parameters you
still need. Do not define another Python model declaration or `workflow.json`
beside these YAML owners.

Competition candidates share one source, target, split, CV and selection policy.
Their entries can override only `model`, `tuning`, `decision_threshold`,
`preprocessing_recipe` and `explainability`. Competition does not accept
`features_path` or `pre_split_recipe` in its defaults; use the default pre-split
recipe. Independent target models can have different labels, inputs, recipes,
models and gates, but must share the exact source snapshot and record keys.
Never include any sibling target in a branch's inputs.

## Root structure

| Key | Type/default | Meaning |
| --- | --- | --- |
| `version` | Integer; required `1` | Version of this YAML document, independent of the runtime config version below. |
| `defaults` | Mapping; `{}` if omitted | Fields inherited by every named model. Complete effective settings still need the required fields. |
| `models` | Nonempty mapping; required | Safe names starting with a letter, at most 64 ASCII letters/digits/underscore/hyphen. Single layout has one entry; competition needs at least two. |
| `pipeline` | Optional mapping; usually omit | Only empty `preprocessing: []`, empty `modeling: {}` and shared `explainability` are accepted. Recipe lists belong in their own files. Do not duplicate explainability here and in defaults/models. |

YAML uses real numbers, booleans and `null`, rather than the string answers used
by the init wizard. Quote dates such as `"2026-01-01T00:00:00Z"` so YAML does not
turn them into timestamp objects. Unknown/duplicate keys, aliases, nonfinite
numbers and documents larger than 64 KiB are rejected.

## Training fields

**Default** means the generated quick-project value, or the runtime default
where a field is omitted. Required fields must be present after inheritance.
Inactive optional fields should be omitted or `null`, as specified. The last
column identifies the corresponding init answer; **YAML** means edit the field
directly. Detailed setup exposes more init answers than quick setup. Branch
answers use the same question name with a `branch_N_` prefix.

<!-- training-fields:start -->

### Identity, columns and runtime

| Field | Type | Default/required | Values and conditions | What changes | Init answer or owner |
| --- | --- | --- | --- | --- | --- |
| `config_version` | Integer | Required `1` | Runtime workflow contract | Keep `1`; not a model version. | Generated |
| `training_layout` | String | `single_model` | `single_model`, `model_competition`, `multi_target` | Chooses one model, one-target competition or independent target branches. Changing it requires the matching generated task graph. | `training_layout` |
| `task` | String | `regression` | `regression`, `classification`; must match estimator | Chooses label/metric semantics. Multi-target entries can differ. | `task` |
| `engine` | String | `pandas` | `pandas`, `polars`; distributed fitted scoring requires pandas | Chooses local training and fitted preprocessing engine, not compute size. | `engine` |
| `training_table` | String | Required | Three-part Delta name after target binding | Source of labeled rows; generation/deployment does not create it. | `source_table_name` plus catalog/input schema |
| `training_version` | Integer or null | Omitted/null | Nonnegative Delta version, including `0` | Null resolves latest once per invocation; an integer pins that snapshot. | `training_version` |
| `record_key_columns` | String list | Required | Nonempty, nonnull, unique source identities; distinct roles | Stable row membership, split evidence and output alignment. Generated publication uses string/BIGINT keys. | `record_key_columns` |
| `input_columns` | String list | Required | Ordered raw model inputs; excludes keys, targets, weights and time/group control roles | Selects features before saved preprocessing. | `input_columns` |
| `target_column` | String | Required | One source label, distinct from other roles | Chooses the prediction target; omit it from future scoring input. | `target_column` |
| `model_name` | String | Required | Three-part registry name in the active metadata schema; include deployment suffix | Names the registered candidate. Multi-target branches need distinct names. | Generated from project/branch name; YAML override |
| `tracking_uri` | String | Runtime default `databricks`; usually omitted | Tracking store setting, credentials supplied separately | Chooses experiment/run storage. | YAML |
| `registry_uri` | String | Runtime default `databricks-uc`; usually omitted | Registry store setting, credentials supplied separately | Chooses version/alias storage independently of tracking. | YAML |

### Final holdout, sampling and memory

| Field | Type | Default/required | Values and conditions | What changes | Init answer or owner |
| --- | --- | --- | --- | --- | --- |
| `split_strategy` | String | `random` | `random`, `temporal` | Random selects a fraction; temporal fits earlier rows and holds out the later interval. | `split_strategy` |
| `test_size` | Number or null | `0.2` | Random: strictly between 0 and 1; temporal: omit/null or retain inactive `0.2` | Final holdout fraction, separate from CV folds. | `test_size` |
| `random_state` | Integer or null | `42` | Random: 0 through 2**32-1; temporal: omit/null or retain inactive `42` | Random holdout seed. | `random_state` |
| `stratify` | Boolean or null | `false` | True only for random classification; group holdout requires false | Retains class proportions where supported. | `stratify` |
| `training_sample_rows` | Integer or null | Omitted/null | At least 4 and at most `max_rows` | Seeded bounded Spark-side selection before local transfer and splitting; does not sample predictions. | `training_sample_rows` |
| `training_sample_seed` | Integer | `42` | 0 through 2**32-1 | Reproduces requested sample membership; saved evidence pins it for approval. | `training_sample_seed` |
| `max_rows` | Integer | `10000` | Positive; aggregate fit/holdout input must fit | Rejects a larger local training population instead of silently truncating it. | `max_rows` |
| `max_input_mb` | Integer | `64` | Positive whole MiB; one unit is 1,048,576 bytes | Bounds decoded/local training data, not Spark wire buffers or total process memory. | `max_input_mb` |

### Source windows and result availability

| Field | Type | Default/required | Values and conditions | What changes | Init answer or owner |
| --- | --- | --- | --- | --- | --- |
| `training_window_mode` | String | Quick: `full_snapshot` | `full_snapshot`, `fixed_window`, `rolling_calendar`, `rolling_days`; init-only `auto` resolves before emission | Selects source rows independently of final splitting and scheduling. | `training_window_mode` |
| `event_column` | String or null | Omitted/null | Required for window selection or temporal split; absent for full snapshot | Observation clock used for selection/ordering, not label availability. | `event_column` |
| `event_time_parsing` | Mapping or null | Omitted/null | `format`, `timezone`, `date_only`; see nested table | Explicitly interprets non-native event dates; no guessed string parsing. | `event_time_format`, `event_time_timezone`, `event_date_only` |
| `start` | Quoted timestamp or null | Omitted/null | Aware ISO instant; required for fixed window | Inclusive observation-window start. Rolling modes derive it at invocation. | `start` |
| `cutoff` | Quoted timestamp or null | Omitted/null | Aware ISO instant, after start; required for fixed window | Exclusive observation-window end; independent of source commit time. | `cutoff` |
| `holdout_start` | Quoted timestamp or null | Omitted/null | Temporal: start < holdout_start < cutoff; null for random | Boundary between earlier fit rows and final temporal holdout. | `holdout_start` |
| `monthly_lookback_months` | Integer or null | Rolling calendar: `4`; otherwise omitted | 1..120 for random, 2..120 for temporal; only rolling_calendar | Number of complete calendar months read, including temporal holdout months. | `monthly_lookback_months` |
| `holdout_months` | Integer or null | Rolling temporal: `1`; otherwise omitted | At least 1 and less than monthly_lookback_months | Final complete months reserved for holdout. | `holdout_months` |
| `window_timezone` | String or null | Rolling calendar: `UTC`; otherwise omitted | Valid IANA zone; only rolling_calendar | Defines calendar-month boundaries, including DST. Cron timezone does not set this. | `window_timezone` |
| `lookback_days` | Integer or null | Rolling days: `90`; otherwise omitted | 1..36500; only rolling_days | Number of elapsed UTC days before invocation. | `lookback_days` |
| `holdout_days` | Integer or null | Rolling-days temporal: `14`; otherwise omitted | At least 1 and less than lookback_days; temporal only | Latest elapsed days reserved for final holdout. | `holdout_days` |
| `filter_unavailable_results` | Boolean | `false` | Enable only with result column and cutoff policy | Excludes unknown/not-yet-available labels from both train and holdout. | `filter_unavailable_results` |
| `result_available_at_column` | String or null | Omitted/null | Required when filtering; distinct from event/key/input/target roles | Records when each label became available. | `result_available_at_column` |
| `result_cutoff` | Quoted timestamp or null | Omitted/null | Aware ISO instant; filtering only | Keep labels available by this instant, including equality; null derives invocation minus lag. | `result_cutoff` |
| `result_availability_lag_hours` | Integer or null | Filtering: `0`; otherwise omitted | 0..87600; omit/null when filtering is off | Delays the derived result cutoff; an explicit result_cutoff takes precedence. | `result_availability_lag_hours` |
| `result_time_parsing` | Mapping or null | Omitted/null | Same nested fields as event parsing; filtering only | Interprets label-availability values independently of event dates. | `result_time_format`, `result_time_timezone`, `result_date_only` |

### Cross-validation on training rows

| Field | Type | Default/required | Values and conditions | What changes | Init answer or owner |
| --- | --- | --- | --- | --- | --- |
| `cv_enabled` | Boolean | `false`; competition forces true | Competition requires true | Fold-local model evaluation/search; final holdout stays outside it. | `cv_enabled` |
| `cv_type` | String | `k_fold` | k_fold, stratified_k_fold, time_series_split, shuffle_split, nested_cv, group_k_fold, stratified_group_k_fold | Chooses fold membership. Stratified methods require classification; temporal/group methods require their controls. | `cv_type` |
| `cv_folds` | Integer | `5` | 2..20 | Number of folds, or outer folds for nested CV. Each fold needs at least two fit and two validation rows. | `cv_folds` |
| `cv_shuffle` | Boolean | `true` | Temporal false; shuffle_split true | Controls permitted fold randomization. | `cv_shuffle` |
| `cv_random_state` | Integer | `42` | 0 through 2**32-1 | CV seed, independent of the final holdout seed. | `cv_random_state` |
| `cv_inner_folds` | Integer or null | Omitted/null | 2..20; nested CV | Explicit inner folds; omitted selects min(3, outer-1), with minimum 2. | `cv_inner_folds` |
| `cv_nested_type` | String | `auto` | auto, k_fold, stratified_k_fold, time_series_split, group_k_fold, stratified_group_k_fold | Inner/outer policy for nested CV; non-auto requires nested_cv. | `cv_nested_type` |
| `cv_group_column` | String or null | Omitted/null | Required for group methods; otherwise null; distinct control column | Keeps entities separate; final group holdout also excludes shared groups. Not a model input. | `cv_group_column` |
| `cv_gap` | Integer | `0` | Nonnegative row count; temporal CV only | Gap between fit and validation windows; not a duration. | `cv_gap` |
| `cv_test_size` | Integer or null | Omitted/null | Positive rows; temporal CV only | Explicit fold validation length; different from fractional final test_size. | `cv_test_size` |
| `cv_max_train_size` | Integer or null | Omitted/null | Positive rows; temporal CV only | Maximum training rows in each temporal fold. | `cv_max_train_size` |

### Quality, weights and bounded reports

| Field | Type | Default/required | Values and conditions | What changes | Init answer or owner |
| --- | --- | --- | --- | --- | --- |
| `metric` | String | heldout_rmse / heldout_accuracy | Task-compatible supported heldout_ metric; competition has a stricter shared metric subset | Primary final comparison objective; error/loss minimizes, other scores maximize. | `regression_metric` / `classification_metric` |
| `min_improvement` | Number | `0.0` | Finite, nonnegative, absolute metric units | Required improvement over champion; a tie never passes. | `min_improvement` |
| `quality_threshold` | Number or null | Omitted/null | Finite task-valid bound; required for automatic promotion and first-champion approval | Maximum acceptable error/loss or minimum acceptable score. Set before training; later edits cannot relax a saved candidate. | `quality_threshold` |
| `quality_gates` | Mapping or null | Omitted/null | Additional task-compatible metric: bound pairs | Every configured absolute gate must pass, including first activation. | YAML; branch quality-gates answer where offered |
| `promotion_policy` | String | `manual_approval` | manual_approval, automatic | Individual candidate activation. Multi-target branches stay manual; the complete set's policy belongs in inference.yml. | `promotion_policy` |
| `champion_version` | String or null | Omitted/null | Concrete positive version string | Optional expectation guard; generated lifecycle resolves current controlled champion and rejects a mismatching explicit value. Does not move an alias. | YAML |
| `weight_column` | String or null | Omitted/null | Existing numeric source column, distinct from all protected/input roles | Supplies training sample weights; support depends on estimator. Holdout metrics and threshold selection remain unweighted. | `sample_weight_enabled`, `weight_column` |
| `risk_category` | String or null | Omitted | At most 256 UTF-8 bytes | Metadata tag only; does not change gates or compute policy. | `risk_category` |
| `preprocessing_probe` | Boolean | `false` | Explicit opt-in | Saved/reloaded preprocessing diagnostic on at most 256 holdout rows, 8 MiB per frame; not worker admission. | YAML |
| `evaluation_charts` | Mapping or null | `{enabled: false}` | enabled plus bounded sample/display controls below | Saves an evaluation sample for the independent chart task. Enabling a previously omitted task also requires matching resources/notebook. | `evaluation_charts_enabled`; detailed limits in YAML |
| `competition_max_candidates` | Integer | `8` | 2..32 in loader; init wizard offers up to 8 | Bounds distinct candidate declarations in a one-target competition. | Generated; YAML for explicit validated limits |
| `competition_max_trials` | Integer | Generated `1000`; loader fallback `100` | 1..10000; competition only | Aggregate search-work bound including nested/halving work, not wall-clock seconds. | `competition_max_trials` |
| `feature_lookup` | Mapping or null | Omitted/null | Native lookup declaration; pandas and admitted Spark/online path | Adds explicit offline feature binding and source evidence; not an arbitrary join. | Online opt-in creates placeholders; complete YAML |
| `online_features` | Mapping or null | Omitted/null | Requires feature_lookup, pandas, non-temporal supported request shape | Saves required-feature and freshness guards with the model; does not create a store or endpoint. | `include_online_serving` creates placeholders; complete YAML |

<!-- training-fields:end -->

### Class and row weights

Set classifier class weights in `model.params`, for example
`params: {class_weight: balanced}`. `null` disables them. Keep a class weight fixed
by removing its search axis; to tune it, omit the fixed parameter and retain the
axis under `tuning.search_space`. Supported classifiers may also use row weights.

Set `weight_column: importance` in `defaults` or a target entry. Omit it or use
`null` for unweighted fitting. Competition uses one shared column. Every active
fold requires finite nonnegative values with positive total weight. Weights are
not normalized. Declared weight columns remain reserved across branches and cannot
be features, targets, keys, group columns or timestamps.

Weights affect estimator fitting; selection, holdout and threshold metrics remain
unweighted. Built-in row changes carry explicit row positions; unsupported custom
row mapping fails. Synthetic sampling needs an explicit supported weight policy.
Source snapshots preserve the training column and saved evidence records its
aggregate statistics and ordered digest. Logs do not print raw weight vectors.
Scoring uses the saved model and needs no weight column. Weight-only edits do not
count as fresh labels for automatic retraining; train explicitly after such edits.

## Model and recipe fields

<!-- model-fields:start -->

| Field | Type | Default/required | Choices and effect | Init answer or owner |
| --- | --- | --- | --- | --- |
| `model` | Mapping | Required here or inherited | `type` is a registered task-compatible Core model ID; `params` is an optional mapping. Omit tuning to fit this model once. | regression_model / classification_model and model_params; branch/candidate variants |
| `tuning` | Mapping | Omitted disables search; generated projects normally include it | Wraps the model in the existing tuner. Fields below use native search metrics, not heldout_ names. | Search questions and YAML |
| `preprocessing_recipe` | String | `default` | Named list in preprocessing.yml. Each candidate/branch learns its own fitted state even when sharing a recipe. | preprocessing_recipe; YAML |
| `pre_split_recipe` | String | `default` | Named fixed eligibility list in pre_split.yml. Competition always uses `recipes.default` and rejects this field. | pre_split_recipe; YAML |
| `features_path` | String | Generated src/features | Alternative project feature package; supported only for multi-target model entries. Its source, assets and exact dependency pins are captured. | YAML |
| `decision_threshold` | Mapping | `{mode: off}` | Classification only: off, manual or auto. Changes hard-label decisions without changing probabilities; nested fields below. | decision-threshold questions; YAML |
| `explainability` | Mapping | Omitted disables it | SHAP method and bounded sample/display limits below; packages and report task must be present. | shap_enabled and SHAP limit questions |

<!-- model-fields:end -->

`model.params` uses the selected estimator's public parameters. For example,
`ridge_regression` accepts `alpha`, while a random forest accepts `n_estimators`
and `max_depth`. Parameters do not transfer automatically when model type
changes. Use `python src/tools/preview.py --list-models` and the
[model catalog](databricks_model_catalog.md) for parameters, default search spaces,
weights and ready-to-use YAML for each model. `class_weight` belongs in the compatible
classifier's params; it is distinct from a source `weight_column`.

## Tuning fields

These keys go inside `models.<name>.tuning` or a shared `defaults.tuning`.
Deleting the entire mapping disables search. An empty mapping still requests a
tuner and needs a compatible objective. Learned preprocessing is refitted within
the training folds; neither search nor calibration can use the final holdout.
The [search and ensemble guide](databricks_tuning.md) explains the five strategies,
their additional controls, CV, competition budgets and complete examples.

<!-- tuning-fields:start -->

| Field | Type | Default/required | Choices and effect | Init answer or owner |
| --- | --- | --- | --- | --- |
| `strategy` | String | `random` | grid, random, optuna, halving_grid, halving_random. Optuna needs the matching optional dependencies. | search_strategy |
| `metric` | String | Required; generated rmse / accuracy | Native Core metric such as rmse, r2, accuracy or f1; not heldout_rmse. Must match task and competition objective where applicable. | regression_search_metric / classification_search_metric |
| `n_trials` | Integer | `10` | 1..1000; trial/candidate budget for the selected strategy. Grid enumeration additionally respects max_candidates. | search_n_trials |
| `random_state` | Integer | `42` | 0 through 2**32-1; search seed independent of final split/CV seeds. | search_random_state |
| `max_candidates` | Integer | `1000` | 1..10000; rejects exhaustive grids above this bound before fitting. | search_max_candidates |
| `search_space` | Mapping | `{}` | Parameter to nonempty finite scalar candidate list; empty means registered default space with fixed params preserved. At most 128 axes, 100 values/axis and 64 KiB. | search_space |
| `timeout` | Number or null | Omitted/null | Positive finite seconds, optuna only; not an overall job deadline. | search_timeout |
| `strategy_params` | Mapping | `{}` | Optuna sampler/pruner/pruning or halving factor/resource/min/max controls below; nonempty rejected for grid/random. | Advanced search questions |
| `tune_threshold` | Boolean | `false` | Binary-classification nested-CV threshold search only; incompatible with an active decision_threshold. Prefer an explicit decision_threshold. | search_tune_threshold |

<!-- tuning-fields:end -->

For a fixed value use `model.params`, for searched values use candidate lists:
`search_space: {alpha: [0.1, 1.0, 10.0]}`. A manual search axis cannot contradict
a fixed parameter. Unknown parameter names fail instead of being silently
ignored. The outer job's CPU/memory limits still apply; search uses one estimator
worker to avoid nested parallel fits.

## Nested settings and conditional controls

| Mapping | Fields, defaults and valid choices | Meaning |
| --- | --- | --- |
| event_time_parsing / result_time_parsing | `format: null`, `timezone: null`, `date_only: reject` (or midnight) | Native timestamps need no format. Naive local clocks need an IANA timezone; offset-bearing formats must not also set one. Date-only values require midnight plus timezone. DST gaps/overlaps are rejected. |
| decision_threshold, mode off | Only `mode: off` | Keep the estimator's normal hard-label decision. |
| decision_threshold, mode manual | Binary: `value` in [0,1] plus typed `positive_class`; or `thresholds: [{class: ..., value: ...}, ...]` covering every class with positive values | Bind original class labels, not probability column indexes. Per-class decisions use the saved threshold policy. |
| decision_threshold, mode auto | `metric: balanced_accuracy` (also f1, f1_macro, f1_weighted, matthews_corrcoef), `validation_fraction: 0.2` strictly between 0 and 1, `random_state: 42`, optional `positive_class` | Uses separate training-side calibration; temporal models retain chronological ordering. |
| explainability | Required `method: shap`; max_samples 100 (1..200), max_features 30 (1..50), max_display_samples min(10,max_samples) (0..50 and no larger than max_samples) | Bounded training explanation and report display. |
| evaluation_charts | Required boolean `enabled`; max_rows 2000 (2..10000), max_features 20 (1..50), max_classes 20 (2..50) | Bounded saved holdout chart sample; does not change full holdout metrics. |
| feature_lookup | `lookups` list, each with table_name, lookup_key and feature_names; optional timestamp fields only on supported offline paths | Declare every fetched feature in input_columns. Keep lookup controls/source identities explicit. See online guide for exact native contract. |
| online_features | `required_features` and `freshness: [{column: ..., max_age_seconds: ...}]` | Cover all fetched features; each lookup needs its saved DOUBLE UTC-epoch freshness input. The age bound is an explicit domain choice, not an automatic default. |
| tuning.strategy_params for optuna | sampler: tpe (random, cmaes); pruner: median (hyperband, none); optional boolean pruning | Configures the installed Optuna integration. |
| tuning.strategy_params for halving | factor: 3 (2..10); resource: n_samples; min_resources and max_resources depend on resource | Custom estimator resource needs a compatible parameter and explicit positive maximum. Bounds and supported count strings are validated before fitting. |

See [online feature lookup](databricks_online_features.md),
[classification threshold policy](databricks_bundle.md#optional-classification-decision-thresholds)
and [preprocessing context](preprocessing_context.md) for their complete execution
contracts. A valid field set does not certify a fitted recipe for Spark or serving.

## Three complete training files

Start with a newly generated project of the **same layout**. Keep its
`config/inference.yml`, `src/features/`, tool scripts and resource graph. Replace
only training.yml with the chosen complete block below, and use these companion
recipes. Both files require a `default` recipe, even if it is empty.

In `config/pre_split.yml`:

```yaml
version: 1
recipes:
  default: []
```

In `config/preprocessing.yml`:

```yaml
version: 1
recipes:
  default:
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [income, age], strategy: mean}
    - name: scale_numeric
      transformer: StandardScaler
      params: {columns: [income, age]}
```

The examples use numeric regression targets to keep estimator/metric roles easy
to compare. The source is an existing Delta table with nonnull unique BIGINT
`customer_id`, DOUBLE `income`/`age`, and DOUBLE `revenue`/`cost` labels. For local
fit verification you can create the following frame; it does not upload a table:

```python
import pandas as pd

frame = pd.DataFrame({
    "customer_id": range(1, 41),
    "income": [float(i) for i in range(1, 41)],
    "age": [float(20 + i % 10) for i in range(1, 41)],
    "revenue": [float(2 * i + 5) for i in range(1, 41)],
    "cost": [float(i + 2) for i in range(1, 41)],
})
assert frame.customer_id.is_unique and frame.shape == (40, 5)
```

### One model for one target

Save this entire block as `config/training.yml` in a single_model project.
The imputer/scaler are learned separately on training rows; the final 20%
holdout does not fit them. The absolute RMSE gate is illustrative.

<!-- example:single_model -->
```yaml
version: 1
defaults:
  config_version: 1
  training_layout: single_model
  engine: pandas
  task: regression
  training_table: "{catalog}.{input_schema}.customer_training"
  training_version: null             # Resolve latest once when invoked.
  record_key_columns: [customer_id]
  input_columns: [income, age]        # Keys and labels are not features.
  target_column: revenue
  model_name: "{catalog}.{metadata_schema}.revenue_model{resource_suffix}"
  split_strategy: random
  training_window_mode: full_snapshot # No event timestamps are needed.
  test_size: 0.2
  random_state: 42
  stratify: false
  filter_unavailable_results: false
  max_rows: 10000
  max_input_mb: 64
  cv_enabled: false
  metric: heldout_rmse
  min_improvement: 0.0
  quality_threshold: 5.0              # Needed for first-champion approval.
  promotion_policy: manual_approval
  preprocessing_recipe: default
  pre_split_recipe: default
models:
  revenue:
    model:
      type: ridge_regression
      params: {alpha: 1.0}
```

There is no tuning mapping, so this example fits the stated fixed ridge model.
A generated wizard project normally includes tuning; remove that mapping
deliberately if a single fixed fit is intended.

### Several algorithms competing for one target

Save this in a model_competition project. Candidates share final holdout and
three-fold training CV. Candidate overrides change only model/search/recipe
settings. The winner alone proceeds to final holdout comparison and registration.

<!-- example:model_competition -->
```yaml
version: 1
defaults:
  config_version: 1
  training_layout: model_competition
  engine: pandas
  task: regression
  training_table: "{catalog}.{input_schema}.customer_training"
  training_version: null
  record_key_columns: [customer_id]
  input_columns: [income, age]
  target_column: revenue
  model_name: "{catalog}.{metadata_schema}.revenue_competition{resource_suffix}"
  split_strategy: random
  training_window_mode: full_snapshot
  test_size: 0.2
  random_state: 42
  stratify: false
  filter_unavailable_results: false
  max_rows: 10000
  max_input_mb: 64
  cv_enabled: true                   # Required by competition.
  cv_type: k_fold
  cv_folds: 3
  cv_shuffle: true
  cv_random_state: 42
  competition_max_candidates: 8
  competition_max_trials: 100
  metric: heldout_rmse               # Shared objective, not training accuracy.
  min_improvement: 0.0
  quality_threshold: 5.0
  promotion_policy: manual_approval
  preprocessing_recipe: default
models:
  ridge:
    model: {type: ridge_regression, params: {alpha: 1.0}}
  forest:
    model:
      type: random_forest_regressor
      params: {n_estimators: 20, max_depth: 4, random_state: 42}
```

To search ridge alpha instead, remove fixed alpha from that candidate's params
and add `tuning: {strategy: grid, metric: rmse, max_candidates: 10,
search_space: {alpha: [0.1, 1.0, 10.0]}}`. Search metric is `rmse`; the final
comparison metric remains `heldout_rmse`. Keep aggregate work inside the
competition budget. Do not tune parameters against the final heldout rows.

### Independent models for two targets

Save this in a multi_target project. Both branches share one source snapshot and
keys; each has its own target, model and model name. Missing labels are excluded
per target. No branch uses the sibling label as a feature.

<!-- example:multi_target -->
```yaml
version: 1
defaults:
  config_version: 1
  training_layout: multi_target
  engine: pandas
  task: regression
  training_table: "{catalog}.{input_schema}.customer_training"
  training_version: null
  record_key_columns: [customer_id]
  input_columns: [income, age]
  split_strategy: random
  training_window_mode: full_snapshot
  test_size: 0.2
  random_state: 42
  stratify: false
  filter_unavailable_results: false
  max_rows: 10000
  max_input_mb: 64
  cv_enabled: false
  metric: heldout_rmse
  min_improvement: 0.0
  quality_threshold: 5.0
  promotion_policy: manual_approval    # Set activation is separate in inference.yml.
  preprocessing_recipe: default
  pre_split_recipe: default
models:
  revenue:
    target_column: revenue
    model_name: "{catalog}.{metadata_schema}.revenue_model{resource_suffix}"
    model: {type: linear_regression, params: {}}
  cost:
    target_column: cost
    model_name: "{catalog}.{metadata_schema}.cost_model{resource_suffix}"
    model: {type: ridge_regression, params: {alpha: 0.5}}
```

The `model_set` block in `inference.yml` owns complete-set activation and output
storage; choose its output mode as described in
[model sets](databricks_model_sets.md#choose-the-output-mode). Individual
component aliases are not the set's release mechanism. Branches do not compete
against each other: they predict
different targets. A failed branch leaves earlier immutable candidates available
but produces no complete successful set.

For the generated graph, MLflow parameters, evaluation charts and SHAP reports,
see [training reports](databricks_training_reports.md). Configure classification
[decision thresholds](databricks_decision_thresholds.md) separately from quality gates.

## Check edits before training

Run from the generated project with the matching Core installed:

```powershell
uv pip install -r deployment/train-requirements.txt
python src/tools/smoke.py
python src/tools/preview.py --action train --catalog workspace --input-schema demo --output-schema demo --metadata-schema demo
python src/tools/refresh_training_graph.py
```

Replace the preview bindings with the intended target's values; see the
[local tools reference](databricks_project_setup.md#local-tools-and-preview)
for every flag and default. Preview does not read deployment/targets.yml for you.
After changing model names, candidates
or branch count, refresh the graph and validate/deploy it before running the job.
Scoring source/model selector/model-set settings remain in the existing
inference.yml; update them to the intended trained names and concrete versions.
The training file alone is not a complete deployment or scoring configuration.

Smoke checks configuration/source syntax; preview resolves trusted recipe
builders and displays each candidate/branch's effective model, inputs, window,
CV and metric. Neither fits a model or reads Delta data. Expect one model preview,
two competition candidates or two target branches for the examples above.
Use a real fit/save/predict check on representative rows before production.

If a pre-split recipe reads the target, choose scoring behavior **before** preview:
in src/features/scoring.py set `SKIP_TARGET_PRE_SPLIT_STEPS=True` to retain
feature-only reuse while skipping target-reading filters, or choose
`SCORING_MODE="custom"` for separately defined scoring eligibility. Mixed
target/feature filters are skipped whole. An imputer does not override a scoring
eligibility exclusion; excluded outcomes retain their keys and null predictions.

| Error or surprising result | Action |
| --- | --- |
| Unknown/duplicate key or multiple owners | Correct the exact field or remove the competing declaration; values are not merged silently. |
| Recipe missing/default recipe absent | Add a default recipe and select an existing named list in the matching phase file. |
| Target-reading pre-split recipe blocks preview | Choose the explicit scoring switch/mode above; the examples' empty pre-split recipe needs neither change. |
| First-champion approval requires a bound | Set quality_threshold before training and create a new candidate with saved matching evidence. |
| Automatic policy lacks quality_threshold | Supply a domain-appropriate absolute bound before training. |
| Window controls active in the wrong mode | Omit/null inactive calendar/day/event/result fields; choose the intended window and split independently. |
| No tuning intended but search runs | Remove tuning mapping, rather than leaving an empty one. |
| Budget exceeded | Deliberately reduce the eligible source/sample or review a larger limit; prediction never samples automatically. |
| Model/params changed but scoring is unchanged | Train a new version, select/activate it deliberately, and run scoring; old artifacts keep their saved state. |

Continue to [preprocessing recipes](preprocessing_recipes.md) for node choices,
[SDK workflows](databricks_sdk.md) for Python use, and the
[operator walkthrough](databricks_bundle_walkthrough.md) for actual run/task
inspection and copyable approval/rollback fields.
