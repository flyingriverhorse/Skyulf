# Databricks Bundle tuning and ensembles

Choose an estimator in the [model catalog](databricks_model_catalog.md), then
choose how to search its parameters here. This guide describes the generated
Bundle's `config/training.yml` contract. The
[training reference](databricks_training_config.md) owns the complete workflow
field list; [Core tuning](hyperparameter_tuning.md) describes the Python API.
The Bundle accepts a bounded subset of the underlying sklearn and Optuna APIs.

| Strategy | Use when | Candidate selection |
| --- | --- | --- |
| [grid](#strategy-grid) | A small, deliberate set of combinations covers the question. | Every Cartesian-product combination. |
| [random](#strategy-random) | Several axes make a full grid too expensive. | Up to `n_trials` sampled combinations. |
| [halving_grid](#strategy-halving_grid) | A small grid has a useful cheap training budget. | Full initial grid, then fewer candidates at larger resources. |
| [halving_random](#strategy-halving_random) | A large space has a useful cheap training budget. | Sampled initial candidates, then successive elimination. |
| [optuna](#strategy-optuna) | Adaptive trial selection or supported pruning is useful. | TPE, random or CMA-ES sampling. |

```mermaid
flowchart LR
    A["Choose estimator and fixed params"] --> B["Choose search space"]
    B --> C["grid / random / halving / optuna"]
    C --> D["Training-side CV and budget"]
    D --> E["Select parameters and refit"]
    E --> F["Separate heldout evaluation"]
```

Search selection uses training-side validation. The final holdout measures the
selected fitted model; it does not choose the winning trial.

## Where settings belong

Use `models.<name>.model.params` for fixed constructor values and
`models.<name>.tuning.search_space` for searched values. For example, keep
`fit_intercept: true` fixed and search `alpha: [0.1, 1.0, 10.0]`. Do not also fix
`alpha: 1.0`: a conflicting explicit axis is rejected. A matching singleton
axis is allowed, but adds no search. Defaults inherited from `defaults` are
shallow overrides: a model's `tuning` mapping replaces the shared mapping.

Omit `tuning` entirely to fit once. `{}` requests tuning; it does not disable it.
An empty `search_space` asks Core to construct the selected model's registered
space, with explicit fixed parameters pinned. The model catalog shows the
registered grid/random families and halving resources. Supplying a nonempty
space replaces the automatic axes. Model defaults still supply omitted
constructor settings.

Each axis must be a nonempty list of finite JSON scalars: numbers, strings,
booleans or `null`. At most 128 axes, 100 values per axis and 65,536 encoded JSON
bytes are admitted. Use native YAML `null`, `true` and `false`. Distribution
objects, Python callables, conditional list-of-dictionaries grids and arbitrary
Optuna constructor options are not YAML search-space values. Unknown estimator
parameter names fail admission. A list such as `weights: [[1, 2], [2, 1]]` is
therefore not an accepted search axis; set ensemble weights structurally in
`model.params` instead.

All strategies use these fields inside `tuning`; their cards explain which
controls affect that strategy.

<!-- tuning-fields:start -->

| Field | Type | Default | Accepted values and effect |
| --- | --- | --- | --- |
| `strategy` | String | `random` | `grid`, `random`, `halving_grid`, `halving_random`, `optuna`. |
| `metric` | String | Required for a single model or branch; competition can inherit its objective | Native Core task-compatible objective, for example `rmse`, `mae`, `mse`, `r2`, `accuracy`, `balanced_accuracy`, `f1`, `f1_weighted`, `precision`, `recall`, `roc_auc`. Do not use `heldout_` names here. The wizard supplies `rmse` or `accuracy` outside competition. |
| `n_trials` | Integer | `10` | 1 through 1000. Random/Optuna trial count or halving-random initial candidate count. Grid variants still validate this field but enumerate their grid independently of it. |
| `random_state` | Integer | `42` | 0 through 4,294,967,295. Seeds search sampling, independently of workflow split and CV seeds. Does not automatically seed every estimator. |
| `max_candidates` | Integer | `1000` | 1 through 10,000. Rejects an exhaustive grid whose Cartesian product exceeds this limit. It is not the random/Optuna trial budget. |
| `search_space` | Mapping | `{}` | Empty means the registered space; otherwise parameter names map to finite scalar candidate lists subject to the bounds above. |
| `timeout` | Number or null | `null` | Positive finite seconds, supported only by Optuna. Soft search stop, not an interrupt or a whole-job deadline. |
| `strategy_params` | Mapping | `{}` | Empty for grid/random; only the exact halving or Optuna controls in the cards below are admitted. |
| `tune_threshold` | Boolean | `false` | Legacy binary-classification threshold tuning, requiring enabled nested CV and probability output. Incompatible with an active `decision_threshold` policy. Prefer the explicit policy documented in the training reference. |

<!-- tuning-fields:end -->

The Bundle fixes search `n_jobs` to 1 and requires all explicitly configured
estimator and nested-estimator workers to be 1. `n_jobs`, `parallel_backend`,
custom scorers, `refit`, `error_score`, and CV keys are not accepted inside the
YAML `tuning` mapping. Set shared CV fields at workflow/defaults level. This
keeps candidate fits bounded; a valid configuration is not a promise that its
grid is cheap.

Loss objectives such as RMSE are ranked with sklearn's negative-loss scorer.
Saved tuning `best_score` can therefore be negative: higher is better in the
search evidence. Final lifecycle metrics such as `heldout_rmse` keep their own
meaning and direction. The final holdout is never the search validation set.

The complete admitted metric names are:

| Task | Native tuning metrics |
| --- | --- |
| Regression | `mse`, `mae`, `rmse`, `r2`, `explained_variance` |
| Classification | `accuracy`, `balanced_accuracy`, `f1`, `f1_weighted`, `precision`, `recall`, `roc_auc`, `roc_auc_ovr`, `roc_auc_ovo`, `roc_auc_ovr_weighted`, `roc_auc_ovo_weighted`, `pr_auc`, `pr_auc_weighted`, `log_loss`, `matthews_corrcoef`, `g_score` |

These are admission choices, not a promise that every model/label distribution
can compute every metric. Probability metrics need appropriate model outputs;
multiclass AUC needs a compatible variant and class coverage. `g_score` requires
`imbalanced-learn`. The wizard exposes a smaller common metric menu; edit YAML
for the other admitted names.

## Common setup for the worked examples

In a generated **single_model** project, replace `config/training.yml` with this
header followed by exactly one strategy card's complete `models` block. Keep the
generated `config/inference.yml`, Python package and job resources. The source
table needs unique nonnull BIGINT `customer_id`, numeric DOUBLE `income` and
numeric DOUBLE `revenue`. Replace the illustrative table/model names and quality
bound for your data.

<!-- tuning-example:header -->
```yaml
version: 1
defaults:
  config_version: 1
  training_layout: single_model
  engine: pandas
  task: regression
  training_table: "{catalog}.{input_schema}.customer_training"
  training_version: null
  record_key_columns: [customer_id]
  input_columns: [income]
  target_column: revenue
  model_name: "{catalog}.{metadata_schema}.revenue_model{resource_suffix}"
  split_strategy: random
  training_window_mode: full_snapshot
  test_size: 0.2
  random_state: 42
  stratify: false
  filter_unavailable_results: false
  max_rows: 10000
  max_input_mb: 64
  cv_enabled: true
  cv_type: k_fold
  cv_folds: 2
  cv_shuffle: true
  cv_random_state: 42
  metric: heldout_rmse
  min_improvement: 0.0
  quality_threshold: 5.0
  promotion_policy: manual_approval
  preprocessing_recipe: default
  pre_split_recipe: default
```

Use this companion content in **both** `config/pre_split.yml` and
`config/preprocessing.yml`. Empty recipes keep the examples focused on search;
choose fitted preprocessing appropriate to your real data before training.

```yaml
version: 1
recipes:
  default: []
```

After saving the selected example, run from the generated project:

```powershell
uv pip install -r deployment/train-requirements.txt
python src/tools/smoke.py
python src/tools/preview.py --action train --catalog workspace --input-schema demo --output-schema demo --metadata-schema demo
python src/tools/refresh_training_graph.py
```

Use bindings for your intended target. Preview resolves model/search/recipe
settings but does not read Delta or fit the model. Deployment and an actual run
follow the [Bundle walkthrough](databricks_bundle_walkthrough.md). The examples
are locally exercised with 40 numeric rows, a separate 8-row final holdout and
saved/reloaded predictors. Those checks do not create a table, deploy a job or
establish Spark-worker/serving admission for a model or recipe.

<a id="strategy-grid"></a>
## Grid search

**Choose `grid`** when every combination has a reason to exist. Four alpha
values times two intercept values mean eight candidates, each evaluated on the
configured training folds, then a best-model refit. Cost grows multiplicatively.

| Control | Grid behavior |
| --- | --- |
| `strategy` | Set `grid`; omission selects random. |
| `search_space` | Finite Cartesian product; empty resolves the model's grid defaults. |
| `max_candidates` | Default 1000; admission bound on that full product, not truncation. |
| `n_trials` | Default 10, still validated, but does not limit grid enumeration. |
| `random_state` | Default 42; does not randomize grid enumeration. CV and estimator seeds remain separate. |
| `strategy_params` / `timeout` | Only `{}` and omitted/null, respectively. |

All other common fields retain the contract above. This four-candidate example
deliberately has `n_trials: 1` to make its non-capping behavior explicit:

<!-- tuning-example:grid -->
```yaml
models:
  revenue:
    model:
      type: ridge_regression
      params: {fit_intercept: true}
    tuning:
      strategy: grid
      metric: rmse
      n_trials: 1
      max_candidates: 4
      random_state: 42
      search_space: {alpha: [0.1, 1.0, 10.0, 100.0]}
```

Common mistake: reducing `n_trials` to shrink an expensive grid. Reduce its axes
or use random search; raising `max_candidates` permits more work rather than
making it faster.

<a id="strategy-random"></a>
## Random search

**Choose `random`** to explore several axes within a fixed candidate budget.
With Bundle finite-list axes, sampled combinations are drawn without replacement;
if the Cartesian product is smaller than `n_trials`, only that many distinct
combinations are available. This is parameter sampling, not row sampling.

| Control | Random behavior |
| --- | --- |
| `strategy` | `random`, also the default. |
| `search_space` | Empty resolves the model's random defaults; nonempty lists replace them. |
| `n_trials` | Default 10; 1..1000 sampled candidates, limited by available combinations. |
| `random_state` | Default 42; seeds candidate sampling. |
| `max_candidates` | Validated but does not cap random candidates; use `n_trials`. |
| `strategy_params` / `timeout` | Only `{}` and omitted/null, respectively. |

All other common fields apply. Four trials here exhaust the small example space;
for a larger space they would select only four combinations.

<!-- tuning-example:random -->
```yaml
models:
  revenue:
    model:
      type: ridge_regression
      params: {fit_intercept: true}
    tuning:
      strategy: random
      metric: rmse
      n_trials: 4
      random_state: 42
      search_space: {alpha: [0.1, 1.0, 10.0, 100.0]}
```

Common mistake: expecting a search seed to control the estimator's randomness.
For a random forest also choose its fixed `model.params.random_state`.

<a id="strategy-halving_grid"></a>
## Successive halving over a grid

**Choose `halving_grid`** when ranking candidates with a smaller budget is a
useful approximation to ranking them with the full budget. It begins with the
grid, increases the resource by `factor`, and retains roughly one in `factor`
candidates. An early weak candidate is eliminated even if it would have won at
full resources. The chosen model is subsequently refitted on the training data.

The two halving strategies accept exactly these `strategy_params`:

| Field | Type | Default | Meaning and bounds |
| --- | --- | --- | --- |
| `factor` | Integer | `3` | 2..10; resource multiplier and approximate elimination ratio. |
| `resource` | String | `n_samples` | Training-row resource, or a real compatible estimator parameter such as `n_estimators`. The wizard offers those two; custom YAML names must exist on the prepared estimator. |
| `min_resources` | Integer or string | `exhaust` | With `n_samples`: `exhaust`, `smallest`, or 2..1,000,000. `exhaust` selects a starting budget intended to use the last round efficiently; `smallest` starts at the estimator/task minimum. With an estimator resource, an explicit integer is required. |
| `max_resources` | Integer or string | `auto` | With `n_samples`: `auto` uses available training rows; integers 2..1,000,000 are capped to the current search population, including nested folds. Estimator resources require an explicit integer in that range. |

ASCII decimal strings such as `"16"` normalize to integer resource counts.
`min_resources` must not exceed `max_resources` or the available sample budget.
Small numeric sample budgets must still leave viable training/validation folds
and enough rows per class. There is no `aggressive_elimination`, `n_candidates`
or other sklearn scheduler setting in Bundle `strategy_params`.

For an estimator resource, remove that parameter from **both** fixed params and
manual search axes. For example, `resource: n_estimators` requires explicit
`min_resources: 10` and `max_resources: 40` on a compatible forest, without a
fixed/searched `n_estimators`. Automatic spaces remove the resource axis, but an
explicit fixed value still conflicts. Non-count parameters may exist on an
estimator yet be unsuitable resources; use a positive training-budget parameter.

| Control | Halving-grid behavior |
| --- | --- |
| `strategy` | Set `halving_grid`. |
| `search_space` / `max_candidates` | Same initial grid/product bound as grid search. |
| `n_trials` | Default 10, validated but does not cap the initial grid or later rounds. |
| `random_state` | Default 42; controls halving sampling where used. |
| `timeout` | Omitted/null; no halving timeout control. |

All remaining common fields apply. With enough rows, this example evaluates
four candidates at 8 rows, two survivors at 16 and one at 32: seven candidate
evaluations, each with two CV folds. The resource describes the population
budget before fold fitting, not eight rows per individual fold.

<!-- tuning-example:halving_grid -->
```yaml
models:
  revenue:
    model:
      type: ridge_regression
      params: {fit_intercept: true}
    tuning:
      strategy: halving_grid
      metric: rmse
      max_candidates: 4
      random_state: 42
      search_space: {alpha: [0.1, 1.0, 10.0, 100.0]}
      strategy_params:
        factor: 2
        resource: n_samples
        min_resources: 8
        max_resources: 32
```

Common mistake: using the default `max_resources: auto` with `n_estimators`, or
fixing the same resource in `model.params`. Both fail admission.

<a id="strategy-halving_random"></a>
## Successive halving over random candidates

**Choose `halving_random`** when a full grid is too large and a small resource
budget provides a useful initial ranking. It first samples up to `n_trials`
candidates, then eliminates candidates and increases resources just as
halving-grid does. Its registered automatic search space uses the random family.

| Control | Halving-random behavior |
| --- | --- |
| `strategy` | Set `halving_random`. |
| `n_trials` | Default 10; 1..1000 initial candidates, not total round evaluations. |
| `random_state` | Default 42; seeds initial candidates and resource sampling. |
| `max_candidates` | Validated but does not cap sampled candidates or later rounds. |
| `strategy_params` | Exactly `factor`, `resource`, `min_resources`, `max_resources`, with the same types/defaults/bounds as halving-grid. |
| `timeout` | Omitted/null. |

All remaining common fields apply. The example has the same possible four,
two, one round counts as halving-grid; a larger candidate space makes its first
round a sample instead of an exhaustive grid.

<!-- tuning-example:halving_random -->
```yaml
models:
  revenue:
    model:
      type: ridge_regression
      params: {fit_intercept: true}
    tuning:
      strategy: halving_random
      metric: rmse
      n_trials: 4
      random_state: 42
      search_space: {alpha: [0.1, 1.0, 10.0, 100.0]}
      strategy_params:
        factor: 2
        resource: n_samples
        min_resources: 8
        max_resources: 32
```

Common mistake: treating `n_trials: 4` as four total fits. Surviving candidates
are evaluated again, and every evaluation includes the selected CV splits.

<a id="strategy-optuna"></a>
## Optuna search

**Choose `optuna`** for adaptive TPE/CMA-ES proposals or supported early pruning.
The training environment needs `optuna` and the sklearn integration
`optuna-integration`; CMA-ES additionally needs `cmaes`. Select the wizard's
matching optional dependencies and check the generated training requirements.
Installing Optuna alone is insufficient for Bundle admission.

| Control | Type | Default | Meaning |
| --- | --- | --- | --- |
| `n_trials` | Integer | `10` | 1..1000 attempted trials; repeated configurations are possible. |
| `random_state` | Integer | `42` | Seeds the sampler; the common integer bounds apply. |
| `timeout` | Number or null | `null` | Positive finite seconds. A running fit is not interrupted; nested CV can start separate studies, so this is not a global deadline. |
| `strategy_params.sampler` | String | `tpe` | `tpe`, `random`, `cmaes`; no additional sampler-constructor kwargs. |
| `strategy_params.pruner` | String | `median` | `median`, `hyperband`, `none`; no additional pruner-constructor kwargs. |
| `strategy_params.pruning` | Boolean | Omitted permits supported pruning; wizard emits `true` | `false` disables pruning; `pruner: none` also disables it. |

All remaining common fields apply; `max_candidates` does not cap Optuna.
`search_space: {}` uses the registered random-family space. With TPE or the
random sampler, each list is categorical, including numeric lists. CMA-ES
instead converts a numeric list into an inclusive min/max integer range or a
continuous float range. Thus `[0.1, 10.0]` can produce intermediate values under
CMA-ES. Strings, booleans and null-containing lists remain categorical and use
independent random fallback where CMA-ES cannot handle them.

Pruning depends on actual estimator capabilities and validation. Supported
native boosters can report iteration checkpoints; incremental estimators can
report training updates; ordinary estimators can prune between completed CV
folds. A single complete fold has no remaining fold work to prune. The logs
state which mode is available. Pruned and failed trials cannot become the
selected model; if none completes, there is no winner to refit. Choosing a
pruner does not guarantee that any trial will be pruned, especially in a small
study with its pruner's default warm-up behavior.

This example disables pruning to make the four-trial local exercise predictable:

<!-- tuning-example:optuna -->
```yaml
models:
  revenue:
    model:
      type: ridge_regression
      params: {fit_intercept: true}
    tuning:
      strategy: optuna
      metric: rmse
      n_trials: 4
      random_state: 42
      search_space: {alpha: [0.1, 1.0, 10.0, 100.0]}
      strategy_params: {sampler: tpe, pruner: none, pruning: false}
```

Common mistake: assuming numeric lists always mean discrete choices, or passing
an Optuna distribution object in YAML. Check the sampler's semantics above.

## CV, nested CV and total work

All five strategies use the workflow's shared CV contract. Configure it beside
`task` and `target_column`, usually under `defaults`, rather than inside `tuning`.

| Workflow control | Type/default | Search behavior |
| --- | --- | --- |
| `cv_enabled` | Boolean, `false` | False still searches on one training-only 80/20 validation split. It does not disable tuning or use the final holdout. |
| `cv_type` | String, `k_fold` | `k_fold`, `stratified_k_fold`, `shuffle_split`, `time_series_split`, `group_k_fold`, `stratified_group_k_fold`, `nested_cv`. |
| `cv_folds` | Integer, `5`; 2..20 | Fold/repeated-split count; outer-fold count for nested CV. |
| `cv_shuffle` | Boolean, `true` | Must be false for temporal CV and true for shuffle-split. |
| `cv_random_state` | Integer, `42`; 0..4,294,967,295 | CV seed, distinct from search sampling and final holdout seeds. |
| `cv_inner_folds` | Integer or null, `null`; explicit 2..20 | Nested search folds. Null chooses `min(3, cv_folds - 1)` when outer folds exceed two, otherwise two. |
| `cv_nested_type` | String, `auto` | Nested only: auto selects stratified k-fold for classification, k-fold for regression. Explicit k-fold, stratified, time-series, group or stratified-group policies are supported. |
| `cv_group_column` | String or null, `null` | Required for group policies; not a model feature. Group search requires enabled CV. |
| `cv_gap` | Integer, `0` | Nonnegative row gap for temporal CV. |
| `cv_test_size` | Integer or null, `null` | Positive validation-window row count for temporal CV. |
| `cv_max_train_size` | Integer or null, `null` | Positive maximum training-window row count for temporal CV. |

Temporal search additionally needs the workflow's explicit `event_column` and
enabled CV; it preserves chronological boundaries. Stratified policies require
classification. Group/time metadata, class counts and actual fold sizes are
validated before fitting. Every admitted CV fold needs at least two training
and two validation rows; model-specific requirements can be stricter. Learned
preprocessing is fitted anew inside each training fold. Preprocessing splitter
nodes are rejected because the Bundle owns splitting.

For nested CV, set `cv_type: nested_cv`, for example `cv_folds: 3` and
`cv_inner_folds: 2`. Each outer training partition runs its own search; its
winner is evaluated on that outer validation partition. After outer evaluation,
a fresh full-training search selects the saved model. The outer scores estimate
the search procedure; the final search's best score is not the outer estimate.
The untouched final holdout remains separate. Stacking/calibration internal
folds add further fits; they do not replace these validation levels.

### Temporal and grouped evaluation

Time-series CV needs a selected observation window, an `event_column` and
`cv_shuffle: false`. Normalized timestamps order the folds and are removed before
preprocessing/model fitting. Equal timestamps across a fold boundary fail;
adjust folds or aggregate observations instead of allowing time leakage.
Nested temporal CV requires a temporal final holdout. `cv_gap`, `cv_test_size`
and `cv_max_train_size` are row counts at both nested levels and the separate
final search. A null maximum training size expands the window; an integer rolls it.

Grouped CV requires `cv_group_column`, excluded from model inputs. A random
final holdout selects whole groups using the saved seed; `test_size` is the
proportion of groups and `stratify` must be false. A temporal final holdout must
also contain disjoint groups. Missing groups/classes or insufficient groups fail
before fitting; row filters retain aligned group/time metadata.

During initialization, ordinary or nested temporal CV fixes the final split to
`temporal`, hides random/shuffle questions and opens the clock/window questions.
The default becomes a rolling-calendar holdout; explicit fixed-window dates
remain pinned. Group and other non-temporal CV retain the selected final split.
These wizard rules do not rewrite an existing `config/training.yml`.
See `nested-temporal-init.example.json` and
`nested-group-threshold-init.example.json` in the template's examples directory.

Nested binary `tune_threshold` generates inner out-of-fold probabilities for
each selected recipe; outer and final holdout labels never choose its threshold.
Multiclass or models without probabilities fail this legacy mode. Use the
explicit [decision-threshold policy](databricks_decision_thresholds.md) for its
supported alternatives. `tuning.json` records fold thresholds and provenance.

`cross_validation.json` separates outer selection scores, selected parameters
and mean/std from the final search. Fixed-model CV retains stability diagnostics;
historical artifacts without nested-search evidence remain diagnostic. Negative
loss scorers stay negative and larger is better. The independent `heldout_*`
metrics determine promotion. Search settings/results appear in `tuning.json`
and the `train_and_tune` summary. pandas and Polars use this same evaluation path.

### Competition budget

In `model_competition`, candidates share CV and the selection objective.
`competition_max_candidates` defaults to 8 (2..32); the wizard offers up to
eight candidate slots. `competition_max_trials` defaults to 100 (1..10,000).
It rejects excessive planned work before source access; it does not truncate
candidates or measure CPU time. The trial bound is calculated as follows:

1. A fixed model contributes one candidate evaluation. Grid contributes its
   Cartesian product; random and Optuna contribute requested `n_trials`.
2. Halving adds survivor rounds to that starting count. With starting count
   `N` and factor `f`, it sums `1 + floor(log_f(N))` rounds, repeatedly retaining
   `ceil(previous_count / f)`. This is conservative when resources stop early.
3. Nested CV multiplies this count by `cv_folds + 1`, covering outer searches
   plus the final full-training search. Ordinary CV folds still multiply actual
   estimator fits; they are not another multiplier in this trial-bound field.
4. Sum across all candidates. An active explicit decision-threshold policy can
   add independent outer policy fits/searches; allow for that extra work too.

For the four-candidate, factor-two halving example, the ordinary bound is
`4 + 2 + 1 = 7`; with two outer folds it is `7 * (2 + 1) = 21`. A budget of 6
or 20 respectively is rejected even if one expects an early stop. A second
candidate adds its own bound. Competition parallelism remains one candidate at
a time. Review preview's effective settings before increasing the budget.

## Voting and stacking ensembles

An ensemble is one model candidate containing several learners. In contrast,
competition evaluates separate candidates and selects one winner. The
[ensemble guide](databricks_ensembles.md) covers the `voting_*` and
`stacking_*` `model.params` fields, nested member search axes, worked voting,
stacking and calibrated examples, and the wizard questions for each slot.

Continue with the [model catalog](databricks_model_catalog.md),
[training field reference](databricks_training_config.md), and
[preprocessing recipes](preprocessing_recipes.md) to choose a complete training
recipe. A successful local search does not grant its fitted preprocessing chain
Spark-worker or online-serving support; use the selected inference mode's own
admission and verification path.
