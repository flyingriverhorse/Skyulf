# Databricks model catalog

Choose a model by its behavior, then copy one of its `models:` entries into
`config/training.yml`. This catalog covers all 16 regression and 18 classification
choices in the project wizard, including voting and stacking.

Keep the surrounding `version: 1`, `defaults:` and your existing data, target,
validation and layout settings from the [training reference](databricks_training_config.md).
Each example below replaces the `models:` mapping; do not concatenate the tuned
and untuned mappings. In a competition, give candidates distinct names. In a
multi-target project, retain each branch's target and feature settings.

The **untuned** example fixes the estimator configuration. The **tuned** example
uses a deliberately small explicit grid to demonstrate valid YAML; it is not the
full automatic search space. Validation/CV comes from `defaults`, not from a
`tuning.cv` field. Read [search strategies and ensembles](databricks_tuning.md) for
all strategy fields, nested CV, budgets, member parameters and wizard questions.

## Reading defaults and search spaces

The parameter tables are generated from the same Core hyperparameter registry
used by the wizard's `build_model_spaces.py`. They show key supported controls,
not every parameter accepted by an upstream estimator. `Core catalog default`
is the registered control value, **not a promise about an omitted YAML field**.
The adjacent column is generated from the calculator's actual constructor
settings. `Library/resolver` means the calculator delegates that default to the
installed estimator or its composite resolver. For example, Logistic Regression
registers `max_iter=100` in the catalog but an omitted YAML value runs with Core's
`max_iter=1000`. Random Forest has similar explicit Core overrides. Supply a
parameter explicitly when reproducing a catalog setting matters. Core also
injects its deterministic seed into compatible estimators when no seed is given.

Table literals use JSON notation (`null`, `true`, `false`); the explicit
constructor column preserves Python source (`None`, `True`, `False`). Use YAML
`null`, `true`, `false` in configuration. A table's default does not imply that
all combinations of its controls are valid; solver/penalty pairs and estimator
capabilities are still checked.

Grid and halving-grid use the compact grid column. Random, halving-random and
Optuna use the richer finite-choice column when the wizard supplies its automatic
space. The wizard writes these axes into the generated configuration. A missing
or empty YAML `search_space` also asks the runtime to derive the Core space for
the selected model, respecting its fixed parameters and ensemble members.
For halving, the default resource is `n_samples`; a resource
parameter cannot also appear in its search space. The strategy guide explains
minimum/maximum resources and how to limit the full Cartesian grid.
Large automatic grids, especially ensembles with several members, can exceed
`max_candidates` before fitting. Use a bounded explicit space such as the
examples below, or a random strategy with an explicit trial budget.

## Weights, optional packages and execution boundaries

`weight_column` is a training workflow setting, not a constructor parameter.
The column is removed from model features; weights must be finite, nonnegative,
and have a positive sum in every fitted fold. Neighbor models do not accept row
weights. Ensembles require every selected base model and the final estimator to
support them. Metrics and threshold selection remain unweighted. An ensemble's
`weights` controls member averaging and is a separate concept.

Every card's Spark/serving note describes the **current admission policy**, not
proof that a particular saved artifact will pass it. Training these estimators
uses local pandas/Polars data within the configured row/byte bounds; choosing
Spark inference does not turn fitting into Spark ML training. Spark worker and
pinned serving admission inspect a complete pandas-fitted artifact, exact model
classes, preprocessing, schemas, runtime and saved state. An otherwise eligible
model can still be rejected by its preprocessing or custom scoring rules.
See [Spark execution](spark.md) and [serving rollout](databricks_serving_rollout.md).

Optional XGBoost/LightGBM packages must be installed in training and inference
environments. Installing a package does not extend the admission allowlist.
Local fit/predict examples do not establish Spark, REST, or `ai_query` acceptance.

## ridge_regression

**Ridge regression. Task: regression.** A linear baseline with L2 shrinkage, useful when numeric predictors are correlated or there are many modest signals. Scale numeric features before comparing alpha values.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:ridge_regression:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | `1.0` | Regularization strength; must be a positive float. |
| `solver` | `"auto"` | `'auto'` | Solver to use in the computational routines. |
| `fit_intercept` | `true` | Library/resolver | Whether to calculate the intercept for this model. |
| `random_state` | `42` | Library/resolver | Seed used by the stochastic sag/saga solvers. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `fit_intercept` | `[true, false]` | `[true, false]` |
| `max_iter` | `[2000]` | `[1000, 2000, 5000]` |
| `solver` | `["auto", "saga"]` | `["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:ridge_regression:end -->

### Minimal training entries

Untuned:

<!-- model-example:ridge_regression:plain -->
```yaml
models:
  candidate:
    model:
      type: ridge_regression
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:ridge_regression:tuned -->
```yaml
models:
  candidate:
    model:
      type: ridge_regression
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        alpha: [0.01, 0.1]
```

**Common mistake:** Assuming the same alpha has the same effect before and after feature scaling.

## linear_regression

**Ordinary least squares. Task: regression.** A transparent baseline when a numeric target changes approximately linearly with the features. Coefficients are useful for understanding direction, provided the features have meaningful units.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:linear_regression:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `fit_intercept` | `true` | `True` | Whether to calculate the intercept for this model. |
| `copy_X` | Not a catalog control | `True` | Core constructor setting. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `fit_intercept` | `[true, false]` | `[true, false]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:linear_regression:end -->

### Minimal training entries

Untuned:

<!-- model-example:linear_regression:plain -->
```yaml
models:
  candidate:
    model:
      type: linear_regression
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:linear_regression:tuned -->
```yaml
models:
  candidate:
    model:
      type: linear_regression
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        fit_intercept: [true, false]
```

**Common mistake:** Treating correlated coefficients as independent causal effects; remove redundant features or compare Ridge.

## adaboost_regressor

**AdaBoost regression. Task: regression.** Sequentially reweights the learning problem to focus later learners on difficult examples. A compact boosting baseline when simple weak learners capture useful signal.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:adaboost_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `50` | `50` | The maximum number of estimators at which boosting is terminated. |
| `learning_rate` | `1.0` | `1.0` | Weight applied to each classifier at each boosting iteration. |
| `random_state` | `42` | Library/resolver | Seed passed to the base estimator's randomness, if it has any. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `learning_rate` | `[0.1, 0.5, 1.0]` | `[0.01, 0.1, 1.0, 1.5]` |
| `n_estimators` | `[50, 100, 200]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:adaboost_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:adaboost_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: adaboost_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:adaboost_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: adaboost_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [50, 100]
```

**Common mistake:** Allowing mislabeled rows or severe target outliers to dominate the repeatedly reweighted fit.

## decision_tree_regressor

**Decision tree regression. Task: regression.** A readable nonlinear baseline that splits feature space into regions. Depth and leaf constraints control how finely it partitions the data.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:decision_tree_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `max_depth` | `null` | `None` | The maximum depth of the tree. |
| `min_samples_split` | `2` | `2` | The minimum number of samples required to split an internal node. |
| `min_samples_leaf` | `1` | Library/resolver | The minimum number of samples required to be at a leaf node. |
| `random_state` | `42` | Library/resolver | Seed for split tie-breaking and feature sub-sampling. |
| `criterion` | `"squared_error"` | `'squared_error'` | The function to measure the quality of a split. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `criterion` | `["squared_error", "friedman_mse"]` | `["squared_error", "friedman_mse", "absolute_error"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 3, 5, 10, 20]` |
| `min_samples_leaf` | `[]` | `[1, 2, 4]` |
| `min_samples_split` | `[2, 5, 10]` | `[2, 5, 10]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:decision_tree_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:decision_tree_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: decision_tree_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:decision_tree_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: decision_tree_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        max_depth: [5, 10]
```

**Common mistake:** Leaving a deep tree unconstrained on a small noisy dataset and treating the training score as generalization.

## elasticnet_regression

**Elastic Net regression. Task: regression.** Combines L1 feature selection and L2 shrinkage. Useful when correlated groups of features make pure Lasso selection unstable; alpha controls total shrinkage and l1_ratio its mix.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:elasticnet_regression:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | `1.0` | Constant that multiplies the penalty terms. |
| `l1_ratio` | `0.5` | `0.5` | The ElasticNet mixing parameter (0 <= l1_ratio <= 1). |
| `selection` | `"cyclic"` | `'cyclic'` | If set to 'random', a random coefficient is updated every iteration. |
| `random_state` | `42` | Library/resolver | Seed for the coefficient update order when selection='random'. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.01, 0.1, 1.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0]` |
| `l1_ratio` | `[0.3, 0.5, 0.7]` | `[0.1, 0.5, 0.7, 0.9]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:elasticnet_regression:end -->

### Minimal training entries

Untuned:

<!-- model-example:elasticnet_regression:plain -->
```yaml
models:
  candidate:
    model:
      type: elasticnet_regression
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:elasticnet_regression:tuned -->
```yaml
models:
  candidate:
    model:
      type: elasticnet_regression
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        alpha: [0.01, 0.1]
```

**Common mistake:** Tuning l1_ratio without tuning alpha or scaling the input features.

## extra_trees_regressor

**Extremely randomized trees regression. Task: regression.** Averages trees with extra split randomization, providing a useful tree-ensemble comparison to Random Forest. Its default bootstrap is false.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:extra_trees_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of trees in the forest. |
| `max_depth` | `null` | `None` | Maximum depth of the tree. None = unlimited. |
| `min_samples_split` | `2` | `2` | Minimum samples required to split an internal node. |
| `min_samples_leaf` | `1` | `1` | Minimum samples required at a leaf node. |
| `bootstrap` | `false` | `False` | Whether bootstrap samples are used (False = use full dataset). |
| `random_state` | `42` | Library/resolver | Seed for bootstrap sampling and random split selection. |
| `criterion` | `"squared_error"` | `'squared_error'` | Function to measure the quality of a split. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `bootstrap` | `[]` | `[false, true]` |
| `criterion` | `["squared_error", "friedman_mse"]` | `["squared_error", "absolute_error", "friedman_mse"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 50]` |
| `min_samples_leaf` | `[]` | `[1, 2, 4]` |
| `min_samples_split` | `[2, 5]` | `[2, 5, 10]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:extra_trees_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:extra_trees_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: extra_trees_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:extra_trees_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: extra_trees_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Assuming bootstrap sampling is always active, or that more trees will fix excessive leaf-level overfitting.

## gradient_boosting_regressor

**Gradient boosting regression. Task: regression.** Builds small trees sequentially to correct earlier errors. Useful for tabular nonlinear relationships; learning_rate and n_estimators jointly set the fitting budget.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:gradient_boosting_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | The number of boosting stages to perform. |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree by learning_rate. |
| `max_depth` | `3` | `3` | Maximum depth of the individual regression estimators. |
| `subsample` | `1.0` | Library/resolver | The fraction of samples to be used for fitting the individual base learners. |
| `min_samples_split` | `2` | Library/resolver | Minimum samples required to split an internal node. |
| `min_samples_leaf` | `1` | Library/resolver | Minimum samples required at a leaf node. |
| `random_state` | `42` | Library/resolver | Seed for the row subsampling used when subsample < 1.0. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:gradient_boosting_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:gradient_boosting_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: gradient_boosting_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:gradient_boosting_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: gradient_boosting_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Increasing learning_rate and tree count together without validation, which can make overfitting much worse.

## hist_gradient_boosting_regressor

**Histogram gradient boosting regression. Task: regression.** Uses feature bins to speed up boosted-tree fitting on larger tabular datasets. max_iter is the boosting-round count; leaves and regularization control capacity.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:hist_gradient_boosting_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `max_iter` | `100` | `100` | Maximum number of iterations (boosting rounds). |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree. |
| `max_leaf_nodes` | `31` | `31` | Maximum number of leaves per tree. |
| `max_depth` | `null` | `None` | Maximum depth per tree. None = unlimited (max_leaf_nodes is the effective limit). |
| `min_samples_leaf` | `20` | `20` | Minimum samples at a leaf node. |
| `l2_regularization` | `0.0` | `0.0` | L2 penalty applied to the leaves' values. |
| `max_bins` | `255` | `255` | Maximum number of bins for feature discretisation. Higher = more precise but slower. |
| `random_state` | `42` | Library/resolver | Seed for the histogram-based gradient boosting randomness. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `l2_regularization` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[]` | `[null, 3, 5, 10]` |
| `max_iter` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `max_leaf_nodes` | `[31, 63]` | `[15, 31, 63, 127]` |
| `min_samples_leaf` | `[20, 50]` | `[10, 20, 50, 100]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:hist_gradient_boosting_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:hist_gradient_boosting_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: hist_gradient_boosting_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:hist_gradient_boosting_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: hist_gradient_boosting_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        max_iter: [100, 200]
```

**Common mistake:** Using n_estimators as a constructor field or halving resource; this family uses max_iter and the wizard default resource is n_samples.

## k_neighbors_regressor

**Nearest-neighbor regression. Task: regression.** Predicts by averaging nearby training targets. A useful local-pattern baseline when distance between scaled feature vectors has a clear meaning; prediction retains dependence on the training set.

**Sample weights:** No; neighbor-distance `weights` is a different setting.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:k_neighbors_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_neighbors` | `5` | `5` | Number of neighbors to use. |
| `weights` | `"uniform"` | `'uniform'` | Weight function used in prediction. |
| `algorithm` | `"auto"` | `'auto'` | Algorithm used to compute the nearest neighbors. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `algorithm` | `["auto"]` | `["auto", "ball_tree", "kd_tree", "brute"]` |
| `n_neighbors` | `[3, 5, 9]` | `[3, 5, 7, 9, 11]` |
| `weights` | `["uniform", "distance"]` | `["uniform", "distance"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:k_neighbors_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:k_neighbors_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: k_neighbors_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:k_neighbors_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: k_neighbors_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_neighbors: [3, 5]
```

**Common mistake:** Confusing distance-based neighbor weights with sample_weight; this estimator cannot fit explicit row weights.

## lasso_regression

**Lasso regression. Task: regression.** L1 regularization can shrink some coefficients to zero, giving a compact linear model when only a subset of features carries signal. Scale inputs and check convergence.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:lasso_regression:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | `1.0` | Constant that multiplies the L1 term. |
| `selection` | `"cyclic"` | `'cyclic'` | If set to 'random', a random coefficient is updated every iteration. |
| `random_state` | `42` | Library/resolver | Seed for the coefficient update order when selection='random'. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.001, 0.01, 0.1, 1.0, 10.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:lasso_regression:end -->

### Minimal training entries

Untuned:

<!-- model-example:lasso_regression:plain -->
```yaml
models:
  candidate:
    model:
      type: lasso_regression
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:lasso_regression:tuned -->
```yaml
models:
  candidate:
    model:
      type: lasso_regression
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        alpha: [0.001, 0.01]
```

**Common mistake:** Reading a zero coefficient as proof that the feature has no signal when correlated substitutes are present.

## lgbm_regressor

**LightGBM regression. Task: regression.** An optional histogram-based boosted-tree model. num_leaves controls leaf-wise model capacity; min_child_samples helps keep small leaves from overfitting.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** `lightgbm`.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:lgbm_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of boosting rounds. |
| `num_leaves` | `31` | `31` | Maximum number of leaves per tree. Controls model complexity; increase for more accuracy. |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree. |
| `max_depth` | `-1` | `-1` | Maximum tree depth. -1 = unlimited (controlled by num_leaves). |
| `min_child_samples` | `20` | `20` | Minimum samples required in a leaf. Regularises against overfitting. |
| `subsample` | `1.0` | `1.0` | Fraction of training rows sampled each iteration for GBDT/DART. GOSS uses gradient sampling instead. |
| `colsample_bytree` | `1.0` | `1.0` | Fraction of features sampled per tree. |
| `reg_alpha` | `0.0` | `0.0` | L1 regularization term on leaf weights. |
| `reg_lambda` | `0.0` | `0.0` | L2 regularization term on leaf weights. |
| `boosting_type` | `"gbdt"` | `'gbdt'` | GBDT is standard gradient boosting. DART adds dropout. GOSS samples by gradient magnitude. |
| `random_state` | `42` | Library/resolver | Seed for feature/row subsampling (bagging, colsample, DART). |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |
| `subsample_freq` | Not a catalog control | `None` | Core constructor setting. |
| `verbose` | Not a catalog control | `-1` | Core constructor setting. |
| `verbosity` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `boosting_type` | `[]` | `["gbdt", "dart", "goss"]` |
| `colsample_bytree` | `[]` | `[0.6, 0.8, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[-1, 5, 10]` | `[-1, 5, 10, 20]` |
| `min_child_samples` | `[]` | `[5, 10, 20, 50]` |
| `n_estimators` | `[100, 200, 500]` | `[100, 200, 500, 1000]` |
| `num_leaves` | `[31, 63, 127]` | `[15, 31, 63, 127, 255]` |
| `reg_alpha` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `reg_lambda` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:lgbm_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:lgbm_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: lgbm_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:lgbm_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: lgbm_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Setting a large num_leaves on a small dataset, or assuming library availability alone grants Spark or serving admission.

## random_forest_regressor

**Random forest regression. Task: regression.** Averages many randomized trees to capture nonlinear interactions with less variance than a single tree. It usually does not need numeric feature scaling.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:random_forest_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `50` | The number of trees in the forest. |
| `max_depth` | `null` | `10` | The maximum depth of the tree. If None, nodes are expanded until all leaves are pure. |
| `min_samples_split` | `2` | `5` | The minimum number of samples required to split an internal node. |
| `min_samples_leaf` | `1` | `2` | The minimum number of samples required to be at a leaf node. |
| `bootstrap` | `true` | Library/resolver | Whether bootstrap samples are used when building trees. |
| `random_state` | `42` | Library/resolver | Seed for bootstrap sampling and tree building. Same data + same seed = identical forest. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `bootstrap` | `[]` | `[true, false]` |
| `criterion` | `["squared_error", "friedman_mse"]` | `["squared_error", "friedman_mse", "absolute_error"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:random_forest_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:random_forest_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: random_forest_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:random_forest_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: random_forest_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Assuming an omitted parameter means the upstream library default: Core starts at 50 trees, depth 10, split size 5 and leaf size 2.

## stacking_regressor

**Stacking ensemble regression. Task: regression.** Learns a final estimator from out-of-fold base predictions. Its internal cv prevents the final learner from training on overoptimistic in-sample predictions; passthrough also includes original features.

**Sample weights:** Yes for the default configuration; every selected child and final estimator must support it.

**Optional package:** None beyond the standard Core scikit-learn runtime; add `xgboost` or `lightgbm` when selected as a member.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:stacking_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `base_estimators` | `["random_forest", "gradient_boosting", "ridge"]` | Library/resolver | Base learners combined by the ensemble. |
| `final_estimator` | `"ridge"` | Library/resolver | Meta-learner trained on the base models' out-of-fold predictions. |
| `cv` | `5` | `5` | Cross-validation folds used to generate out-of-fold predictions for the final estimator. Keep small (e.g. 3) when also running an outer hyperparameter search to limit nested-CV cost. |
| `passthrough` | `false` | Library/resolver | When on, the final estimator sees the original input features alongside the base models' predictions instead of the predictions alone. Can help when the base learners miss signal in the raw data. |
| `n_jobs` | `1` | Library/resolver | Number of base models fit in parallel. 1 = sequential, -1 = use all CPU cores. Speeds up fitting when you have several independent base learners. |

### Default search spaces

These axes use the catalog's default members shown above. The wizard
expands your selected members instead; it keeps composition, voting and
internal CV fixed. Calibrated members add `estimator__` after their name.

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `final_estimator__alpha` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `final_estimator__fit_intercept` | `[true, false]` | `[true, false]` |
| `final_estimator__max_iter` | `[2000]` | `[1000, 2000, 5000]` |
| `final_estimator__solver` | `["auto", "saga"]` | `["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]` |
| `gradient_boosting__learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `gradient_boosting__max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `gradient_boosting__min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `gradient_boosting__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `gradient_boosting__subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |
| `random_forest__bootstrap` | `[]` | `[true, false]` |
| `random_forest__criterion` | `["squared_error", "friedman_mse"]` | `["squared_error", "friedman_mse", "absolute_error"]` |
| `random_forest__max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `random_forest__min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `random_forest__min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `random_forest__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `ridge__alpha` | `[0.01, 0.1, 1.0, 10.0, 100.0]` | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `ridge__fit_intercept` | `[true, false]` | `[true, false]` |
| `ridge__max_iter` | `[2000]` | `[1000, 2000, 5000]` |
| `ridge__solver` | `["auto", "saga"]` | `["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:stacking_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:stacking_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: stacking_regressor
      params: {"base_estimators": ["ridge", "decision_tree"], "final_estimator": "ridge", "cv": 3}
```

Tuned (small explicit grid):

<!-- model-example:stacking_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: stacking_regressor
      params: {"base_estimators": ["ridge", "decision_tree"], "final_estimator": "ridge", "cv": 3}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        ridge__alpha: [0.01, 0.1]
```

**Common mistake:** Confusing internal cv with the external model-selection folds, or multiplying both by a large search budget without checking the resulting fit count.

## svr

**Support vector regression. Task: regression.** Kernel regression for moderate-sized datasets with smooth nonlinear relationships. Scale features and compare the kernel, C and gamma together; large datasets can make fitting expensive.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:svr:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `C` | `1.0` | `1.0` | Regularization parameter. The strength of the regularization is inversely proportional to C. |
| `kernel` | `"rbf"` | `'rbf'` | Specifies the kernel type to be used in the algorithm. |
| `gamma` | `"scale"` | `'scale'` | Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `C` | `[0.1, 1.0, 10.0]` | `[0.1, 1.0, 10.0, 100.0]` |
| `gamma` | `["scale", "auto"]` | `["scale", "auto"]` |
| `kernel` | `["linear", "rbf"]` | `["linear", "rbf", "poly", "sigmoid"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:svr:end -->

### Minimal training entries

Untuned:

<!-- model-example:svr:plain -->
```yaml
models:
  candidate:
    model:
      type: svr
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:svr:tuned -->
```yaml
models:
  candidate:
    model:
      type: svr
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        C: [0.1, 1.0]
```

**Common mistake:** Applying an RBF kernel to features with wildly different units, or assuming its training cost scales like a linear model.

## voting_regressor

**Voting ensemble regression. Task: regression.** Combines independently fitted base learners. Regression averages predictions; classification can average probabilities (soft) or use a majority label vote (hard). weights follow the order of base_estimators.

**Sample weights:** Yes for the default configuration; every selected child and final estimator must support it.

**Optional package:** None beyond the standard Core scikit-learn runtime; add `xgboost` or `lightgbm` when selected as a member.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:voting_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `base_estimators` | `["linear_regression", "random_forest", "gradient_boosting"]` | Library/resolver | Fixed base learners combined by the ensemble; enable base model tuning to search their parameters. |
| `n_jobs` | `1` | Library/resolver | Number of base models fit in parallel. 1 = sequential, -1 = use all CPU cores. Speeds up fitting when you have several independent base learners. |

### Default search spaces

These axes use the catalog's default members shown above. The wizard
expands your selected members instead; it keeps composition, voting and
internal CV fixed. Calibrated members add `estimator__` after their name.

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `gradient_boosting__learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `gradient_boosting__max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `gradient_boosting__min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `gradient_boosting__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `gradient_boosting__subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |
| `linear_regression__fit_intercept` | `[true, false]` | `[true, false]` |
| `random_forest__bootstrap` | `[]` | `[true, false]` |
| `random_forest__criterion` | `["squared_error", "friedman_mse"]` | `["squared_error", "friedman_mse", "absolute_error"]` |
| `random_forest__max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `random_forest__min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `random_forest__min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `random_forest__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:voting_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:voting_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: voting_regressor
      params: {"base_estimators": ["ridge", "decision_tree"]}
```

Tuned (small explicit grid):

<!-- model-example:voting_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: voting_regressor
      params: {"base_estimators": ["ridge", "decision_tree"]}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        ridge__alpha: [0.01, 0.1]
```

**Common mistake:** Using full registry IDs as member names: use ridge or random_forest, for example, rather than ridge_regression or random_forest_regressor. Voting regression averages numeric predictions; member weights must follow the member order.

## xgboost_regressor

**XGBoost regression. Task: regression.** An optional gradient-boosted tree model with row/column sampling and regularization controls. Useful for tabular interactions when the matching xgboost package is available in every runtime.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** `xgboost`.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission. Only reviewed CPU `gbtree` regressors with supported built-in objectives qualify.

<!-- model-tables:xgboost_regressor:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of gradient boosted trees. |
| `max_depth` | `6` | `6` | Maximum tree depth for base learners. |
| `learning_rate` | `0.3` | `0.3` | Boosting learning rate (eta). |
| `subsample` | `1.0` | `0.8` | Subsample ratio of the training instances. |
| `colsample_bytree` | `1.0` | `0.8` | Subsample ratio of columns when constructing each tree. |
| `min_child_weight` | `1` | Library/resolver | Minimum sum of instance weights in a child. Higher values regularize against overfitting. |
| `gamma` | `0.0` | Library/resolver | Minimum loss reduction required to make a split. Higher = more conservative tree growth. |
| `reg_alpha` | `0.0` | Library/resolver | L1 regularization term on leaf weights. |
| `reg_lambda` | `1.0` | Library/resolver | L2 regularization term on leaf weights. Default sklearn XGBoost is 1. |
| `random_state` | `42` | Library/resolver | Seed for row/column subsampling when subsample or colsample_bytree < 1.0. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `colsample_bytree` | `[]` | `[0.6, 0.8, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.3]` |
| `max_depth` | `[3, 5, 7]` | `[3, 5, 7, 9]` |
| `min_child_weight` | `[1, 3, 5]` | `[1, 3, 5, 7]` |
| `n_estimators` | `[100, 200, 500]` | `[100, 200, 500, 1000]` |
| `reg_alpha` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `reg_lambda` | `[0.1, 1.0, 5.0]` | `[0.1, 1.0, 5.0, 10.0]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:xgboost_regressor:end -->

### Minimal training entries

Untuned:

<!-- model-example:xgboost_regressor:plain -->
```yaml
models:
  candidate:
    model:
      type: xgboost_regressor
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:xgboost_regressor:tuned -->
```yaml
models:
  candidate:
    model:
      type: xgboost_regressor
      params: {}
    tuning:
      strategy: grid
      metric: r2
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Installing xgboost only on the training driver, or assuming every classifier/booster configuration is admitted for Spark workers.

## logistic_regression

**Logistic regression. Task: classification.** A probabilistic linear classification baseline for binary or multiclass targets. Scale numeric features; C is inverse regularization strength, so smaller C means stronger shrinkage.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:logistic_regression:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `C` | `1.0` | Library/resolver | Smaller values specify stronger regularization. |
| `penalty` | `"l2"` | Library/resolver | Norm used in the penalization. |
| `solver` | `"lbfgs"` | `'lbfgs'` | Algorithm to use in the optimization problem. |
| `max_iter` | `100` | `1000` | Maximum number of iterations taken for the solvers to converge. |
| `l1_ratio` | `null` | Library/resolver | The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. |
| `random_state` | `42` | Library/resolver | Seed used by stochastic solvers (saga, liblinear shuffling). |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `C` | `[0.01, 0.1, 1.0, 10.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` |
| `max_iter` | `[500, 2000]` | `[100, 200, 500, 1000, 2000, 5000]` |
| `penalty` | `["l1", "l2"]` | `["l1", "l2"]` |
| `solver` | `["saga"]` | `["saga"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:logistic_regression:end -->

### Minimal training entries

Untuned:

<!-- model-example:logistic_regression:plain -->
```yaml
models:
  candidate:
    model:
      type: logistic_regression
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:logistic_regression:tuned -->
```yaml
models:
  candidate:
    model:
      type: logistic_regression
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        C: [0.01, 0.1]
```

**Common mistake:** Choosing an incompatible solver and penalty pair. The default generated grid switches to saga so its L1 and L2 candidates are compatible.

## adaboost_classifier

**AdaBoost classification. Task: classification.** Sequentially reweights the learning problem to focus later learners on difficult examples. A compact boosting baseline when simple weak learners capture useful signal.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:adaboost_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `50` | `50` | The maximum number of estimators at which boosting is terminated. |
| `learning_rate` | `1.0` | `1.0` | Weight applied to each classifier at each boosting iteration. |
| `random_state` | `42` | Library/resolver | Seed passed to the base estimator's randomness, if it has any. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `learning_rate` | `[0.1, 0.5, 1.0]` | `[0.01, 0.1, 1.0, 1.5]` |
| `n_estimators` | `[50, 100, 200]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:adaboost_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:adaboost_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: adaboost_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:adaboost_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: adaboost_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [50, 100]
```

**Common mistake:** Allowing mislabeled rows or severe target outliers to dominate the repeatedly reweighted fit.

## bernoulli_nb

**Bernoulli naive Bayes. Task: classification.** A classifier for binary occurrence features, such as whether a token is present. binarize defines the threshold applied to numeric inputs; null means inputs are already binary.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:bernoulli_nb:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | `1.0` | Additive (Laplace/Lidstone) smoothing parameter. |
| `binarize` | `0.0` | `0.0` | Threshold for binarizing features. Features above this value are set to 1, others to 0. Set to None to skip binarization (assume features are already binary). |
| `fit_prior` | `true` | `True` | Whether to learn class prior probabilities. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.1, 0.5, 1.0, 2.0]` | `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0]` |
| `binarize` | `[0.0, 0.5]` | `[0.0, 0.5, 1.0]` |
| `fit_prior` | `[]` | `[true, false]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:bernoulli_nb:end -->

### Minimal training entries

Untuned:

<!-- model-example:bernoulli_nb:plain -->
```yaml
models:
  candidate:
    model:
      type: bernoulli_nb
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:bernoulli_nb:tuned -->
```yaml
models:
  candidate:
    model:
      type: bernoulli_nb
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        alpha: [0.1, 0.5]
```

**Common mistake:** Using count magnitude as a signal while binarization discards it; consider MultinomialNB for counts.

## calibrated_classifier

**Calibrated classifier. Task: classification.** Wraps a base classifier with an internal cross-validated probability calibrator. Useful when probability quality matters in addition to class ranking; sigmoid is a compact starting point.

**Sample weights:** Yes for the default configuration; every selected child and final estimator must support it.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:calibrated_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `base_estimator` | `"logistic_regression"` | Library/resolver | Classifier whose probabilities are calibrated. Uses sensible defaults for the chosen estimator. During tuning, each selected classifier is a search candidate. |
| `method` | `"sigmoid"` | `'sigmoid'` | How to map raw scores to calibrated probabilities. Sigmoid (Platt) suits small datasets; isotonic is non-parametric and needs more data. |
| `cv` | `5` | `5` | Number of cross-validation folds used to fit the calibrator. |
| `random_state` | `42` | Library/resolver | Seed for stochastic base estimators. Does not change calibration CV splits or deterministic base estimators. |
| `estimator` | Not a catalog control | `LogisticRegression(max_iter=1000)` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `cv` | `[3, 5]` | `[3, 5, 10]` |
| `method` | `["sigmoid", "isotonic"]` | `["sigmoid", "isotonic"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:calibrated_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:calibrated_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: calibrated_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:calibrated_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: calibrated_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        method: ["sigmoid", "isotonic"]
```

**Common mistake:** Using isotonic calibration with very little data, or forgetting that calibration cv adds fits inside an outer search.

## decision_tree_classifier

**Decision tree classification. Task: classification.** A readable nonlinear baseline that splits feature space into regions. Depth and leaf constraints control how finely it partitions the data.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:decision_tree_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `max_depth` | `null` | `None` | The maximum depth of the tree. |
| `min_samples_split` | `2` | `2` | The minimum number of samples required to split an internal node. |
| `min_samples_leaf` | `1` | Library/resolver | The minimum number of samples required to be at a leaf node. |
| `random_state` | `42` | Library/resolver | Seed for split tie-breaking and feature sub-sampling. |
| `criterion` | `"gini"` | `'gini'` | The function to measure the quality of a split. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `criterion` | `["gini", "entropy"]` | `["gini", "entropy"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 3, 5, 10, 20]` |
| `min_samples_leaf` | `[]` | `[1, 2, 4]` |
| `min_samples_split` | `[2, 5, 10]` | `[2, 5, 10]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:decision_tree_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:decision_tree_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: decision_tree_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:decision_tree_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: decision_tree_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        max_depth: [5, 10]
```

**Common mistake:** Leaving a deep tree unconstrained on a small noisy dataset and treating the training score as generalization.

## extra_trees_classifier

**Extremely randomized trees classification. Task: classification.** Averages trees with extra split randomization, providing a useful tree-ensemble comparison to Random Forest. Its default bootstrap is false.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:extra_trees_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of trees in the forest. |
| `max_depth` | `null` | `None` | Maximum depth of the tree. None = unlimited. |
| `min_samples_split` | `2` | `2` | Minimum samples required to split an internal node. |
| `min_samples_leaf` | `1` | `1` | Minimum samples required at a leaf node. |
| `bootstrap` | `false` | `False` | Whether bootstrap samples are used (False = use full dataset). |
| `random_state` | `42` | Library/resolver | Seed for bootstrap sampling and random split selection. |
| `criterion` | `"gini"` | `'gini'` | Function to measure the quality of a split. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `bootstrap` | `[]` | `[false, true]` |
| `criterion` | `["gini", "entropy"]` | `["gini", "entropy", "log_loss"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 50]` |
| `min_samples_leaf` | `[]` | `[1, 2, 4]` |
| `min_samples_split` | `[2, 5]` | `[2, 5, 10]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:extra_trees_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:extra_trees_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: extra_trees_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:extra_trees_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: extra_trees_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Assuming bootstrap sampling is always active, or that more trees will fix excessive leaf-level overfitting.

## gaussian_nb

**Gaussian naive Bayes. Task: classification.** A fast classification baseline for continuous features, modeling a separate Gaussian distribution per feature and class. Its conditional-independence assumption can still give useful classification.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:gaussian_nb:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `var_smoothing` | `1e-09` | `1e-09` | Portion of the largest variance of all features that is added to variances for calculation stability. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `var_smoothing` | `[1e-09, 1e-08, 1e-07, 1e-06]` | `[1e-09, 1e-08, 1e-07, 1e-06]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:gaussian_nb:end -->

### Minimal training entries

Untuned:

<!-- model-example:gaussian_nb:plain -->
```yaml
models:
  candidate:
    model:
      type: gaussian_nb
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:gaussian_nb:tuned -->
```yaml
models:
  candidate:
    model:
      type: gaussian_nb
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        var_smoothing: [1e-09, 1e-08]
```

**Common mistake:** Interpreting confident probabilities as calibrated when highly correlated features repeat the same evidence.

## gradient_boosting_classifier

**Gradient boosting classification. Task: classification.** Builds small trees sequentially to correct earlier errors. Useful for tabular nonlinear relationships; learning_rate and n_estimators jointly set the fitting budget.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:gradient_boosting_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | The number of boosting stages to perform. |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree by learning_rate. |
| `max_depth` | `3` | `3` | Maximum depth of the individual regression estimators. |
| `subsample` | `1.0` | Library/resolver | The fraction of samples to be used for fitting the individual base learners. |
| `min_samples_split` | `2` | Library/resolver | Minimum samples required to split an internal node. |
| `min_samples_leaf` | `1` | Library/resolver | Minimum samples required at a leaf node. |
| `random_state` | `42` | Library/resolver | Seed for the row subsampling used when subsample < 1.0. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:gradient_boosting_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:gradient_boosting_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: gradient_boosting_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:gradient_boosting_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: gradient_boosting_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Increasing learning_rate and tree count together without validation, which can make overfitting much worse.

## hist_gradient_boosting_classifier

**Histogram gradient boosting classification. Task: classification.** Uses feature bins to speed up boosted-tree fitting on larger tabular datasets. max_iter is the boosting-round count; leaves and regularization control capacity.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:hist_gradient_boosting_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `max_iter` | `100` | `100` | Maximum number of iterations (boosting rounds). |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree. |
| `max_leaf_nodes` | `31` | `31` | Maximum number of leaves per tree. |
| `max_depth` | `null` | `None` | Maximum depth per tree. None = unlimited (max_leaf_nodes is the effective limit). |
| `min_samples_leaf` | `20` | `20` | Minimum samples at a leaf node. |
| `l2_regularization` | `0.0` | `0.0` | L2 penalty applied to the leaves' values. |
| `max_bins` | `255` | `255` | Maximum number of bins for feature discretisation. Higher = more precise but slower. |
| `random_state` | `42` | Library/resolver | Seed for the histogram-based gradient boosting randomness. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `l2_regularization` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[]` | `[null, 3, 5, 10]` |
| `max_iter` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `max_leaf_nodes` | `[31, 63]` | `[15, 31, 63, 127]` |
| `min_samples_leaf` | `[20, 50]` | `[10, 20, 50, 100]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:hist_gradient_boosting_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:hist_gradient_boosting_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: hist_gradient_boosting_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:hist_gradient_boosting_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: hist_gradient_boosting_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        max_iter: [100, 200]
```

**Common mistake:** Using n_estimators as a constructor field or halving resource; this family uses max_iter and the wizard default resource is n_samples.

## k_neighbors_classifier

**Nearest-neighbor classifier. Task: classification.** Classifies from nearby labeled rows. Useful as a distance-based baseline with appropriately scaled features and enough neighbors from every class.

**Sample weights:** No; neighbor-distance `weights` is a different setting.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:k_neighbors_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_neighbors` | `5` | `5` | Number of neighbors to use. |
| `weights` | `"uniform"` | `'uniform'` | Weight function used in prediction. |
| `algorithm` | `"auto"` | `'auto'` | Algorithm used to compute the nearest neighbors. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `algorithm` | `["auto"]` | `["auto", "ball_tree", "kd_tree", "brute"]` |
| `n_neighbors` | `[3, 5, 9]` | `[3, 5, 7, 9, 11]` |
| `weights` | `["uniform", "distance"]` | `["uniform", "distance"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:k_neighbors_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:k_neighbors_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: k_neighbors_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:k_neighbors_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: k_neighbors_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_neighbors: [3, 5]
```

**Common mistake:** Setting n_neighbors above the number of rows in the smallest CV training fold, or expecting row-weight support.

## lgbm_classifier

**LightGBM classification. Task: classification.** An optional histogram-based boosted-tree model. num_leaves controls leaf-wise model capacity; min_child_samples helps keep small leaves from overfitting.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** `lightgbm`.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:lgbm_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of boosting rounds. |
| `num_leaves` | `31` | `31` | Maximum number of leaves per tree. Controls model complexity; increase for more accuracy. |
| `learning_rate` | `0.1` | `0.1` | Shrinks the contribution of each tree. |
| `max_depth` | `-1` | `-1` | Maximum tree depth. -1 = unlimited (controlled by num_leaves). |
| `min_child_samples` | `20` | `20` | Minimum samples required in a leaf. Regularises against overfitting. |
| `subsample` | `1.0` | `1.0` | Fraction of training rows sampled each iteration for GBDT/DART. GOSS uses gradient sampling instead. |
| `colsample_bytree` | `1.0` | `1.0` | Fraction of features sampled per tree. |
| `reg_alpha` | `0.0` | `0.0` | L1 regularization term on leaf weights. |
| `reg_lambda` | `0.0` | `0.0` | L2 regularization term on leaf weights. |
| `boosting_type` | `"gbdt"` | `'gbdt'` | GBDT is standard gradient boosting. DART adds dropout. GOSS samples by gradient magnitude. |
| `random_state` | `42` | Library/resolver | Seed for feature/row subsampling (bagging, colsample, DART). |
| `class_weight` | `null` | Library/resolver | Automatically adjusts weights inversely proportional to class frequencies in the training data, useful for imbalanced targets. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |
| `subsample_freq` | Not a catalog control | `None` | Core constructor setting. |
| `verbose` | Not a catalog control | `-1` | Core constructor setting. |
| `verbosity` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `boosting_type` | `[]` | `["gbdt", "dart", "goss"]` |
| `colsample_bytree` | `[]` | `[0.6, 0.8, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `max_depth` | `[-1, 5, 10]` | `[-1, 5, 10, 20]` |
| `min_child_samples` | `[]` | `[5, 10, 20, 50]` |
| `n_estimators` | `[100, 200, 500]` | `[100, 200, 500, 1000]` |
| `num_leaves` | `[31, 63, 127]` | `[15, 31, 63, 127, 255]` |
| `reg_alpha` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `reg_lambda` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:lgbm_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:lgbm_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: lgbm_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:lgbm_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: lgbm_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Setting a large num_leaves on a small dataset, or assuming library availability alone grants Spark or serving admission.

## multinomial_nb

**Multinomial naive Bayes. Task: classification.** A fast classifier for nonnegative counts or frequency-like features, such as token counts. alpha smooths rare feature counts and fit_prior controls learned class priors.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:multinomial_nb:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `alpha` | `1.0` | `1.0` | Additive (Laplace/Lidstone) smoothing parameter. 0 = no smoothing. |
| `fit_prior` | `true` | `True` | Whether to learn class prior probabilities. If False, uniform priors are used. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.1, 0.5, 1.0, 2.0]` | `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0]` |
| `fit_prior` | `[true, false]` | `[true, false]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:multinomial_nb:end -->

### Minimal training entries

Untuned:

<!-- model-example:multinomial_nb:plain -->
```yaml
models:
  candidate:
    model:
      type: multinomial_nb
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:multinomial_nb:tuned -->
```yaml
models:
  candidate:
    model:
      type: multinomial_nb
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        alpha: [0.1, 0.5]
```

**Common mistake:** Passing negative values produced by centering or standard scaling; use nonnegative features.

## random_forest_classifier

**Random forest classification. Task: classification.** Averages many randomized trees to capture nonlinear interactions with less variance than a single tree. It usually does not need numeric feature scaling.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Eligible exact model family, subject to complete artifact admission.

<!-- model-tables:random_forest_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `50` | The number of trees in the forest. |
| `max_depth` | `null` | `10` | The maximum depth of the tree. If None, nodes are expanded until all leaves are pure. |
| `min_samples_split` | `2` | `5` | The minimum number of samples required to split an internal node. |
| `min_samples_leaf` | `1` | `2` | The minimum number of samples required to be at a leaf node. |
| `bootstrap` | `true` | Library/resolver | Whether bootstrap samples are used when building trees. |
| `random_state` | `42` | Library/resolver | Seed for bootstrap sampling and tree building. Same data + same seed = identical forest. |
| `criterion` | `"gini"` | Library/resolver | The function to measure the quality of a split. |
| `class_weight` | `null` | Library/resolver | Automatically adjusts weights inversely proportional to class frequencies in the training data, useful for imbalanced targets. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `bootstrap` | `[]` | `[true, false]` |
| `criterion` | `["gini", "entropy"]` | `["gini", "entropy", "log_loss"]` |
| `max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:random_forest_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:random_forest_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: random_forest_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:random_forest_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: random_forest_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Assuming an omitted parameter means the upstream library default: Core starts at 50 trees, depth 10, split size 5 and leaf size 2.

## sgd_classifier

**Stochastic gradient classifier. Task: classification.** Fits a linear decision rule with stochastic optimization, useful for large feature matrices. The Core default loss is log_loss, which exposes probabilities; other losses change the output capabilities.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:sgd_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `loss` | `"log_loss"` | `'log_loss'` | The loss function to be used. |
| `penalty` | `"l2"` | `'l2'` | The regularization term to use. |
| `alpha` | `0.0001` | `0.0001` | Constant that multiplies the regularization term. Smaller means weaker regularization. |
| `l1_ratio` | `0.15` | `0.15` | The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. |
| `max_iter` | `1000` | `1000` | The maximum number of passes over the training data (epochs). |
| `random_state` | `42` | Library/resolver | Seed for the SGD shuffle and weight initialization. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `alpha` | `[0.0001, 0.001]` | `[1e-05, 0.0001, 0.001, 0.01]` |
| `loss` | `["log_loss", "hinge"]` | `["log_loss", "hinge", "modified_huber"]` |
| `max_iter` | `[1000]` | `[500, 1000, 2000]` |
| `penalty` | `["l2"]` | `["l1", "l2"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:sgd_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:sgd_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: sgd_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:sgd_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: sgd_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        loss: ["log_loss", "hinge"]
```

**Common mistake:** Selecting hinge loss while expecting predict_proba or soft voting. Check probability requirements when changing loss.

## stacking_classifier

**Stacking ensemble classification. Task: classification.** Learns a final estimator from out-of-fold base predictions. Its internal cv prevents the final learner from training on overoptimistic in-sample predictions; passthrough also includes original features.

**Sample weights:** Yes for the default configuration; every selected child and final estimator must support it.

**Optional package:** None beyond the standard Core scikit-learn runtime; add `xgboost` or `lightgbm` when selected as a member.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:stacking_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `base_estimators` | `["random_forest", "gradient_boosting", "svc"]` | Library/resolver | Base learners combined by the ensemble. |
| `final_estimator` | `"logistic_regression"` | Library/resolver | Meta-learner trained on the base models' out-of-fold predictions. |
| `cv` | `5` | `5` | Cross-validation folds used to generate out-of-fold predictions for the final estimator. Keep small (e.g. 3) when also running an outer hyperparameter search to limit nested-CV cost. |
| `passthrough` | `false` | Library/resolver | When on, the final estimator sees the original input features alongside the base models' predictions instead of the predictions alone. Can help when the base learners miss signal in the raw data. |
| `n_jobs` | `1` | Library/resolver | Number of base models fit in parallel. 1 = sequential, -1 = use all CPU cores. Speeds up fitting when you have several independent base learners. |
| `calibrate_base_models` | `false` | Library/resolver | Wrap each base classifier in CalibratedClassifierCV so its predicted probabilities are well-calibrated. Improves soft voting and stacking when base models give over/under-confident probabilities. |
| `calibration_method` | `"sigmoid"` | Library/resolver | Sigmoid (Platt scaling) is robust on small data; isotonic is more flexible but needs more samples to avoid overfitting. |
| `calibration_cv` | `3` | Library/resolver | Cross-validation folds used to fit each base model's calibrator. |

### Default search spaces

These axes use the catalog's default members shown above. The wizard
expands your selected members instead; it keeps composition, voting and
internal CV fixed. Calibrated members add `estimator__` after their name.

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `final_estimator__C` | `[0.01, 0.1, 1.0, 10.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` |
| `final_estimator__max_iter` | `[500, 2000]` | `[100, 200, 500, 1000, 2000, 5000]` |
| `final_estimator__penalty` | `["l1", "l2"]` | `["l1", "l2"]` |
| `final_estimator__solver` | `["saga"]` | `["saga"]` |
| `gradient_boosting__learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `gradient_boosting__max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `gradient_boosting__min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `gradient_boosting__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `gradient_boosting__subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |
| `random_forest__bootstrap` | `[]` | `[true, false]` |
| `random_forest__criterion` | `["gini", "entropy"]` | `["gini", "entropy", "log_loss"]` |
| `random_forest__max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `random_forest__min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `random_forest__min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `random_forest__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `svc__C` | `[0.1, 1.0, 10.0]` | `[0.1, 1.0, 10.0, 100.0]` |
| `svc__gamma` | `["scale", "auto"]` | `["scale", "auto"]` |
| `svc__kernel` | `["linear", "rbf"]` | `["linear", "rbf", "poly", "sigmoid"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:stacking_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:stacking_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: stacking_classifier
      params: {"base_estimators": ["logistic_regression", "decision_tree"], "final_estimator": "logistic_regression", "cv": 3}
```

Tuned (small explicit grid):

<!-- model-example:stacking_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: stacking_classifier
      params: {"base_estimators": ["logistic_regression", "decision_tree"], "final_estimator": "logistic_regression", "cv": 3}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        logistic_regression__C: [0.01, 0.1]
```

**Common mistake:** Confusing internal cv with the external model-selection folds, or multiplying both by a large search budget without checking the resulting fit count.

## svc

**Support vector classifier. Task: classification.** Kernel classification for moderate-sized data where a nonlinear boundary improves on a linear baseline. Scale inputs; Core enables probability=True, which adds calibration work to training.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** None beyond the standard Core scikit-learn runtime.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:svc:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `C` | `1.0` | `1.0` | Regularization parameter. The strength of the regularization is inversely proportional to C. |
| `kernel` | `"rbf"` | `'rbf'` | Specifies the kernel type to be used in the algorithm. |
| `gamma` | `"scale"` | `'scale'` | Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. |
| `probability` | Not a catalog control | `True` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `C` | `[0.1, 1.0, 10.0]` | `[0.1, 1.0, 10.0, 100.0]` |
| `gamma` | `["scale", "auto"]` | `["scale", "auto"]` |
| `kernel` | `["linear", "rbf"]` | `["linear", "rbf", "poly", "sigmoid"]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:svc:end -->

### Minimal training entries

Untuned:

<!-- model-example:svc:plain -->
```yaml
models:
  candidate:
    model:
      type: svc
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:svc:tuned -->
```yaml
models:
  candidate:
    model:
      type: svc
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        C: [0.1, 1.0]
```

**Common mistake:** Assuming the upstream SVC probability=False default is also the Core default, or underestimating fitting cost on large data.

## voting_classifier

**Voting ensemble classification. Task: classification.** Combines independently fitted base learners. Regression averages predictions; classification can average probabilities (soft) or use a majority label vote (hard). weights follow the order of base_estimators.

**Sample weights:** Yes for the default configuration; every selected child and final estimator must support it.

**Optional package:** None beyond the standard Core scikit-learn runtime; add `xgboost` or `lightgbm` when selected as a member.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:voting_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `base_estimators` | `["random_forest", "logistic_regression", "gradient_boosting"]` | Library/resolver | Base learners combined by the ensemble. |
| `voting` | `"soft"` | `'soft'` | Soft averages predicted probabilities (needs predict_proba on every base model); hard takes the majority class vote. |
| `n_jobs` | `1` | Library/resolver | Number of base models fit in parallel. 1 = sequential, -1 = use all CPU cores. Speeds up fitting when you have several independent base learners. |
| `calibrate_base_models` | `false` | Library/resolver | Wrap each base classifier in CalibratedClassifierCV so its predicted probabilities are well-calibrated. Improves soft voting and stacking when base models give over/under-confident probabilities. |
| `calibration_method` | `"sigmoid"` | Library/resolver | Sigmoid (Platt scaling) is robust on small data; isotonic is more flexible but needs more samples to avoid overfitting. |
| `calibration_cv` | `3` | Library/resolver | Cross-validation folds used to fit each base model's calibrator. |

### Default search spaces

These axes use the catalog's default members shown above. The wizard
expands your selected members instead; it keeps composition, voting and
internal CV fixed. Calibrated members add `estimator__` after their name.

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `gradient_boosting__learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.2]` |
| `gradient_boosting__max_depth` | `[3, 5]` | `[3, 5, 7, 9]` |
| `gradient_boosting__min_samples_leaf` | `[1, 5]` | `[1, 5, 10, 20]` |
| `gradient_boosting__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |
| `gradient_boosting__subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |
| `logistic_regression__C` | `[0.01, 0.1, 1.0, 10.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` |
| `logistic_regression__max_iter` | `[500, 2000]` | `[100, 200, 500, 1000, 2000, 5000]` |
| `logistic_regression__penalty` | `["l1", "l2"]` | `["l1", "l2"]` |
| `logistic_regression__solver` | `["saga"]` | `["saga"]` |
| `random_forest__bootstrap` | `[]` | `[true, false]` |
| `random_forest__criterion` | `["gini", "entropy"]` | `["gini", "entropy", "log_loss"]` |
| `random_forest__max_depth` | `[5, 10, 20]` | `[null, 5, 10, 20, 30, 50]` |
| `random_forest__min_samples_leaf` | `[1, 4]` | `[1, 2, 4, 8]` |
| `random_forest__min_samples_split` | `[2, 10]` | `[2, 5, 10, 20]` |
| `random_forest__n_estimators` | `[100, 200, 500]` | `[50, 100, 200, 500]` |

**Halving resource:** `n_samples` (the wizard default).
This model has no top-level `n_estimators` resource. Use `n_samples` in the wizard; another estimator parameter needs an explicit compatible configuration.
<!-- model-tables:voting_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:voting_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: voting_classifier
      params: {"base_estimators": ["logistic_regression", "decision_tree"]}
```

Tuned (small explicit grid):

<!-- model-example:voting_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: voting_classifier
      params: {"base_estimators": ["logistic_regression", "decision_tree"]}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        logistic_regression__C: [0.01, 0.1]
```

**Common mistake:** Using full registry IDs as member names: use ridge or random_forest, for example, rather than ridge_regression or random_forest_regressor. Soft voting requires probabilities from every member.

## xgboost_classifier

**XGBoost classification. Task: classification.** An optional gradient-boosted tree model with row/column sampling and regularization controls. Useful for tabular interactions when the matching xgboost package is available in every runtime.

**Sample weights:** Yes; configure the workflow `weight_column`.

**Optional package:** `xgboost`.

**Spark workers / serving:** Not admitted by the current Spark-worker/pinned-serving model allowlist; use local inference.

<!-- model-tables:xgboost_classifier:start -->
### Parameters and defaults

| Parameter | Core catalog default | Explicit Core constructor default | Meaning |
| --- | --- | --- | --- |
| `n_estimators` | `100` | `100` | Number of gradient boosted trees. |
| `max_depth` | `6` | `6` | Maximum tree depth for base learners. |
| `learning_rate` | `0.3` | `0.3` | Boosting learning rate (eta). |
| `subsample` | `1.0` | `0.8` | Subsample ratio of the training instances. |
| `colsample_bytree` | `1.0` | `0.8` | Subsample ratio of columns when constructing each tree. |
| `min_child_weight` | `1` | Library/resolver | Minimum sum of instance weights in a child. Higher values regularize against overfitting. |
| `gamma` | `0.0` | Library/resolver | Minimum loss reduction required to make a split. Higher = more conservative tree growth. |
| `reg_alpha` | `0.0` | Library/resolver | L1 regularization term on leaf weights. |
| `reg_lambda` | `1.0` | Library/resolver | L2 regularization term on leaf weights. Default sklearn XGBoost is 1. |
| `random_state` | `42` | Library/resolver | Seed for row/column subsampling when subsample or colsample_bytree < 1.0. |
| `class_weight` | `null` | Library/resolver | Automatically adjusts weights inversely proportional to class frequencies in the training data, useful for imbalanced targets. XGBoost has no native support for this, so it is applied as a computed sample_weight at fit time. |
| `n_jobs` | Not a catalog control | `-1` | Core constructor setting. |

### Default search spaces

| Axis | Grid / halving_grid | Random / halving_random / Optuna |
| --- | --- | --- |
| `colsample_bytree` | `[]` | `[0.6, 0.8, 1.0]` |
| `learning_rate` | `[0.05, 0.1, 0.2]` | `[0.01, 0.05, 0.1, 0.3]` |
| `max_depth` | `[3, 5, 7]` | `[3, 5, 7, 9]` |
| `min_child_weight` | `[1, 3, 5]` | `[1, 3, 5, 7]` |
| `n_estimators` | `[100, 200, 500]` | `[100, 200, 500, 1000]` |
| `reg_alpha` | `[]` | `[0.0, 0.01, 0.1, 1.0]` |
| `reg_lambda` | `[0.1, 1.0, 5.0]` | `[0.1, 1.0, 5.0, 10.0]` |
| `subsample` | `[0.8, 1.0]` | `[0.6, 0.8, 1.0]` |

**Halving resource:** `n_samples` (the wizard default).
This model also exposes `n_estimators`; selecting it requires explicit `max_resources` and removing that axis from the search space.
<!-- model-tables:xgboost_classifier:end -->

### Minimal training entries

Untuned:

<!-- model-example:xgboost_classifier:plain -->
```yaml
models:
  candidate:
    model:
      type: xgboost_classifier
      params: {}
```

Tuned (small explicit grid):

<!-- model-example:xgboost_classifier:tuned -->
```yaml
models:
  candidate:
    model:
      type: xgboost_classifier
      params: {}
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 4
      search_space:
        n_estimators: [100, 200]
```

**Common mistake:** Installing xgboost only on the training driver, or assuming every classifier/booster configuration is admitted for Spark workers.

## Keeping this catalog current

The explanatory text and examples are maintained here; marked parameter/search
regions are regenerated from Core. From a repository checkout, run:

```bash
python skyulf-core/templates/databricks/build_model_catalog.py
python skyulf-core/templates/databricks/build_model_catalog.py --check
```

The generator needs only Python's standard library. Its check and the model-card
coverage/example tests detect drift without needing workspace credentials.
