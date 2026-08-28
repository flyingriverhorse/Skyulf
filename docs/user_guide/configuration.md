# Configuration

This page documents the configuration schema consumed by `SkyulfPipeline` and `FeatureEngineer` (the **skyulf-core** library).

> **Running the full platform?** See the [Backend Configuration Reference](../guides/backend_configuration.md) for all `.env` / environment variable settings (database, security, Celery, uploads, etc.).

## Pipeline config

`SkyulfPipeline` expects:

```python
{
  "preprocessing": [ ... ],
  "modeling": { ... }
}
```

### Preprocessing config

The preprocessing list is executed in order.

Each step is:

```python
{
  "name": "step_name",
  "transformer": "TransformerType",
  "params": { ... }
}
```

`TransformerType` is a string key resolved via the `NodeRegistry`.
For the full list and per-node parameters, see:

- Reference → Preprocessing Nodes
- Reference → API → Preprocessing → pipeline

#### Minimal examples

```python
# Split to avoid leakage
{"name": "split", "transformer": "TrainTestSplitter", "params": {"test_size": 0.2, "random_state": 42, "target_column": "target"}}
```

```python
# Impute missing numeric values
{"name": "impute", "transformer": "SimpleImputer", "params": {"strategy": "mean", "columns": ["age"]}}
```

```python
# Encode categoricals
{"name": "encode", "transformer": "OneHotEncoder", "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"}}
```

```python
# Scale numeric columns
{"name": "scale", "transformer": "StandardScaler", "params": {"auto_detect": True}}
```

### Modeling config

`SkyulfPipeline` supports the following model types via the `NodeRegistry`.

#### Classification (9 models)

| Key | Algorithm |
|---|---|
| `logistic_regression` | Logistic Regression |
| `random_forest_classifier` | Random Forest Classifier |
| `svc` | Support Vector Classifier |
| `k_neighbors_classifier` | K-Nearest Neighbors Classifier |
| `decision_tree_classifier` | Decision Tree Classifier |
| `gradient_boosting_classifier` | Gradient Boosting Classifier |
| `adaboost_classifier` | AdaBoost Classifier |
| `xgboost_classifier` | XGBoost Classifier *(requires `skyulf-core[modeling-xgboost]`)* |
| `gaussian_nb` | Gaussian Naive Bayes |

#### Regression (11 models)

| Key | Algorithm |
|---|---|
| `linear_regression` | Linear Regression |
| `ridge_regression` | Ridge Regression |
| `lasso_regression` | Lasso Regression |
| `elasticnet_regression` | ElasticNet Regression |
| `random_forest_regressor` | Random Forest Regressor |
| `svr` | Support Vector Regressor |
| `k_neighbors_regressor` | K-Nearest Neighbors Regressor |
| `decision_tree_regressor` | Decision Tree Regressor |
| `gradient_boosting_regressor` | Gradient Boosting Regressor |
| `adaboost_regressor` | AdaBoost Regressor |
| `xgboost_regressor` | XGBoost Regressor *(requires `skyulf-core[modeling-xgboost]`)* |

#### Meta

| Key | Purpose |
|---|---|
| `hyperparameter_tuner` | Wraps any model above with grid, random, Optuna, or halving search |

Example:

```python
{
  "type": "random_forest_classifier",
  "node_id": "model_node",
  "params": {
    "n_estimators": 200,
    "max_depth": 10
  }
}
```

Tuner example:

```python
{
  "type": "hyperparameter_tuner",
  "base_model": {"type": "logistic_regression"},
  "strategy": "optuna",
  "search_space": {"C": [0.1, 1.0, 10.0]},
  "n_trials": 25,
  "metric": "accuracy"
}
```

#### Reproducibility and seeds

Seeding has exactly one owner: `skyulf.types.DEFAULT_RANDOM_STATE` (currently
`42`). Every stochastic component (model fitting, CV folds, tuning search,
splitting, encoding cross-folds) falls back to it, so runs are reproducible
out of the box. Explicit configuration always wins, in this precedence order:

1. **Node params** — `"params": {"random_state": 7}` on a model/preprocessing
   node controls that node only. Pass `"random_state": null` to opt out of
   seeding a specific estimator.
2. **Tuner config** — `random_state` and `cv_random_state` on the
   `hyperparameter_tuner` config seed the search, the CV folds, and the final
   refit.
3. **`DEFAULT_RANDOM_STATE`** — the shared fallback, injected at model
   construction.

Two exceptions, both fixed and reproducible:

- **Iterative imputation** defaults its seed to `0` (sklearn's own default for
  `IterativeImputer`), not `42`. The canvas shows the seed in the imputation
  settings; set it to `42` if you want every node aligned.
- **Internal helpers** (feature-selection importance estimators, target/WoE
  encoding CV folds, profiling analyzers) carry fixed seeds that are not
  user-configurable — they are implementation details, not pipeline choices.

```python
# Pin one seed for the whole tuning run:
{
  "type": "hyperparameter_tuner",
  "base_model": {"type": "random_forest_classifier"},
  "strategy": "random",
  "search_space": {"n_estimators": [100, 200]},
  "random_state": 123,
  "cv_random_state": 123
}
```

##### Where to set seeds in the canvas (per node)

| Node | Control | Where in the UI | What it seeds |
|---|---|---|---|
| Train/Test Split | Random State | Split settings | Which rows go to train/test/validation |
| Classification, Regression, Text Classification (basic mode) | Random State | Hyperparameters → Customize | The model's own randomness (bootstrap sampling, tree splits, stochastic solvers, initialization) |
| Classification, Regression, Text Classification (advanced mode) | Random State | Tuning Strategy section | The tuning search and the final refit |
| Any training/ensemble node | Fold Split Seed | Cross Validation section | How rows are dealt to CV folds (only when shuffling is on) |
| Ensemble | Random State | Hyperparameter Tuning section | The ensemble tuning search and refit |
| Segmentation (K-Means, Mini-Batch K-Means, Gaussian Mixture) | Random State | Hyperparameters → Customize | Centroid/component initialization |
| Imputation (Iterative only) | Random State | Imputation settings (default `0`) | The iterative imputer's estimator |
| Resampling (over/undersampling) | Random State | Resampling settings | Synthetic sample generation / sampling |

Notes:

- **Basic mode:** the seed appears in the model's hyperparameter list once you
  switch on **Customize**; leaving it untouched keeps the default `42`.
- **Tuning never searches over seeds.** The seed is a fixed control, not a
  search-space candidate — same seed + same data = identical tuning outcome.
- **Models without a seed control** are deterministic by construction:
  Linear Regression, K-Nearest Neighbors, Naive Bayes family, SVC/SVR, BIRCH.
- **Ensemble base learners** (voting/stacking) are seeded automatically at
  construction; per-base-learner seeds can be supplied via
  `base_estimator_params` (API) or by wiring in a model node with a customized
  Random State.
- **Opt-out:** in a hand-written config, `"random_state": null` leaves that
  estimator unseeded (run-to-run variation). The canvas always sends a number.

See "Modeling Nodes" in Reference for details.
