# Modeling Nodes

This page documents modeling configuration for `SkyulfPipeline`.

## Common config shape

`SkyulfPipeline` expects a modeling block like:

```python
{
  "type": "logistic_regression",
  "node_id": "model_node",  # optional
  "params": { ... }          # optional; estimator hyperparameters
}
```

The sklearn wrapper supports both:

- Nested params (preferred): `{ "params": {"C": 1.0} }`
- Flat params (legacy): `{ "C": 1.0, "type": "..." }`

## Classification

### logistic_regression

Backed by `sklearn.linear_model.LogisticRegression`.

Defaults:

- `max_iter=1000`
- `solver=lbfgs`
- `random_state=42`

Learned params:

- fitted sklearn estimator (stored in-memory and pickled when saving the pipeline)

### random_forest_classifier

Backed by `sklearn.ensemble.RandomForestClassifier`.

Defaults include:

- `n_estimators=50`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `n_jobs=-1`, `random_state=42`

Learned params:

- fitted sklearn estimator

## Regression

### ridge_regression

Backed by `sklearn.linear_model.Ridge`.

Defaults:

- `alpha=1.0`, `solver=auto`, `random_state=42`

### random_forest_regressor

Backed by `sklearn.ensemble.RandomForestRegressor`.

Defaults include:

- `n_estimators=50`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `n_jobs=-1`, `random_state=42`

## Hyperparameter tuning

### hyperparameter_tuner

This mode wraps a base model and performs search.

Config:

- `type`: `hyperparameter_tuner`
- `base_model`: dict with a supported base model type (e.g., logistic regression)
- tuning options such as:
  - `strategy`: `grid` | `random` | `halving_grid` | `halving_random` | `optuna` (availability depends on installed packages)
  - `search_space`: dict of parameter → list/range
  - `metric`: e.g., `accuracy`, `f1`, `roc_auc`, `rmse`, `r2`
  - `cv_enabled`, `cv_type`, `cv_folds`, `random_state`

Learned params:

- a tuple `(best_model, tuning_result)` where `best_model` is a fitted estimator.

## Cross-validation

`StatefulEstimator.cross_validate()` can perform CV on the train split and returns aggregated fold metrics.
