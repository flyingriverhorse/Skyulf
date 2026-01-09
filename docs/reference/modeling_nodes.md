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

Example (RandomForestClassifier):

```python
{
  "type": "random_forest_classifier",
  "params": {"n_estimators": 50, "random_state": 42}
}
```

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

### svc

Backed by `sklearn.svm.SVC`.

Defaults:
- `C=1.0`, `kernel=rbf`, `gamma=scale`
- `probability=True`, `random_state=42`

### k_neighbors_classifier

Backed by `sklearn.neighbors.KNeighborsClassifier`.

Defaults:
- `n_neighbors=5`, `weights=uniform`
- `algorithm=auto`, `n_jobs=-1`

### decision_tree_classifier

Backed by `sklearn.tree.DecisionTreeClassifier`.

Defaults:
- `max_depth=None`, `min_samples_split=2`
- `criterion=gini`, `random_state=42`

### gradient_boosting_classifier

Backed by `sklearn.ensemble.GradientBoostingClassifier`.

Defaults:
- `n_estimators=100`, `learning_rate=0.1`
- `max_depth=3`, `random_state=42`

### adaboost_classifier

Backed by `sklearn.ensemble.AdaBoostClassifier`.

Defaults:
- `n_estimators=50`, `learning_rate=1.0`
- `random_state=42`

### xgboost_classifier

Backed by `xgboost.XGBClassifier`.

Defaults:
- `n_estimators=100`, `max_depth=6`
- `learning_rate=0.3`, `n_jobs=-1`
- `random_state=42`

### gaussian_nb

Backed by `sklearn.naive_bayes.GaussianNB`.

Defaults:
- `var_smoothing=1e-9`

## Regression

### ridge_regression

Backed by `sklearn.linear_model.Ridge`.

Defaults:

- `alpha=1.0`, `solver=auto`, `random_state=42`

### lasso_regression

Backed by `sklearn.linear_model.Lasso`.

Defaults:
- `alpha=1.0`, `selection=cyclic`
- `random_state=42`

### elasticnet_regression

Backed by `sklearn.linear_model.ElasticNet`.

Defaults:
- `alpha=1.0`, `l1_ratio=0.5`
- `selection=cyclic`, `random_state=42`

### random_forest_regressor

Backed by `sklearn.ensemble.RandomForestRegressor`.

Defaults include:

- `n_estimators=50`, `max_depth=10`
- `min_samples_split=5`, `min_samples_leaf=2`
- `n_jobs=-1`, `random_state=42`

### svr

Backed by `sklearn.svm.SVR`.

Defaults:
- `C=1.0`, `kernel=rbf`, `gamma=scale`

### k_neighbors_regressor

Backed by `sklearn.neighbors.KNeighborsRegressor`.

Defaults:
- `n_neighbors=5`, `weights=uniform`
- `algorithm=auto`, `n_jobs=-1`

### decision_tree_regressor

Backed by `sklearn.tree.DecisionTreeRegressor`.

Defaults:
- `max_depth=None`, `min_samples_split=2`
- `criterion=squared_error`, `random_state=42`

### gradient_boosting_regressor

Backed by `sklearn.ensemble.GradientBoostingRegressor`.

Defaults:
- `n_estimators=100`, `learning_rate=0.1`
- `max_depth=3`, `random_state=42`

### adaboost_regressor

Backed by `sklearn.ensemble.AdaBoostRegressor`.

Defaults:
- `n_estimators=50`, `learning_rate=1.0`
- `random_state=42`

### xgboost_regressor

Backed by `xgboost.XGBRegressor`.

Defaults:
- `n_estimators=100`, `max_depth=6`
- `learning_rate=0.3`, `n_jobs=-1`
- `random_state=42`

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

Note: `SkyulfPipeline` performs modeling through the same building blocks (a calculator + applier); `StatefulEstimator`
is the lightweight wrapper exposed for low-level usage.
