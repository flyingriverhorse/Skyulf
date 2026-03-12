# Configuration

This page documents the configuration schema consumed by `SkyulfPipeline` and `FeatureEngineer`.

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

See "Modeling Nodes" in Reference for details.
