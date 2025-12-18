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

`TransformerType` is a string key that `FeatureEngineer` dispatches to a Calculator/Applier pair.
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

`SkyulfPipeline` currently supports these model types:

- `logistic_regression`
- `random_forest_classifier`
- `ridge_regression`
- `random_forest_regressor`
- `hyperparameter_tuner`

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
  "strategy": "random",
  "search_space": {"C": [0.1, 1.0, 10.0]},
  "n_trials": 25,
  "metric": "accuracy"
}
```

See “Modeling Nodes” in Reference for details.
