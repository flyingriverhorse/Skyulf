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

The supported `TransformerType` strings are defined in the `FeatureEngineer` dispatcher:

- `TrainTestSplitter`
- `feature_target_split`
- `TextCleaning`
- `ValueReplacement`
- `Deduplicate`
- `DropMissingColumns`
- `DropMissingRows`
- `MissingIndicator`
- `AliasReplacement`
- `InvalidValueReplacement`
- `SimpleImputer`
- `KNNImputer`
- `IterativeImputer`
- `OneHotEncoder`
- `DummyEncoder`
- `OrdinalEncoder`
- `LabelEncoder`
- `TargetEncoder`
- `HashEncoder`
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `MaxAbsScaler`
- `IQR`
- `ZScore`
- `Winsorize`
- `ManualBounds`
- `EllipticEnvelope`
- `PowerTransformer`
- `SimpleTransformation`
- `GeneralTransformation`
- `GeneralBinning`
- `CustomBinning`
- `KBinsDiscretizer`
- `VarianceThreshold`
- `CorrelationThreshold`
- `UnivariateSelection`
- `ModelBasedSelection`
- `feature_selection`
- `Casting`
- `PolynomialFeatures`
- `FeatureMath` / `FeatureGenerationNode`
- `Oversampling`
- `Undersampling`
- `DatasetProfile`
- `DataSnapshot`

See “Preprocessing Nodes” in Reference for per-node parameters.

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
