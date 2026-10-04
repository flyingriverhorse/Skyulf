# Weighted Training and Support Matrix

Use a training column to control each observation's contribution to model fitting.
Weighting is optional. You can use sample weights, class weights, or both with
compatible models.

This guide describes the current development implementation, verified on
2026-10-02 with Python 3.12.10, scikit-learn 1.8.0 and imbalanced-learn 0.14.1.
The support tables describe specific contracts, not every possible combination
of data, parameters and custom code.

## Which setting belongs where?

| Setting | Purpose | Where to configure it |
| --- | --- | --- |
| `weight_column` / `WEIGHT_COLUMN` | Name of the source column containing row weights | Bundle model file; see the layout table below |
| `sample_weight` | Numeric vector passed to the model's `fit` call | Bundle extracts it automatically; direct Core callers pass it explicitly |
| `class_weight` | Class-level weighting for a supported classifier | Model `params`; for tuning, `base_model.params` |
| `synthetic_weight` | Weight assigned to new synthetic training rows | The sampling step's `params`, only when combining sample weights with synthetic sampling |

Selecting a weight column does **not** automatically enable `synthetic_weight`.
Neither `weight_column` nor `synthetic_weight` belongs in the estimator's params.
The source weight column must never become an input feature.

## Configure a Bundle

The initialization wizard asks whether to use sample weights before model
selection. The default is disabled. When enabled, enter the column name and
choose from compatible models. Ensemble member choices are also filtered.
Multi-target branches choose independently. Runtime checks still validate edited
model definitions after initialization.

| Generated file | Setting |
| --- | --- |
| `src/modeling/single_model.py` | `WEIGHT_COLUMN = "importance"` |
| `src/modeling/model_competition.py` | One shared `WEIGHT_COLUMN` for all candidates |
| `src/modeling/multi_model.py` | Each `MODELS[branch_name]["workflow"]["weight_column"]` |

For single-model training, edit the existing assignment:

```python
# src/modeling/single_model.py
WEIGHT_COLUMN = "importance"  # Use None to disable sample weights.
```

For an existing multi-target branch, edit its workflow setting:

```python
# src/modeling/multi_model.py, after the existing MODELS declaration
MODELS["churn"]["workflow"]["weight_column"] = "importance"
```

Replace `churn` with an actual branch name. Other branches retain their settings.
There is no separate `weights.py`. The loader captures the model source and
selected settings for training replay; scoring uses the saved fitted artifact.

The weight column must exist in training data. Keep it out of `input_columns`
and do not reuse it as a target, record key, group or time column. Weights must
be numeric, finite, nonnegative, aligned with the rows, and have a positive sum
for each actual fit. Missing values, booleans, strings, negative/infinite values,
incorrect lengths and zero-total fit vectors are rejected.

See [Databricks Bundle setup](databricks_bundle.md) for the generated layout.

### Optional class weights

For a direct classifier, add `"class_weight": "balanced"` to its existing
`params`. For a generated tuner, edit the existing base model:

```python
# src/modeling/single_model.py, after the existing MODELING declaration
# This example assumes MODELING is a tuner with a compatible classifier.
MODELING["base_model"]["params"]["class_weight"] = "balanced"
MODELING["search_space"].pop("class_weight", None)
```

Removing the search axis keeps the value fixed. To tune class weights instead,
keep the search axis and remove the fixed value from `base_model.params`.
Use `None` to disable class weights. They apply to classification, not regression.
When both are enabled, sample and class weights are multiplied once. Class
weights are computed from the labels used by that fit, after any resampling.

### Add SMOTE with sample weights

Keep `WEIGHT_COLUMN = "importance"` in the model file. Separately, add this step
to the list returned by `build_preprocessing()` in
`src/features/preprocessing.py`:

```python
{
    "name": "balance",
    "transformer": "Oversampling",
    "params": {
        "method": "smote",
        "synthetic_weight": "class_mean",
        "random_state": 42,
    },
}
```

Place it after the transformations needed to produce valid numeric features.
Preserve the other steps in your recipe. Use the training preprocessing recipe,
not `pre_split.py`: synthetic sampling must not learn from held-out rows.

The resulting flow is:

1. The Bundle extracts `importance` into sample weights and excludes it from features.
2. SMOTE produces synthetic rows using feature values, without the weight column.
3. Existing rows keep their weights. Each new row receives the arithmetic mean
   weight of its class in that training fold.
4. The model receives both original and synthetic rows with their aligned
   `sample_weight` vector.

For example, if source training rows of class 1 have weights `[2, 4, 6]`, new
class-1 rows receive weight `4`. Validation/test rows do not enter this average.
This is Skyulf's explicit policy; it is not native SMOTE sample-weight support.

| Sampling use | Is `synthetic_weight` required? |
| --- | --- |
| Sampling without sample weights | No |
| Sample weights + `random_over` | No; repeat the source row's weight |
| Sample weights + undersampling | No; select the retained rows' weights |
| Sample weights + synthetic methods below | Yes; omission raises an error |

`"uniform"` is an alternative that assigns weight `1` to synthetic rows.
Neither policy changes sampling probabilities or neighbor distances. Standard
sampler requirements, including enough minority examples in each training fold,
still apply.

## Direct Core example

Core callers explicitly separate the weight column before passing data to the
pipeline. This complete example uses a train/test split before SMOTE:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from skyulf.pipeline import SkyulfPipeline

X, y = make_classification(
    n_samples=240,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    weights=[0.7, 0.3],
    random_state=42,
)
frame = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
frame["target"] = y
frame["importance"] = np.where(y == 1, 3.0, 1.0)
weights = frame.pop("importance").to_numpy()  # Remove it from model inputs.

pipeline = SkyulfPipeline({
    "preprocessing": [
        {
            "name": "split",
            "transformer": "TrainTestSplitter",
            "params": {
                "target_column": "target",
                "test_size": 0.2,
                "validation_size": 0.0,
                "stratify": True,
                "random_state": 42,
            },
        },
        {
            "name": "balance",
            "transformer": "Oversampling",
            "params": {
                "method": "smote",
                "synthetic_weight": "class_mean",
                "random_state": 42,
            },
        },
    ],
    "modeling": {
        "type": "logistic_regression",
        "params": {"max_iter": 500, "class_weight": None},
    },
})
pipeline.fit(frame, target_column="target", sample_weight=weights)
predictions = pipeline.predict(frame.drop(columns="target"))
assert len(predictions) == len(frame)
```

The splitter carries the corresponding training weights forward. Sampling runs
only during training; prediction neither resamples rows nor requires a weight
column. For already split data, use `SplitDataset.train_sample_weight` and omit
the separate `fit(sample_weight=...)` argument.

## Model support

The catalog audit fitted 34 model choices and compared supported models with
independent estimator references: 32 passed; the two KNN final models were
correctly rejected. Optional XGBoost/LightGBM packages must be installed.

| Group | Models |
| --- | --- |
| Classification | logistic_regression, adaboost_classifier, bernoulli_nb, decision_tree_classifier, extra_trees_classifier, gaussian_nb, gradient_boosting_classifier, hist_gradient_boosting_classifier, lgbm_classifier, multinomial_nb, random_forest_classifier, sgd_classifier, svc, xgboost_classifier |
| Regression | linear_regression, adaboost_regressor, decision_tree_regressor, elasticnet_regression, extra_trees_regressor, gradient_boosting_regressor, hist_gradient_boosting_regressor, lasso_regression, lgbm_regressor, random_forest_regressor, ridge_regression, svr, xgboost_regressor |
| Conditional ensembles | voting_classifier, voting_regressor, stacking_classifier, stacking_regressor, calibrated_classifier |
| Unsupported final models | k_neighbors_classifier, k_neighbors_regressor |

All ensemble estimators receiving row weights, including a stacking final model,
must support them. An incompatible child raises an error. Accepting arbitrary
`**kwargs` alone is not evidence of support. Voting's `weights` controls the
combination of model predictions; KNN's `weights="distance"` controls neighbor
contributions. Neither is `sample_weight`.

## Preprocessing support

All 67 registered names below, including aliases, have a supported weight
transport contract. Required columns, feature types and optional dependencies
still apply. This means weights reach the model correctly; it does not mean
each preprocessor learns weighted statistics.

| Group | Registered names |
| --- | --- |
| Scaling | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler |
| Imputation | SimpleImputer, KNNImputer, IterativeImputer, GroupImputer |
| Encoding | OneHotEncoder, OrdinalEncoder, LabelEncoder, DummyEncoder, HashEncoder, TargetEncoder, WOEEncoder |
| Vectorization | count_vectorizer, tfidf_vectorizer, hashing_vectorizer, sentence_embedder, tokenizer |
| Feature selection | feature_selection, VarianceThreshold, UnivariateSelection, ModelBasedSelection, CorrelationThreshold |
| Feature generation | FeatureGeneration, FeatureGenerationNode, FeatureMath, PolynomialFeatures, PolynomialFeaturesNode, FeatureInteraction |
| Binning | CustomBinning, GeneralBinning, KBinsDiscretizer |
| Transformations | Casting, SimpleTransformation, GeneralTransformation, PowerTransformer |
| Geo | GeoDistance, H3Index |
| Time | DateFeatures, LagFeatures, RollingAggregate |
| Missing values / rows | DropMissingRows, DropMissingColumns, MissingIndicator, Deduplicate |
| Functions | ColumnFunction, FittedFunction, RowFilterFunction |
| Sampling | Oversampling, Undersampling |
| Inspection | DataSnapshot, DatasetProfile |
| Splitting | TrainTestSplitter, Split, feature_target_split |
| Outliers | ClipValues, IQR, ZScore, Winsorize, EllipticEnvelope, ManualBounds |
| Cleaning | TextCleaning, AliasReplacement, ValueReplacement, InvalidValueReplacement |

Row-preserving operations retain the weights. Row removal also removes the
corresponding labels and weights. Lag/rolling sorting and filtering use the
same positional mapping for features, labels and weights. Inspection nodes
preserve the underlying data. DataFrame index labels and equal row counts are
not sufficient evidence of correct alignment.

**KNNImputer is supported before a weighted model.** It is a preprocessing step;
the unsupported KNN classifiers/regressors above are final estimators.

### Custom preprocessing is optional

| Situation | Behavior |
| --- | --- |
| No sample weights supplied | Normal custom preprocessing runs; no new mapping hook is required |
| Sample weights + audited built-in step | Core transports weights automatically |
| Sample weights + custom applier with valid mapping | Core maps the original labels and weights to output rows |
| Sample weights + custom step without mapping | Explicit error; the step and weights are not silently skipped |
| Invalid mapping | Explicit error |

Custom node authors can implement
`apply_with_row_mapping(data, artifact) -> (transformed_data, positions)`.
`positions` contains one zero-based input row position per output row, for
example `[2, 0, 2]` for a reorder with repetition. Core validates integer type,
shape, length and bounds, and rebuilds labels and weights from the original
positions. The author is responsible for returning the true row provenance.

Normal users of `ColumnFunction`, `FittedFunction` and `RowFilterFunction` do not
need to add this hook; use their existing row-preserving/filter contracts.
Custom `fit_transform_train` behavior is not automatically admitted. Column
merges with different row selections or arbitrary custom mappings are rejected.
An arbitrary custom window operation is not covered by built-in lag/rolling
support. See [custom nodes](extending_custom_nodes.md).

## Supported sampling methods

| Method | Weight handling |
| --- | --- |
| random_over | Repeat source weights using the sampler's selected indices |
| random_under_sampling, nearmiss, tomek_links, edited_nearest_neighbours | Select weights using the actual retained indices |
| smote, adasyn, borderline_smote, svm_smote, kmeans_smote | Preserve original weights; assign synthetic weights using the explicit policy |
| smote_tomek | Assign synthetic weights, then apply the Tomek cleaning selection |

## CV, evaluation and remaining limits

- Training weights are sliced and transported through direct fitting, CV, grid,
  random, halving-grid, halving-random, Optuna, nested CV and final refit.
- Preprocessing and sampling fit within training folds. Held-out rows do not
  determine synthetic weights or learned preprocessing parameters.
- **Evaluation metrics remain unweighted**, including CV selection and threshold
  objectives. `f1_weighted` averages classes by support; it does not consume
  user-provided row weights.
- **Imputation/scaling statistics remain unweighted.** Normal imputation followed
  by a weighted model is supported; weighted imputation is a separate feature.
- **The Canvas has no new weight-column selector.** This guide describes Bundle
  and Core configuration.
- Existing inference row-count/order restrictions still apply.

See [cross-validation](cross_validation.md),
[hyperparameter tuning](hyperparameter_tuning.md) and
[preprocessing placement](preprocessing_placement.md).

## Verification scope

For a concrete two-case cloud proof with actual fit inputs, saved artifacts and
independent reference comparisons, see
[Single-Model Sampling Weight Acceptance](sampling_weight_acceptance.md).

The weight-focused suite recorded 679 passes and three optional CLI skips; those
CLI checks passed separately using the real CLI. Tests inspect actual estimator
fit inputs, compare sampling geometry with imbalanced-learn, and check exact
weight alignment through CV/refit. The preprocessing inventory used the real H3
package; its sentence-embedding transport check used a controlled encoder.

Live Databricks acceptance covered SMOTE with voting, stacking and calibration,
plus two single-model Bundle trainings with different weight columns/class-weight
settings. Saved-model predictions matched independent references; scoring worked
without weight columns. Competition and multi-target generation/configuration
were tested locally, not through a complete cloud lifecycle for every layout.

The final clean full-Core rerun on 2026-10-03 passed: **13,848 passed, 662 skipped,
zero failures/errors**, in 28 minutes 39 seconds; process exit code **0**.
Branch-aware total coverage reached **90.01%**, passing the unchanged **90%** CI
gate. This replaces the earlier incomplete verification and 89.02% result.
The margin above the gate is small.

The run used fresh coverage data and temporary/cache directories. Hashes of all
817 measured source/test/config files were identical before and after the run.
Ruff, formatting, full CI Ty scope, CCN <= 10 and Bundle schema freshness also
passed. No production code or tests needed changes during this verification.

Skipped tests include optional Spark/Delta runtimes, opt-in Databricks CLI
checks and benchmarks; they are not counted as passes. This was a local Windows
run with the CI Core scope, not execution of every repository CI job or every
optional-runtime lane.
