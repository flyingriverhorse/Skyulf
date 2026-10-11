# Weighted Training and Support Matrix

Use a training column to control each observation's contribution to model fitting.
Weighting is optional. You can use sample weights, class weights, or both with
compatible models.

The support tables describe specific contracts, not every possible combination
of data, parameters and custom code.

## Which setting belongs where?

| Setting | Purpose | Where to configure it |
| --- | --- | --- |
| `weight_column` | Name of the source column containing row weights | `config/training.yml`; see the layout table below |
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

| Layout | Setting in `config/training.yml` |
| --- | --- |
| `single_model` | `defaults.weight_column` or the named model's `weight_column` |
| `model_competition` | One shared `defaults.weight_column` for all candidates |
| `multi_target` | Each named `models.<name>.weight_column` |

For a single model or shared competition setting:

```yaml
defaults:
  weight_column: importance  # Use null to disable sample weights.
```

For a multi-target model override:

```yaml
models:
  churn:
    weight_column: importance
```

Replace `churn` with the existing model name and retain its other fields. Other
models retain their own settings. There is no separate weights file. The loader
captures selected settings for replay; scoring uses the saved fitted artifact.
Older Python-configured projects can retain their existing `WEIGHT_COLUMN`
setting, but must not also declare a competing YAML model owner.

The weight column must exist in training data. Keep it out of `input_columns`
and do not reuse it as a target, record key, group or time column. Weights must
be numeric, finite, nonnegative, aligned with the rows, and have a positive sum
for each actual fit. Missing values, booleans, strings, negative/infinite values,
incorrect lengths and zero-total fit vectors are rejected.

See [Databricks Bundle setup](databricks_bundle.md) for the generated layout.

### Optional class weights

For a direct classifier, set `class_weight` in the model's params. In a Bundle:

```yaml
models:
  churn:
    model:
      type: logistic_regression
      params:
        class_weight: balanced
```

For tuning, keep this value fixed in the model params and remove any competing
`class_weight` search axis, or remove the fixed value and supply that axis in
`tuning.search_space`. Direct Core tuner configurations place fixed parameters
under `base_model.params`. Use YAML `null` (Python `None`) to disable class
weights. They apply to classification, not regression.
When both are enabled, sample and class weights are combined once, using the
rows supplied to that fit after any resampling. Native estimators retain their
own class-weight rules. Since the [scikit-learn 1.7 change](https://scikit-learn.org/1.7/whats_new/v1.7.html#sklearn-linear-model), LogisticRegression with
`class_weight="balanced"` uses weighted class frequencies when sample weights
are supplied; it does not multiply a count-only class factor by those weights.
For estimators without native class-weight support, Skyulf computes class
factors from label counts and multiplies them by the supplied row weights.

### Add SMOTE with sample weights

Keep `weight_column: importance` in the training settings. Separately, add
this step to the selected recipe in `config/preprocessing.yml`:

```yaml
- name: balance
  transformer: Oversampling
  params:
    method: smote
    synthetic_weight: class_mean
    random_state: 42
```

Place it after the transformations needed to produce valid numeric features.
Preserve the other steps in your recipe. Use the training preprocessing recipe,
not `config/pre_split.yml`: synthetic sampling must not learn from held-out rows.

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

The models below support the documented sample-weight contract. KNN final
estimators do not accept row weights. Install the optional XGBoost/LightGBM
packages when selecting those model families.

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

## Verify your configured recipe

Follow [Verify sampling and sample weights](sampling_weight_acceptance.md) to
compare actual fit inputs, saved/reloaded predictions and an independent estimator
reference. Keep the source snapshot and training/holdout membership fixed, and
check alignment after sampling rather than relying only on metadata or row counts.

Validate the paths your workflow uses: direct fitting, CV/search, final refit,
registered-model replay and publication. A check of one recipe or runtime does
not establish every estimator/parameter combination or optional compute backend.
Scoring should use the saved feature schema without requiring training weights.
