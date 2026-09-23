# Node and model coverage inventory

Source snapshot: `63dd3e21`, inspected 2026-09-22. **SM-17 follows the first local Bundle:**
this complete source-ID list is its starting inventory, not completed per-option
runtime validation. Changes after this snapshot must update the inventory.
Current work is the [Databricks integration lane](04-databricks-integration-plan.md)
using fitted local packages; the later Spark coverage is not a prerequisite.

An AST scan of literal `NodeRegistry.register` calls found **100 IDs**: **62
preprocessing** and **38 modeling**. Aliases and optional-library registrations
are retained deliberately. This is not a count of distinct algorithms or of
nodes available with only base dependencies.

## Coverage rules

- Every row needs configuration-level fit, apply, state export/import, inference,
  row/schema effects and runtime evidence. A family-level check cannot certify
  every option in that family.
- OPEN is planned missing coverage, not accepted permanent exclusion. SM-17 does
  not close while requested coverage remains open without an explicit user deferral.
- Only the imputer/scaler declarations listed below currently advertise native
  Spark FE. Their already-tested configurations remain supported during expansion.
- Model rows require separate local Python worker compatibility and native Spark
  training decisions. Generic sklearn regression/classification support exists,
  but this inventory does not certify every model ID or configured variant.
- XGBoost/LightGBM, clustering, custom prediction semantics and classifiers without
  probabilities require explicit bundle/output adapters. Composite sklearn paths
  require per-configuration auditing; add adapters only where their semantics need one.
- Native model backend changes are explicit choices; no silent replacement of
  an existing sklearn estimator by a different Spark estimator.
- Existing local pandas/Polars behavior must retain its regression coverage.

Detailed work and acceptance: [gap review](reports/2026-09-22-spark-databricks-gap-review.md).

## Source registrations

| Registration ID | Source | Owning task | Current Spark/portable coverage |
| --- | --- | --- | --- |
| `adaboost_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L571) | SM-17g/h/i | OPEN: per-model contract audit |
| `calibrated_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L260) | SM-17g/h/i | OPEN: per-model contract audit |
| `decision_tree_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L500) | SM-17g/h/i | OPEN: per-model contract audit |
| `extra_trees_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L664) | SM-17g/h/i | OPEN: per-model contract audit |
| `gaussian_nb` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L857) | SM-17g/h/i | OPEN: per-model contract audit |
| `gradient_boosting_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L536) | SM-17g/h/i | OPEN: per-model contract audit |
| `hist_gradient_boosting_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L705) | SM-17g/h/i | OPEN: per-model contract audit |
| `k_neighbors_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L463) | SM-17g/h/i | OPEN: per-model contract audit |
| `lgbm_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L775) | SM-17g/h/i | OPEN: per-model contract audit |
| `logistic_regression` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L67) | SM-17g/h/i | OPEN: per-model contract audit |
| `random_forest_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L385) | SM-17g/h/i | OPEN: per-model contract audit |
| `sgd_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L890) | SM-17g/h/i | OPEN: per-model contract audit |
| `svc` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L424) | SM-17g/h/i | OPEN: per-model contract audit |
| `xgboost_classifier` | [modeling/classification.py](../../skyulf-core/skyulf/modeling/classification.py#L607) | SM-17g/h/i | OPEN: per-model contract audit |
| `birch` | [modeling/clustering.py](../../skyulf-core/skyulf/modeling/clustering.py#L274) | SM-17g/h/i | OPEN: per-model contract audit |
| `gaussian_mixture` | [modeling/clustering.py](../../skyulf-core/skyulf/modeling/clustering.py#L241) | SM-17g/h/i | OPEN: per-model contract audit |
| `kmeans` | [modeling/clustering.py](../../skyulf-core/skyulf/modeling/clustering.py#L171) | SM-17g/h/i | OPEN: per-model contract audit |
| `minibatch_kmeans` | [modeling/clustering.py](../../skyulf-core/skyulf/modeling/clustering.py#L201) | SM-17g/h/i | OPEN: per-model contract audit |
| `stacking_classifier` | [modeling/ensemble.py](../../skyulf-core/skyulf/modeling/ensemble.py#L570) | SM-17g/h/i | OPEN: per-model contract audit |
| `stacking_regressor` | [modeling/ensemble.py](../../skyulf-core/skyulf/modeling/ensemble.py#L646) | SM-17g/h/i | OPEN: per-model contract audit |
| `voting_classifier` | [modeling/ensemble.py](../../skyulf-core/skyulf/modeling/ensemble.py#L532) | SM-17g/h/i | OPEN: per-model contract audit |
| `voting_regressor` | [modeling/ensemble.py](../../skyulf-core/skyulf/modeling/ensemble.py#L610) | SM-17g/h/i | OPEN: per-model contract audit |
| `bernoulli_nb` | [modeling/naive_bayes.py](../../skyulf-core/skyulf/modeling/naive_bayes.py#L71) | SM-17g/h/i | OPEN: per-model contract audit |
| `multinomial_nb` | [modeling/naive_bayes.py](../../skyulf-core/skyulf/modeling/naive_bayes.py#L33) | SM-17g/h/i | OPEN: per-model contract audit |
| `adaboost_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L384) | SM-17g/h/i | OPEN: per-model contract audit |
| `decision_tree_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L311) | SM-17g/h/i | OPEN: per-model contract audit |
| `elasticnet_regression` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L203) | SM-17g/h/i | OPEN: per-model contract audit |
| `extra_trees_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L420) | SM-17g/h/i | OPEN: per-model contract audit |
| `gradient_boosting_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L349) | SM-17g/h/i | OPEN: per-model contract audit |
| `hist_gradient_boosting_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L462) | SM-17g/h/i | OPEN: per-model contract audit |
| `k_neighbors_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L274) | SM-17g/h/i | OPEN: per-model contract audit |
| `lasso_regression` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L170) | SM-17g/h/i | OPEN: per-model contract audit |
| `lgbm_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L526) | SM-17g/h/i | OPEN: per-model contract audit |
| `linear_regression` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L58) | SM-17g/h/i | OPEN: per-model contract audit |
| `random_forest_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L131) | SM-17g/h/i | OPEN: per-model contract audit |
| `ridge_regression` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L95) | SM-17g/h/i | OPEN: per-model contract audit |
| `svr` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L241) | SM-17g/h/i | OPEN: per-model contract audit |
| `xgboost_regressor` | [modeling/regression.py](../../skyulf-core/skyulf/modeling/regression.py#L609) | SM-17g/h/i | OPEN: per-model contract audit |
| `CustomBinning` | [preprocessing/bucketing.py](../../skyulf-core/skyulf/preprocessing/bucketing.py#L608) | SM-17b | OPEN: Spark implementation |
| `GeneralBinning` | [preprocessing/bucketing.py](../../skyulf-core/skyulf/preprocessing/bucketing.py#L544) | SM-17b | OPEN: Spark implementation |
| `KBinsDiscretizer` | [preprocessing/bucketing.py](../../skyulf-core/skyulf/preprocessing/bucketing.py#L666) | SM-17b | OPEN: Spark implementation |
| `Casting` | [preprocessing/casting.py](../../skyulf-core/skyulf/preprocessing/casting.py#L449) | SM-17a | OPEN: Spark implementation |
| `AliasReplacement` | [preprocessing/cleaning/alias.py](../../skyulf-core/skyulf/preprocessing/cleaning/alias.py#L174) | SM-17a | OPEN: Spark implementation |
| `InvalidValueReplacement` | [preprocessing/cleaning/invalid_value.py](../../skyulf-core/skyulf/preprocessing/cleaning/invalid_value.py#L251) | SM-17a | OPEN: Spark implementation |
| `TextCleaning` | [preprocessing/cleaning/text.py](../../skyulf-core/skyulf/preprocessing/cleaning/text.py#L213) | SM-17a | OPEN: Spark implementation |
| `ValueReplacement` | [preprocessing/cleaning/value_replacement.py](../../skyulf-core/skyulf/preprocessing/cleaning/value_replacement.py#L216) | SM-17a | OPEN: Spark implementation |
| `Deduplicate` | [preprocessing/drop_and_missing/deduplicate.py](../../skyulf-core/skyulf/preprocessing/drop_and_missing/deduplicate.py#L90) | SM-17a | OPEN: Spark implementation |
| `DropMissingColumns` | [preprocessing/drop_and_missing/drop_columns.py](../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_columns.py#L107) | SM-17a | OPEN: Spark implementation |
| `DropMissingRows` | [preprocessing/drop_and_missing/drop_rows.py](../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_rows.py#L137) | SM-17a | OPEN: Spark implementation |
| `MissingIndicator` | [preprocessing/drop_and_missing/missing_indicator.py](../../skyulf-core/skyulf/preprocessing/drop_and_missing/missing_indicator.py#L127) | SM-17a | OPEN: Spark implementation |
| `DummyEncoder` | [preprocessing/encoding/dummy.py](../../skyulf-core/skyulf/preprocessing/encoding/dummy.py#L263) | SM-17b | OPEN: Spark implementation |
| `HashEncoder` | [preprocessing/encoding/hash.py](../../skyulf-core/skyulf/preprocessing/encoding/hash.py#L138) | SM-17b | OPEN: Spark implementation |
| `LabelEncoder` | [preprocessing/encoding/label.py](../../skyulf-core/skyulf/preprocessing/encoding/label.py#L310) | SM-17b | OPEN: Spark implementation |
| `OneHotEncoder` | [preprocessing/encoding/one_hot.py](../../skyulf-core/skyulf/preprocessing/encoding/one_hot.py#L230) | SM-17b | OPEN: Spark implementation |
| `OrdinalEncoder` | [preprocessing/encoding/ordinal.py](../../skyulf-core/skyulf/preprocessing/encoding/ordinal.py#L338) | SM-17b | OPEN: Spark implementation |
| `TargetEncoder` | [preprocessing/encoding/target.py](../../skyulf-core/skyulf/preprocessing/encoding/target.py#L326) | SM-17d | OPEN: Spark implementation |
| `WOEEncoder` | [preprocessing/encoding/woe.py](../../skyulf-core/skyulf/preprocessing/encoding/woe.py#L315) | SM-17d | OPEN: Spark implementation |
| `FeatureGeneration` | [preprocessing/feature_generation/generation.py](../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py#L105) | SM-17a | OPEN: Spark implementation |
| `FeatureGenerationNode` | [preprocessing/feature_generation/generation.py](../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py#L107) | SM-17a | OPEN: Spark implementation |
| `FeatureMath` | [preprocessing/feature_generation/generation.py](../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py#L106) | SM-17a | OPEN: Spark implementation |
| `FeatureInteraction` | [preprocessing/feature_generation/interaction.py](../../skyulf-core/skyulf/preprocessing/feature_generation/interaction.py#L162) | SM-17a | OPEN: Spark implementation |
| `PolynomialFeatures` | [preprocessing/feature_generation/polynomial.py](../../skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py#L100) | SM-17a | OPEN: Spark implementation |
| `PolynomialFeaturesNode` | [preprocessing/feature_generation/polynomial.py](../../skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py#L101) | SM-17a | OPEN: Spark implementation |
| `CorrelationThreshold` | [preprocessing/feature_selection/correlation.py](../../skyulf-core/skyulf/preprocessing/feature_selection/correlation.py#L162) | SM-17c | OPEN: Spark implementation |
| `feature_selection` | [preprocessing/feature_selection/facade.py](../../skyulf-core/skyulf/preprocessing/feature_selection/facade.py#L67) | SM-17c | OPEN: Spark implementation |
| `ModelBasedSelection` | [preprocessing/feature_selection/model_based.py](../../skyulf-core/skyulf/preprocessing/feature_selection/model_based.py#L45) | SM-17c | OPEN: Spark implementation |
| `UnivariateSelection` | [preprocessing/feature_selection/univariate.py](../../skyulf-core/skyulf/preprocessing/feature_selection/univariate.py#L46) | SM-17c | OPEN: Spark implementation |
| `VarianceThreshold` | [preprocessing/feature_selection/variance.py](../../skyulf-core/skyulf/preprocessing/feature_selection/variance.py#L33) | SM-17c | OPEN: Spark implementation |
| `GeoDistance` | [preprocessing/geo/distance.py](../../skyulf-core/skyulf/preprocessing/geo/distance.py#L156) | SM-17e | OPEN: Spark implementation |
| `H3Index` | [preprocessing/geo/h3_index.py](../../skyulf-core/skyulf/preprocessing/geo/h3_index.py#L123) | SM-17e | OPEN: Spark implementation |
| `IterativeImputer` | [preprocessing/imputation/iterative.py](../../skyulf-core/skyulf/preprocessing/imputation/iterative.py#L52) | SM-17b | OPEN: Spark implementation |
| `KNNImputer` | [preprocessing/imputation/knn.py](../../skyulf-core/skyulf/preprocessing/imputation/knn.py#L49) | SM-17b | OPEN: Spark implementation |
| `SimpleImputer` | [preprocessing/imputation/simple.py](../../skyulf-core/skyulf/preprocessing/imputation/simple.py#L117) | SM-17b | PARTIAL: native fit/apply mean/constant; other strategies OPEN |
| `DataSnapshot` | [preprocessing/inspection.py](../../skyulf-core/skyulf/preprocessing/inspection.py#L156) | SM-17f | OPEN: Spark implementation |
| `DatasetProfile` | [preprocessing/inspection.py](../../skyulf-core/skyulf/preprocessing/inspection.py#L88) | SM-17f | OPEN: Spark implementation |
| `EllipticEnvelope` | [preprocessing/outliers/elliptic.py](../../skyulf-core/skyulf/preprocessing/outliers/elliptic.py#L124) | SM-17c | OPEN: Spark implementation |
| `IQR` | [preprocessing/outliers/iqr.py](../../skyulf-core/skyulf/preprocessing/outliers/iqr.py#L70) | SM-17c | OPEN: Spark implementation |
| `ManualBounds` | [preprocessing/outliers/manual_bounds.py](../../skyulf-core/skyulf/preprocessing/outliers/manual_bounds.py#L88) | SM-17c | OPEN: Spark implementation |
| `Winsorize` | [preprocessing/outliers/winsorize.py](../../skyulf-core/skyulf/preprocessing/outliers/winsorize.py#L76) | SM-17c | OPEN: Spark implementation |
| `ZScore` | [preprocessing/outliers/zscore.py](../../skyulf-core/skyulf/preprocessing/outliers/zscore.py#L74) | SM-17c | OPEN: Spark implementation |
| `Oversampling` | [preprocessing/resampling.py](../../skyulf-core/skyulf/preprocessing/resampling.py#L270) | SM-17f | OPEN: Spark implementation |
| `Undersampling` | [preprocessing/resampling.py](../../skyulf-core/skyulf/preprocessing/resampling.py#L413) | SM-17f | OPEN: Spark implementation |
| `MaxAbsScaler` | [preprocessing/scaling/maxabs.py](../../skyulf-core/skyulf/preprocessing/scaling/maxabs.py#L67) | SM-17a | OPEN: Spark implementation |
| `MinMaxScaler` | [preprocessing/scaling/minmax.py](../../skyulf-core/skyulf/preprocessing/scaling/minmax.py#L67) | SM-17a | OPEN: Spark implementation |
| `RobustScaler` | [preprocessing/scaling/robust.py](../../skyulf-core/skyulf/preprocessing/scaling/robust.py#L86) | SM-17b | OPEN: Spark implementation |
| `StandardScaler` | [preprocessing/scaling/standard.py](../../skyulf-core/skyulf/preprocessing/scaling/standard.py#L124) | SM-17a | IMPLEMENTED: native fit/apply mean/std flags; retain regression gates |
| `Split` | [preprocessing/split.py](../../skyulf-core/skyulf/preprocessing/split.py#L165) | SM-17f | OPEN: Spark implementation |
| `TrainTestSplitter` | [preprocessing/split.py](../../skyulf-core/skyulf/preprocessing/split.py#L166) | SM-17f | OPEN: Spark implementation |
| `feature_target_split` | [preprocessing/split.py](../../skyulf-core/skyulf/preprocessing/split.py#L553) | SM-17f | OPEN: Spark implementation |
| `DateFeatures` | [preprocessing/time_series/date_features.py](../../skyulf-core/skyulf/preprocessing/time_series/date_features.py#L149) | SM-17a | OPEN: Spark implementation |
| `LagFeatures` | [preprocessing/time_series/lag.py](../../skyulf-core/skyulf/preprocessing/time_series/lag.py#L108) | SM-17d | OPEN: Spark implementation |
| `RollingAggregate` | [preprocessing/time_series/rolling.py](../../skyulf-core/skyulf/preprocessing/time_series/rolling.py#L146) | SM-17d | OPEN: Spark implementation |
| `GeneralTransformation` | [preprocessing/transformations/general.py](../../skyulf-core/skyulf/preprocessing/transformations/general.py#L189) | SM-17c | OPEN: Spark implementation |
| `PowerTransformer` | [preprocessing/transformations/power.py](../../skyulf-core/skyulf/preprocessing/transformations/power.py#L151) | SM-17c | OPEN: Spark implementation |
| `SimpleTransformation` | [preprocessing/transformations/simple.py](../../skyulf-core/skyulf/preprocessing/transformations/simple.py#L64) | SM-17c | OPEN: Spark implementation |
| `count_vectorizer` | [preprocessing/vectorization/count_vectorizer.py](../../skyulf-core/skyulf/preprocessing/vectorization/count_vectorizer.py#L125) | SM-17e | OPEN: Spark implementation |
| `hashing_vectorizer` | [preprocessing/vectorization/hashing_vectorizer.py](../../skyulf-core/skyulf/preprocessing/vectorization/hashing_vectorizer.py#L107) | SM-17e | OPEN: Spark implementation |
| `sentence_embedder` | [preprocessing/vectorization/sentence_embedder.py](../../skyulf-core/skyulf/preprocessing/vectorization/sentence_embedder.py#L206) | SM-17e | OPEN: Spark implementation |
| `tfidf_vectorizer` | [preprocessing/vectorization/tfidf_vectorizer.py](../../skyulf-core/skyulf/preprocessing/vectorization/tfidf_vectorizer.py#L119) | SM-17e | OPEN: Spark implementation |
| `tokenizer` | [preprocessing/vectorization/tokenizer.py](../../skyulf-core/skyulf/preprocessing/vectorization/tokenizer.py#L153) | SM-17e | OPEN: Spark implementation |

## SM-17-00 remaining acceptance

- [ ] Expand each registration to its actual configuration options and aliases.
- [ ] Record independent native-fit, native-apply and worker-Python capabilities.
- [ ] Record codec/state, column order, label/probability and row effects.
- [ ] Reconcile optional runtime registrations and fa?ade/dynamic model choices.
- [ ] Add a registry/inventory guard against new untracked nodes.
- [ ] Link each supported combination to tests and selected-runtime evidence.

The inventory is documentation only. It grants no capabilities and starts no jobs.
