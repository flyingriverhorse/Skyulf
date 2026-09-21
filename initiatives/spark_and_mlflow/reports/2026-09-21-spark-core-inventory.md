# Spark incelemesi: kaynak ve node envanteri

Tarih: 2026-09-21. Ana yorumlar ve sinirlar: [Inceleme raporu](2026-09-21-spark-databricks-readiness.md).

Bu dosya AST/import/registry taramasidir. Bir dosyanin listelenmesi, her satirinin
bagimsiz dogruluk denetiminden gectigi veya Spark destegi oldugu anlamina gelmez.
Isaretler arama bulgularidir; docstring icinde de gecebilirler.


Ta??ma notu (2026-09-21): Bu envanter tarihsel snapshot kayd?d?r. ? ile
i?aretli 88 kaynak/test ba?lant?s?n?n hedefi plan haz?rlan?rken g?ncel
checkout'ta bulunmuyordu; kay?tlar silinmedi, k?r?k ba?lant?lar d?z yol etiketi
olarak korundu. Dosya say?lar? ve eski bulgular yeniden ?l??lm?? say?lmaz.

## Kaynak dosyalari

| Dosya | Satir | Kayitlar | Yerel veri / yurutme isaretleri |
| --- | ---: | --- | --- |
| [__init__.py](../../../skyulf-core/skyulf/__init__.py) | 54 |  |  |
| [_validation.py](../../../skyulf-core/skyulf/_validation.py) | 12 |  |  |
| [config_validation.py](../../../skyulf-core/skyulf/config_validation.py) | 111 |  |  |
| [core/__init__.py](../../../skyulf-core/skyulf/core/__init__.py) | 47 |  |  |
| [core/artifacts.py](../../../skyulf-core/skyulf/core/artifacts.py) | 653 |  |  |
| [core/compute.py](../../../skyulf-core/skyulf/core/compute.py) | 80 |  |  |
| [core/deprecation.py](../../../skyulf-core/skyulf/core/deprecation.py) | 93 |  |  |
| [core/meta/__init__.py](../../../skyulf-core/skyulf/core/meta/__init__.py) | 1 |  |  |
| [core/meta/decorators.py](../../../skyulf-core/skyulf/core/meta/decorators.py) | 54 |  |  |
| [core/model_registry.py](../../../skyulf-core/skyulf/core/model_registry.py) | 92 |  |  |
| [core/protocols.py](../../../skyulf-core/skyulf/core/protocols.py) | 77 |  |  |
| [core/schema.py](../../../skyulf-core/skyulf/core/schema.py) | 303 |  |  |
| [core/serialization.py](../../../skyulf-core/skyulf/core/serialization.py) | 94 |  | pickle, joblib |
| `core/validation.py` ? | 30 |  |  |
| [core/warnings.py](../../../skyulf-core/skyulf/core/warnings.py) | 52 |  |  |
| [data/__init__.py](../../../skyulf-core/skyulf/data/__init__.py) | 8 |  |  |
| [data/catalog.py](../../../skyulf-core/skyulf/data/catalog.py) | 47 |  |  |
| [data/dataset.py](../../../skyulf-core/skyulf/data/dataset.py) | 74 |  |  |
| [engines/__init__.py](../../../skyulf-core/skyulf/engines/__init__.py) | 26 |  |  |
| [engines/pandas_engine.py](../../../skyulf-core/skyulf/engines/pandas_engine.py) | 148 | pandas | to_pandas, to_numpy |
| [engines/polars_engine.py](../../../skyulf-core/skyulf/engines/polars_engine.py) | 209 | polars | to_pandas, to_numpy |
| [engines/protocol.py](../../../skyulf-core/skyulf/engines/protocol.py) | 136 |  | to_pandas, iloc |
| [engines/registry.py](../../../skyulf-core/skyulf/engines/registry.py) | 170 |  | to_numpy |
| [engines/sklearn_bridge.py](../../../skyulf-core/skyulf/engines/sklearn_bridge.py) | 164 |  | to_numpy |
| [leakage.py](../../../skyulf-core/skyulf/leakage.py) | 326 |  |  |
| [modeling/__init__.py](../../../skyulf-core/skyulf/modeling/__init__.py) | 103 |  | cross_val |
| [modeling/_boosting_progress.py](../../../skyulf-core/skyulf/modeling/_boosting_progress.py) | 107 |  |  |
| `modeling/_class_weights.py` ? | 42 |  |  |
| [modeling/_evaluation/__init__.py](../../../skyulf-core/skyulf/modeling/_evaluation/__init__.py) | 45 |  |  |
| [modeling/_evaluation/classification.py](../../../skyulf-core/skyulf/modeling/_evaluation/classification.py) | 174 |  |  |
| [modeling/_evaluation/clustering.py](../../../skyulf-core/skyulf/modeling/_evaluation/clustering.py) | 309 |  | to_pandas, iter_rows |
| [modeling/_evaluation/common.py](../../../skyulf-core/skyulf/modeling/_evaluation/common.py) | 77 |  |  |
| [modeling/_evaluation/metrics.py](../../../skyulf-core/skyulf/modeling/_evaluation/metrics.py) | 500 |  |  |
| [modeling/_evaluation/regression.py](../../../skyulf-core/skyulf/modeling/_evaluation/regression.py) | 86 |  |  |
| [modeling/_evaluation/schemas.py](../../../skyulf-core/skyulf/modeling/_evaluation/schemas.py) | 88 |  |  |
| [modeling/_evaluation/thresholds.py](../../../skyulf-core/skyulf/modeling/_evaluation/thresholds.py) | 281 |  |  |
| [modeling/_explainability/__init__.py](../../../skyulf-core/skyulf/modeling/_explainability/__init__.py) | 7 |  |  |
| [modeling/_explainability/shap_explanation.py](../../../skyulf-core/skyulf/modeling/_explainability/shap_explanation.py) | 443 |  | to_numpy, iloc |
| `modeling/_lightgbm.py` ? | 33 |  |  |
| [modeling/_sklearn_compat.py](../../../skyulf-core/skyulf/modeling/_sklearn_compat.py) | 54 |  |  |
| [modeling/_tuning/__init__.py](../../../skyulf-core/skyulf/modeling/_tuning/__init__.py) | 15 |  |  |
| [modeling/_tuning/engine.py](../../../skyulf-core/skyulf/modeling/_tuning/engine.py) | 1044 |  | to_pandas, to_numpy, cross_val |
| [modeling/_tuning/fold_pipeline.py](../../../skyulf-core/skyulf/modeling/_tuning/fold_pipeline.py) | 234 |  | to_pandas, to_numpy, iloc |
| `modeling/_tuning/fold_scoring.py` ? | 56 |  | to_numpy |
| [modeling/_tuning/grid_random.py](../../../skyulf-core/skyulf/modeling/_tuning/grid_random.py) | 315 |  | to_numpy, iloc |
| [modeling/_tuning/metrics.py](../../../skyulf-core/skyulf/modeling/_tuning/metrics.py) | 243 |  |  |
| [modeling/_tuning/params.py](../../../skyulf-core/skyulf/modeling/_tuning/params.py) | 148 |  |  |
| [modeling/_tuning/refit.py](../../../skyulf-core/skyulf/modeling/_tuning/refit.py) | 234 |  |  |
| [modeling/_tuning/reporter.py](../../../skyulf-core/skyulf/modeling/_tuning/reporter.py) | 97 |  |  |
| [modeling/_tuning/schemas.py](../../../skyulf-core/skyulf/modeling/_tuning/schemas.py) | 65 |  | cross_val |
| [modeling/_tuning/splitters.py](../../../skyulf-core/skyulf/modeling/_tuning/splitters.py) | 175 |  | to_pandas |
| [modeling/_tuning/strategies/__init__.py](../../../skyulf-core/skyulf/modeling/_tuning/strategies/__init__.py) | 7 |  |  |
| [modeling/_tuning/strategies/halving.py](../../../skyulf-core/skyulf/modeling/_tuning/strategies/halving.py) | 93 |  |  |
| [modeling/_tuning/strategies/optuna.py](../../../skyulf-core/skyulf/modeling/_tuning/strategies/optuna.py) | 323 |  |  |
| `modeling/_tuning/strategies/optuna_folds.py` ? | 187 |  | to_numpy |
| `modeling/_tuning/strategies/optuna_search.py` ? | 211 |  |  |
| [modeling/_tuning/strategies/runner.py](../../../skyulf-core/skyulf/modeling/_tuning/strategies/runner.py) | 207 |  | joblib |
| [modeling/base.py](../../../skyulf-core/skyulf/modeling/base.py) | 650 |  | to_numpy, pickle, cross_val |
| [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | 937 | logistic_regression, calibrated_classifier, random_forest_classifier, svc, k_neighbors_classifier, decision_tree_classifier, gradient_boosting_classifier, adaboost_classifier, extra_trees_classifier, hist_gradient_boosting_classifier, gaussian_nb, sgd_classifier, xgboost_classifier, lgbm_classifier |  |
| [modeling/clustering.py](../../../skyulf-core/skyulf/modeling/clustering.py) | 300 | kmeans, minibatch_kmeans, gaussian_mixture, birch | to_pandas, pickle |
| [modeling/cross_validation.py](../../../skyulf-core/skyulf/modeling/cross_validation.py) | 694 |  | to_numpy, iloc, cross_val |
| [modeling/ensemble.py](../../../skyulf-core/skyulf/modeling/ensemble.py) | 678 | voting_classifier, stacking_classifier, voting_regressor, stacking_regressor |  |
| [modeling/fold_preprocessing.py](../../../skyulf-core/skyulf/modeling/fold_preprocessing.py) | 33 |  | cross_val |
| [modeling/hyperparameters/__init__.py](../../../skyulf-core/skyulf/modeling/hyperparameters/__init__.py) | 106 |  |  |
| [modeling/hyperparameters/_bayes.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_bayes.py) | 68 |  |  |
| [modeling/hyperparameters/_calibration.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_calibration.py) | 55 |  |  |
| [modeling/hyperparameters/_clustering.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_clustering.py) | 121 |  |  |
| [modeling/hyperparameters/_ensemble.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_ensemble.py) | 205 |  |  |
| [modeling/hyperparameters/_field.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_field.py) | 71 |  |  |
| [modeling/hyperparameters/_linear.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_linear.py) | 241 |  |  |
| [modeling/hyperparameters/_neighbors.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_neighbors.py) | 40 |  |  |
| [modeling/hyperparameters/_registry.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_registry.py) | 665 |  |  |
| [modeling/hyperparameters/_svm.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_svm.py) | 39 |  |  |
| [modeling/hyperparameters/_tree.py](../../../skyulf-core/skyulf/modeling/hyperparameters/_tree.py) | 656 |  |  |
| [modeling/naive_bayes.py](../../../skyulf-core/skyulf/modeling/naive_bayes.py) | 99 | multinomial_nb, bernoulli_nb |  |
| `modeling/pruning.py` ? | 152 |  |  |
| [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | 658 | linear_regression, ridge_regression, random_forest_regressor, lasso_regression, elasticnet_regression, svr, k_neighbors_regressor, decision_tree_regressor, gradient_boosting_regressor, adaboost_regressor, extra_trees_regressor, hist_gradient_boosting_regressor, lgbm_regressor, xgboost_regressor | joblib |
| [modeling/sklearn_wrapper.py](../../../skyulf-core/skyulf/modeling/sklearn_wrapper.py) | 268 |  | to_pandas, pickle |
| [pipeline/__init__.py](../../../skyulf-core/skyulf/pipeline/__init__.py) | 17 |  |  |
| [pipeline/_pipeline.py](../../../skyulf-core/skyulf/pipeline/_pipeline.py) | 786 |  | to_pandas, pickle |
| [pipeline/diagram.py](../../../skyulf-core/skyulf/pipeline/diagram.py) | 134 |  |  |
| [pipeline/seal.py](../../../skyulf-core/skyulf/pipeline/seal.py) | 250 |  | pickle |
| [preprocessing/__init__.py](../../../skyulf-core/skyulf/preprocessing/__init__.py) | 261 |  |  |
| [preprocessing/_artifacts.py](../../../skyulf-core/skyulf/preprocessing/_artifacts.py) | 121 |  |  |
| [preprocessing/_category_keys.py](../../../skyulf-core/skyulf/preprocessing/_category_keys.py) | 75 |  |  |
| [preprocessing/_helpers.py](../../../skyulf-core/skyulf/preprocessing/_helpers.py) | 319 |  | to_pandas, to_numpy, iloc |
| `preprocessing/_output_names.py` ? | 29 |  |  |
| [preprocessing/_schema.py](../../../skyulf-core/skyulf/preprocessing/_schema.py) | 11 |  |  |
| [preprocessing/base.py](../../../skyulf-core/skyulf/preprocessing/base.py) | 341 |  | tracemalloc |
| [preprocessing/bucketing.py](../../../skyulf-core/skyulf/preprocessing/bucketing.py) | 699 | GeneralBinning, CustomBinning, KBinsDiscretizer | to_pandas |
| [preprocessing/casting.py](../../../skyulf-core/skyulf/preprocessing/casting.py) | 535 | Casting | to_pandas, to_numpy |
| [preprocessing/cleaning/__init__.py](../../../skyulf-core/skyulf/preprocessing/cleaning/__init__.py) | 35 |  |  |
| [preprocessing/cleaning/_common.py](../../../skyulf-core/skyulf/preprocessing/cleaning/_common.py) | 48 |  |  |
| [preprocessing/cleaning/alias.py](../../../skyulf-core/skyulf/preprocessing/cleaning/alias.py) | 216 | AliasReplacement |  |
| [preprocessing/cleaning/invalid_value.py](../../../skyulf-core/skyulf/preprocessing/cleaning/invalid_value.py) | 303 | InvalidValueReplacement |  |
| [preprocessing/cleaning/text.py](../../../skyulf-core/skyulf/preprocessing/cleaning/text.py) | 250 | TextCleaning |  |
| [preprocessing/cleaning/value_replacement.py](../../../skyulf-core/skyulf/preprocessing/cleaning/value_replacement.py) | 254 | ValueReplacement |  |
| [preprocessing/dispatcher.py](../../../skyulf-core/skyulf/preprocessing/dispatcher.py) | 273 |  | to_pandas |
| [preprocessing/drop_and_missing/__init__.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/__init__.py) | 29 |  |  |
| [preprocessing/drop_and_missing/_common.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/_common.py) | 59 |  | to_numpy, iloc |
| [preprocessing/drop_and_missing/deduplicate.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/deduplicate.py) | 119 | Deduplicate | to_numpy, iloc |
| [preprocessing/drop_and_missing/drop_columns.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_columns.py) | 150 | DropMissingColumns |  |
| [preprocessing/drop_and_missing/drop_rows.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_rows.py) | 170 | DropMissingRows | to_numpy, iloc |
| [preprocessing/drop_and_missing/missing_indicator.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/missing_indicator.py) | 177 | MissingIndicator |  |
| [preprocessing/encoding/__init__.py](../../../skyulf-core/skyulf/preprocessing/encoding/__init__.py) | 44 |  |  |
| [preprocessing/encoding/_common.py](../../../skyulf-core/skyulf/preprocessing/encoding/_common.py) | 148 |  |  |
| [preprocessing/encoding/dummy.py](../../../skyulf-core/skyulf/preprocessing/encoding/dummy.py) | 297 | DummyEncoder | to_list |
| [preprocessing/encoding/hash.py](../../../skyulf-core/skyulf/preprocessing/encoding/hash.py) | 186 | HashEncoder | to_list |
| [preprocessing/encoding/label.py](../../../skyulf-core/skyulf/preprocessing/encoding/label.py) | 350 | LabelEncoder | to_numpy |
| [preprocessing/encoding/one_hot.py](../../../skyulf-core/skyulf/preprocessing/encoding/one_hot.py) | 271 | OneHotEncoder | to_numpy, toarray |
| [preprocessing/encoding/ordinal.py](../../../skyulf-core/skyulf/preprocessing/encoding/ordinal.py) | 384 | OrdinalEncoder | to_numpy |
| [preprocessing/encoding/target.py](../../../skyulf-core/skyulf/preprocessing/encoding/target.py) | 416 | TargetEncoder | to_pandas, to_numpy |
| [preprocessing/encoding/woe.py](../../../skyulf-core/skyulf/preprocessing/encoding/woe.py) | 385 | WOEEncoder | to_pandas, to_numpy, to_list, iloc |
| [preprocessing/feature_generation/__init__.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/__init__.py) | 31 |  |  |
| [preprocessing/feature_generation/_common.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/_common.py) | 197 |  | to_numpy, iloc |
| [preprocessing/feature_generation/_pandas_ops.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/_pandas_ops.py) | 256 |  |  |
| [preprocessing/feature_generation/_polars_ops.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/_polars_ops.py) | 315 |  |  |
| [preprocessing/feature_generation/generation.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py) | 153 | FeatureGeneration, FeatureMath, FeatureGenerationNode | to_pandas |
| [preprocessing/feature_generation/interaction.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/interaction.py) | 215 | FeatureInteraction | to_pandas |
| [preprocessing/feature_generation/polynomial.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py) | 153 | PolynomialFeatures, PolynomialFeaturesNode | to_numpy |
| [preprocessing/feature_selection/__init__.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/__init__.py) | 25 |  |  |
| [preprocessing/feature_selection/_common.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/_common.py) | 365 |  | to_pandas, to_numpy |
| [preprocessing/feature_selection/correlation.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/correlation.py) | 213 | CorrelationThreshold | to_pandas |
| [preprocessing/feature_selection/facade.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/facade.py) | 97 | feature_selection |  |
| [preprocessing/feature_selection/model_based.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/model_based.py) | 117 | ModelBasedSelection | to_pandas |
| [preprocessing/feature_selection/univariate.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/univariate.py) | 117 | UnivariateSelection | to_pandas |
| [preprocessing/feature_selection/variance.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/variance.py) | 96 | VarianceThreshold | to_numpy |
| [preprocessing/fold_adapter.py](../../../skyulf-core/skyulf/preprocessing/fold_adapter.py) | 377 |  | to_pandas |
| [preprocessing/geo/__init__.py](../../../skyulf-core/skyulf/preprocessing/geo/__init__.py) | 20 |  |  |
| [preprocessing/geo/distance.py](../../../skyulf-core/skyulf/preprocessing/geo/distance.py) | 226 | GeoDistance | to_pandas |
| [preprocessing/geo/h3_index.py](../../../skyulf-core/skyulf/preprocessing/geo/h3_index.py) | 165 | H3Index | to_pandas, to_numpy |
| [preprocessing/imputation/__init__.py](../../../skyulf-core/skyulf/preprocessing/imputation/__init__.py) | 25 |  |  |
| [preprocessing/imputation/_common.py](../../../skyulf-core/skyulf/preprocessing/imputation/_common.py) | 159 |  | to_numpy |
| [preprocessing/imputation/iterative.py](../../../skyulf-core/skyulf/preprocessing/imputation/iterative.py) | 103 | IterativeImputer | to_numpy |
| [preprocessing/imputation/knn.py](../../../skyulf-core/skyulf/preprocessing/imputation/knn.py) | 97 | KNNImputer | to_numpy |
| [preprocessing/imputation/simple.py](../../../skyulf-core/skyulf/preprocessing/imputation/simple.py) | 236 | SimpleImputer |  |
| [preprocessing/inspection.py](../../../skyulf-core/skyulf/preprocessing/inspection.py) | 187 | DatasetProfile, DataSnapshot |  |
| [preprocessing/outliers/__init__.py](../../../skyulf-core/skyulf/preprocessing/outliers/__init__.py) | 32 |  |  |
| [preprocessing/outliers/_common.py](../../../skyulf-core/skyulf/preprocessing/outliers/_common.py) | 39 |  | to_numpy, iloc |
| [preprocessing/outliers/elliptic.py](../../../skyulf-core/skyulf/preprocessing/outliers/elliptic.py) | 179 | EllipticEnvelope | to_pandas, to_numpy, iloc |
| [preprocessing/outliers/iqr.py](../../../skyulf-core/skyulf/preprocessing/outliers/iqr.py) | 124 | IQR | to_pandas, to_numpy |
| [preprocessing/outliers/manual_bounds.py](../../../skyulf-core/skyulf/preprocessing/outliers/manual_bounds.py) | 111 | ManualBounds |  |
| [preprocessing/outliers/winsorize.py](../../../skyulf-core/skyulf/preprocessing/outliers/winsorize.py) | 129 | Winsorize | to_pandas |
| [preprocessing/outliers/zscore.py](../../../skyulf-core/skyulf/preprocessing/outliers/zscore.py) | 130 | ZScore | to_pandas |
| [preprocessing/pipeline.py](../../../skyulf-core/skyulf/preprocessing/pipeline.py) | 724 |  | to_pandas |
| [preprocessing/resampling.py](../../../skyulf-core/skyulf/preprocessing/resampling.py) | 455 | Oversampling, Undersampling | to_pandas |
| [preprocessing/scaling/__init__.py](../../../skyulf-core/skyulf/preprocessing/scaling/__init__.py) | 29 |  |  |
| [preprocessing/scaling/_common.py](../../../skyulf-core/skyulf/preprocessing/scaling/_common.py) | 54 |  |  |
| [preprocessing/scaling/maxabs.py](../../../skyulf-core/skyulf/preprocessing/scaling/maxabs.py) | 119 | MaxAbsScaler | to_numpy |
| [preprocessing/scaling/minmax.py](../../../skyulf-core/skyulf/preprocessing/scaling/minmax.py) | 125 | MinMaxScaler | to_numpy |
| [preprocessing/scaling/robust.py](../../../skyulf-core/skyulf/preprocessing/scaling/robust.py) | 158 | RobustScaler | to_numpy |
| [preprocessing/scaling/standard.py](../../../skyulf-core/skyulf/preprocessing/scaling/standard.py) | 177 | StandardScaler | to_numpy |
| [preprocessing/split.py](../../../skyulf-core/skyulf/preprocessing/split.py) | 593 | Split, TrainTestSplitter, feature_target_split | to_pandas, to_numpy |
| [preprocessing/time_series/__init__.py](../../../skyulf-core/skyulf/preprocessing/time_series/__init__.py) | 23 |  |  |
| [preprocessing/time_series/_common.py](../../../skyulf-core/skyulf/preprocessing/time_series/_common.py) | 112 |  | to_numpy, iloc |
| [preprocessing/time_series/date_features.py](../../../skyulf-core/skyulf/preprocessing/time_series/date_features.py) | 202 | DateFeatures |  |
| [preprocessing/time_series/lag.py](../../../skyulf-core/skyulf/preprocessing/time_series/lag.py) | 148 | LagFeatures | to_numpy, iloc |
| [preprocessing/time_series/rolling.py](../../../skyulf-core/skyulf/preprocessing/time_series/rolling.py) | 194 | RollingAggregate |  |
| [preprocessing/transformations/__init__.py](../../../skyulf-core/skyulf/preprocessing/transformations/__init__.py) | 19 |  |  |
| [preprocessing/transformations/_ops.py](../../../skyulf-core/skyulf/preprocessing/transformations/_ops.py) | 93 |  |  |
| [preprocessing/transformations/_power_common.py](../../../skyulf-core/skyulf/preprocessing/transformations/_power_common.py) | 69 |  |  |
| [preprocessing/transformations/general.py](../../../skyulf-core/skyulf/preprocessing/transformations/general.py) | 269 | GeneralTransformation | to_pandas, to_numpy |
| [preprocessing/transformations/power.py](../../../skyulf-core/skyulf/preprocessing/transformations/power.py) | 198 | PowerTransformer | to_pandas, to_numpy |
| [preprocessing/transformations/simple.py](../../../skyulf-core/skyulf/preprocessing/transformations/simple.py) | 116 | SimpleTransformation |  |
| [preprocessing/vectorization/__init__.py](../../../skyulf-core/skyulf/preprocessing/vectorization/__init__.py) | 31 |  |  |
| [preprocessing/vectorization/_common.py](../../../skyulf-core/skyulf/preprocessing/vectorization/_common.py) | 327 |  | to_pandas, toarray, to_list |
| [preprocessing/vectorization/count_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/count_vectorizer.py) | 175 | count_vectorizer | toarray |
| [preprocessing/vectorization/hashing_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/hashing_vectorizer.py) | 160 | hashing_vectorizer |  |
| [preprocessing/vectorization/sentence_embedder.py](../../../skyulf-core/skyulf/preprocessing/vectorization/sentence_embedder.py) | 252 | sentence_embedder | to_list |
| [preprocessing/vectorization/tfidf_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/tfidf_vectorizer.py) | 169 | tfidf_vectorizer |  |
| [preprocessing/vectorization/tokenizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/tokenizer.py) | 204 | tokenizer | to_list |
| [profiling/__init__.py](../../../skyulf-core/skyulf/profiling/__init__.py) | 43 |  |  |
| [profiling/_analyzer/__init__.py](../../../skyulf-core/skyulf/profiling/_analyzer/__init__.py) | 36 |  |  |
| [profiling/_analyzer/_utils.py](../../../skyulf-core/skyulf/profiling/_analyzer/_utils.py) | 114 |  | collect( |
| [profiling/_analyzer/categorical.py](../../../skyulf-core/skyulf/profiling/_analyzer/categorical.py) | 36 |  |  |
| [profiling/_analyzer/causal.py](../../../skyulf-core/skyulf/profiling/_analyzer/causal.py) | 157 |  | to_numpy |
| [profiling/_analyzer/column.py](../../../skyulf-core/skyulf/profiling/_analyzer/column.py) | 356 |  | to_numpy |
| [profiling/_analyzer/dates.py](../../../skyulf-core/skyulf/profiling/_analyzer/dates.py) | 181 |  |  |
| [profiling/_analyzer/decomposition.py](../../../skyulf-core/skyulf/profiling/_analyzer/decomposition.py) | 230 |  | iter_rows |
| [profiling/_analyzer/geo.py](../../../skyulf-core/skyulf/profiling/_analyzer/geo.py) | 162 |  | collect( |
| [profiling/_analyzer/multivariate.py](../../../skyulf-core/skyulf/profiling/_analyzer/multivariate.py) | 418 |  | to_numpy, to_list |
| [profiling/_analyzer/numeric.py](../../../skyulf-core/skyulf/profiling/_analyzer/numeric.py) | 141 |  | to_numpy |
| [profiling/_analyzer/recommendations.py](../../../skyulf-core/skyulf/profiling/_analyzer/recommendations.py) | 283 |  | to_list, collect( |
| [profiling/_analyzer/rules.py](../../../skyulf-core/skyulf/profiling/_analyzer/rules.py) | 372 |  | to_numpy, to_list |
| [profiling/_analyzer/target.py](../../../skyulf-core/skyulf/profiling/_analyzer/target.py) | 231 |  | iter_rows, collect( |
| [profiling/_analyzer/temporal.py](../../../skyulf-core/skyulf/profiling/_analyzer/temporal.py) | 286 |  | to_numpy, iter_rows, collect( |
| [profiling/_analyzer/text.py](../../../skyulf-core/skyulf/profiling/_analyzer/text.py) | 136 |  | to_list, iter_rows |
| [profiling/analyzer.py](../../../skyulf-core/skyulf/profiling/analyzer.py) | 680 |  | collect( |
| [profiling/correlations.py](../../../skyulf-core/skyulf/profiling/correlations.py) | 184 |  | collect( |
| [profiling/distributions.py](../../../skyulf-core/skyulf/profiling/distributions.py) | 87 |  | iter_rows, collect( |
| [profiling/drift.py](../../../skyulf-core/skyulf/profiling/drift.py) | 630 |  | to_numpy, to_list |
| [profiling/expect.py](../../../skyulf-core/skyulf/profiling/expect.py) | 237 |  | to_pandas |
| [profiling/schemas.py](../../../skyulf-core/skyulf/profiling/schemas.py) | 466 |  |  |
| [profiling/visualizer.py](../../../skyulf-core/skyulf/profiling/visualizer.py) | 867 |  | to_pandas |
| [registry.py](../../../skyulf-core/skyulf/registry.py) | 136 |  |  |
| [types.py](../../../skyulf-core/skyulf/types.py) | 52 |  |  |
| [utils.py](../../../skyulf-core/skyulf/utils.py) | 425 |  | to_pandas |

## Runtime node kayitlari

Kurulu optional paketlere gore kayit sayisi degisebilir. Aliaslar dahildir.
Asagida bugunku ortamda gozlenen kayitlar vardir; Spark destek listesi degildir.

| Kayit | Kategori | Kaynak | Veri ogreniyor |
| --- | --- | --- | --- |
| `AliasReplacement` | Cleaning | [preprocessing/cleaning/alias.py](../../../skyulf-core/skyulf/preprocessing/cleaning/alias.py) | False |
| `Casting` | Data Operations | [preprocessing/casting.py](../../../skyulf-core/skyulf/preprocessing/casting.py) | True |
| `CorrelationThreshold` | Feature Selection | [preprocessing/feature_selection/correlation.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/correlation.py) | True |
| `CustomBinning` | Preprocessing | [preprocessing/bucketing.py](../../../skyulf-core/skyulf/preprocessing/bucketing.py) | True |
| `DataSnapshot` | Inspection | [preprocessing/inspection.py](../../../skyulf-core/skyulf/preprocessing/inspection.py) | False |
| `DatasetProfile` | Inspection | [preprocessing/inspection.py](../../../skyulf-core/skyulf/preprocessing/inspection.py) | False |
| `DateFeatures` | Preprocessing | [preprocessing/time_series/date_features.py](../../../skyulf-core/skyulf/preprocessing/time_series/date_features.py) | False |
| `Deduplicate` | Data Operations | [preprocessing/drop_and_missing/deduplicate.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/deduplicate.py) | True |
| `DropMissingColumns` | Cleaning | [preprocessing/drop_and_missing/drop_columns.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_columns.py) | True |
| `DropMissingRows` | Cleaning | [preprocessing/drop_and_missing/drop_rows.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/drop_rows.py) | False |
| `DummyEncoder` | Preprocessing | [preprocessing/encoding/dummy.py](../../../skyulf-core/skyulf/preprocessing/encoding/dummy.py) | True |
| `EllipticEnvelope` | Preprocessing | [preprocessing/outliers/elliptic.py](../../../skyulf-core/skyulf/preprocessing/outliers/elliptic.py) | True |
| `FeatureGeneration` | Feature Engineering | [preprocessing/feature_generation/generation.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py) | True |
| `FeatureGenerationNode` | Feature Engineering | [preprocessing/feature_generation/generation.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py) | True |
| `FeatureInteraction` | Feature Engineering | [preprocessing/feature_generation/interaction.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/interaction.py) | False |
| `FeatureMath` | Feature Engineering | [preprocessing/feature_generation/generation.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/generation.py) | True |
| `GeneralBinning` | Preprocessing | [preprocessing/bucketing.py](../../../skyulf-core/skyulf/preprocessing/bucketing.py) | True |
| `GeneralTransformation` | Preprocessing | [preprocessing/transformations/general.py](../../../skyulf-core/skyulf/preprocessing/transformations/general.py) | True |
| `GeoDistance` | Feature Engineering | [preprocessing/geo/distance.py](../../../skyulf-core/skyulf/preprocessing/geo/distance.py) | False |
| `H3Index` | Feature Engineering | [preprocessing/geo/h3_index.py](../../../skyulf-core/skyulf/preprocessing/geo/h3_index.py) | False |
| `HashEncoder` | Preprocessing | [preprocessing/encoding/hash.py](../../../skyulf-core/skyulf/preprocessing/encoding/hash.py) | True |
| `IQR` | Preprocessing | [preprocessing/outliers/iqr.py](../../../skyulf-core/skyulf/preprocessing/outliers/iqr.py) | True |
| `InvalidValueReplacement` | Cleaning | [preprocessing/cleaning/invalid_value.py](../../../skyulf-core/skyulf/preprocessing/cleaning/invalid_value.py) | False |
| `IterativeImputer` | Preprocessing | [preprocessing/imputation/iterative.py](../../../skyulf-core/skyulf/preprocessing/imputation/iterative.py) | True |
| `KBinsDiscretizer` | Preprocessing | [preprocessing/bucketing.py](../../../skyulf-core/skyulf/preprocessing/bucketing.py) | True |
| `KNNImputer` | Preprocessing | [preprocessing/imputation/knn.py](../../../skyulf-core/skyulf/preprocessing/imputation/knn.py) | True |
| `LabelEncoder` | Preprocessing | [preprocessing/encoding/label.py](../../../skyulf-core/skyulf/preprocessing/encoding/label.py) | True |
| `LagFeatures` | Preprocessing | [preprocessing/time_series/lag.py](../../../skyulf-core/skyulf/preprocessing/time_series/lag.py) | False |
| `ManualBounds` | Preprocessing | [preprocessing/outliers/manual_bounds.py](../../../skyulf-core/skyulf/preprocessing/outliers/manual_bounds.py) | False |
| `MaxAbsScaler` | Preprocessing | [preprocessing/scaling/maxabs.py](../../../skyulf-core/skyulf/preprocessing/scaling/maxabs.py) | True |
| `MinMaxScaler` | Preprocessing | [preprocessing/scaling/minmax.py](../../../skyulf-core/skyulf/preprocessing/scaling/minmax.py) | True |
| `MissingIndicator` | Feature Engineering | [preprocessing/drop_and_missing/missing_indicator.py](../../../skyulf-core/skyulf/preprocessing/drop_and_missing/missing_indicator.py) | True |
| `ModelBasedSelection` | Feature Selection | [preprocessing/feature_selection/model_based.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/model_based.py) | True |
| `OneHotEncoder` | Preprocessing | [preprocessing/encoding/one_hot.py](../../../skyulf-core/skyulf/preprocessing/encoding/one_hot.py) | True |
| `OrdinalEncoder` | Preprocessing | [preprocessing/encoding/ordinal.py](../../../skyulf-core/skyulf/preprocessing/encoding/ordinal.py) | True |
| `Oversampling` | Preprocessing | [preprocessing/resampling.py](../../../skyulf-core/skyulf/preprocessing/resampling.py) | True |
| `PolynomialFeatures` | Feature Engineering | [preprocessing/feature_generation/polynomial.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py) | True |
| `PolynomialFeaturesNode` | Feature Engineering | [preprocessing/feature_generation/polynomial.py](../../../skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py) | True |
| `PowerTransformer` | Preprocessing | [preprocessing/transformations/power.py](../../../skyulf-core/skyulf/preprocessing/transformations/power.py) | True |
| `RobustScaler` | Preprocessing | [preprocessing/scaling/robust.py](../../../skyulf-core/skyulf/preprocessing/scaling/robust.py) | True |
| `RollingAggregate` | Preprocessing | [preprocessing/time_series/rolling.py](../../../skyulf-core/skyulf/preprocessing/time_series/rolling.py) | False |
| `SimpleImputer` | Preprocessing | [preprocessing/imputation/simple.py](../../../skyulf-core/skyulf/preprocessing/imputation/simple.py) | True |
| `SimpleTransformation` | Preprocessing | [preprocessing/transformations/simple.py](../../../skyulf-core/skyulf/preprocessing/transformations/simple.py) | False |
| `Split` | Data Operations | [preprocessing/split.py](../../../skyulf-core/skyulf/preprocessing/split.py) | False |
| `StandardScaler` | Preprocessing | [preprocessing/scaling/standard.py](../../../skyulf-core/skyulf/preprocessing/scaling/standard.py) | True |
| `TargetEncoder` | Preprocessing | [preprocessing/encoding/target.py](../../../skyulf-core/skyulf/preprocessing/encoding/target.py) | True |
| `TextCleaning` | Cleaning | [preprocessing/cleaning/text.py](../../../skyulf-core/skyulf/preprocessing/cleaning/text.py) | False |
| `TrainTestSplitter` | Data Operations | [preprocessing/split.py](../../../skyulf-core/skyulf/preprocessing/split.py) | False |
| `Undersampling` | Preprocessing | [preprocessing/resampling.py](../../../skyulf-core/skyulf/preprocessing/resampling.py) | True |
| `UnivariateSelection` | Feature Selection | [preprocessing/feature_selection/univariate.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/univariate.py) | True |
| `ValueReplacement` | Cleaning | [preprocessing/cleaning/value_replacement.py](../../../skyulf-core/skyulf/preprocessing/cleaning/value_replacement.py) | False |
| `VarianceThreshold` | Feature Selection | [preprocessing/feature_selection/variance.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/variance.py) | True |
| `WOEEncoder` | Preprocessing | [preprocessing/encoding/woe.py](../../../skyulf-core/skyulf/preprocessing/encoding/woe.py) | True |
| `Winsorize` | Preprocessing | [preprocessing/outliers/winsorize.py](../../../skyulf-core/skyulf/preprocessing/outliers/winsorize.py) | True |
| `ZScore` | Preprocessing | [preprocessing/outliers/zscore.py](../../../skyulf-core/skyulf/preprocessing/outliers/zscore.py) | True |
| `adaboost_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `adaboost_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `bernoulli_nb` | Modeling | [modeling/naive_bayes.py](../../../skyulf-core/skyulf/modeling/naive_bayes.py) | True |
| `birch` | Modeling | [modeling/clustering.py](../../../skyulf-core/skyulf/modeling/clustering.py) | True |
| `calibrated_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `count_vectorizer` | Text | [preprocessing/vectorization/count_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/count_vectorizer.py) | True |
| `decision_tree_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `decision_tree_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `elasticnet_regression` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `extra_trees_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `extra_trees_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `feature_selection` | Feature Selection | [preprocessing/feature_selection/facade.py](../../../skyulf-core/skyulf/preprocessing/feature_selection/facade.py) | True |
| `feature_target_split` | Data Operations | [preprocessing/split.py](../../../skyulf-core/skyulf/preprocessing/split.py) | False |
| `gaussian_mixture` | Modeling | [modeling/clustering.py](../../../skyulf-core/skyulf/modeling/clustering.py) | True |
| `gaussian_nb` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `gradient_boosting_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `gradient_boosting_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `hashing_vectorizer` | Text | [preprocessing/vectorization/hashing_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/hashing_vectorizer.py) | False |
| `hist_gradient_boosting_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `hist_gradient_boosting_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `k_neighbors_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `k_neighbors_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `kmeans` | Modeling | [modeling/clustering.py](../../../skyulf-core/skyulf/modeling/clustering.py) | True |
| `lasso_regression` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `lgbm_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `lgbm_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `linear_regression` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `logistic_regression` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `minibatch_kmeans` | Modeling | [modeling/clustering.py](../../../skyulf-core/skyulf/modeling/clustering.py) | True |
| `multinomial_nb` | Modeling | [modeling/naive_bayes.py](../../../skyulf-core/skyulf/modeling/naive_bayes.py) | True |
| `random_forest_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `random_forest_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `ridge_regression` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `sentence_embedder` | Text | [preprocessing/vectorization/sentence_embedder.py](../../../skyulf-core/skyulf/preprocessing/vectorization/sentence_embedder.py) | False |
| `sgd_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `stacking_classifier` | Ensemble | [modeling/ensemble.py](../../../skyulf-core/skyulf/modeling/ensemble.py) | True |
| `stacking_regressor` | Ensemble | [modeling/ensemble.py](../../../skyulf-core/skyulf/modeling/ensemble.py) | True |
| `svc` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `svr` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |
| `tfidf_vectorizer` | Text | [preprocessing/vectorization/tfidf_vectorizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/tfidf_vectorizer.py) | True |
| `tokenizer` | Text | [preprocessing/vectorization/tokenizer.py](../../../skyulf-core/skyulf/preprocessing/vectorization/tokenizer.py) | False |
| `voting_classifier` | Ensemble | [modeling/ensemble.py](../../../skyulf-core/skyulf/modeling/ensemble.py) | True |
| `voting_regressor` | Ensemble | [modeling/ensemble.py](../../../skyulf-core/skyulf/modeling/ensemble.py) | True |
| `xgboost_classifier` | Modeling | [modeling/classification.py](../../../skyulf-core/skyulf/modeling/classification.py) | True |
| `xgboost_regressor` | Modeling | [modeling/regression.py](../../../skyulf-core/skyulf/modeling/regression.py) | True |

## Test dosyasi envanteri

269 test dosyasi. Tum core pytest sonucu ana rapordadir.

- `tests/integration/test_alias_nontext_contract.py` ?
- `tests/integration/test_arithmetic_missing_values.py` ?
- `tests/integration/test_binning_contract_regressions.py` ?
- `tests/integration/test_binning_missing_labels.py` ?
- [tests/integration/test_bucketing.py](../../../skyulf-core/tests/integration/test_bucketing.py)
- [tests/integration/test_casting.py](../../../skyulf-core/tests/integration/test_casting.py)
- `tests/integration/test_casting_text_inputs.py` ?
- [tests/integration/test_classification_gaps.py](../../../skyulf-core/tests/integration/test_classification_gaps.py)
- [tests/integration/test_cleaning_alias.py](../../../skyulf-core/tests/integration/test_cleaning_alias.py)
- [tests/integration/test_cleaning_invalid_value.py](../../../skyulf-core/tests/integration/test_cleaning_invalid_value.py)
- [tests/integration/test_cleaning_text.py](../../../skyulf-core/tests/integration/test_cleaning_text.py)
- [tests/integration/test_clustering.py](../../../skyulf-core/tests/integration/test_clustering.py)
- [tests/integration/test_core_pipeline_tuning_leakage.py](../../../skyulf-core/tests/integration/test_core_pipeline_tuning_leakage.py)
- [tests/integration/test_cross_validation.py](../../../skyulf-core/tests/integration/test_cross_validation.py)
- `tests/integration/test_cross_validation_date_objects.py` ?
- [tests/integration/test_cross_validation_time_sort_integrity.py](../../../skyulf-core/tests/integration/test_cross_validation_time_sort_integrity.py)
- [tests/integration/test_cv_per_fold_refit.py](../../../skyulf-core/tests/integration/test_cv_per_fold_refit.py)
- `tests/integration/test_date_features_timezone.py` ?
- `tests/integration/test_decimal_numeric.py` ?
- [tests/integration/test_drop_and_missing_gaps.py](../../../skyulf-core/tests/integration/test_drop_and_missing_gaps.py)
- `tests/integration/test_drop_missing_rows_policy.py` ?
- [tests/integration/test_drop_rows.py](../../../skyulf-core/tests/integration/test_drop_rows.py)
- [tests/integration/test_encoding_dummy.py](../../../skyulf-core/tests/integration/test_encoding_dummy.py)
- [tests/integration/test_encoding_hash.py](../../../skyulf-core/tests/integration/test_encoding_hash.py)
- [tests/integration/test_encoding_label.py](../../../skyulf-core/tests/integration/test_encoding_label.py)
- [tests/integration/test_encoding_one_hot.py](../../../skyulf-core/tests/integration/test_encoding_one_hot.py)
- [tests/integration/test_encoding_ordinal.py](../../../skyulf-core/tests/integration/test_encoding_ordinal.py)
- [tests/integration/test_encoding_target.py](../../../skyulf-core/tests/integration/test_encoding_target.py)
- [tests/integration/test_encoding_woe.py](../../../skyulf-core/tests/integration/test_encoding_woe.py)
- [tests/integration/test_ensemble.py](../../../skyulf-core/tests/integration/test_ensemble.py)
- `tests/integration/test_ensemble_calibrated_tuning_refit.py` ?
- [tests/integration/test_ensemble_config_immutability.py](../../../skyulf-core/tests/integration/test_ensemble_config_immutability.py)
- `tests/integration/test_enum_text_cleaning.py` ?
- `tests/integration/test_evaluation_class_axis.py` ?
- [tests/integration/test_evaluation_classification.py](../../../skyulf-core/tests/integration/test_evaluation_classification.py)
- [tests/integration/test_evaluation_regression.py](../../../skyulf-core/tests/integration/test_evaluation_regression.py)
- `tests/integration/test_extended_pruning_support.py` ?
- `tests/integration/test_feature_generation_datetime_names.py` ?
- [tests/integration/test_feature_generation_full.py](../../../skyulf-core/tests/integration/test_feature_generation_full.py)
- [tests/integration/test_feature_generation_gaps.py](../../../skyulf-core/tests/integration/test_feature_generation_gaps.py)
- [tests/integration/test_feature_generation_group_agg.py](../../../skyulf-core/tests/integration/test_feature_generation_group_agg.py)
- [tests/integration/test_feature_generation_operation_validation.py](../../../skyulf-core/tests/integration/test_feature_generation_operation_validation.py)
- [tests/integration/test_feature_generation_polars_ops.py](../../../skyulf-core/tests/integration/test_feature_generation_polars_ops.py)
- [tests/integration/test_feature_generation_ratio_missing.py](../../../skyulf-core/tests/integration/test_feature_generation_ratio_missing.py)
- [tests/integration/test_feature_generation_ratio_sign.py](../../../skyulf-core/tests/integration/test_feature_generation_ratio_sign.py)
- [tests/integration/test_feature_interaction.py](../../../skyulf-core/tests/integration/test_feature_interaction.py)
- [tests/integration/test_feature_selection_common.py](../../../skyulf-core/tests/integration/test_feature_selection_common.py)
- [tests/integration/test_feature_selection_facade.py](../../../skyulf-core/tests/integration/test_feature_selection_facade.py)
- [tests/integration/test_feature_selection_gaps.py](../../../skyulf-core/tests/integration/test_feature_selection_gaps.py)
- `tests/integration/test_feature_selection_target_dtypes.py` ?
- [tests/integration/test_geo_nodes.py](../../../skyulf-core/tests/integration/test_geo_nodes.py)
- `tests/integration/test_group_aggregate_null_keys.py` ?
- [tests/integration/test_imputation_common_knn_iterative_simple.py](../../../skyulf-core/tests/integration/test_imputation_common_knn_iterative_simple.py)
- `tests/integration/test_imputation_text_column_contract.py` ?
- [tests/integration/test_inspection.py](../../../skyulf-core/tests/integration/test_inspection.py)
- `tests/integration/test_invalid_value_integer_precision.py` ?
- `tests/integration/test_invalid_value_numeric_contract.py` ?
- [tests/integration/test_leakage_fixture_contract.py](../../../skyulf-core/tests/integration/test_leakage_fixture_contract.py)
- [tests/integration/test_leakage_operation_contract.py](../../../skyulf-core/tests/integration/test_leakage_operation_contract.py)
- `tests/integration/test_logistic_elasticnet_defaults.py` ?
- `tests/integration/test_logistic_penalty_consistency.py` ?
- `tests/integration/test_model_seal_supported_state.py` ?
- [tests/integration/test_modeling_all.py](../../../skyulf-core/tests/integration/test_modeling_all.py)
- [tests/integration/test_naive_bayes.py](../../../skyulf-core/tests/integration/test_naive_bayes.py)
- `tests/integration/test_optuna_fold_pruning.py` ?
- `tests/integration/test_optuna_incremental_pruning.py` ?
- `tests/integration/test_optuna_native_pruning.py` ?
- `tests/integration/test_optuna_public_pruning.py` ?
- [tests/integration/test_outliers_elliptic_winsorize_zscore_iqr.py](../../../skyulf-core/tests/integration/test_outliers_elliptic_winsorize_zscore_iqr.py)
- [tests/integration/test_outliers_manual_bounds.py](../../../skyulf-core/tests/integration/test_outliers_manual_bounds.py)
- [tests/integration/test_pipeline_integration_modeling.py](../../../skyulf-core/tests/integration/test_pipeline_integration_modeling.py)
- [tests/integration/test_pipeline_integration_multi_model.py](../../../skyulf-core/tests/integration/test_pipeline_integration_multi_model.py)
- [tests/integration/test_pipeline_integration_preprocessing.py](../../../skyulf-core/tests/integration/test_pipeline_integration_preprocessing.py)
- [tests/integration/test_pipeline_integration_tuning.py](../../../skyulf-core/tests/integration/test_pipeline_integration_tuning.py)
- `tests/integration/test_prediction_row_contract.py` ?
- `tests/integration/test_prediction_temporal_order.py` ?
- [tests/integration/test_preprocessing_behavioral_reaudit.py](../../../skyulf-core/tests/integration/test_preprocessing_behavioral_reaudit.py)
- [tests/integration/test_preprocessing_time_series_common.py](../../../skyulf-core/tests/integration/test_preprocessing_time_series_common.py)
- [tests/integration/test_profiling_analyzer.py](../../../skyulf-core/tests/integration/test_profiling_analyzer.py)
- [tests/integration/test_profiling_column_name_collisions.py](../../../skyulf-core/tests/integration/test_profiling_column_name_collisions.py)
- `tests/integration/test_profiling_constant_columns.py` ?
- `tests/integration/test_profiling_correlation_omissions.py` ?
- [tests/integration/test_profiling_correlations_distributions.py](../../../skyulf-core/tests/integration/test_profiling_correlations_distributions.py)
- [tests/integration/test_profiling_dates.py](../../../skyulf-core/tests/integration/test_profiling_dates.py)
- `tests/integration/test_profiling_dates_lossless.py` ?
- [tests/integration/test_profiling_decomposition.py](../../../skyulf-core/tests/integration/test_profiling_decomposition.py)
- `tests/integration/test_profiling_decomposition_missing.py` ?
- `tests/integration/test_profiling_decomposition_names.py` ?
- [tests/integration/test_profiling_drift.py](../../../skyulf-core/tests/integration/test_profiling_drift.py)
- `tests/integration/test_profiling_drift_dtypes.py` ?
- [tests/integration/test_profiling_geo.py](../../../skyulf-core/tests/integration/test_profiling_geo.py)
- [tests/integration/test_profiling_multivariate.py](../../../skyulf-core/tests/integration/test_profiling_multivariate.py)
- [tests/integration/test_profiling_null_enum.py](../../../skyulf-core/tests/integration/test_profiling_null_enum.py)
- [tests/integration/test_profiling_numeric.py](../../../skyulf-core/tests/integration/test_profiling_numeric.py)
- `tests/integration/test_profiling_outlier_population.py` ?
- `tests/integration/test_profiling_recommendation_safety.py` ?
- [tests/integration/test_profiling_recommendations.py](../../../skyulf-core/tests/integration/test_profiling_recommendations.py)
- [tests/integration/test_profiling_repeated_exclusions.py](../../../skyulf-core/tests/integration/test_profiling_repeated_exclusions.py)
- `tests/integration/test_profiling_repeated_filters.py` ?
- [tests/integration/test_profiling_rules.py](../../../skyulf-core/tests/integration/test_profiling_rules.py)
- `tests/integration/test_profiling_seasonality_measure.py` ?
- `tests/integration/test_profiling_string_categories.py` ?
- [tests/integration/test_profiling_target.py](../../../skyulf-core/tests/integration/test_profiling_target.py)
- [tests/integration/test_profiling_target_contract.py](../../../skyulf-core/tests/integration/test_profiling_target_contract.py)
- [tests/integration/test_profiling_target_outlier_regressions.py](../../../skyulf-core/tests/integration/test_profiling_target_outlier_regressions.py)
- `tests/integration/test_profiling_target_recommendation_inference.py` ?
- [tests/integration/test_profiling_temporal.py](../../../skyulf-core/tests/integration/test_profiling_temporal.py)
- `tests/integration/test_profiling_temporal_decomposition.py` ?
- [tests/integration/test_profiling_temporal_missing_values.py](../../../skyulf-core/tests/integration/test_profiling_temporal_missing_values.py)
- [tests/integration/test_profiling_text.py](../../../skyulf-core/tests/integration/test_profiling_text.py)
- `tests/integration/test_profiling_unsupported_dtypes.py` ?
- `tests/integration/test_pruning_capability.py` ?
- `tests/integration/test_release_tuning_roundtrip.py` ?
- [tests/integration/test_repeated_split_boundary.py](../../../skyulf-core/tests/integration/test_repeated_split_boundary.py)
- `tests/integration/test_resampling_advanced_settings.py` ?
- [tests/integration/test_row_filter_column_collisions.py](../../../skyulf-core/tests/integration/test_row_filter_column_collisions.py)
- [tests/integration/test_scaling.py](../../../skyulf-core/tests/integration/test_scaling.py)
- `tests/integration/test_scaling_range_validation.py` ?
- [tests/integration/test_sentence_embedder.py](../../../skyulf-core/tests/integration/test_sentence_embedder.py)
- [tests/integration/test_split.py](../../../skyulf-core/tests/integration/test_split.py)
- `tests/integration/test_split_config_warnings.py` ?
- [tests/integration/test_stateful_estimator_tuple_targets.py](../../../skyulf-core/tests/integration/test_stateful_estimator_tuple_targets.py)
- `tests/integration/test_temporal_tuning_boundaries.py` ?
- [tests/integration/test_text_target_context.py](../../../skyulf-core/tests/integration/test_text_target_context.py)
- [tests/integration/test_time_series_gaps.py](../../../skyulf-core/tests/integration/test_time_series_gaps.py)
- [tests/integration/test_time_series_missing_values.py](../../../skyulf-core/tests/integration/test_time_series_missing_values.py)
- [tests/integration/test_time_series_nodes.py](../../../skyulf-core/tests/integration/test_time_series_nodes.py)
- [tests/integration/test_transformations_general.py](../../../skyulf-core/tests/integration/test_transformations_general.py)
- [tests/integration/test_transformations_power_simple.py](../../../skyulf-core/tests/integration/test_transformations_power_simple.py)
- [tests/integration/test_tuning.py](../../../skyulf-core/tests/integration/test_tuning.py)
- `tests/integration/test_tuning_class_axis.py` ?
- `tests/integration/test_tuning_class_weights.py` ?
- `tests/integration/test_tuning_filtered_validation.py` ?
- `tests/integration/test_tuning_native_missing_values.py` ?
- [tests/integration/test_tuning_per_fold_refit.py](../../../skyulf-core/tests/integration/test_tuning_per_fold_refit.py)
- `tests/integration/test_tuning_positive_class.py` ?
- [tests/integration/test_tuning_time_series_holdout.py](../../../skyulf-core/tests/integration/test_tuning_time_series_holdout.py)
- [tests/integration/test_value_replacement.py](../../../skyulf-core/tests/integration/test_value_replacement.py)
- [tests/integration/test_vectorization.py](../../../skyulf-core/tests/integration/test_vectorization.py)
- [tests/integration/test_woe_and_calibration.py](../../../skyulf-core/tests/integration/test_woe_and_calibration.py)
- [tests/integration/test_wrapped_polars_frames.py](../../../skyulf-core/tests/integration/test_wrapped_polars_frames.py)
- [tests/integration/test_xy_row_alignment.py](../../../skyulf-core/tests/integration/test_xy_row_alignment.py)
- [tests/test_preprocessing_split_callback.py](../../../skyulf-core/tests/test_preprocessing_split_callback.py)
- [tests/unit/test_all_nodes_smoke.py](../../../skyulf-core/tests/unit/test_all_nodes_smoke.py)
- [tests/unit/test_artifact_shapes.py](../../../skyulf-core/tests/unit/test_artifact_shapes.py)
- [tests/unit/test_artifact_snapshots.py](../../../skyulf-core/tests/unit/test_artifact_snapshots.py)
- [tests/unit/test_benchmarks.py](../../../skyulf-core/tests/unit/test_benchmarks.py)
- [tests/unit/test_boosting_progress.py](../../../skyulf-core/tests/unit/test_boosting_progress.py)
- [tests/unit/test_calendar_and_row_target_contract.py](../../../skyulf-core/tests/unit/test_calendar_and_row_target_contract.py)
- `tests/unit/test_calibrated_classifier_random_state.py` ?
- [tests/unit/test_cleaning_operation_leakage.py](../../../skyulf-core/tests/unit/test_cleaning_operation_leakage.py)
- [tests/unit/test_core_compute.py](../../../skyulf-core/tests/unit/test_core_compute.py)
- [tests/unit/test_core_model_registry.py](../../../skyulf-core/tests/unit/test_core_model_registry.py)
- [tests/unit/test_core_schema.py](../../../skyulf-core/tests/unit/test_core_schema.py)
- [tests/unit/test_core_seams.py](../../../skyulf-core/tests/unit/test_core_seams.py)
- [tests/unit/test_core_serialization.py](../../../skyulf-core/tests/unit/test_core_serialization.py)
- [tests/unit/test_data_catalog.py](../../../skyulf-core/tests/unit/test_data_catalog.py)
- [tests/unit/test_dataset.py](../../../skyulf-core/tests/unit/test_dataset.py)
- [tests/unit/test_deprecation.py](../../../skyulf-core/tests/unit/test_deprecation.py)
- `tests/unit/test_eda_profile_alias.py` ?
- [tests/unit/test_encoding_category_keys_regression_20260909.py](../../../skyulf-core/tests/unit/test_encoding_category_keys_regression_20260909.py)
- [tests/unit/test_encoding_common.py](../../../skyulf-core/tests/unit/test_encoding_common.py)
- [tests/unit/test_encoding_operation_leakage.py](../../../skyulf-core/tests/unit/test_encoding_operation_leakage.py)
- `tests/unit/test_encoding_output_names_regression_20260912.py` ?
- `tests/unit/test_encoding_temporal_enum_regression_20260912.py` ?
- [tests/unit/test_encoding_text_deep_audit_20260908.py](../../../skyulf-core/tests/unit/test_encoding_text_deep_audit_20260908.py)
- `tests/unit/test_engine_context.py` ?
- `tests/unit/test_engine_empty_frames.py` ?
- [tests/unit/test_engine_parity.py](../../../skyulf-core/tests/unit/test_engine_parity.py)
- `tests/unit/test_engine_wrapper_serialization.py` ?
- [tests/unit/test_engines_pandas.py](../../../skyulf-core/tests/unit/test_engines_pandas.py)
- [tests/unit/test_engines_polars.py](../../../skyulf-core/tests/unit/test_engines_polars.py)
- [tests/unit/test_engines_registry.py](../../../skyulf-core/tests/unit/test_engines_registry.py)
- [tests/unit/test_engines_sklearn_bridge.py](../../../skyulf-core/tests/unit/test_engines_sklearn_bridge.py)
- [tests/unit/test_ensemble_nodes.py](../../../skyulf-core/tests/unit/test_ensemble_nodes.py)
- [tests/unit/test_evaluation_clustering.py](../../../skyulf-core/tests/unit/test_evaluation_clustering.py)
- [tests/unit/test_evaluation_clustering_polars.py](../../../skyulf-core/tests/unit/test_evaluation_clustering_polars.py)
- [tests/unit/test_evaluation_common.py](../../../skyulf-core/tests/unit/test_evaluation_common.py)
- [tests/unit/test_evaluation_metrics.py](../../../skyulf-core/tests/unit/test_evaluation_metrics.py)
- [tests/unit/test_evaluation_thresholds.py](../../../skyulf-core/tests/unit/test_evaluation_thresholds.py)
- [tests/unit/test_expect.py](../../../skyulf-core/tests/unit/test_expect.py)
- `tests/unit/test_expect_empty.py` ?
- [tests/unit/test_explainability.py](../../../skyulf-core/tests/unit/test_explainability.py)
- [tests/unit/test_feature_operation_leakage.py](../../../skyulf-core/tests/unit/test_feature_operation_leakage.py)
- [tests/unit/test_fold_audit.py](../../../skyulf-core/tests/unit/test_fold_audit.py)
- [tests/unit/test_fold_merge_guide_example.py](../../../skyulf-core/tests/unit/test_fold_merge_guide_example.py)
- [tests/unit/test_fold_merged_adapter.py](../../../skyulf-core/tests/unit/test_fold_merged_adapter.py)
- `tests/unit/test_fold_merged_row_contract.py` ?
- [tests/unit/test_fold_pipeline.py](../../../skyulf-core/tests/unit/test_fold_pipeline.py)
- [tests/unit/test_fold_preprocessing.py](../../../skyulf-core/tests/unit/test_fold_preprocessing.py)
- `tests/unit/test_fold_scoring.py` ?
- `tests/unit/test_general_transformation_sequential_fit.py` ?
- `tests/unit/test_geo_distance_schema.py` ?
- [tests/unit/test_hyperparameters_class_weight.py](../../../skyulf-core/tests/unit/test_hyperparameters_class_weight.py)
- [tests/unit/test_hyperparameters_random_state.py](../../../skyulf-core/tests/unit/test_hyperparameters_random_state.py)
- [tests/unit/test_hyperparameters_registry.py](../../../skyulf-core/tests/unit/test_hyperparameters_registry.py)
- [tests/unit/test_infer_output_schema.py](../../../skyulf-core/tests/unit/test_infer_output_schema.py)
- `tests/unit/test_label_target_containers_regression_20260912.py` ?
- [tests/unit/test_leakage_enforcement.py](../../../skyulf-core/tests/unit/test_leakage_enforcement.py)
- [tests/unit/test_leakage_safety_validation.py](../../../skyulf-core/tests/unit/test_leakage_safety_validation.py)
- `tests/unit/test_lightgbm_subsampling.py` ?
- `tests/unit/test_manual_bounds_missing_columns.py` ?
- [tests/unit/test_modeling.py](../../../skyulf-core/tests/unit/test_modeling.py)
- [tests/unit/test_modeling_base.py](../../../skyulf-core/tests/unit/test_modeling_base.py)
- [tests/unit/test_modeling_classification_gaps.py](../../../skyulf-core/tests/unit/test_modeling_classification_gaps.py)
- [tests/unit/test_modeling_clustering.py](../../../skyulf-core/tests/unit/test_modeling_clustering.py)
- [tests/unit/test_modeling_ensemble_gaps.py](../../../skyulf-core/tests/unit/test_modeling_ensemble_gaps.py)
- [tests/unit/test_modeling_naive_bayes.py](../../../skyulf-core/tests/unit/test_modeling_naive_bayes.py)
- [tests/unit/test_modeling_regression_gaps.py](../../../skyulf-core/tests/unit/test_modeling_regression_gaps.py)
- [tests/unit/test_modeling_sklearn_wrapper.py](../../../skyulf-core/tests/unit/test_modeling_sklearn_wrapper.py)
- [tests/unit/test_multi_output_audit.py](../../../skyulf-core/tests/unit/test_multi_output_audit.py)
- [tests/unit/test_no_inline_engine_dispatch.py](../../../skyulf-core/tests/unit/test_no_inline_engine_dispatch.py)
- [tests/unit/test_numeric_preprocessing_behavior_audit.py](../../../skyulf-core/tests/unit/test_numeric_preprocessing_behavior_audit.py)
- [tests/unit/test_outlier_failure_branches.py](../../../skyulf-core/tests/unit/test_outlier_failure_branches.py)
- [tests/unit/test_patch_coverage_core_round5.py](../../../skyulf-core/tests/unit/test_patch_coverage_core_round5.py)
- [tests/unit/test_patch_coverage_core_round6.py](../../../skyulf-core/tests/unit/test_patch_coverage_core_round6.py)
- [tests/unit/test_pipeline.py](../../../skyulf-core/tests/unit/test_pipeline.py)
- [tests/unit/test_pipeline_card.py](../../../skyulf-core/tests/unit/test_pipeline_card.py)
- [tests/unit/test_pipeline_config_validation.py](../../../skyulf-core/tests/unit/test_pipeline_config_validation.py)
- [tests/unit/test_pipeline_coverage.py](../../../skyulf-core/tests/unit/test_pipeline_coverage.py)
- [tests/unit/test_pipeline_describe.py](../../../skyulf-core/tests/unit/test_pipeline_describe.py)
- [tests/unit/test_pipeline_fit_lifecycle.py](../../../skyulf-core/tests/unit/test_pipeline_fit_lifecycle.py)
- [tests/unit/test_pipeline_seal.py](../../../skyulf-core/tests/unit/test_pipeline_seal.py)
- [tests/unit/test_pipeline_split_extraction.py](../../../skyulf-core/tests/unit/test_pipeline_split_extraction.py)
- `tests/unit/test_pipeline_threshold_alignment.py` ?
- `tests/unit/test_pipeline_threshold_fingerprint.py` ?
- [tests/unit/test_pipeline_threshold_tuning.py](../../../skyulf-core/tests/unit/test_pipeline_threshold_tuning.py)
- [tests/unit/test_polynomial_leakage_policy.py](../../../skyulf-core/tests/unit/test_polynomial_leakage_policy.py)
- [tests/unit/test_postfix_imputation_selection_leakage.py](../../../skyulf-core/tests/unit/test_postfix_imputation_selection_leakage.py)
- [tests/unit/test_preprocessing.py](../../../skyulf-core/tests/unit/test_preprocessing.py)
- [tests/unit/test_preprocessing_base.py](../../../skyulf-core/tests/unit/test_preprocessing_base.py)
- [tests/unit/test_preprocessing_dispatcher.py](../../../skyulf-core/tests/unit/test_preprocessing_dispatcher.py)
- [tests/unit/test_preprocessing_helpers.py](../../../skyulf-core/tests/unit/test_preprocessing_helpers.py)
- [tests/unit/test_preprocessing_inspection.py](../../../skyulf-core/tests/unit/test_preprocessing_inspection.py)
- [tests/unit/test_preprocessing_nodes.py](../../../skyulf-core/tests/unit/test_preprocessing_nodes.py)
- [tests/unit/test_preprocessing_pipeline.py](../../../skyulf-core/tests/unit/test_preprocessing_pipeline.py)
- `tests/unit/test_preprocessing_refactor_branches.py` ?
- [tests/unit/test_preprocessing_selection_leakage.py](../../../skyulf-core/tests/unit/test_preprocessing_selection_leakage.py)
- [tests/unit/test_profiling_column.py](../../../skyulf-core/tests/unit/test_profiling_column.py)
- [tests/unit/test_profiling_expect_gap.py](../../../skyulf-core/tests/unit/test_profiling_expect_gap.py)
- [tests/unit/test_profiling_geo_exclusions.py](../../../skyulf-core/tests/unit/test_profiling_geo_exclusions.py)
- `tests/unit/test_profiling_json_contract.py` ?
- `tests/unit/test_profiling_outlier_summary.py` ?
- `tests/unit/test_profiling_timeseries_alignment.py` ?
- [tests/unit/test_profiling_utils.py](../../../skyulf-core/tests/unit/test_profiling_utils.py)
- [tests/unit/test_profiling_visualizer.py](../../../skyulf-core/tests/unit/test_profiling_visualizer.py)
- [tests/unit/test_profiling_visualizer_labels.py](../../../skyulf-core/tests/unit/test_profiling_visualizer_labels.py)
- [tests/unit/test_protocols.py](../../../skyulf-core/tests/unit/test_protocols.py)
- [tests/unit/test_public_api_exports.py](../../../skyulf-core/tests/unit/test_public_api_exports.py)
- [tests/unit/test_pyarrow_dtypes.py](../../../skyulf-core/tests/unit/test_pyarrow_dtypes.py)
- [tests/unit/test_registry_contract.py](../../../skyulf-core/tests/unit/test_registry_contract.py)
- [tests/unit/test_registry_toplevel.py](../../../skyulf-core/tests/unit/test_registry_toplevel.py)
- [tests/unit/test_resampling.py](../../../skyulf-core/tests/unit/test_resampling.py)
- `tests/unit/test_resampling_smote_tomek.py` ?
- [tests/unit/test_schema_contract.py](../../../skyulf-core/tests/unit/test_schema_contract.py)
- [tests/unit/test_schema_inference_fuzz.py](../../../skyulf-core/tests/unit/test_schema_inference_fuzz.py)
- `tests/unit/test_sentence_embedder_concurrency.py` ?
- `tests/unit/test_shap_refactor_branches.py` ?
- `tests/unit/test_temporal_model_input.py` ?
- [tests/unit/test_text_vectorization.py](../../../skyulf-core/tests/unit/test_text_vectorization.py)
- [tests/unit/test_tuning_engine.py](../../../skyulf-core/tests/unit/test_tuning_engine.py)
- [tests/unit/test_tuning_engine_failure_branches.py](../../../skyulf-core/tests/unit/test_tuning_engine_failure_branches.py)
- [tests/unit/test_tuning_failed_fold_semantics.py](../../../skyulf-core/tests/unit/test_tuning_failed_fold_semantics.py)
- `tests/unit/test_tuning_refactor_branches.py` ?
- [tests/unit/test_tuning_reporter.py](../../../skyulf-core/tests/unit/test_tuning_reporter.py)
- [tests/unit/test_utils.py](../../../skyulf-core/tests/unit/test_utils.py)
- [tests/unit/test_vectorization_gaps.py](../../../skyulf-core/tests/unit/test_vectorization_gaps.py)
- `tests/unit/test_vectorization_output_names_regression_20260912.py` ?
- [tests/utils/test_case_loader.py](../../../skyulf-core/tests/utils/test_case_loader.py)
