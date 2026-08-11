# Enterprise Readiness — Algorithm/Feature Coverage Gaps in `skyulf-core`

**Date:** 2026-08-11
**Status:** Investigation complete
**Scope:** Preprocessing, feature-engineering, and modeling node coverage in
`skyulf-core/skyulf/` compared against sklearn, feature-engine,
category_encoders, imbalanced-learn, and common statistical/EDA/time-series
tooling. Companion to
[2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md) (UI/backend
option-mismatch, extensibility) and
[../deep-learning/2026-08-11-architecture-design.md](../deep-learning/2026-08-11-architecture-design.md)
(planned DL model family — not duplicated here).

## How this was produced

Full-tree `grep -rn "@NodeRegistry.register" skyulf/` to get the authoritative
node-id inventory, cross-checked against directory listings of
`preprocessing/` and `modeling/`, then compared feature-by-feature against
what a user would otherwise reach for in sklearn/feature-engine/
category_encoders/imbalanced-learn/statsmodels.

## Inventory: all currently-registered node ids

### Preprocessing — Imputation
| Node id | File |
|---|---|
| `SimpleImputer` | `preprocessing/imputation/simple.py:77` |
| `KNNImputer` | `preprocessing/imputation/knn.py:40` |
| `IterativeImputer` | `preprocessing/imputation/iterative.py:42` |

### Preprocessing — Missing/Duplicate handling
| Node id | File |
|---|---|
| `MissingIndicator` | `preprocessing/drop_and_missing/missing_indicator.py:81` |
| `DropMissingColumns` | `preprocessing/drop_and_missing/drop_columns.py:85` |
| `DropMissingRows` | `preprocessing/drop_and_missing/drop_rows.py:68` |
| `Deduplicate` | `preprocessing/drop_and_missing/deduplicate.py:63` |

### Preprocessing — Cleaning
| Node id | File |
|---|---|
| `AliasReplacement` | `preprocessing/cleaning/alias.py:118` |
| `InvalidValueReplacement` | `preprocessing/cleaning/invalid_value.py:202` |
| `TextCleaning` | `preprocessing/cleaning/text.py:181` |
| `ValueReplacement` | `preprocessing/cleaning/value_replacement.py:142` |

### Preprocessing — Scaling
| Node id | File |
|---|---|
| `StandardScaler` | `preprocessing/scaling/standard.py:98` |
| `MinMaxScaler` | `preprocessing/scaling/minmax.py:58` |
| `MaxAbsScaler` | `preprocessing/scaling/maxabs.py:57` |
| `RobustScaler` | `preprocessing/scaling/robust.py:73` |

### Preprocessing — Transformations
| Node id | File |
|---|---|
| `PowerTransformer` | `preprocessing/transformations/power.py:108` |
| `SimpleTransformation` | `preprocessing/transformations/simple.py:59` |
| `GeneralTransformation` | `preprocessing/transformations/general.py:151` |

### Preprocessing — Bucketing/Binning
| Node id | File |
|---|---|
| `GeneralBinning` | `preprocessing/bucketing.py:424` |
| `CustomBinning` | `preprocessing/bucketing.py:469` |
| `KBinsDiscretizer` | `preprocessing/bucketing.py:507` |

### Preprocessing — Encoding
| Node id | File |
|---|---|
| `OneHotEncoder` | `preprocessing/encoding/one_hot.py:205` |
| `OrdinalEncoder` | `preprocessing/encoding/ordinal.py:269` |
| `LabelEncoder` | `preprocessing/encoding/label.py:259` |
| `DummyEncoder` | `preprocessing/encoding/dummy.py:126` |
| `HashEncoder` | `preprocessing/encoding/hash.py:88` |
| `TargetEncoder` | `preprocessing/encoding/target.py:296` |
| `WOEEncoder` | `preprocessing/encoding/woe.py:185` |

### Preprocessing — Outliers
| Node id | File |
|---|---|
| `IQR` | `preprocessing/outliers/iqr.py:63` |
| `ZScore` | `preprocessing/outliers/zscore.py:67` |
| `Winsorize` | `preprocessing/outliers/winsorize.py:61` |
| `EllipticEnvelope` | `preprocessing/outliers/elliptic.py:84` |
| `ManualBounds` | `preprocessing/outliers/manual_bounds.py:85` |

### Preprocessing — Resampling (class imbalance)
| Node id | File |
|---|---|
| `Oversampling` (smote/adasyn/borderline_smote/svm_smote/kmeans_smote/smote_tomek/random_over) | `preprocessing/resampling.py:234` |
| `Undersampling` (random_under_sampling/nearmiss/tomek_links) | `preprocessing/resampling.py:341` |

### Preprocessing — Feature generation / selection
| Node id | File |
|---|---|
| `FeatureGeneration` / `FeatureMath` / `FeatureGenerationNode` | `preprocessing/feature_generation/generation.py:24-26` |
| `FeatureInteraction` | `preprocessing/feature_generation/interaction.py:154` |
| `PolynomialFeatures` / `PolynomialFeaturesNode` | `preprocessing/feature_generation/polynomial.py:89-90` |
| `CorrelationThreshold` | `preprocessing/feature_selection/correlation.py:153` |
| `feature_selection` (facade) | `preprocessing/feature_selection/facade.py:58` |
| `ModelBasedSelection` | `preprocessing/feature_selection/model_based.py:35` |
| `UnivariateSelection` | `preprocessing/feature_selection/univariate.py:37` |
| `VarianceThreshold` | `preprocessing/feature_selection/variance.py:23` |

### Preprocessing — Time series
| Node id | File |
|---|---|
| `DateFeatures` | `preprocessing/time_series/date_features.py:142` |
| `LagFeatures` | `preprocessing/time_series/lag.py:82` |
| `RollingAggregate` | `preprocessing/time_series/rolling.py:124` |

### Preprocessing — Geo
| Node id | File |
|---|---|
| `GeoDistance` | `preprocessing/geo/distance.py:143` |
| `H3Index` | `preprocessing/geo/h3_index.py:106` |

### Preprocessing — Vectorization / NLP
| Node id | File |
|---|---|
| `count_vectorizer` | `preprocessing/vectorization/count_vectorizer.py:101` |
| `hashing_vectorizer` | `preprocessing/vectorization/hashing_vectorizer.py:89` |
| `tfidf_vectorizer` | `preprocessing/vectorization/tfidf_vectorizer.py:95` |
| `tokenizer` | `preprocessing/vectorization/tokenizer.py:105` |
| `sentence_embedder` | `preprocessing/vectorization/sentence_embedder.py:122` |

### Preprocessing — Inspection / Split
| Node id | File |
|---|---|
| `DatasetProfile` | `preprocessing/inspection.py:85` |
| `DataSnapshot` | `preprocessing/inspection.py:132` |
| `Casting` | `preprocessing/casting.py:353` |
| `Split` / `TrainTestSplitter` | `preprocessing/split.py:126-127` |
| `feature_target_split` | `preprocessing/split.py:372` |

### Modeling — Classification
`logistic_regression`, `calibrated_classifier`, `random_forest_classifier`,
`svc`, `k_neighbors_classifier`, `decision_tree_classifier`,
`gradient_boosting_classifier`, `adaboost_classifier`,
`xgboost_classifier` (optional dep), `extra_trees_classifier`,
`hist_gradient_boosting_classifier`, `lgbm_classifier` (optional dep),
`gaussian_nb`, `sgd_classifier` — `modeling/classification.py`

### Modeling — Regression
`linear_regression`, `ridge_regression`, `random_forest_regressor`,
`lasso_regression`, `elasticnet_regression`, `svr`,
`k_neighbors_regressor`, `decision_tree_regressor`,
`gradient_boosting_regressor`, `adaboost_regressor`,
`extra_trees_regressor`, `hist_gradient_boosting_regressor`,
`lgbm_regressor` (optional dep), `xgboost_regressor` (optional dep) —
`modeling/regression.py`

### Modeling — Clustering
`kmeans`, `minibatch_kmeans`, `gaussian_mixture`, `birch` —
`modeling/clustering.py`

### Modeling — Ensemble (meta-estimators)
`voting_classifier`, `stacking_classifier`, `voting_regressor`,
`stacking_regressor` — `modeling/ensemble.py`

### Modeling — Naive Bayes
`multinomial_nb`, `bernoulli_nb` — `modeling/naive_bayes.py`

---

## Findings — genuine gaps

### 1. No forecasting-model family despite a dedicated time-series preprocessing folder
**Missing:** ARIMA/SARIMAX, exponential smoothing (Holt-Winters), and
Prophet-style forecasters. There is no statsmodels or Prophet dependency
anywhere in the repo (`grep -rn "arima\|prophet\|sarimax" skyulf/` returns
zero hits outside this doc).
**Why it matters:** `preprocessing/time_series/` already ships
`DateFeatures`, `LagFeatures`, and `RollingAggregate`
(`time_series/date_features.py:142`, `lag.py:82`, `rolling.py:124`),
signaling first-class time-series intent, but every registered model in
`modeling/` is a generic regressor/classifier fit on the lag/rolling
features — there is no node that models seasonality/trend natively or
produces a native forecast horizon with prediction intervals. A user
building a demand-forecasting pipeline has to leave the platform for
statsmodels/Prophet/sktime entirely for the actual forecasting step, then
has no way to bring results back into the canvas.
**Closest existing analog:** `gradient_boosting_regressor`
(`modeling/regression.py:283`) or `hist_gradient_boosting_regressor`
(`modeling/regression.py:376`) fit on lag features — works but has no
native multi-step-ahead recursive forecast, confidence intervals, or
seasonal decomposition.
**Effort:** L. Model the API after `statsmodels.tsa` (`SARIMAX`,
`ExponentialSmoothing`) for a first cut; a `Prophet`/`sktime`-style node
could follow. This is explicitly flagged as future-DL-only in the
deep-learning architecture doc (§4.1 `timeseries/sequence_forecaster.py`),
but that plan only covers a neural LSTM/TCN forecaster — classical
statistical forecasting (ARIMA/ETS) is not on that roadmap at all and
should probably land first since it's far cheaper.

### 2. No CatBoost model, despite XGBoost and LightGBM both being supported as optional deps
**Missing:** `CatBoostClassifier`/`CatBoostRegressor`. Confirmed via
`grep -rln "catboost" skyulf/` → zero results.
**Why it matters:** CatBoost is one of the three "big three" GBM libraries
(XGBoost, LightGBM, CatBoost) and is specifically preferred by many teams
for native categorical handling (no encoding step needed) and often
stronger out-of-the-box defaults. Its absence means categorical-heavy
tabular datasets must go through an encoding node first, which CatBoost
users would normally skip.
**Closest existing analog:** `lgbm_classifier`/`lgbm_regressor`
(`modeling/classification.py:576`, `modeling/regression.py:426`) — same
optional-dependency lazy-import pattern (`classification.py:23`,
`regression.py:16`) that a CatBoost node could directly copy.
**Effort:** S–M (the lazy-optional-dependency + `SklearnCalculator`/
`SklearnApplier` pattern is already established and just needs a new
model class plugged in).

### 3. category_encoders-style advanced encoders are absent beyond Target/WOE/Hash
**Missing:** Leave-One-Out encoding, James-Stein encoding, CatBoost
(ordered target) encoding, and rare-label grouping (feature-engine's
`RareLabelEncoder`). `grep -rn "catboost|leave_one_out|james_stein|rare" -i
skyulf/preprocessing/encoding/` returns nothing.
**Why it matters:** `TargetEncoder` (`encoding/target.py:296`) and
`WOEEncoder` (`encoding/woe.py:185`) cover the two most common Bayesian
encoders, but high-cardinality categorical pipelines commonly need
leave-one-out encoding to avoid target leakage during training (the
in-fold target-mean leakage risk with a plain target encoder is
well-known), and rare-label grouping is a standard first step before any
encoding to control cardinality explosion. Today a user must handle rare
categories manually via `ValueReplacement`
(`cleaning/value_replacement.py:142`) with a manually curated value list,
which doesn't scale to hundreds of rare categories.
**Closest existing analog:** `TargetEncoder`
(`preprocessing/encoding/target.py:296`) — closest structurally; a
leave-one-out variant is a small delta (per-row exclusion of that row's own
target when computing the mean) on the same fit/apply skeleton.
**Effort:** S (leave-one-out encoder, same skeleton as TargetEncoder) to
M (rare-label grouping node, new but simple frequency-threshold logic).

### 4. No IterativeImputer/KNNImputer competitor gaps — this area is actually strong (not a finding, confirmed for completeness)
`IterativeImputer` (`preprocessing/imputation/iterative.py:42`) and
`KNNImputer` (`preprocessing/imputation/knn.py:40`) both exist, matching
sklearn's most requested imputers. No gap here; see
node-flexibility.md §3 for UI-only param exposure gaps (not a coverage
gap).

### 5. No QuantileTransformer
**Missing:** sklearn's `QuantileTransformer` (uniform/normal output
distribution mapping). `grep -n "QuantileTransformer" -r
skyulf/preprocessing/` returns zero hits; only `PowerTransformer`
(`transformations/power.py:108`) exists for distribution-shape correction.
**Why it matters:** QuantileTransformer is commonly reached for when a
feature's distribution is multi-modal or has extreme outliers that
Box-Cox/Yeo-Johnson (PowerTransformer's methods) don't handle well — it's
a standard, distinct tool in the sklearn preprocessing toolbox, not a
niche one.
**Closest existing analog:** `PowerTransformer`
(`preprocessing/transformations/power.py:108`) — same
`BaseCalculator`/`BaseApplier` shape, a QuantileTransformer node is a
straightforward sibling.
**Effort:** S.

### 6. No cyclical (sin/cos) encoding for calendar features
**Missing:** feature-engine/common-practice cyclical encoding of periodic
calendar components (hour-of-day, day-of-week, month) as
`sin(2πx/period)`/`cos(2πx/period)` pairs. `DateFeatures`
(`preprocessing/time_series/date_features.py:1-40`) extracts raw
year/month/day/dayofweek/weekofyear as plain (nullable) integers — grep
for `sin|cos|cyclical` in that file returns nothing.
**Why it matters:** feeding raw `month` (1–12) or `dayofweek` (0–6) into a
linear/distance-based model treats December and January as maximally far
apart, which is wrong for genuinely periodic signals — cyclical encoding
is the standard fix and is a very common feature-engineering step for any
seasonality-sensitive tabular model (not just true forecasting models).
**Closest existing analog:** `DateFeatures`
(`preprocessing/time_series/date_features.py:142`) — the natural place to
add a `cyclical: bool` or `encoding: "raw"|"cyclical"` option per feature.
**Effort:** S (pure feature-computation addition to an existing node).

### 7. No VIF/multicollinearity node, and correlation stops at Pearson/Spearman (no Kendall) in the pipeline-composable node
**Missing:** a Variance Inflation Factor node for multicollinearity
diagnosis. `grep -rln "vif|variance_inflation" skyulf/` (excluding this
doc) returns nothing anywhere in `skyulf/preprocessing/` or
`skyulf/modeling/`. Separately, `CorrelationThreshold`
(`preprocessing/feature_selection/correlation.py:153`) hardcodes its
native/accelerated method set to `_NATIVE_POLARS_METHODS = frozenset(("pearson",
"spearman"))` (`correlation.py:19`) and its pandas fallback method default
is `"pearson"` (`correlation.py:73`) with no `kendall` example/validation
anywhere in the file — Kendall's tau (useful for small/ordinal/rank data)
is not exposed as a documented option.
**Why it matters:** VIF is the standard quantitative multicollinearity
check before fitting any linear/logistic model (more rigorous than a
pairwise-correlation threshold, since it captures multivariate
collinearity, not just pairwise), and is a routine step in regulated/
statistically-literate teams' workflows (e.g. econometrics, credit
scoring). Its total absence means this diagnostic must be done outside the
platform (statsmodels `variance_inflation_factor`) with no way to feed the
result back as a reusable pipeline node.
**Closest existing analog:** `CorrelationThreshold`
(`preprocessing/feature_selection/correlation.py:153`) is the closest
selection node, but it's pairwise-only, not a substitute for VIF.
**Effort:** S–M for a VIF node (statsmodels dependency,
straightforward fit-and-report-then-drop pattern mirroring
`CorrelationThreshold`); S for adding `"kendall"` to the accepted method
list (would fall through to the existing pandas `.corr(method=...)` path
since pandas already supports it — `correlation.py:78` — only the Polars
native-path frozenset and any UI/validation allow-list need updating).

### 8. Hypothesis tests exist but are report-only (profiling), not pipeline nodes
**Found:** Shapiro-Wilk/KS normality tests (`profiling/_analyzer/
column.py:57-68`) and one-way ANOVA (`profiling/_analyzer/target.py:182-199`,
using `scipy.stats.f_oneway`) are implemented — but confirmed via
`grep -n "NodeRegistry" skyulf/profiling/_analyzer/*.py skyulf/profiling/*.py`
(zero matches) that **none of this is registered as a `NodeRegistry` node**.
It's exclusively wired into the standalone dataset-profiling report
generator, not composable/branchable inside a pipeline graph (e.g. "drop
this feature if it fails a normality test" or "gate a step on a chi-square
test of independence").
**Why it matters:** Users doing rigorous EDA/feature-vetting inside the
canvas (the tool's whole premise) cannot use these tests as a graph node
whose pass/fail or p-value output feeds a downstream decision — they can
only read a static profiling report. A chi-square test of independence
between two categorical columns (a very common categorical-feature-vetting
check) doesn't exist at all, registered or not — `grep -n "chi2_contingency"
skyulf/` outside this doc returns nothing.
**Closest existing analog:** `DatasetProfile`
(`preprocessing/inspection.py:85`) is registered and pipeline-composable,
but it's a generic descriptive-stats node, not a hypothesis-test node with
a pass/fail or p-value output column.
**Effort:** M — the underlying scipy calls already exist in
`profiling/_analyzer/`, so a new "StatisticalTest" node family is mostly
plumbing (wrapping existing calls in `BaseCalculator`/`BaseApplier`,
adding a chi-square-of-independence variant) rather than new statistics.

### 9. No group-aware or blocked/purged time-series cross-validation strategy
**Missing:** `GroupKFold`/`StratifiedGroupKFold`. `grep -n
"GroupKFold\|StratifiedGroupKFold" skyulf/modeling/cross_validation.py`
returns nothing; only `TimeSeriesSplit` (`cross_validation.py:12,394`) is
implemented for the non-standard-KFold cases.
**Why it matters:** without group-aware CV, a dataset with repeated
entities (e.g. multiple rows per customer/patient) can leak the same
entity across train and validation folds, silently inflating validation
metrics — a correctness bug users can easily fall into with no warning.
This is a smaller, more surgical version of the group-split gap already
flagged for `TrainTestSplitNode` in node-flexibility.md §3, but applies to
k-fold CV specifically, which is a separate code path
(`modeling/cross_validation.py`).
**Closest existing analog:** `TimeSeriesSplit` support
(`cross_validation.py:394`) — same file, same dispatch pattern; adding
`GroupKFold` is a parallel branch.
**Effort:** S (sklearn already implements `GroupKFold`; this is wiring +
requiring a group-column param, not new algorithm work).

### 10. No PMML/ONNX/MLflow export — noted here only to cross-reference, not duplicated
Already covered in node-flexibility.md §7 (`artifacts/local.py:6-44`,
`artifacts/factory.py:24-37`); not re-analyzed here since it's an
export/interoperability concern rather than an algorithm-coverage gap.

## What's already strong (confirmed, not a gap)

- **Imbalanced-learn coverage is comprehensive**, not absent as the audit
  brief hypothesized: `Oversampling` supports
  `random_over/smote/adasyn/borderline_smote/svm_smote/kmeans_smote/
  smote_tomek` and `Undersampling` supports
  `random_under_sampling/nearmiss/tomek_links`
  (`preprocessing/resampling.py:165-171, 289-291`).
- **No stubbed/`NotImplementedError` nodes found.** A repo-wide grep for
  `NotImplementedError|TODO|FIXME` inside `preprocessing/` and `modeling/`
  turned up only one unrelated TODO (`transformations/power.py:130`, a
  pandas-removal migration note, not a coverage stub) — every registered
  node has a real, working `infer_output_schema`/`fit`/`apply`
  implementation.
- **Model calibration exists**: `calibrated_classifier`
  (`modeling/classification.py:181`, wrapping sklearn's
  `CalibratedClassifierCV`) — contradicts the "no calibration" gap
  hypothesized in the audit brief.
- **XGBoost and LightGBM are both supported** as optional, lazily-imported
  dependencies (`modeling/classification.py:23,576`,
  `modeling/regression.py:16,426`) — only CatBoost (Finding #1... #2 above)
  is missing from the "big three" GBM set.
- Text/NLP: TF-IDF (`tfidf_vectorizer`), count/hashing vectorizers, and
  dense sentence embeddings via `sentence-transformers`
  (`sentence_embedder.py:122`) are all present — this is a fairly complete
  basic-NLP feature-extraction set for a tabular-first platform.
