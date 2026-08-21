# Engine Data Paths — Where Polars, NumPy, and pandas Run

Skyulf runs on a configurable dataframe engine (`SKYULF_ENGINE`, default
**polars**, pandas first-class — see [Engine Mechanics](engine_mechanics.md)).
This page answers a more specific question: **for every part of skyulf-core,
which library actually touches the data, and what path does the data take?**

## Legend

Every fit/apply path gets one of these labels:

| Label | Data flow | Meaning |
| :--- | :--- | :--- |
| **P** | Polars → Polars | Pure Polars expressions. The frame never leaves the Polars engine. |
| **P→N→P** | Polars → NumPy/SciPy → Polars | Columns extracted to NumPy, computed there, results re-attached as Polars series. No pandas involved. |
| **P→sk→P** | Polars → NumPy → scikit-learn → Polars | scikit-learn fit/transform on NumPy extracted straight from Polars (`SklearnBridge`), results converted back. **No pandas hop.** |
| **P→Py→P** | Polars → Python objects → Polars | Values pulled as `list[str]` (or per-row Python calls), processed, rebuilt as Polars columns. |
| **P→pd→P** | Polars → pandas → Polars | A genuine pandas hop, kept only where exact pandas semantics are required (documented per site). |
| **pd** | pandas only | Runs on pandas by design, regardless of `SKYULF_ENGINE`, because the underlying library is pandas-bound. Converts back to the configured engine afterwards. |

When `SKYULF_ENGINE=pandas`, every node simply runs its pandas implementation
(the right-hand path everywhere) — the table below describes the **Polars
input** paths, which is the default.

## What "the sklearn/XGBoost boundary" means

scikit-learn, XGBoost, LightGBM and imblearn do not read dataframe objects —
they train on raw NumPy arrays. So there is a line every model eventually
crosses: the data leaves the dataframe world (Polars **or** pandas) as a NumPy
matrix, the library does its work, and the results come back out. That line is
**the boundary**.

Two consequences:

1. **Where the crossing happens is the only thing Skyulf controls.** It
   crosses at the cheapest point: `SklearnBridge.to_sklearn` converts
   Polars → NumPy *directly*, skipping pandas entirely.
2. **Engine choice can't change how long the model itself takes.** The heavy
   compute happens inside the library, on NumPy, identically for both engines —
   which is exactly why every model row in the
   [benchmark](../performance.md) lands at ~1.0x.

## Preprocessing nodes

### Cleaning, rows & columns — all native Polars

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| AliasReplacement, ValueReplacement, InvalidValueReplacement, TextCleaning | P | **P** | Expression-based replacements. |
| Casting | — | **P** | `cast()` with strict/error handling options. |
| Deduplicate, DropMissingRows, DropMissingColumns, MissingIndicator | P | **P** | Native filters / `is_null` expressions. |
| DataSnapshot, DatasetProfile (inspection) | — | **P** | Stats computed with Polars aggregations. |

### Imputation

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| SimpleImputer | **P** | **P** | mean/median/mode computed as Polars aggregations; apply is a `fill_null` expression — the fastest node in the [benchmark](../performance.md). |
| KNNImputer, IterativeImputer | P→sk | **P→sk→P** | sklearn imputer fitted/transformed on NumPy subsets; unrelated columns stay untouched in Polars. |

### Scaling — all native Polars

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler | **P** | **P** | Statistics (mean/std, min/max, quantiles) via Polars expressions; apply rebuilds columns from the stored constants. |

### Transformations

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| SimpleTransformation (log/sqrt/…) | P | **P** | Expression math. |
| GeneralTransformation | P→N | **P→N→P** | Power-family functions computed on extracted NumPy arrays. |
| PowerTransformer (Yeo-Johnson/Box-Cox) | P→sk | **P→N→P** | Fit via sklearn on NumPy; apply re-applies the stored lambdas with SciPy on NumPy, re-attached as Polars series. |

### Outlier handling

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| IQR, ZScore, Winsorize, ManualBounds | **P** | **P** | Bounds from Polars quantile/mean/std expressions; apply filters or clips with expressions. |
| EllipticEnvelope | P→sk | **P→N→P** | sklearn fit on the selected subset; apply builds a NumPy boolean mask and filters the Polars frame natively — unrelated column dtypes are preserved. |

### Encoding

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| OneHotEncoder, OrdinalEncoder, TargetEncoder | P→sk | **P→sk→P** | sklearn encoder fitted on NumPy from Polars; transform output (dense NumPy) re-attached via horizontal concat. No pandas hop. |
| HashEncoder | P | **P** | Hashing trick as expressions. |
| LabelEncoder | P→N | **P** | Fit reads the column as NumPy (sklearn `LabelEncoder`); apply is a Polars mapping expression. |
| DummyEncoder | P | **P** | Native indicator expressions. |
| WOEEncoder | P* | **P** | *Fit aggregates in Polars but keys are normalized to match pandas `astype(str)` exactly, so artifacts stay interchangeable between engines. |

### Binning & feature engineering

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| GeneralBinning, CustomBinning, KBinsDiscretizer | P | **P** | `cut`-style binning via expressions. |
| FeatureMath, FeatureInteraction, FeatureGeneration (`FeatureGenerationNode` alias) | P | **P** | Expression arithmetic. |
| PolynomialFeatures (`PolynomialFeaturesNode` alias) | P | **P→N→P** | Cross-products computed on NumPy for speed. |
| DateFeatures, LagFeatures, RollingAggregate | — | **P** | Native datetime/`shift`/`rolling` expressions. |
| CorrelationThreshold | **P** | **P** | Correlation matrix from Polars; apply drops columns. |
| UnivariateSelection, ModelBasedSelection, VarianceThreshold, `feature_selection` dispatcher | P→sk | **P** | Fit scores via sklearn on NumPy; apply is a native column drop. |

### Splitting

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| TrainTestSplitter, Split, FeatureTargetSplitter (`feature_target_split`) | — | **P→N→P** | `train_test_split` runs on row *indices* (`np.arange(n)`, incl. stratified and validation splits); the Polars frame is partitioned with native `gather` — rows and dtypes untouched. |

### Geo & text

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| GeoDistance | — | **P** | Haversine as expressions. |
| H3Index | — | **P→N→P** | Coordinates pulled as NumPy, `h3` called per row, result attached as a String column. |
| Tokenizer | — | **P→Py→P** | Text columns pulled as Python strings, sklearn analyzer applied, token columns attached back. |
| CountVectorizer (`count_vectorizer`), TfidfVectorizer (`tfidf_vectorizer`), HashingVectorizer (`hashing_vectorizer`) | P→Py→sk | **P→Py→sk→P** | Text joined to `list[str]`, sklearn fit/transform, dense matrix rebuilt with `pl.from_numpy` and `hstack`. Non-String text columns fall back to the pandas path for `astype(str)` parity. |
| SentenceEmbedder (`sentence_embedder`) | P→Py | **P→Py→P** | sentence-transformers encodes Python strings; embedding matrix re-attached from NumPy. |

## Models & training

Every model node crosses the boundary the same way (see explainer above):
fit extracts NumPy from the configured engine's frame, the library trains,
and predictions come back as columns in the configured engine's frame type.
**This is the complete list of registered model nodes** — the path is
identical for all of them:

| Family | Nodes (registry IDs) | Fit | Predict |
| :--- | :--- | :--- | :--- |
| Linear | `linear_regression`, `ridge_regression`, `lasso_regression`, `elasticnet_regression`, `logistic_regression`, `sgd_classifier` | **P→sk** | **P→sk→P** |
| Trees | `decision_tree_classifier`, `decision_tree_regressor` | **P→sk** | **P→sk→P** |
| Forests | `random_forest_classifier`, `random_forest_regressor`, `extra_trees_classifier`, `extra_trees_regressor` | **P→sk** | **P→sk→P** |
| Boosting (sklearn) | `gradient_boosting_classifier`, `gradient_boosting_regressor`, `hist_gradient_boosting_classifier`, `hist_gradient_boosting_regressor`, `adaboost_classifier`, `adaboost_regressor` | **P→sk** | **P→sk→P** |
| Boosting (libraries) | `xgboost_classifier`, `xgboost_regressor`, `lgbm_classifier`, `lgbm_regressor` | **P→N→lib** | **P→N→lib→P** |
| Naive Bayes | `gaussian_nb`, `multinomial_nb`, `bernoulli_nb` | **P→sk** | **P→sk→P** |
| SVM | `svc`, `svr` | **P→sk** | **P→sk→P** |
| Neighbors | `k_neighbors_classifier`, `k_neighbors_regressor` | **P→sk** | **P→sk→P** |
| Calibration | `calibrated_classifier` | **P→sk** | **P→sk→P** |
| Clustering | `kmeans`, `minibatch_kmeans`, `birch`, `gaussian_mixture` | **P→sk** | labels: **P→sk→P** |
| Meta-ensembles | `stacking_classifier`, `stacking_regressor`, `voting_classifier`, `voting_regressor` | **P→sk** (base learners on the same bridge, per fold) | **P→sk→P** |

Supporting machinery uses the same bridge:

| Stage | Path | Notes |
| :--- | :--- | :--- |
| Hyperparameter tuning | **P→sk** | One bridge crossing per candidate/fold. |
| Cross-validation | **P→sk** | Same bridge per fold. |
| SHAP explanations | P→pd→P | SHAP internals are pandas-based; converted at the boundary, result handed back in the engine's frame type. |

## Deliberate pandas islands

These stay on pandas **by design**, whatever `SKYULF_ENGINE` is set to:

| Area | Why |
| :--- | :--- |
| **Resampling** (Oversampling / Undersampling) | imblearn's SMOTE & co. are pandas/NumPy-bound; the node converts in, resamples, and hands the configured engine's frame type back. |
| **Matplotlib visualizations** (e.g. scatter-matrix in the visualizer) | matplotlib's DataFrame API expects pandas. |
| **great_expectations profiling** (`profiling/expect.py`) | The library consumes pandas. |

(Model fits are **not** a pandas island — they cross to NumPy directly, see
the boundary explainer above.)

## Boundary services (convert at the edge, hand the engine's type back)

These internals are pandas-based but expose dual-engine behavior: they accept
the configured engine's frame, convert at their boundary, and return the
configured engine's frame type — so a `SKYULF_ENGINE=polars` pipeline stays
Polars end to end:

- Merge logic (`pipeline` merge internals)
- SHAP explanations
- Data-preview statistics
- Drift reference normalization & the `/monitoring` reference-data consumer
- Clustering evaluation (has a native Polars path alongside the pandas one)

## How to verify

- The [performance benchmarks](../performance.md) include a **parity check**:
  every benchmarked node is run on both engines and outputs are compared
  value-for-value before timings are reported.
- The backend suite passes under both `SKYULF_ENGINE=polars` and
  `SKYULF_ENGINE=pandas`.
- Dtype fidelity: Phase 4 of the Polars migration removed all whole-frame
  `to_pandas()` + `from_pandas()` round-trips, which silently upcast nullable
  `Int64` columns to `Float64`. Red-green dtype-preservation tests guard every
  formerly round-tripping node.
