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
| FeatureMath, FeatureInteraction, FeatureGeneration | P | **P** | Expression arithmetic. |
| PolynomialFeatures | P | **P→N→P** | Cross-products computed on NumPy for speed. |
| DateFeatures, LagFeatures, RollingAggregate | — | **P** | Native datetime/`shift`/`rolling` expressions. |
| CorrelationThreshold | **P** | **P** | Correlation matrix from Polars; apply drops columns. |
| UnivariateSelection, ModelBasedSelection, VarianceThreshold | P→sk | **P** | Fit scores via sklearn on NumPy; apply is a native column drop. |

### Splitting

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| TrainTestSplitter, Split, FeatureTargetSplitter | — | **P→N→P** | `train_test_split` runs on row *indices* (`np.arange(n)`, incl. stratified and validation splits); the Polars frame is partitioned with native `gather` — rows and dtypes untouched. |

### Geo & text

| Node | Fit | Apply | Notes |
| :--- | :--- | :--- | :--- |
| GeoDistance | — | **P** | Haversine as expressions. |
| H3Index | — | **P→N→P** | Coordinates pulled as NumPy, `h3` called per row, result attached as a String column. |
| Tokenizer | — | **P→Py→P** | Text columns pulled as Python strings, sklearn analyzer applied, token columns attached back. |
| CountVectorizer, TfidfVectorizer, HashingVectorizer | P→Py→sk | **P→Py→sk→P** | Text joined to `list[str]`, sklearn fit/transform, dense matrix rebuilt with `pl.from_numpy` and `hstack`. Non-String text columns fall back to the pandas path for `astype(str)` parity. |
| SentenceEmbedder | P→Py | **P→Py→P** | sentence-transformers encodes Python strings; embedding matrix re-attached from NumPy. |

## Models & training

| Stage | Path | Notes |
| :--- | :--- | :--- |
| fit (sklearn models) | **P→sk** | `SklearnBridge.to_sklearn` extracts NumPy straight from Polars — no pandas in between. |
| fit (XGBoost / LightGBM) | **P→N→lib** | Both libraries accept NumPy directly. |
| predict / predict_proba | **P→sk→P** | Predictions come back as columns in the configured engine's frame type. |
| Tuning, cross-validation | **P→sk** | Same bridge per fold. |

## Deliberate pandas islands

These stay on pandas **by design**, whatever `SKYULF_ENGINE` is set to:

| Area | Why |
| :--- | :--- |
| **Resampling** (Oversampling / Undersampling) | imblearn's SMOTE & co. are pandas/NumPy-bound; the node converts in, resamples, and hands the configured engine's frame type back. |
| **Matplotlib visualizations** (e.g. scatter-matrix in the visualizer) | matplotlib's DataFrame API expects pandas. |
| **great_expectations profiling** (`profiling/expect.py`) | The library consumes pandas. |
| **sklearn model fit boundaries** | covered above — NumPy bridge, not pandas. |

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
