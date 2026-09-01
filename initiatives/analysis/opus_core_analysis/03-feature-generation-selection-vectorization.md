# Skyulf Deep Audit (Opus) — Feature generation, selection & vectorization

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

---

## Feature generation, selection, vectorization, transformations

### OC-23
### 🟠 High — Polars `ratio` flips the sign of near-zero negative denominators

**File:** `preprocessing/feature_generation/_polars_ops.py:97-112`

The pandas path uses `_safe_divide`, which preserves the denominator's sign when
clamping to epsilon. The polars `ratio` path always substitutes positive
`epsilon`. (Polars `divide` gets this right — `ratio` is the outlier.)

```text
ratio pandas = [0.1, -1999999999.9999998, 0.0]
ratio polars = [0.1,  1999999999.9999998, 0.0]
```

**Impact:** The same pipeline produces **opposite-signed** features depending on
the engine.

**Fix:** Reuse the signed-epsilon logic from `_polars_divide()` in `_polars_ratio()`.

---

### OC-24
### 🟠 High — Polars group aggregates treat null group keys differently from pandas

**File:** `preprocessing/feature_generation/_polars_ops.py:222-234`

Pandas `groupby(...).transform()` drops null group keys, returning `NaN`. Polars
`over()` groups nulls together and emits real aggregates.

```text
group_count pandas = [1.0, nan, 1.0]      polars = [1, 1, 1]
group_mean  pandas = [1.0, nan, 1.0]      polars = [1.0, 2.0, 1.0]
```

**Fix:** Choose a contract and enforce it. For pandas parity, wrap the polars
output in `when(col(group_col).is_null()).then(None).otherwise(...)`.

---

### OC-25
### 🟠 High — RFE "K" chosen in the UI is ignored by the backend

**File:** `preprocessing/feature_selection/_common.py:236-240`

The UI exposes `k`; sklearn's `RFE` takes `n_features_to_select`. Nothing maps
between them, so RFE silently uses its default (half the features):

```text
config: {'method':'rfe', 'k':2, 6 input features}
selected: ['c','d','e']   count 3      # user asked for 2
```

**Fix:** Map `k → n_features_to_select` in the converter, or accept `k` as a
backend alias.

---

### OC-26
### 🟠 High — `HashingVectorizer` UI "none" norm is an invalid sklearn value

**File:** `preprocessing/vectorization/hashing_vectorizer.py:59`

```python
norm = config.get("norm", "l2") or None
```

The UI sends the **string** `"none"`, which is truthy, so it reaches sklearn:

```text
InvalidParameterError: The 'norm' parameter of normalize must be a str among
{'l1','l2','max'}. Got 'none' instead.
```

**Impact:** Selecting "None" normalization in the canvas is a guaranteed runtime
failure. Unlike the other frontend-sync bugs this one is at least loud.

**Fix:** Normalise the string `"none"` to `None` in the backend (and have the UI
store `null`).

---

### OC-27
### 🟠 High — `GeneralTransformation` ignores the UI `standardize` toggle

**File:** `preprocessing/transformations/general.py:34-39,138-139`

The UI exposes `standardize` for Yeo-Johnson/Box-Cox and the converter forwards
it, but the backend always fits and applies `PowerTransformer(standardize=True)`:

```text
config standardize=False
output mean -0.0  std 1.0     # standardized anyway
```

The dedicated `PowerTransformer` node honours `standardize=False` correctly —
the bug is confined to `GeneralTransformation`.

**Fix:** Store `standardize` in the artifact and pass it to both fit and the
apply-time reconstruction.

---

### OC-28
### 🟠 High — Box-Cox transform failures silently return untransformed data

**File:** `preprocessing/transformations/power.py:97-104`

Fit selects Box-Cox columns using **training** positivity. If transform-time data
contains zero or negative values, sklearn raises, the applier logs, and returns
the **original dataframe unchanged**:

```text
train a=[1,2,3], test a=[0,4,5]
PowerTransformer (Pandas) application failed: Box-Cox requires strictly positive data
output: [0.0, 4.0, 5.0]      # untransformed
```

**Impact:** Classic train/serve skew. The model was trained on Box-Cox-transformed
features and is scored on raw ones, with only a log line to indicate it.

**Fix:** Validate transform-time positivity and raise a clear error, or null out
the invalid rows while still transforming unaffected columns. Failing open is
the wrong default for a fitted transform.

---

### OC-29
### 🟡 Medium — `FeatureGeneration` advertises `polynomial` but silently skips it

**File:** `preprocessing/feature_generation/_common.py:24-31`

`FEATURE_MATH_ALLOWED_TYPES` includes `"polynomial"`, but neither engine's
handler dict implements it, so it no-ops without error.

**Fix:** Remove it from the allow-list, or route it to `PolynomialFeatures`.

---

### OC-30
### 🟡 Medium — Datetime extraction ignores the UI output name and overwrites collisions

**Files:** `feature_generation/_pandas_ops.py:173-184`, `_polars_ops.py:181-205`

The UI shows "Output Column Name" for every operation, but datetime extraction
always writes `{source}_{feature}` and never consults `output_column` or
`_resolve_output_col()`. Existing columns are overwritten even when
`allow_overwrite=False`:

```text
input columns: ['dt','dt_year'],  output_column='custom_year'
pandas: {'dt': [...], 'dt_year': [2024]}    # custom name ignored, dt_year clobbered
```

**Fix:** Support `output_prefix`/per-feature names with collision resolution, or
hide the output-name control for multi-output datetime operations.

---

### OC-31
### 🟡 Medium — Frontend wrongly requires a target for unsupervised CorrelationThreshold

**File:** `frontend/.../FeatureSelectionNode.tsx:564-566`

Validation requires `target_column` for every method except
`variance_threshold`, but `CorrelationThresholdCalculator.fit()` ignores `_y`
entirely and is unsupervised.

**Impact:** Valid pipelines are blocked in the canvas unless the user picks a
meaningless target. (Note this is the mirror image of the other findings — here
the frontend is *stricter* than the backend.)

**Fix:** Exempt `correlation_threshold` from the target requirement.

---

### OC-32
### 🟡 Medium — `VarianceThreshold` crashes when all candidates are constant

**File:** `preprocessing/feature_selection/variance.py:38-47`

```text
all_constant  ValueError: No feature in X meets the variance threshold 0.00000
all_null      ValueError: No feature in X meets the variance threshold 0.00000
```

**Impact:** "Remove all zero-variance columns" — a legitimate and common
request — aborts the pipeline exactly when it would be most useful.

**Fix:** Catch the sklearn no-feature `ValueError` and return
`selected_columns=[]` with `candidate_columns=cols`.

---

### OC-33
### 🟡 Medium — `FeatureInteraction` cannot generate single-column self-products

**File:** `preprocessing/feature_generation/interaction.py:173-178`

`_resolve_combinations()` supports self-products via
`combinations_with_replacement` when `interaction_only=False`, but `fit()` skips
everything unless `len(cols) >= degree`, so a single column with `degree=2`
produces nothing.

**Fix:** Only require `len(cols) >= degree` when `interaction_only=True`.

---

### OC-34
### 🟡 Medium — Count/TF-IDF vectorizers crash on empty or stop-word-only corpora

**Files:** `vectorization/count_vectorizer.py:79-80`, `tfidf_vectorizer.py:73-74`

```text
CountVectorizerCalculator.fit(pd.DataFrame({'txt': ['', None]}), {'columns': ['txt']})
ValueError: empty vocabulary; perhaps the documents only contain stop words
```

**Fix:** Catch the empty-vocabulary error and return an empty artifact with a
node warning.

---
