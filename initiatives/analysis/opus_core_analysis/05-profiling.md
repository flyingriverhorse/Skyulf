# Skyulf Deep Audit (Opus) — Profiling: analyzer, drift & visualisation

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

---

## Profiling — analyzer, statistics, drift, expectations, visualisation

### OC-39
### 🟠 High — NaN-bearing numeric columns publish `nan` stats and leak non-finite JSON

**Files:** `profiling/analyzer.py:215-224`, `backend/eda/tasks.py:221-222`

NaN is counted as missing, but the aggregations still run on the raw float
column, so mean/std/variance/skew/kurtosis all become `nan`:

```text
EDAAnalyzer([1.0, 2.0, nan, 4.0]).x stats:
  {'mean': nan, 'std': nan, 'variance': nan, 'skewness': nan}
pandas skipna equivalent:
  {'mean': 2.3333, 'std': 1.5275, 'skew': 0.9352}
```

The backend then persists `profile.model_dump(mode="json")` — which still
contains Python `nan`.

**Impact:** A single NaN blanks the entire statistics panel, and strict JSON
consumers or DB JSON encoders can reject the payload outright.

**Fix:** Normalise float NaN to null before aggregation (`fill_nan(None)`), and
sanitize `DatasetProfile` with a shared finite-float normaliser before dump.

---

### OC-40
### 🟠 High — PCA/clustering "mean imputation" actually replaces NaN with `0.0`

**File:** `profiling/_analyzer/multivariate.py:46-60`

`_impute_matrix()` claims mean imputation but only fills **polars nulls**. Float
NaNs survive into NumPy and are then zeroed by `np.nan_to_num`:

```text
Xdf = {'a': [10.0, 20.0, nan], 'b': [1.0, 2.0, 3.0]}
_impute_matrix →  [[10.0,1.0], [20.0,2.0], [0.0,3.0]]
expected mean-imputed a → [10.0, 20.0, 15.0]
```

**Impact:** Every PCA projection, component loading, cluster centre, and cluster
assignment is wrong whenever the data contains NaN. `0.0` is not a neutral value
for unscaled data — it is an extreme outlier.

**Fix:** Mirror the sibling `_impute_matrix_drop_empty()`, which already does
this correctly (cast → `fill_nan(None)` → fill nulls with column means).

---

### OC-41
### 🟠 High — Quartiles use nearest-rank, not linear interpolation

**Files:** `profiling/analyzer.py:221-222`, `_analyzer/target.py:154-156`

`pl.Expr.quantile()` is called without `interpolation=`, so polars defaults to
nearest-rank. Pandas/NumPy default to linear:

```text
data = [1, 2, 3, 10]
analyzer q25/q75:  2.0  / 3.0
pandas   q25/q75:  1.75 / 4.75
```

**Impact:** Boxplot hinges and IQR outlier fences are materially wrong on
small/medium datasets, and disagree with what a user computes in pandas.

**Fix:** Pass `interpolation="linear"` consistently.

---

### OC-42
### 🟠 High — Skewness/kurtosis use biased estimators, breaking a hardcoded threshold

**Files:** `profiling/analyzer.py:223-224`, `_analyzer/recommendations.py:66-78`

Polars `skew()`/`kurtosis()` default to the biased (SciPy `bias=True`)
estimators; pandas reports bias-corrected values:

```text
data = [1, 2, 3, 10]
analyzer:         skew 1.0182  kurt -0.7696
pandas:           skew 1.7636  kurt  3.2280
scipy bias=True:  skew 1.0182  kurt -0.7696
scipy bias=False: skew 1.7636  kurt  3.2280
```

The recommendation engine applies a hardcoded `abs(skewness) > 1.5` rule to the
**biased** value, so this example (pandas skew 1.76, clearly skewed) is **not**
flagged.

**Fix:** Pick an explicit public convention. For pandas parity use bias-corrected
estimators; otherwise rename the fields and re-tune the threshold.

---

### OC-43
### 🟠 High — Correlation drops valid columns/rows instead of a defined missing policy

**File:** `profiling/correlations.py:41-44,100-110`

Two compounding problems: a column containing any NaN gets `std() == nan`, fails
the `> 1e-9` check, and is discarded **as if constant**; and the matrix uses
listwise deletion across all selected columns, whereas pandas is pairwise.

```text
x=[1,2,nan,4], y=[1,2,3,4]
pandas corr_xy:          1.0
calculate_correlations:  None

sparse 3-col frame — complete rows after drop_nulls: 0
pandas pairwise corr: all 1.0
calculate_correlations: None
```

**Impact:** The correlation panel and leakage hints vanish entirely on ordinary
partially-missing data.

**Fix:** Convert NaN → null, then compute pairwise with a documented minimum
overlap count.

---

### OC-44
### 🟠 High — Wasserstein drift thresholds a normalized value but reports the raw one

**Files:** `profiling/drift.py:181-195`,
`frontend/.../drift/_hooks/useDriftReport.ts:77-78`

The backend decides drift with `norm_wd = wd / std_ref` but serializes
`value=float(wd)` alongside the **normalized** threshold. The frontend then
re-applies the threshold to the raw value:

```text
reference_std=1004.771  raw_wd=50.000  normalized_wd=0.049763  threshold=0.1
backend has_drift=False        value > threshold = True
```

**Impact:** A report shows `wasserstein_distance=50.0, threshold=0.1,
has_drift=false` — self-contradictory on its face. Touching the threshold slider
flips the same column to "drifted" without any data changing.

**Fix:** Emit both `raw_value` and `normalized_value`, and have the frontend
compare the normalized field.

---

### OC-45
### 🟠 High — Schema drift is computed but never counted or rendered

**Files:** `profiling/drift.py:76-98`, `frontend/.../drift/DriftTable.tsx:281-288`

`missing_columns` and `new_columns` are computed, but `drifted_columns_count`
counts only per-common-column metric drift, and the UI never references either
field:

```text
missing_columns = ['dropped']
new_columns     = ['new_col']
drifted_columns_count = 0
drift_detected = False
UI empty state: "No drifted columns found — all features are stable."
```

**Impact:** Dropping a training feature in production — one of the most serious
drift events possible — is reported as "all features are stable".

**Fix:** Add first-class schema-drift records; count and render them.

---

### OC-46
### 🟠 High — Non-finite floats enter public profile payloads and break strict JSON

**File:** `profiling/schemas.py:7-17,263-302`

Public schema fields are plain `float`, so `NaN`/`Infinity` pass validation:

```text
json.dumps(profile.model_dump(mode='json'), allow_nan=False)
→ ValueError: Out of range float values are not JSON compliant: inf
```

**Fix:** Sanitize all profile floats to finite values (or `None`) before model
construction. Add a `json.dumps(..., allow_nan=False)` test.

---

### OC-47
### 🟡 Medium — Common-column dtype drift can silently disappear

**File:** `profiling/drift.py:136-153`

A reference numeric column that becomes non-numeric in production is cast with
`strict=False`; all values become null, the column returns `None`, and it is
omitted from `column_drifts`:

```text
reference: a=[1,2,3]      current: a=['x','y','z']
missing=[]  new=[]  column_drifts={}  drifted_count=0
```

**Fix:** Compare dtypes before metric calculation and emit an explicit
`type_drift` alert when a cast fails.

---

### OC-48
### 🟡 Medium — Expectations pass vacuously on empty frames

**File:** `profiling/expect.py:92-209`

`expect_no_nulls`, `expect_value_range`, and `expect_unique` all **pass** on an
empty frame.

**Impact:** A failed ingestion producing zero rows sails through the data-quality
gate.

**Fix:** Add `expect_non_empty`, or default `allow_empty=False`.

---

### OC-49
### 🟡 Medium — Valid partially-unlabelled PCA payloads crash plotting

**Files:** `profiling/schemas.py:153-157`, `visualizer.py:716-737`

`PCAPoint.label` is optional, but `_pca_color_values()` filters out `None`,
producing a colour vector shorter than `x`/`y`:

```text
pca labels=[None,'a','b']
ValueError: 'c' argument has 2 elements, inconsistent with 'x' and 'y' with size 3
```

**Fix:** Emit one colour per point using a sentinel for missing labels.

---

### OC-50
### 🟡 Medium — Binary targets miss class-balance advice or flip to regression by sample size

**Files:** `_analyzer/recommendations.py:147-152`, `_analyzer/_utils.py:39-42`

Balance recommendations accept only `"Categorical"`, excluding real Boolean
targets. And integer 0/1 targets are typed by the ratio
`n_unique / row_count < 0.05`, so the **same target** changes task type with
sample size:

```text
bool target 95 False / 5 True  → dtype Boolean, target recs []
int [0,1]*10  n=20  → Numeric,    task Regression,     balance_recs []
int [0,1]*50  n=100 → Categorical, task Classification, balance_recs [Resample]
```

**Impact:** Severe imbalance goes unreported, and rule discovery can train a
*regressor* for a classification target on small data.

**Fix:** Treat Boolean as categorical; detect binary integer targets explicitly
when `target_col` is supplied, independent of row count.

---

### OC-51
### 🟡 Medium — Transform advice can be invalid and self-contradictory

**File:** `_analyzer/recommendations.py:66-78,129-139`

High skew always suggests "Log or Box-Cox" without checking `min`, `zeros_count`,
or `negatives_count` — both are invalid for non-positive data. Worse, the same
profile also reports it is ready for modeling, because `Transform` is not
counted as an issue:

```text
x = [-100,1,2,3,4,5,6,7]   skew=-2.2544  min=-100  negatives=1
recs = ["Apply Log or Box-Cox transformation to 'x'.",
        "No missing values or constant columns found. Data is ready for modeling!"]
```

**Fix:** Recommend Yeo-Johnson for non-positive data; count any Transform /
Encode / Resample recommendation as "not fully ready".

---

### OC-52
### ⚪ Low — Categorical colour mapping is process-nondeterministic

**File:** `profiling/visualizer.py:710-713`

`_label_color_map()` builds labels from a `set`:

```text
PYTHONHASHSEED=1: {'gold':0, 'silver':1, 'bronze':2}
PYTHONHASHSEED=2: {'bronze':0, 'gold':1, 'silver':2}
```

**Fix:** Use `sorted(...)` or `dict.fromkeys(...)` to preserve first-seen order.

---
