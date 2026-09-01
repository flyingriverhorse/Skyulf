# Skyulf Deep Audit (Opus) — Encoding, cleaning, imputation & scaling

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

---

## Encoding, cleaning, imputation, scaling, drop/missing

### OC-12
### 🔴 Critical — Row-dropping desyncs `X` and `y` on non-unique pandas indexes

**Files:** `preprocessing/drop_and_missing/drop_rows.py:60-67`,
`preprocessing/drop_and_missing/deduplicate.py:44-47`

Both pandas paths filter the target with `y.loc[X_clean.index]`. With duplicate
index labels, `.loc` returns **all rows matching each label**, not the
corresponding positions. `y` can come back longer than `X`:

```text
dropmissing  X index [0, 1]              shape (2, 2)
dropmissing  y index [0, 0, 1]  values ['row0','row1','row2']  len 3

dedup        X index [0, 1]              shape (2, 2)
dedup        y index [0, 0, 1]  values ['row0','row1','row2']  len 3
```

The polars paths do this correctly by tracking row positions.

**Impact:** This is the most severe finding in the audit. Downstream training
either fails on a length mismatch or — if lengths happen to align — **trains on
silently misaligned labels**, producing a model that is wrong with no error
anywhere. Non-unique indexes arise routinely from `concat`, `explode`, and
resampling upstream.

**Fix:** Mirror the polars implementation: capture positional row numbers before
dropping and select `y` with `.iloc`, never `.loc`. Add a regression test using
a duplicate-index frame.

---

### OC-13
### 🟠 High — Drop-Rows UI settings ignored; every canvas run becomes "drop any missing"

**Files:** `frontend/ml-canvas/src/core/utils/pipelineConverter.ts:249-253`,
`preprocessing/drop_and_missing/drop_rows.py:38-41,97-103`

The frontend sends `missing_threshold` and `drop_if_any_missing`. Python reads
only `subset`, `how`, and `threshold`. The UI default
`{drop_if_any_missing: false, missing_threshold: 50}` therefore fits as
`{how: "any", threshold: None}`:

```text
drop rows params: {'type':'drop_missing_rows','subset':None,'how':'any','threshold':None}
drop rows kept index: [0]
```

**Impact:** A user who sets "drop rows with more than 50% missing" gets "drop
rows with *any* missing value". On wide data this can delete almost the entire
dataset, and `y` is filtered to that unintended subset.

**Fix:** Translate in `pipelineConverter.ts`: `drop_if_any_missing === true →
how: "any"`; otherwise convert the percentage into an absolute `threshold`.
Better: add a backend `missing_threshold` percentage mode so the UI and core
speak the same language.

---

### OC-14
### 🟠 High — Iterative Imputer UI estimator choices silently fall back to BayesianRidge

**File:** `preprocessing/imputation/_common.py:103-111`

The UI emits `decision_tree`, `extra_trees`, `knn`. `_build_iterative_estimator()`
matches only `DecisionTree`, `ExtraTrees`, `KNeighbors`.

```text
iter estimator decision_tree -> BayesianRidge
iter estimator extra_trees   -> BayesianRidge
iter estimator knn           -> BayesianRidge
iter estimator DecisionTree  -> DecisionTreeRegressor
```

**Impact:** Every MICE imputation configured from the canvas uses Bayesian Ridge
regardless of what the user selected. Results are plausible, so nobody notices.

**Fix:** Normalise aliases case-insensitively and accept both spellings.

---

### OC-15
### 🟠 High — MinMax/Robust scaler range controls in the UI are ignored

**Files:** `preprocessing/scaling/minmax.py:96-100`, `robust.py:116-123`

The UI stores `feature_range_min`/`feature_range_max` and
`quantile_range_min`/`quantile_range_max`. Python expects `feature_range` and
`quantile_range` tuples. Given UI config `feature_range_min=-1, feature_range_max=1`
and `quantile_range_min=10, quantile_range_max=90`:

```text
minmax feature_range: (0, 1)      # user asked for (-1, 1)
robust quantile_range: (25.0, 75.0)  # user asked for (10, 90)
```

**Impact:** Scaling output is silently wrong whenever a user configures a
non-default range — a common action for neural-network inputs needing `[-1, 1]`.

**Fix:** Map in `pipelineConverter.ts`:
`feature_range: [min ?? 0, max ?? 1]`, `quantile_range: [low ?? 25, high ?? 75]`.

---

### OC-16
### 🟠 High — KNN/Iterative imputers crash on all-missing fitted columns

**Files:** `preprocessing/imputation/knn.py:64-76`, `iterative.py:68-84`,
`_common.py:73-99`

The fit artifact retains every requested column, but sklearn drops
all-missing features during `transform()`. `_sklearn_transform_subset()` assumes
the transformed width equals `len(cols)` and writes by index:

```text
KNNImputerCalculator       pd ValueError  Columns must be same length as key
KNNImputerCalculator       pl IndexError  index 1 out of bounds for axis 1 with size 1
IterativeImputerCalculator pd ValueError  Columns must be same length as key
IterativeImputerCalculator pl IndexError  index 1 out of bounds for axis 1 with size 1
```

**Impact:** An all-null column — extremely common in real data — makes
multivariate imputation unusable instead of skipping the column.

**Fix:** Pass `keep_empty_features=True` where sklearn supports it, or drop
empty columns from the artifact at fit time and emit a node warning.

---

### OC-17
### 🟠 High — SimpleImputer polars mean/median crashes on all-null columns

**File:** `preprocessing/imputation/_common.py:32-37,54-56`

Polars `mean()`/`median()` over an all-null column yields `None`, which is stored
in `fill_values`; apply then calls `fill_null(None)` and raises. Pandas skips
the column instead:

```text
pd {'fill_values': {'b': 1.5}, 'columns': ['b'], ...}
pl {'fill_values': {'a': None, 'b': 1.5}, 'columns': ['a','b'], ...}
pl ValueError: must specify either a fill `value` or `strategy`
```

**Impact:** Artifacts diverge between engines, and polars pipelines fail at apply
time on sparse numeric input.

**Fix:** Filter out columns whose computed statistic is `None`/`NaN`, matching
the pandas behaviour, or adopt an explicit documented `keep_empty_features`
policy for both engines.

---

### OC-18
### 🟡 Medium — One-hot/dummy generated names can collide with existing columns

**Files:** `preprocessing/encoding/one_hot.py:68-92`, `dummy.py:76-99`

Generated names such as `city_a` are never checked against existing columns.
Each engine fails differently:

```text
onehot pandas columns: ['city_a','city_a','city_b']   duplicate? True
onehot polars: DuplicateError  column 'city_a' has more than one occurrence
dummy  pandas columns: ['city_a','city_a','city_b']   duplicate? True
dummy  polars columns: ['city_a','city_b']            # silently overwrote the original
```

**Impact:** Three different wrong behaviours for one input — duplicate labels,
a hard crash, and silent data loss.

**Fix:** Precompute output names at fit time; reject or deterministically rename
collisions. Apply the same policy to `MissingIndicator` and multiclass
`TargetEncoder`.

---

### OC-19
### 🟡 Medium — Alias Replacement exposes a `punctuation` mode that does nothing

**File:** `preprocessing/cleaning/alias.py:21-28,45-52,100-114`

The UI offers `mode: 'punctuation'` and describes it as removing punctuation.
Python stores `alias_type: 'punctuation'`, `_resolve_alias_mapping()` returns
`{}`, and apply coalesces every non-match back to the original value:

```text
alias params: {'alias_type': 'punctuation', 'custom_map': {}}
alias out: ['A.B!', 'yes']    # unchanged
```

**Fix:** Either implement the mode or remove the UI option and point users to
`TextCleaning.remove_special`.

---

### OC-20
### 🟡 Medium — Value Replacement's "empty columns = all columns" UI promise is false

**Files:** `frontend/.../ValueReplacementSettings.tsx:175-182`,
`preprocessing/cleaning/value_replacement.py:163-180,208-221`

The UI states that leaving the column selection empty applies replacements to
all compatible columns. Python calls `resolve_columns(X, config)` with no
default selector, gets `[]`, and both apply paths return unchanged data:

```text
value repl params: {'columns': [], 'mapping': {-999: 0}, ...}
value repl out:    {'a': [-999, 1], 'b': [-999, 2]}    # unchanged
```

**Fix:** Align the contract — either require ≥1 column in the UI, or make the
node treat empty `columns` as "all compatible columns".

---

### OC-21
### 🟡 Medium — WOE additive smoothing is not normalized over categories

**File:** `preprocessing/encoding/woe.py:130-145`

`_column_woe()` adds `reg` to each category count but divides by
`total_pos + reg` / `total_neg + reg` instead of `total + reg * n_categories`.
The results are therefore not probability distributions over bins:

```text
woe actual:              {'a': -0.788457, 'b': 0.820981, 'c': -0.277632}
woe laplace-normalized:  {'a': -0.619039, 'b': 0.990399, 'c': -0.108214}
```

**Impact:** WOE values and IV feature-importance scores are shifted for
imbalanced targets with more than two categories.

**Fix:** Use denominators `total_pos + reg * n_bins` and `total_neg + reg * n_bins`.

---

### OC-22
### ⚪ Low — `TargetEncoder.infer_output_schema` checks an impossible value

**File:** `preprocessing/encoding/target.py:340-360`

```python
if config.get("target_type", "auto") not in ("binary", "regression"):
    return None
```

The frontend and sklearn both use `"continuous"` for regression target
encoding; `"regression"` is never produced. Valid continuous configs therefore
always lose schema prediction.

**Fix:** Replace `"regression"` with `"continuous"`.

---
