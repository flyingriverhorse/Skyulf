# 15 — Profiling analyzer leaves (`profiling/_analyzer/`)

**Scope:** all 15 modules of `skyulf/profiling/_analyzer/` (2,817 lines).

I flagged this domain earlier as under-reported — 9 of 11 leaf modules had
produced zero findings while the other 2 yielded 14, which was not credible. I
re-audited it myself. The suspicion was justified: the single most consequential
profiling bug in the codebase lives in `column.py`, and it had been missed.

New ids continue at **OC-110**.

| ID | Severity | Issue | Location |
| --- | --- | --- | --- |
| [OC-110](#oc-110) | 🟠 High | Small datasets misclassify obvious categorical columns as `Text`, which silently defeats task-type inference | `_analyzer/column.py:_dtype_to_semantic_bucket` |
| [OC-111](#oc-111) | 🟡 Medium | The high-cardinality encoding recommendation is unreachable for integer columns (mutually exclusive conditions) | `_analyzer/recommendations.py:86-92` |
| [OC-113](#oc-113) | 🟠 High | Near-perfect multicollinearity silently reports **VIF = 1.0** — a clean bill of health — because `max(1.0, …)` clamps numerical garbage | `_analyzer/numeric.py:_calculate_vif` |
| [OC-114](#oc-114) | 🟡 Medium | An all-null tracked column yields 30 `NaN` autocorrelation lags presented as real analysis (≥1000-row datasets only) | `_analyzer/temporal.py:167-191` |
| [OC-112](#oc-112) | ⚪ Low | `categorical.py` comment promises a missing-value marker; the code silently drops nulls instead | `_analyzer/categorical.py:22-30` |

---

<a id="oc-110"></a>
### OC-110
### 🟠 High — Small datasets misclassify categorical columns as `Text`, defeating task-type inference

**File:** `skyulf-core/skyulf/profiling/_analyzer/column.py`, function
`_dtype_to_semantic_bucket`

The semantic bucketer treats integers and strings asymmetrically:

```python
if dtype in _INT_DTYPES:
    return "Categorical" if (ratio < 0.05 and n_unique < 20) else "Numeric"
...
if dtype in (pl.Utf8, pl.String):
    return "Categorical" if ratio < 0.05 else "Text"        # ratio ONLY
```

The integer branch has two conditions — a *ratio* test **and** an absolute
`n_unique` ceiling. The string branch has only the ratio test and **no absolute
escape hatch**. Since `ratio = n_unique / row_count`, a string column can only
ever be Categorical if the dataset has more than `n_unique / 0.05` rows. On small
frames that is wildly wrong:

```
STRING columns (Categorical if ratio < 0.05 else Text)
    50 rows,   3 uniques (ratio=0.060) -> Text          <-- 3 categories!
   100 rows,   5 uniques (ratio=0.050) -> Text
   100 rows,   6 uniques (ratio=0.060) -> Text
   100 rows,  10 uniques (ratio=0.100) -> Text
  2000 rows,  60 uniques (ratio=0.030) -> Categorical
```

A 50-row frame with a three-value column (`small`/`medium`/`large`) is bucketed
as free **Text**. Note also the boundary is exclusive, so a ratio of *exactly*
0.05 falls to `Text`.

**Why this is High, not cosmetic.** `Text` is excluded from essentially every
downstream analysis:

- **Task type is never inferred.** `analyzer.py:502` gates on
  `if target_type in ["Categorical", "Boolean", "Numeric"]`. A `Text` target
  falls through, so `rule_tree` stays `None` **and `final_task_type` is never
  set** — the profiler simply fails to determine whether the problem is
  classification or regression. For a small dataset with a string target of 3–10
  classes (an extremely common exploratory case) the EDA output is silently
  incomplete.
- No `CategoricalStats` are produced (`unique_count`, `top_k`, rare labels).
- Target association/interaction analysis skips the column (`analyzer.py:400`
  filters on `== "Categorical"`).
- No encoding recommendations are generated.

**Fix:** give the string branch the same absolute escape hatch the integer
branch already has, e.g.
`return "Categorical" if (ratio < 0.05 or n_unique <= 20) else "Text"`.
That preserves the existing behaviour on large frames (where the ratio rule
already dominates) and fixes the small-frame case. A `Text` target should
probably also be tolerated at `analyzer.py:502` rather than silently skipped.

---

<a id="oc-111"></a>
### OC-111
### 🟡 Medium — The high-cardinality encoding recommendation cannot fire for integer columns

**File:** `skyulf-core/skyulf/profiling/_analyzer/recommendations.py:86-92`

```python
if (
    profile.categorical_stats
    and profile.dtype == "Categorical"
    and profile.categorical_stats.unique_count > 50
):
    ... "Use Target Encoding or Hashing for '{col}' instead of One-Hot."
```

For an integer column, `dtype == "Categorical"` is only reachable when
`n_unique < 20` (see OC-110's snippet). Requiring `unique_count > 50`
simultaneously is therefore **unsatisfiable** — the two conditions are mutually
exclusive, and this recommendation is dead code for every integer column:

```
INT columns (Categorical if ratio<0.05 AND n_unique<20)
  1000 rows,  10 uniques -> Categorical     (10 > 50 is False)
  1000 rows,  25 uniques -> Numeric         (dtype check fails)
```

A 1,000-row frame with a 200-value integer ID column — a textbook
target-encoding candidate — is bucketed `Numeric` and never flagged.

The branch *is* reachable for strings (needs > 1,000 rows, since
`ratio < 0.05` with `n_unique > 50` implies `rows > 1000`) and for a true
`pl.Categorical` dtype, so this is not wholly dead — hence Medium, not a
correctness bug. But the integer case it most obviously targets can never
trigger.

**Fix:** drive the recommendation off `unique_count` and the raw dtype rather
than the semantic bucket, which has already discarded the information the rule
needs.

---

<a id="oc-112"></a>
### OC-112
### ⚪ Low — Comment promises a missing-value marker; the code drops nulls

**File:** `skyulf-core/skyulf/profiling/_analyzer/categorical.py:22-30`

```python
if value is None:
    # Polars' `value_counts` includes null as a real
    # category; render it as an actual missing marker
    # rather than the literal string "None", which would
    # otherwise look like (and be indistinguishable
    # from) a genuine category value of that name.
    continue
```

The comment describes rendering a marker. The code `continue`s — the null
category is **discarded entirely**. The stated goal (don't let nulls masquerade
as a literal `"None"` category) is met, but the implementation silently differs
from its own documentation, and the consequence is that `top_k` counts no longer
sum to the column's row count. For a column that is 70% null, the "top values"
list omits by far its largest group, with nothing in the payload indicating the
omission.

**Fix:** either update the comment to say nulls are excluded, or emit an
explicit sentinel entry (e.g. `{"value": None, "count": n, "is_null": true}`) so
consumers can render it deliberately. `missing_percentage` is already reported
separately, so this is presentational — hence Low.

<a id="oc-113"></a>
### OC-113
### 🟠 High — Near-perfect multicollinearity is reported as **VIF = 1.0** (no multicollinearity at all)

**File:** `skyulf/profiling/_analyzer/numeric.py:32-63` (`_calculate_vif`)

VIF is computed from the diagonal of the inverted correlation matrix, guarded by
a `LinAlgError` handler and a final clamp:

```python
try:
    inv_corr = np.linalg.inv(corr_matrix)
except np.linalg.LinAlgError:
    return dict.fromkeys(numeric_cols, 999.0)      # perfect collinearity
return {col: max(1.0, float(inv_corr[i, i])) for i, col in enumerate(numeric_cols)}
```

The guard only fires on **exact** singularity. Real data is almost never exactly
singular — it is *nearly* singular (a duplicated feature with rounding, cm vs
inches, a near-constant column). `np.linalg.inv` does not raise for those; it
returns numerical garbage, and the `max(1.0, …)` clamp then converts that garbage
into the safest possible answer.

**Executed repro** — three columns where `b = 2a + 1e-9·noise`, i.e. essentially
perfect multicollinearity:

```
perfectly collinear (b = 2a)          -> {'a': 999.0, 'b': 999.0, 'c': 999.0}   # correct
near-collinear (b = 2a + 1e-9 noise)  -> {'a': 1.0,   'b': 1.0,   'c': 1.0}     # WRONG
independent                           -> {'a': 1.01,  'b': 1.00,  'c': 1.01}    # correct
```

The middle row is the bug: the *most severe* realistic multicollinearity produces
a result indistinguishable from the *cleanest* possible data.

**Root cause**, isolated numerically:

```
condition number:        1.738e+16     (numerically singular)
smallest eigenvalue:    -6.003e-16     (negative — matrix is not positive definite)
raw inverse diagonal:   [-1.32e+22, -1.32e+22, -0.0]
after max(1.0, …) clamp: [1.0, 1.0, 1.0]
```

The inverse diagonal is **−1.3e+22** — catastrophic numerical failure. The clamp
exists because VIF is mathematically ≥ 1, so small negative rounding noise should
round up to 1.0. But it cannot distinguish "−1e−16, harmless rounding" from
"−1.3e+22, the inversion collapsed", and maps both to *no multicollinearity*.

**Why it matters — this is a false negative on a safety check.** The value is
fully user-reachable: `analyzer.py:563-564` feeds it to `_add_vif_alerts` (which
warns when `VIF > 5`), stores it at `:622` as `vif=vif_data`, and it is typed and
rendered in the frontend (`core/types/edaProfile.ts:131-132`,
`core/api/datasets.ts:250`). A user profiling a dataset with duplicated or
near-duplicated features is told, positively, that there is no multicollinearity
— exactly when the warning matters most. A silent all-clear on a diagnostic is
worse than no diagnostic.

**Fix:** test conditioning *before* inverting, rather than relying on `inv` to
raise. Either

```python
if np.linalg.cond(corr_matrix) > 1e10:          # or eigvalsh(...).min() <= tol
    return dict.fromkeys(numeric_cols, 999.0)
```

or use `np.linalg.pinv` and flag any diagonal entry that is negative or
non-finite instead of clamping it. The clamp should assert, not absorb: a
negative raw diagonal below a small tolerance is a *failure signal*, not a 1.0.

<a id="oc-114"></a>
### OC-114
### 🟡 Medium — An all-null tracked column produces 30 `NaN` autocorrelation lags, presented as real analysis

**File:** `skyulf/profiling/_analyzer/temporal.py:167-191` (`_compute_acf`)

`_compute_acf` fills missing values with the column mean before computing
autocorrelation:

```python
mask = np.isnan(series)
if mask.any():
    series[mask] = np.nanmean(series)     # <- NaN when EVERY value is NaN
```

When the tracked column is entirely null, `np.nanmean` returns `NaN` (emitting
`RuntimeWarning: Mean of empty slice`), so the "cleaned" series is still all-NaN.
`var` is then `NaN`, the `var == 0` guard is `False`, and 30 lags of `NaN`
correlation are appended and returned as `TimeSeriesAnalysis.autocorrelation`.

This is reachable whenever a numeric column is null across the whole resampling
window — a sensor that never reported, a column populated only for later rows, a
join that produced no matches.

**Executed repro** (all-null `sensor` column vs. a healthy control column, both
through the public `_analyze_timeseries`):

```
raw path (<1000 rows)  ALL-NULL col     entries=  0 non-finite=  0 lag1=None
raw path (<1000 rows)  CONTROL healthy  entries= 30 non-finite=  0 lag1=0.972
resampled (>=1000)     ALL-NULL col     entries= 30 non-finite= 30 lag1=nan
      strict JSON FAILS -> Out of range float values are not JSON compliant: nan
resampled (>=1000)     CONTROL healthy  entries= 30 non-finite=  0 lag1=0.979
```

The control column returns finite, correct autocorrelations on both paths, which
is what makes the all-NaN result trustworthy as a finding rather than a
mis-called probe.

**Only the resampled path is affected.** Below 1,000 rows `_build_raw_trend` is
used and the null rows never reach the ACF, so the bug is invisible on small test
fixtures and appears only on realistically sized data — which is likely why no
test caught it.

**The inconsistency is internal to this one module**, which is what makes it a
clear defect rather than a design choice — two of the three consumers of the same
`trend_df` already handle this:

| consumer | behaviour on all-null | |
| --- | --- | --- |
| `_trend_points_from_df:99-109` | explicitly skips rows whose values are all `None` | ✅ safe |
| `_compute_stationarity_test:193-216` | ADF raises, caught, returns `None` (`"ADF test failed: exog contains inf or nans"`) | ✅ fails safe |
| `_compute_acf:167-191` | emits 30 `NaN` lags | ❌ |

**Impact.** Not a crash: per the [OC-46 correction](./00-validation-log.md#oc-46)
the EDA router serialises with `orjson`, which coerces `NaN` to `null`. So the
user is shown 30 autocorrelation lags with empty values — *fabricated analysis
output* rather than an honest "not computable". Any consumer using stdlib `json`
with `allow_nan=False` would raise instead, as shown above.

**Fix:** bail out before the loop when the series has no finite values, mirroring
the sibling guards:

```python
finite = series[np.isfinite(series)]
if finite.size == 0:
    return acf_stats          # no data to correlate
series[mask] = finite.mean()
```

This also silences the `RuntimeWarning: Mean of empty slice` that the current code
emits on every such column.

---

<a id="final-core-sweep"></a>
### Final core sweep — the previously-unread `_analyzer/` submodules

Closing the ~1,000-line gap recorded in the [coverage table](./README.md#audit-coverage).
Findings: [OC-113](#oc-113) (`numeric.py`) and [OC-114](#oc-114) (`temporal.py`).
Everything else read in this sweep was verified sound:

- **`temporal.py` — the ACF estimator is correct.** `sum((y1-mean)*(y2-mean))/n/var`
  divides by `n` (not `n-lag`), which is the standard *biased* ACF estimator —
  the right choice, since it guarantees a positive semi-definite
  autocorrelation sequence. Verified against a sine wave: lags 1-3 =
  0.938 / 0.777 / 0.542, decaying correctly.
- **`temporal.py` — `_resample_interval` is sound.** Targets ~100 points via a
  monotonic `1s → 1m → 1h → 1d → 1w` ladder; degenerate ranges
  (`min_date == max_date`, duration 0) fall to `"1s"` without dividing by zero.
- **`decomposition.py` — the ratio normalisation guards division by zero**
  (`if total_val == 0 or total_val is None: ratio = 0.0`), and null values are
  coalesced to `0` on the way out.
- **`geo.py` — the bounding-box validator is a genuine safety net.** It rejects
  a column merely *named* `lat`/`lon` whose values fall outside `[-90, 90]` /
  `[-180, 180]`, logging which column failed and why. This correctly prevents
  an ID or percentage column from being reported as geospatial.
- **`rules.py` — decision-tree rule extraction handles categoricals correctly.**
  `_split_clauses` maps a numeric split threshold back to category *names*, and
  the invariant it depends on is established and documented at `:109`
  (`physical code i == cat_categories[col][i]`, built from
  `cat_series.cat.get_categories().to_list()`), so the index alignment is
  guaranteed rather than assumed.
- **`optuna.py` — the lazy loader and seeding are correct.** Load state lives on
  a single object guarded by a `threading.Lock` so concurrent tuning runs cannot
  double-import, and every sampler (`RandomSampler`, `CmaEsSampler`,
  `TPESampler`) is constructed with `seed=config.random_state`.

**A note on `geo.py` column detection** (not filed — a feature gap, not a bug):
`_infer_lat_column` matches only `latitude`/`lat` while `_infer_lon_column`
matches `longitude`/`lng`/`lon`/`long`. The asymmetry is harmless because the
common pairings all resolve, but neither matcher does substring matching, so
real-world prefixed names (`pickup_latitude` / `pickup_longitude`) silently
return no geospatial analysis at all.

---

## Checked and found sound

- **`recommendations.py` guard structure.** Every rule correctly null-checks its
  stats object before dereferencing (`profile.numeric_stats and …`), so a column
  missing a stats block cannot raise. The `skewness` guard also safely handles
  `NaN` (`abs(nan) > 1.5` is `False`).
- **`_run_normality_test` (`column.py:58-64`)** correctly refuses to run on
  degenerate input — it requires `len > 20` **and** `std > 1e-10`, avoiding the
  classic Shapiro-Wilk crash on constant columns, and switches to
  Kolmogorov-Smirnov above 5,000 samples where Shapiro-Wilk is unreliable.
- **`_get_semantic_type`'s `n_unique` skip (`column.py:36-43`)** is a genuine
  correctness fix, not just an optimisation: it avoids calling `n_unique()` on
  dtypes such as `Object` where polars raises. The comment explains exactly why.
- **`_add_high_null_alert`** uses a plain `> 5` percent threshold consistently.
- **`multivariate.py` sampling** is seeded (`seed=42`) and therefore
  reproducible across runs.

> **Note — the other `_analyzer` bug is already filed.**
> [OC-40](../opus_core_analysis.md#oc-40) (`multivariate.py::_impute_matrix`
> turning `NaN` into `0.0` rather than the mean) lives in this package. I
> re-verified it during this pass and pinned its root cause to polars' null/NaN
> distinction — see
> [00-validation-log.md](./00-validation-log.md#oc-40).
> **Merged, not re-filed.**

## Improvements

- **Unify the two semantic-type call sites behind property tests.** The
  docstring notes `_dtype_to_semantic_bucket` exists so the per-series and
  vectorized passes "never drift apart" — good design. Add a table-driven test
  over `(dtype, rows, n_unique)` asserting the expected bucket; OC-110 and
  OC-111 would both have surfaced immediately.
- **Make the cardinality thresholds named constants.** `0.05`, `20` and `50` are
  inline magic numbers spread across two files, which is precisely how the
  OC-111 contradiction arose unnoticed.
<a id="oc-148"></a>
### OC-148
### 🟡 Medium — PII detector flags ordinary numeric ID columns as "Email/Phone"; the guard comment's rationale is false

**File:** `profiling/_analyzer/text.py:107-128` (`_check_pii`),
called from `profiling/_analyzer/column.py:218` (Categorical) and `:267` (Text)

The phone heuristic is `^\+?[\d\s().-]{7,20}$` plus a ≥7-digit requirement,
justified in-code as:

```python
# Requires at least 7 digits so plain numeric IDs/years don't false-positive.
```

The digit floor excludes **years** but does nothing for IDs — most real
identifier columns have *more* than 7 digits, not fewer. Any 7–20 character
string of digits/separators matches, and because the check is
`any(... for val in sample)` over just 20 values, **one** matching row flags the
whole column.

**Measured** via `EDAAnalyzer._check_pii` (positive and negative controls
included, so the negatives prove the probe discriminates):

| column contents | flagged as PII |
|---|---|
| `a.b@x.com`, … *(control +)* | ✅ True — correct |
| `+1 (555) 123-4567`, … *(control +)* | ✅ True — correct |
| `1999`, `2001`, `2024` *(control −)* | ✅ False — correct |
| `1234.56`, … *(control −)* | ✅ False — correct |
| **`10000001`** (8-digit customer id) | ⛔ **True** — false positive |
| **`4820019`** (7-digit order ref) | ⛔ **True** — false positive |
| **`90210-1234`** (ZIP+4) | ⛔ **True** — false positive |

Numeric customer/order/account IDs are present in a large share of tabular
datasets, so this fires constantly. The alert text asserts the column contains
"Email/Phone" data — a false privacy claim that trains users to dismiss the
alert, which devalues the *true* positives it also produces.

**Fix:** require positive evidence rather than mere shape. A pure run of digits
with no separator, no `+`, and no grouping is an identifier, not a phone number;
real phone formats have either a leading `+`, or separators in plausible
positions, or a length of exactly 10–11 digits with grouping. Cheap improvement:
require *either* a `+` prefix *or* at least one separator character *and*
10–11 digits total. Also consider requiring ≥2 of the 20 sampled values to match
before alerting, so a single stray value can't flag a column. At minimum, correct
the comment — it currently asserts a guarantee the code does not provide.

---
