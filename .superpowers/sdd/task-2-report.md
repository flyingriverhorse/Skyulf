# Task 2 Report: Native Pearson and Spearman Fit Calculation

## Status
Completed.

## Scope Delivered
Added native Polars fit-time correlation calculation for eligible Pearson and Spearman paths on both raw `pl.DataFrame` and `SkyulfPolarsWrapper`, while preserving the existing artifact shape and apply-time output contract.

## RED Evidence
### Added failing tests
File: `skyulf-core/tests/test_feature_selection_gaps.py`
- `test_correlation_threshold_native_polars_matches_pandas_without_conversion`
- `test_correlation_threshold_apply_preserves_polars_audit_schema_and_order`

### RED command
```bash
source .venv/bin/activate && pytest skyulf-core/tests/test_feature_selection_gaps.py -k "native_polars_matches_pandas_without_conversion" -q
```
Result:
- `FAILED ... eligible Polars fit called to_pandas`
- 2 failed, 32 deselected

This proved the pre-change fit path converted eligible raw and wrapped Polars frames to Pandas.

## GREEN Evidence
### Implementation
File: `skyulf-core/skyulf/preprocessing/feature_selection/correlation.py`
- Added `_as_polars_frame()` for raw/wrapped Polars detection.
- Added `_fit_correlation_threshold_pandas()` to retain the legacy Pandas path unchanged.
- Added `_native_polars_correlation_eligible()` to gate native fitting to supported methods/types.
- Added `_polars_correlation_columns_to_drop()` to compute pairwise-complete upper-triangle Polars correlations.
- Updated `CorrelationThresholdCalculator.fit()` to use native Polars Pearson/Spearman when eligible and fall back to Pandas otherwise.

### GREEN command
```bash
source .venv/bin/activate && pytest skyulf-core/tests/test_feature_selection_gaps.py -k "native_polars_matches_pandas_without_conversion" -q
```
Result:
- 2 passed, 32 deselected

## Focused Verification
### Correlation coverage
```bash
source .venv/bin/activate && pytest skyulf-core/tests/test_feature_selection_gaps.py -k "correlation" -q
```
Result:
- 13 passed, 21 deselected

### Required Python static checks
```bash
source .venv/bin/activate && \
  ruff check . && \
  ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
  ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
```
Result:
- `ruff check .`: passed
- `ruff format --check ...`: passed (`569 files already formatted`)
- `ty check ...`: initially failed on the new native path typing, then passed after narrowing the Polars method literal and resolving columns through `SkyulfPolarsWrapper(frame)`.

## Changed Files
- `skyulf-core/skyulf/preprocessing/feature_selection/correlation.py`
- `skyulf-core/tests/test_feature_selection_gaps.py`

## Commit
- `96bb9bdec415f66e9fa58afb1a202a2f704f94ed` — `feat(skyulf-core): add native Polars correlation fitting`

## Self-Review
- Confirmed the new native path only activates for raw/wrapped Polars inputs, Pearson/Spearman methods, numeric/boolean dtypes, and numeric thresholds.
- Confirmed the fallback comment and retained Pandas helper preserve the legacy contract for unsupported methods like Kendall/callables.
- Confirmed the Pearson audit artifact remains `columns_to_drop == ["b", "c", "mostly_null"]` and apply preserves raw/wrapped Polars output columns/schema exactly.
- Confirmed only the intended task files were committed; `.superpowers/sdd/progress.md` remained dirty and unstaged.

## Concerns
None beyond the intentional fallback behavior retained for unsupported Polars correlation modes.

---

## Task 2 Reviewer Finding Fix

### RED
```bash
python - <<'PY'
import polars as pl
from skyulf.preprocessing.feature_selection.correlation import _native_polars_correlation_eligible
frame = pl.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 3.0, 4.0]})
print(_native_polars_correlation_eligible(frame, ['a', 'b'], 'pearson', True))
print(_native_polars_correlation_eligible(frame, ['a', 'b'], 'pearson', False))
PY
```
Pre-fix controller verification: both calls returned `True`.

### GREEN
```bash
cd /Users/BH7043/Skyulf && source .venv/bin/activate && pytest -q skyulf-core/tests/test_feature_selection_gaps.py -k correlation
```
Result: `15 passed, 21 deselected`.

```bash
cd /Users/BH7043/Skyulf && source .venv/bin/activate && python - <<'PY'
import polars as pl
from skyulf.preprocessing.feature_selection.correlation import _native_polars_correlation_eligible
frame = pl.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 3.0, 4.0]})
print(_native_polars_correlation_eligible(frame, ['a', 'b'], 'pearson', True))
print(_native_polars_correlation_eligible(frame, ['a', 'b'], 'pearson', False))
print(_native_polars_correlation_eligible(frame, ['a', 'b'], 'pearson', 0.95))
PY
```
Result: `False`, `False`, `True`.

### Commit
`f41f2a4ef9442ceed847f369bad22d2427d00bdf`

---

## Remaining Reviewer Finding Fix

### RED
Controller verification: `isinstance(np.int64(1), Real) is True`, but `_native_polars_correlation_eligible()` still rejected the threshold with `isinstance(threshold, (int, float))`.

### GREEN
```bash
cd /Users/BH7043/Skyulf && source .venv/bin/activate && pytest skyulf-core/tests/test_feature_selection_gaps.py -k correlation -q
```
Result: `16 passed, 21 deselected`.

```bash
cd /Users/BH7043/Skyulf && source .venv/bin/activate && ruff check . && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
```
Result: all checks passed.

### Commit
`e381e344bdc7bfe8a5bf25c9ad5838d0ab1735d6`
