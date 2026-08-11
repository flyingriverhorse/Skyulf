# Phase 12 — Confirmed Bugs Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 9 concrete, reproducible bugs confirmed in `initiatives/enterprise-readiness/2026-08-11-bug-hunt.md`, in priority order, each as an independent, fully tested change with no architectural prerequisites.

**Architecture:** No new subsystems. Each task is a surgical fix inside an existing file (calculator/applier pair, Zustand store, or router), guarded by a new or extended unit/integration test that reproduces the bug first (red), then passes after the fix (green).

**Tech Stack:** Python 3.11+ / pandas / polars / SQLAlchemy (async) / FastAPI (backend + skyulf-core); TypeScript / Zustand / React Flow (frontend `ml-canvas`).

## Global Constraints

- Python: target 3.11+, use existing type-hint style (`dict[str, Any]`, `X | None`), no new dependencies.
- Every skyulf-core fix must keep pandas/polars parity (both engines produce equivalent output) — this repo's existing convention per `test_time_series_nodes.py`.
- Every fix must be covered by a **new failing-first test** that reproduces the exact bug-hunt repro before the fix, then passes after.
- Do not touch code unrelated to the specific bug in each task — this is a bug-fix pass, not a refactor.
- Run `ruff check .` and `ruff format --check <touched files>` after each Python task; run `npx eslint <touched files>` and `npx tsc --noEmit -p .` (from `frontend/ml-canvas/`) after each TS task, per repo-wide lint gate.
- Follow the Backend/Core ↔ Frontend Sync Rule: none of these 9 fixes change a node's param schema, enum/allow-list, or output shape, so no frontend node-component cross-check is required — confirmed per-task below.

---

### Task 1: Fix Lag Features target (`y`) misalignment on sort/dropna

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/time_series/lag.py:38-73`
- Test: `skyulf-core/tests/test_time_series_nodes.py` (add new test function)

**Interfaces:**
- Consumes: existing `apply_dual_engine(X, params, _apply_polars, _apply_pandas)` dispatcher (unchanged signature), existing `sort_pandas(df, sort_by)` helper from `_common.py` (unchanged signature: `(pd.DataFrame, str | None) -> pd.DataFrame`).
- Produces: `_apply_polars(X, _y, params) -> tuple[Any, Any]` and `_apply_pandas(X, _y, params) -> tuple[Any, Any]` now both return a **reordered/filtered `y`** aligned row-for-row with the returned `X`, instead of echoing back the original `_y` untouched.

**Bug:** `_apply_pandas`/`_apply_polars` sort and optionally `dropna()`/`drop_nulls()` on `X`, but return the original `_y` unmodified — after a sort or dropna, `X` and `y` refer to different rows.

- [ ] **Step 1: Write the failing test**

Add to `skyulf-core/tests/test_time_series_nodes.py`:

```python
def test_lag_features_reorders_and_filters_y_with_x():
    """Bug-hunt finding #2: sort_by/drop_na must move y in lockstep with X."""
    X = pd.DataFrame({"t": [2, 1, 3], "x": [20, 10, 30]})
    y = pd.Series(["row-t2", "row-t1", "row-t3"])
    art = LagFeaturesCalculator().fit(
        X, {"columns": ["x"], "lags": [1], "sort_by": "t", "drop_na": True}
    )
    X_out, y_out = LagFeaturesApplier().apply((X, y), art)
    # After sorting by t: rows are t=1 (row-t1), t=2 (row-t2), t=3 (row-t3).
    # lag_1 of x is NaN for the first sorted row (t=1) -> dropped by drop_na.
    # Remaining rows, in order: t=2 (row-t2), t=3 (row-t3).
    assert X_out["t"].tolist() == [2, 3]
    assert list(y_out) == ["row-t2", "row-t3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py::test_lag_features_reorders_and_filters_y_with_x -v`
Expected: FAIL — `y_out` will still be `["row-t2", "row-t1", "row-t3"]` (length 3, wrong order), not `["row-t2", "row-t3"]`.

- [ ] **Step 3: Fix `_apply_pandas` and `_apply_polars` to carry `y` through the same reindex/filter**

Replace the pandas path in `skyulf-core/skyulf/preprocessing/time_series/lag.py`:

```python
def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    lags: list[int] = params.get("lags", [])
    group_by: list[str] | None = params.get("group_by") or None
    if not columns or not lags:
        return X, _y

    df = sort_pandas(X.copy(), params.get("sort_by"))
    y_out = _y.reindex(df.index) if _y is not None and hasattr(_y, "reindex") else _y
    for col in columns:
        if col in df.columns:
            _pandas_lag_column(df, col, lags, group_by)
    if params.get("drop_na"):
        keep_mask = df.notna().all(axis=1)
        df = df[keep_mask]
        if y_out is not None and hasattr(y_out, "loc"):
            y_out = y_out.loc[df.index]
    return df, y_out
```

Replace the polars path:

```python
def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    import polars as pl

    columns: list[str] = params.get("columns", [])
    lags: list[int] = params.get("lags", [])
    sort_by: str | None = params.get("sort_by")
    if not columns or not lags:
        return X, _y

    y_out = _y
    if sort_by and sort_by in X.columns:
        # Attach y as a temporary column so the sort permutation carries it
        # along, then split it back off — polars has no positional reindex.
        if y_out is not None and hasattr(y_out, "__len__") and len(y_out) == X.height:
            y_series = y_out if isinstance(y_out, pl.Series) else pl.Series("__y__", list(y_out))
            X_with_y = X.with_columns(y_series.alias("__y__"))
            X_with_y = X_with_y.sort(sort_by)
            y_out = X_with_y["__y__"]
            X_out = X_with_y.drop("__y__")
        else:
            X_out = X.sort(sort_by)
    else:
        X_out = X

    exprs = _polars_lag_exprs(columns, list(X_out.columns), lags, params.get("group_by") or None)
    if exprs:
        X_out = X_out.with_columns(exprs)
    if params.get("drop_na"):
        keep_mask = X_out.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        X_out = X_out.filter(keep_mask)
        if y_out is not None and hasattr(y_out, "filter"):
            y_out = y_out.filter(keep_mask)
    return X_out, y_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py::test_lag_features_reorders_and_filters_y_with_x -v`
Expected: PASS

- [ ] **Step 5: Run the full time-series test module to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py -v`
Expected: all tests PASS (existing `test_lag_features_parity_with_groups` and `test_lag_features_coerces_and_dedups_lags` must still pass unchanged, since they don't pass a `y`).

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/skyulf/preprocessing/time_series/lag.py skyulf-core/tests/test_time_series_nodes.py
git commit -m "fix(skyulf-core): keep y aligned with X in Lag Features sort/drop_na"
```

---

### Task 2: Fix Rolling Aggregate target (`y`) misalignment on sort

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/time_series/rolling.py:56-116`
- Test: `skyulf-core/tests/test_time_series_nodes.py` (add new test function)

**Interfaces:**
- Consumes: same `apply_dual_engine`/`sort_pandas` as Task 1.
- Produces: `_apply_pandas`/`_apply_polars` in `rolling.py` now return `y` reordered to match the sorted `X` (Rolling Aggregate has no `drop_na` param, so only the sort needs to be mirrored — simpler than Task 1).

**Bug:** Same class of bug as Task 1 but for Rolling Aggregate: `X` is sorted by `sort_by`, `y` is returned unsorted.

- [ ] **Step 1: Write the failing test**

Add to `skyulf-core/tests/test_time_series_nodes.py`:

```python
def test_rolling_aggregate_reorders_y_with_x():
    """Bug-hunt finding #3: sort_by must reorder y along with X."""
    X = pd.DataFrame({"t": [2, 1, 3], "x": [20.0, 10.0, 30.0]})
    y = pd.Series(["row-t2", "row-t1", "row-t3"])
    art = RollingAggregateCalculator().fit(
        X, {"columns": ["x"], "aggregations": ["mean"], "window": 2, "sort_by": "t"}
    )
    X_out, y_out = RollingAggregateApplier().apply((X, y), art)
    assert X_out["t"].tolist() == [1, 2, 3]
    assert list(y_out) == ["row-t1", "row-t2", "row-t3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py::test_rolling_aggregate_reorders_y_with_x -v`
Expected: FAIL — `y_out` will be `["row-t2", "row-t1", "row-t3"]`, not `["row-t1", "row-t2", "row-t3"]`.

- [ ] **Step 3: Fix `_apply_pandas` and `_apply_polars` in `rolling.py`**

Replace the pandas path:

```python
def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    aggs: list[str] = params.get("aggregations", [])
    if not columns or not aggs:
        return X, _y

    window = int(params.get("window", 3))
    min_periods = int(params.get("min_periods", 1))
    group_by: list[str] | None = params.get("group_by") or None
    df = sort_pandas(X.copy(), params.get("sort_by"))
    y_out = _y.reindex(df.index) if _y is not None and hasattr(_y, "reindex") else _y
    for col in columns:
        if col in df.columns:
            _pandas_roll_column(df, col, aggs, window, min_periods, group_by)
    return df, y_out
```

Replace the polars path (mirrors Task 1's `__y__` sidecar-column technique since Rolling Aggregate has no `drop_na`):

```python
def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    import polars as pl

    columns: list[str] = params.get("columns", [])
    aggs: list[str] = params.get("aggregations", [])
    sort_by: str | None = params.get("sort_by")
    if not columns or not aggs:
        return X, _y

    y_out = _y
    if sort_by and sort_by in X.columns:
        if y_out is not None and hasattr(y_out, "__len__") and len(y_out) == X.height:
            y_series = y_out if isinstance(y_out, pl.Series) else pl.Series("__y__", list(y_out))
            X_with_y = X.with_columns(y_series.alias("__y__")).sort(sort_by)
            y_out = X_with_y["__y__"]
            X_out = X_with_y.drop("__y__")
        else:
            X_out = X.sort(sort_by)
    else:
        X_out = X

    exprs = _polars_rolling_exprs(
        columns,
        list(X_out.columns),
        aggs,
        int(params.get("window", 3)),
        int(params.get("min_periods", 1)),
        params.get("group_by") or None,
    )
    if exprs:
        X_out = X_out.with_columns(exprs)
    return X_out, y_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py::test_rolling_aggregate_reorders_y_with_x -v`
Expected: PASS

- [ ] **Step 5: Run the full time-series test module to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_time_series_nodes.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/skyulf/preprocessing/time_series/rolling.py skyulf-core/tests/test_time_series_nodes.py
git commit -m "fix(skyulf-core): keep y aligned with X in Rolling Aggregate sort_by"
```

---

### Task 3: Fix Feature Selection facade's default method (`"variance"` → unknown)

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/feature_selection/facade.py:44-77`
- Test: create `skyulf-core/tests/test_feature_selection_facade.py`

**Interfaces:**
- Consumes: existing `_FS_CALCULATORS: dict[str, Callable[[], BaseCalculator]]` map (unchanged keys/values), existing `VarianceThresholdCalculator`/`VarianceThresholdApplier` from `.variance` (unchanged).
- Produces: `FeatureSelectionCalculator.fit` now resolves the node-metadata default `"method": "variance"` to `VarianceThresholdCalculator` instead of logging "Unknown feature selection method" and returning `{}`.

**Bug:** `@node_meta(params={"method": "variance", ...})` advertises `"variance"` as the default, but `_FS_CALCULATORS` only has the key `"variance_threshold"` — the node's own advertised default silently no-ops.

- [ ] **Step 1: Write the failing test**

Create `skyulf-core/tests/test_feature_selection_facade.py`:

```python
"""Regression test for bug-hunt finding #6: FeatureSelection's advertised
default ("variance") must resolve to a real calculator, not silently no-op."""

import pandas as pd

from skyulf.preprocessing.feature_selection.facade import (
    FeatureSelectionApplier,
    FeatureSelectionCalculator,
)


def test_default_method_variance_resolves_to_variance_threshold():
    X = pd.DataFrame({"constant": [1, 1, 1], "variable": [1, 2, 3]})
    artifact = FeatureSelectionCalculator().fit(
        X, {"method": "variance", "threshold": 0.0}
    )
    assert artifact != {}
    assert artifact.get("type") == "variance_threshold"
    result = FeatureSelectionApplier().apply(X, artifact)
    # variance_threshold with threshold=0.0 must drop the constant column.
    assert "constant" not in result.columns
    assert "variable" in result.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd skyulf-core && python -m pytest tests/test_feature_selection_facade.py -v`
Expected: FAIL — `artifact == {}`, so `artifact.get("type")` is `None`, not `"variance_threshold"`.

- [ ] **Step 3: Add the `"variance"` alias to `_FS_CALCULATORS` and fix the default**

Edit `skyulf-core/skyulf/preprocessing/feature_selection/facade.py`:

```python
_FS_CALCULATORS: dict[str, Callable[[], BaseCalculator]] = {
    "variance_threshold": VarianceThresholdCalculator,
    "variance": VarianceThresholdCalculator,  # alias for the node's advertised default
    "correlation_threshold": CorrelationThresholdCalculator,
    "select_k_best": UnivariateSelectionCalculator,
    "select_percentile": UnivariateSelectionCalculator,
    "generic_univariate_select": UnivariateSelectionCalculator,
    "select_fpr": UnivariateSelectionCalculator,
    "select_fdr": UnivariateSelectionCalculator,
    "select_fwe": UnivariateSelectionCalculator,
    "select_from_model": ModelBasedSelectionCalculator,
    "rfe": ModelBasedSelectionCalculator,
}
```

Also update `FeatureSelectionCalculator.fit`'s fallback default (currently `"select_k_best"`, which disagrees with the `@node_meta` default `"variance"`) to match the metadata:

```python
    def fit(
        self,
        df: Any,
        config: dict[str, Any],
    ) -> Mapping[str, Any]:
        method = config.get("method", "variance")
        ctor = _FS_CALCULATORS.get(method)
        if ctor is None:
            logger.warning(f"Unknown feature selection method: {method}")
            return {}
        return ctor().fit(df, config)
```

Note: also check `FeatureSelectionApplier.apply`'s `type_name` dispatch — `VarianceThresholdCalculator().fit(...)` must set `params["type"] = "variance_threshold"` (not `"variance"`) so the applier's existing `elif type_name == "variance_threshold"` branch still matches. Verify this by reading `skyulf-core/skyulf/preprocessing/feature_selection/variance.py`'s `fit` return dict before running the test — if it already hardcodes `"type": "variance_threshold"`, no applier change is needed (the alias only affects fit-time *calculator selection*, not the artifact's own `"type"` tag).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd skyulf-core && python -m pytest tests/test_feature_selection_facade.py -v`
Expected: PASS

- [ ] **Step 5: Run the full feature-selection test suite to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/ -k feature_selection -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/skyulf/preprocessing/feature_selection/facade.py skyulf-core/tests/test_feature_selection_facade.py
git commit -m "fix(skyulf-core): resolve FeatureSelection's advertised 'variance' default method"
```

---

### Task 4: Fix General Binning's advertised `"uniform"` strategy default

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/bucketing.py:338-366, 424-430`
- Test: create `skyulf-core/tests/test_bucketing_general_binning.py`

**Interfaces:**
- Consumes: existing `_fit_equal_width(series, n_bins) -> np.ndarray` (unchanged), existing `_fit_one_column_edges(series, strategy, override, config, defaults) -> tuple[Any, Any]` (unchanged signature).
- Produces: `_fit_one_column_edges` now maps the strategy alias `"uniform"` onto the same code path as `"equal_width"`, so a config built straight from the node's own registered default (`{"n_bins": 5, "strategy": "uniform", "columns": [...]}`) actually fits bin edges instead of returning `(None, None)`.

**Bug:** `@node_meta(params={"n_bins": 5, "strategy": "uniform", ...})` advertises `"uniform"` as the default strategy, but `_fit_one_column_edges` only recognizes `"equal_width"`, `"equal_frequency"`, `"kmeans"`, `"custom"`, `"kbins"` — `"uniform"` falls through to `return None, None`, silently producing an empty `bin_edges_map`.

- [ ] **Step 1: Write the failing test**

Create `skyulf-core/tests/test_bucketing_general_binning.py`:

```python
"""Regression test for bug-hunt finding #7: GeneralBinning's advertised
default strategy ("uniform") must actually produce bin edges."""

import pandas as pd

from skyulf.preprocessing.bucketing import GeneralBinningApplier, GeneralBinningCalculator


def test_default_strategy_uniform_produces_bin_edges():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    artifact = GeneralBinningCalculator().fit(
        X, {"columns": ["x"], "n_bins": 2, "strategy": "uniform"}
    )
    assert artifact.get("bin_edges", {}).get("x") not in (None, [])
    result = GeneralBinningApplier().apply(X, artifact)
    assert "x_binned" in result.columns
    assert result["x_binned"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd skyulf-core && python -m pytest tests/test_bucketing_general_binning.py -v`
Expected: FAIL — `artifact["bin_edges"]` is `{}` (no `"x"` key), because `_fit_one_column_edges` returns `(None, None)` for the unrecognized `"uniform"` strategy.

- [ ] **Step 3: Add the `"uniform"` alias to `_fit_one_column_edges`**

Edit `skyulf-core/skyulf/preprocessing/bucketing.py`, in `_fit_one_column_edges` (around line 347):

```python
def _fit_one_column_edges(
    series: pd.Series,
    strategy: str,
    override: dict[str, Any],
    config: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[Any, Any]:
    """Dispatch one column to its strategy-specific fitter; returns ``(edges, labels)``."""
    default_n_bins = defaults["default_n_bins"]
    if strategy in ("equal_width", "uniform"):  # "uniform" is the node's registered default alias
        return _fit_equal_width(series, override.get("equal_width_bins", defaults["n_bins"])), None
    if strategy == "equal_frequency":
        return (
            _fit_equal_frequency(
                series,
                override.get("equal_frequency_bins", defaults["q_bins"]),
                override.get("duplicates", defaults["duplicates"]),
            ),
            None,
        )
    if strategy == "kmeans":
        return _fit_kmeans(series, override.get("n_bins", default_n_bins)), None
    if strategy == "custom":
        return _resolve_custom_edges(str(series.name or ""), override, config)
    if strategy == "kbins":
        n_bins = override.get("kbins_n_bins", config.get("kbins_n_bins", default_n_bins))
        k_strategy = override.get("kbins_strategy", config.get("kbins_strategy", "quantile"))
        return _fit_kbins(series, n_bins, k_strategy), None
    return None, None
```

Also update the calculator's `global_strategy` default in `_fit_one_column_into_maps` (currently `config.get("strategy", "equal_width")`) — leave it as `"equal_width"` since that's a valid strategy name already; the fix is purely in the dispatch table so both `"uniform"` (metadata default) and `"equal_width"` (frontend's literal value, per the bug-hunt note "the frontend happens to use `equal_width`") resolve identically.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd skyulf-core && python -m pytest tests/test_bucketing_general_binning.py -v`
Expected: PASS

- [ ] **Step 5: Run the full bucketing test suite to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/ -k bucketing -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/skyulf/preprocessing/bucketing.py skyulf-core/tests/test_bucketing_general_binning.py
git commit -m "fix(skyulf-core): map GeneralBinning's 'uniform' default strategy to equal_width fit"
```

---

### Task 5: Fix FeatureMath silently dropping mixed-offset datetime extraction

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/feature_generation/_pandas_ops.py:173-186`
- Test: create `skyulf-core/tests/test_feature_generation_datetime_extract.py`

**Interfaces:**
- Consumes: existing `_PANDAS_DT_FEATURES: dict[str, Callable]` (unchanged), existing `_pandas_datetime_apply(op, df_out) -> None` (unchanged signature, in-place mutation).
- Produces: `_pandas_datetime_apply` now parses timestamps with `utc=True` so mixed-offset ISO strings normalize to a single tz-aware dtype instead of falling back to an `object` dtype that later raises inside the `.dt` accessor and gets silently swallowed by the broad `except Exception`.

**Bug:** `pd.to_datetime(df_out[col], errors="coerce")` on a column with mixed UTC offsets (e.g. `"...+00:00"` and `"...+01:00"`) produces an `object`-dtype Series of `Timestamp` objects (pandas can't infer one consistent tz), so `.dt.hour` inside `builder(dt)` raises `AttributeError: Can only use .dt accessor with datetimelike values`; the surrounding `try/except` logs a warning and returns, silently omitting the requested `{col}_{feat}` column.

- [ ] **Step 1: Write the failing test**

Create `skyulf-core/tests/test_feature_generation_datetime_extract.py`:

```python
"""Regression test for bug-hunt finding #9: FeatureMath must not silently
drop datetime_extract features on mixed-UTC-offset timestamp columns."""

import pandas as pd

from skyulf.preprocessing.feature_generation._pandas_ops import _featgen_apply_pandas


def test_datetime_extract_handles_mixed_offset_timestamps():
    X = pd.DataFrame({
        "when": [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T00:00:00+01:00",
        ]
    })
    df_out, _y = _featgen_apply_pandas(X, None, {
        "operations": [{
            "operation_type": "datetime_extract",
            "input_columns": ["when"],
            "datetime_features": ["hour"],
        }]
    })
    assert "when_hour" in df_out.columns
    # 00:00+00:00 -> hour 0 UTC; 00:00+01:00 -> hour 23 UTC (previous day).
    assert df_out["when_hour"].tolist() == [0, 23]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd skyulf-core && python -m pytest tests/test_feature_generation_datetime_extract.py -v`
Expected: FAIL — `"when_hour" not in df_out.columns` (the column is silently absent; a warning is logged instead).

- [ ] **Step 3: Fix `_pandas_datetime_apply` to normalize to UTC**

Edit `skyulf-core/skyulf/preprocessing/feature_generation/_pandas_ops.py`:

```python
def _pandas_datetime_apply(op: dict[str, Any], df_out: Any) -> None:
    """Materialise datetime-extract features onto ``df_out`` in place.

    Parses with ``utc=True`` so columns mixing timestamp offsets (e.g.
    "...+00:00" and "...+01:00") normalize to one tz-aware dtype instead of
    falling back to an ``object`` dtype that breaks the ``.dt`` accessor.
    """
    valid = [c for c in op.get("input_columns", []) if c in df_out.columns]
    features = op.get("datetime_features", [])
    for col in valid:
        try:
            dt = pd.to_datetime(df_out[col], errors="coerce", utc=True)
            for feat in features:
                builder = _PANDAS_DT_FEATURES.get(feat)
                if builder is None:
                    continue
                df_out[f"{col}_{feat}"] = builder(dt)
        except Exception as e:
            logger.warning(f"Failed to extract datetime features for column {col}: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd skyulf-core && python -m pytest tests/test_feature_generation_datetime_extract.py -v`
Expected: PASS

- [ ] **Step 5: Run the full feature-generation test suite to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/ -k "feature_generation or featgen" -v`
Expected: all tests PASS. In particular, confirm no existing test asserts naive (non-UTC) datetime output from this path — if one does, update its expected values to the UTC-normalized equivalents rather than reverting the fix.

- [ ] **Step 6: Commit**

```bash
git add skyulf-core/skyulf/preprocessing/feature_generation/_pandas_ops.py skyulf-core/tests/test_feature_generation_datetime_extract.py
git commit -m "fix(skyulf-core): normalize mixed-offset timestamps to UTC in datetime_extract"
```

---

### Task 6: Fix upload UI's stricter-than-server 500 MB limit

**Files:**
- Modify: `frontend/ml-canvas/src/modules/nodes/data/FileUpload.tsx:52-56`
- Modify: `backend/config/mixins/files.py` (expose the limit to the frontend)
- Create: `backend/monitoring/router.py` addition, or reuse an existing public config endpoint — see Step 3 for the exact call site
- Test: create `frontend/ml-canvas/src/modules/nodes/data/__tests__/FileUpload.test.tsx` (or extend an existing test file if one already covers `FileUpload.tsx` — check first)

**Interfaces:**
- Consumes: existing `useUploadDataset()` hook from `../../../core/hooks/useDatasets` (unchanged).
- Produces: a new `useMaxUploadSizeBytes()` hook (or equivalent) that returns the server's actual `MAX_UPLOAD_SIZE` (in bytes), which `FileUpload.tsx` uses in place of the hardcoded `500 * 1024 * 1024` constant.

**Bug:** The frontend hardcodes a 500 MB client-side cap and shows `"Maximum size is 500MB"`, while the backend's `Settings.MAX_UPLOAD_SIZE` default is 10 GB (and is separately configurable via the `MAX_UPLOAD_SIZE` env var) — the client rejects and misleads the user about valid uploads.

- [ ] **Step 1: Check whether a public config/health endpoint already exists to extend**

Run: `grep -n "APIRouter\|@router.get" /Users/BH7043/Skyulf/backend/monitoring/router.py | head -20`

If `backend/monitoring/router.py` already has a `GET /health` or `GET /config`-style unauthenticated endpoint, extend its response with `max_upload_size_bytes`. If none exists, add a small new endpoint as shown in Step 2.

- [ ] **Step 2: Add a `GET /api/config/upload-limits` endpoint (backend)**

In `backend/monitoring/router.py`, add (adjust the exact router/prefix to match what's already registered in that file — inspect the top of the file for the existing `router = APIRouter(...)` declaration and prefix before adding):

```python
from backend.config import get_settings


@router.get("/config/upload-limits")
async def get_upload_limits() -> dict[str, int]:
    """Expose the server's effective upload size limit so the client UI
    never advertises a stricter, incorrect cap than the backend enforces.
    """
    settings = get_settings()
    return {"max_upload_size_bytes": settings.MAX_UPLOAD_SIZE}
```

- [ ] **Step 3: Write the failing frontend test**

Create `frontend/ml-canvas/src/modules/nodes/data/__tests__/FileUpload.test.tsx` (check first whether `frontend/ml-canvas/src/modules/nodes/data/__tests__/` already exists and follow its existing test setup/imports pattern — e.g. `vitest` + `@testing-library/react`, matching whatever the repo's other node tests use):

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FileUpload } from '../FileUpload';

// Mock the server-provided limit at 10 GB (matches backend default) so a
// 600MB file — previously rejected by the hardcoded 500MB client cap — is
// accepted by the file-size guard.
vi.mock('../../../../core/hooks/useUploadLimits', () => ({
  useMaxUploadSizeBytes: () => 10 * 1024 * 1024 * 1024,
}));
vi.mock('../../../../core/hooks/useDatasets', () => ({
  useUploadDataset: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

describe('FileUpload size limit', () => {
  it('accepts a 600MB file when the server limit is 10GB', () => {
    render(<FileUpload onUploadComplete={vi.fn()} onCancel={vi.fn()} />);
    const bigFile = new File([new ArrayBuffer(1)], 'big.csv', { type: 'text/csv' });
    Object.defineProperty(bigFile, 'size', { value: 600 * 1024 * 1024 });
    const input = screen.getByTestId('file-upload-input');
    fireEvent.change(input, { target: { files: [bigFile] } });
    expect(screen.queryByText(/Maximum size is 500MB/i)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd frontend/ml-canvas && npx vitest run src/modules/nodes/data/__tests__/FileUpload.test.tsx`
Expected: FAIL — either the `useUploadLimits` module doesn't exist yet (import error) or (once Step 5's hook file is stubbed) the component still shows the 500MB error because `FileUpload.tsx` hasn't been updated yet.

- [ ] **Step 5: Add the `useMaxUploadSizeBytes` hook (frontend)**

Create `frontend/ml-canvas/src/core/hooks/useUploadLimits.ts`:

```ts
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';

interface UploadLimitsResponse {
  max_upload_size_bytes: number;
}

const FALLBACK_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024 * 1024; // matches backend's documented default

/** Server-configured upload size ceiling, so the client never shows a
 * stricter, incorrect limit than the backend actually enforces. */
export function useMaxUploadSizeBytes(): number {
  const { data } = useQuery({
    queryKey: ['upload-limits'],
    queryFn: async () => {
      const res = await apiClient.get<UploadLimitsResponse>('/config/upload-limits');
      return res.data;
    },
    staleTime: Infinity, // server config doesn't change within a session
  });
  return data?.max_upload_size_bytes ?? FALLBACK_MAX_UPLOAD_SIZE_BYTES;
}
```

(If the repo's `apiClient` module has a different export shape than `.get<T>(path)`, inspect `frontend/ml-canvas/src/core/api/client.ts` first and match its actual signature before writing this file.)

- [ ] **Step 6: Update `FileUpload.tsx` to use the hook and add a stable test id**

Edit `frontend/ml-canvas/src/modules/nodes/data/FileUpload.tsx`:

```tsx
import { useMaxUploadSizeBytes } from '../../../core/hooks/useUploadLimits';
```

```tsx
export const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, onCancel }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const maxSizeBytes = useMaxUploadSizeBytes();
```

```tsx
  const handleFiles = async (file: File) => {
    setError(null);
    setProgress(0);

    if (file.size > maxSizeBytes) {
      const maxSizeMb = (maxSizeBytes / (1024 * 1024)).toFixed(0);
      setError(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)}MB). Maximum size is ${maxSizeMb}MB.`);
      return;
    }
```

Find the `<input type="file" ...>` element further down in the same file and add `data-testid="file-upload-input"` to it (needed for the test in Step 3 to locate it).

- [ ] **Step 7: Run test to verify it passes**

Run: `cd frontend/ml-canvas && npx vitest run src/modules/nodes/data/__tests__/FileUpload.test.tsx`
Expected: PASS

- [ ] **Step 8: Lint and type-check**

Run: `cd frontend/ml-canvas && npx eslint src/modules/nodes/data/FileUpload.tsx src/core/hooks/useUploadLimits.ts src/modules/nodes/data/__tests__/FileUpload.test.tsx && npx tsc --project tsconfig.json --noEmit`
Expected: 0 errors, 0 warnings.

- [ ] **Step 9: Commit**

```bash
git add frontend/ml-canvas/src/modules/nodes/data/FileUpload.tsx frontend/ml-canvas/src/core/hooks/useUploadLimits.ts frontend/ml-canvas/src/modules/nodes/data/__tests__/FileUpload.test.tsx backend/monitoring/router.py
git commit -m "fix(frontend,backend): surface server's real upload size limit instead of a hardcoded 500MB cap"
```

---

### Task 7: Block cyclic pipeline connections on the canvas before submission

**Files:**
- Modify: `frontend/ml-canvas/src/core/store/useGraphStore.ts:289-454` (the `onConnect` handler)
- Test: create `frontend/ml-canvas/src/core/store/__tests__/useGraphStore.cycle.test.ts` (check first whether `__tests__` already exists next to `useGraphStore.ts` and match its existing setup)

**Interfaces:**
- Consumes: existing `nodes: Node[]`, `edges: Edge[]` from the store's own `get()` (unchanged).
- Produces: a new pure helper `wouldCreateCycle(connection: Connection, nodes: Node[], edges: Edge[]) -> boolean`, exported from `useGraphStore.ts` (or a new `graphCycleCheck.ts` util, whichever matches this file's existing pattern of inlining vs. extracting helpers — this file already inlines a `rootsOf` BFS helper inline in `onConnect`, so follow that same inline-helper convention), called at the very top of `onConnect` before any of the existing warning branches.

**Bug:** `onConnect` runs several UX warnings (model-to-model, ensemble lineage, X/Y split, fan-in merge) but never checks for a cycle. Connecting `B -> A` when `A -> B` already exists silently succeeds; the pipeline converter's BFS then emits a best-effort order and the backend engine fails late trying to load an artifact that was never produced.

- [ ] **Step 1: Write the failing test**

Create `frontend/ml-canvas/src/core/store/__tests__/useGraphStore.cycle.test.ts`:

```ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useGraphStore } from '../useGraphStore';
import type { Node, Connection } from 'reactflow';

function makeNode(id: string, definitionType: string): Node {
  return { id, type: 'default', position: { x: 0, y: 0 }, data: { definitionType } };
}

describe('onConnect cycle prevention', () => {
  beforeEach(() => {
    useGraphStore.setState({ nodes: [], edges: [] });
  });

  it('rejects a connection that would create a 2-node cycle', () => {
    const nodeA = makeNode('A', 'dataset_node');
    const nodeB = makeNode('B', 'StandardScaler');
    useGraphStore.setState({
      nodes: [nodeA, nodeB],
      edges: [{ id: 'e1', source: 'A', target: 'B' }],
    });

    const cyclicConnection: Connection = { source: 'B', target: 'A', sourceHandle: null, targetHandle: null };
    useGraphStore.getState().onConnect(cyclicConnection);

    // The cyclic edge must NOT have been added.
    const edges = useGraphStore.getState().edges;
    expect(edges.some(e => e.source === 'B' && e.target === 'A')).toBe(false);
    expect(edges).toHaveLength(1); // only the original A->B edge remains
  });

  it('still allows a normal acyclic connection', () => {
    const nodeA = makeNode('A', 'dataset_node');
    const nodeB = makeNode('B', 'StandardScaler');
    const nodeC = makeNode('C', 'MinMaxScaler');
    useGraphStore.setState({
      nodes: [nodeA, nodeB, nodeC],
      edges: [{ id: 'e1', source: 'A', target: 'B' }],
    });

    const acyclicConnection: Connection = { source: 'B', target: 'C', sourceHandle: null, targetHandle: null };
    useGraphStore.getState().onConnect(acyclicConnection);

    const edges = useGraphStore.getState().edges;
    expect(edges.some(e => e.source === 'B' && e.target === 'C')).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/__tests__/useGraphStore.cycle.test.ts`
Expected: FAIL on the first test — `onConnect` currently adds the `B -> A` edge unconditionally (no cycle check exists), so `edges` will have length 2, not 1.

- [ ] **Step 3: Add the cycle check at the top of `onConnect`**

Edit `frontend/ml-canvas/src/core/store/useGraphStore.ts`. Insert this new block as the very first statement inside `onConnect`, before the existing `const nodes = get().nodes;` line:

```ts
  onConnect: (connection: Connection) => {
    const nodes = get().nodes;
    const edges = get().edges;

    // Reject connections that would create a cycle: walk forward from the
    // proposed target and check whether the proposed source is reachable.
    // If it is, adding source->target would close a loop.
    if (connection.source && connection.target) {
      const wouldCreateCycle = (): boolean => {
        const targetId = connection.target!;
        const sourceId = connection.source!;
        if (targetId === sourceId) return true; // self-loop
        const visited = new Set<string>();
        const stack = [targetId];
        while (stack.length > 0) {
          const current = stack.pop()!;
          if (current === sourceId) return true;
          if (visited.has(current)) continue;
          visited.add(current);
          for (const e of edges.filter((ed) => ed.source === current)) {
            stack.push(e.target);
          }
        }
        return false;
      };
      if (wouldCreateCycle()) {
        toast.error(
          'Invalid connection',
          'This connection would create a cycle. Pipelines must be a directed acyclic graph (DAG) — a node cannot (even indirectly) feed back into one of its own upstream nodes.',
        );
        return;
      }
    }

    const sourceNode = nodes.find((n) => n.id === connection.source);
    const targetNode = nodes.find((n) => n.id === connection.target);
```

(This replaces the original two lines `const nodes = get().nodes;` / `const edges = get().edges;` plus the following `const sourceNode = ...` / `const targetNode = ...` lines — the cycle check is inserted between the existing `get()` calls and the existing `sourceNode`/`targetNode` lookups, reusing the same `nodes`/`edges` locals rather than re-fetching them.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/__tests__/useGraphStore.cycle.test.ts`
Expected: PASS (both tests)

- [ ] **Step 5: Run the full store test suite to check for regressions**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/`
Expected: all tests PASS — in particular, confirm the existing ensemble-lineage and X/Y-split `window.confirm` warning tests (if any exist) still fire correctly, since the cycle check now runs before them but returns early only when a cycle is detected.

- [ ] **Step 6: Lint and type-check**

Run: `cd frontend/ml-canvas && npx eslint src/core/store/useGraphStore.ts src/core/store/__tests__/useGraphStore.cycle.test.ts && npx tsc --project tsconfig.json --noEmit`
Expected: 0 errors, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/core/store/useGraphStore.ts frontend/ml-canvas/src/core/store/__tests__/useGraphStore.cycle.test.ts
git commit -m "fix(frontend): reject cyclic canvas connections in onConnect before submission"
```

---

### Task 8: Fix out-of-order job-list polling responses reverting fresher state

**Files:**
- Modify: `frontend/ml-canvas/src/core/store/useJobStore.ts:82-174` (`runPollTick` and `fetchJobs`)
- Test: create `frontend/ml-canvas/src/core/store/__tests__/useJobStore.raceCondition.test.ts`

**Interfaces:**
- Consumes: existing `jobsApi.getJobs(limit, skip) -> Promise<JobInfo[]>` (unchanged signature).
- Produces: `runPollTick` and `fetchJobs` now each stamp every in-flight request with a monotonically increasing sequence number (module-scoped `let requestSeq = 0`) and only apply a response to the store if its sequence number is still the highest one seen — an older, slower response arriving after a newer one is silently discarded instead of overwriting the store.

**Bug:** `runPollTick`/`fetchJobs` have no request-ordering guard. If request A (slow) and request B (fast, triggered by a WS event refresh) both call `jobsApi.getJobs`, and B resolves first with fresher data, A's later resolution unconditionally overwrites the store with stale data.

- [ ] **Step 1: Write the failing test**

Create `frontend/ml-canvas/src/core/store/__tests__/useJobStore.raceCondition.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useJobStore } from '../useJobStore';
import { jobsApi } from '../../api/jobs';

vi.mock('../../api/jobs', () => ({
  jobsApi: { getJobs: vi.fn() },
}));
vi.mock('../../realtime/jobEventsSocket', () => ({
  jobEventsSocket: { on: vi.fn(), off: vi.fn(), isConnected: () => false },
}));

describe('useJobStore out-of-order response guard', () => {
  beforeEach(() => {
    useJobStore.setState({ jobs: [] });
    vi.clearAllMocks();
  });

  it('discards a slow, stale fetchJobs response that resolves after a fresher one', async () => {
    let resolveSlow!: (v: unknown) => void;
    const slow = new Promise((resolve) => { resolveSlow = resolve; });
    const fresh = [{ job_id: 'job-1', status: 'completed', created_at: new Date().toISOString() }];
    const stale = [{ job_id: 'job-1', status: 'running', created_at: new Date().toISOString() }];

    (jobsApi.getJobs as ReturnType<typeof vi.fn>)
      .mockImplementationOnce(() => slow)              // request A: slow, started first
      .mockImplementationOnce(() => Promise.resolve(fresh)); // request B: fast, started second

    const callA = useJobStore.getState().fetchJobs();  // kicks off A (pending)
    const callB = useJobStore.getState().fetchJobs();  // kicks off B (resolves immediately)
    await callB;
    expect(useJobStore.getState().jobs).toEqual(fresh);

    resolveSlow(stale); // A resolves last, with stale data
    await callA;

    // A's stale response must NOT have overwritten B's fresher result.
    expect(useJobStore.getState().jobs).toEqual(fresh);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/__tests__/useJobStore.raceCondition.test.ts`
Expected: FAIL — `useJobStore.getState().jobs` will equal `stale` after `callA` resolves, since `fetchJobs` currently has no sequencing guard and unconditionally `set({ jobs, ... })`.

- [ ] **Step 3: Add a sequence-number guard to `fetchJobs` and `runPollTick`**

Edit `frontend/ml-canvas/src/core/store/useJobStore.ts`. Add a module-scoped counter near the other module-scoped `let`s inside the `create<JobState>((set, get) => { ... })` closure:

```ts
export const useJobStore = create<JobState>((set, get) => {
  let pollingInterval: ReturnType<typeof setInterval> | null = null;
  let pollingDeadline: number | null = null;   // hard stop timestamp
  let unsubscribeWs: (() => void) | null = null;
  let unsubscribeStatus: (() => void) | null = null;
  let wsConnected = false;
  let refreshTimer: ReturnType<typeof setTimeout> | null = null;

  // Guards against out-of-order getJobs(page 0) responses: only the response
  // to the most recently issued request is allowed to update `jobs`. A
  // slower, older request that resolves after a newer one is discarded.
  let latestListRequestSeq = 0;
```

Then update `runPollTick`'s fetch (replace lines 89-98):

```ts
    try {
      const requestSeq = ++latestListRequestSeq;
      const latestJobs = await jobsApi.getJobs(PAGE_SIZE, 0);
      if (requestSeq !== latestListRequestSeq) {
        return; // a newer request has since been issued; discard this stale response
      }

      set(state => {
          if (state.jobs.length <= PAGE_SIZE) {
              return { jobs: latestJobs };
          } else {
              return { jobs: [...latestJobs, ...state.jobs.slice(PAGE_SIZE)] };
          }
      });
```

And update `fetchJobs` (replace lines 145-154):

```ts
    fetchJobs: async () => {
      set({ isLoading: true, skip: 0 });
      const requestSeq = ++latestListRequestSeq;
      try {
        const jobs = await jobsApi.getJobs(PAGE_SIZE, 0);
        if (requestSeq !== latestListRequestSeq) {
          return; // a newer request has since been issued; discard this stale response
        }
        set({ jobs, isLoading: false, hasMore: jobs.length === PAGE_SIZE });
      } catch (error) {
        console.error('Failed to fetch jobs:', error);
        if (requestSeq === latestListRequestSeq) {
          set({ isLoading: false });
        }
      }
    },
```

(`loadMoreJobs` is intentionally left untouched — it appends to a specific `skip` offset rather than replacing the head of the list, so it isn't affected by this specific "page-0 refresh" race.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/__tests__/useJobStore.raceCondition.test.ts`
Expected: PASS

- [ ] **Step 5: Run the full job-store test suite to check for regressions**

Run: `cd frontend/ml-canvas && npx vitest run src/core/store/`
Expected: all tests PASS.

- [ ] **Step 6: Lint and type-check**

Run: `cd frontend/ml-canvas && npx eslint src/core/store/useJobStore.ts src/core/store/__tests__/useJobStore.raceCondition.test.ts && npx tsc --project tsconfig.json --noEmit`
Expected: 0 errors, 0 warnings.

- [ ] **Step 7: Commit**

```bash
git add frontend/ml-canvas/src/core/store/useJobStore.ts frontend/ml-canvas/src/core/store/__tests__/useJobStore.raceCondition.test.ts
git commit -m "fix(frontend): discard out-of-order job-list responses with a request sequence guard"
```

---

### Task 9: Enforce a database-level idempotency claim for job submission (cross-process duplicate prevention)

**Files:**
- Modify: `backend/database/models.py` (add a unique idempotency column to `TrainingJob`/`MLJob`)
- Modify: `backend/database/engine.py` (add a migration entry + unique index creation)
- Modify: `backend/ml_pipeline/_execution/jobs.py` (`JobManager.create_job`)
- Modify: `backend/ml_pipeline/_internal/_routers/run_pipeline.py` (catch the uniqueness violation and return the existing job)
- Test: create `backend/tests/test_job_idempotency.py` (check first for an existing `backend/tests/` structure/conftest pattern and match it)

**Interfaces:**
- Consumes: existing `TrainingJob` model fields `dataset_source_id: str`, `node_id: str`, `status: str` (unchanged), existing `JobManager.create_job(session, pipeline_id, node_id, job_type, dataset_id, user_id, model_type, graph, branch_index) -> str` signature (unchanged — only its internals change).
- Produces: a new `idempotency_key: Mapped[str]` column on `MLJob` (the shared base class both `TrainingJob` and other job tables inherit from — confirm this by reading `backend/database/models.py` around line 254 before editing) with a **partial unique index** scoped to active statuses, computed as `f"{dataset_source_id}:{node_id}:{branch_index}:{window_bucket}"`; `JobManager.create_job` now catches the resulting `IntegrityError` on insert and returns the pre-existing job's id instead of a new one, making duplicate submission safe even across two separate API processes (not just within one event loop, unlike the current `_submit_locks` dict).

**Bug:** `_submit_locks: dict[str, asyncio.Lock] = {}` in `run_pipeline.py` is process-local. Two FastAPI worker processes behind a load balancer each have their own empty dict, so neither serializes against the other; `JobManager.find_active_job`'s `SELECT ... FOR UPDATE SKIP LOCKED` only locks *existing* rows — it does nothing to prevent two processes from concurrently observing "no active row" and both proceeding to `INSERT`.

- [ ] **Step 1: Confirm the shared base class and existing columns**

Run: `sed -n '240,300p' /Users/BH7043/Skyulf/backend/database/models.py`

Confirm `MLJob` (the class at line ~254) is the shared abstract/mixin base that `TrainingJob` (line 302) inherits from, and that it already has `node_id`, `dataset_source_id`, `status`. This confirms where to add the new column so it lands on every job subtype, not just `TrainingJob`.

- [ ] **Step 2: Write the failing test**

Create `backend/tests/test_job_idempotency.py` (adapt imports/fixtures to match whatever async-SQLAlchemy test session fixture already exists elsewhere in `backend/tests/` — search first: `grep -rl "async_sessionmaker\|AsyncSession" backend/tests/conftest.py`):

```python
"""Regression test for bug-hunt finding #1: two concurrent create_job calls
for the same (dataset, node, branch) must not both succeed — the second
must return the first job's id instead of creating a duplicate row."""

import asyncio

import pytest

from backend.ml_pipeline._execution.jobs import JobManager


@pytest.mark.asyncio
async def test_concurrent_create_job_is_idempotent(async_session_factory):
    """Simulates two 'processes' each with their own session, racing to
    create a job for the same (dataset_id, node_id, branch_index) with no
    existing active job yet — i.e. the exact race the process-local
    asyncio.Lock in run_pipeline.py cannot prevent across processes."""
    async def submit():
        async with async_session_factory() as session:
            job_id = await JobManager.create_job(
                session,
                pipeline_id="pipeline-1",
                node_id="node-1",
                job_type="training",
                dataset_id="dataset-1",
                model_type="classification",
                graph={},
                branch_index=0,
            )
            await session.commit()
            return job_id

    job_id_a, job_id_b = await asyncio.gather(submit(), submit())
    assert job_id_a == job_id_b, (
        "Two concurrent create_job calls for the same "
        "(dataset_id, node_id, branch_index) must resolve to one job id."
    )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_job_idempotency.py -v`
Expected: FAIL — `job_id_a != job_id_b`, since `create_job` currently always inserts a new row regardless of concurrent duplicates.

- [ ] **Step 4: Add the idempotency key column to `MLJob`**

Edit `backend/database/models.py`, in the `MLJob` class (confirmed location from Step 1), add:

```python
    # Deterministic per-submission key: f"{dataset_source_id}:{node_id}:{branch_index}".
    # A unique index on this column (created via a raw-SQL partial index in
    # _run_migrations, scoped to status IN ('queued','running')) is the
    # actual concurrency guard — the asyncio.Lock in run_pipeline.py only
    # serializes requests within one process and cannot prevent two
    # separate API processes from both observing "no active job" and both
    # inserting. See initiatives/enterprise-readiness/2026-08-11-bug-hunt.md
    # finding #1.
    idempotency_key: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)
```

- [ ] **Step 5: Populate `idempotency_key` in `BasicTrainingManager.create_training_job` and `AdvancedTuningManager.create_tuning_job`**

Run: `grep -n "def create_training_job\|def create_tuning_job\|TrainingJob(" /Users/BH7043/Skyulf/backend/ml_pipeline/_execution/basic_training_manager.py /Users/BH7043/Skyulf/backend/ml_pipeline/_execution/advanced_tuning_manager.py`

Read both matched constructor call sites, then add `idempotency_key=f"{dataset_id}:{node_id}:{branch_index}"` as a field passed into each `TrainingJob(...)` instantiation, alongside the existing `dataset_source_id=dataset_id, node_id=node_id, ...` kwargs.

- [ ] **Step 6: Add the partial unique index migration**

Edit `backend/database/engine.py`'s `_run_migrations`, appending to `_MIGRATIONS` (the exact list of tuples shown at line 206 in the current file):

```python
        # v0.8.0 — Phase 12 finding #1: DB-level idempotency guard for job
        # submission. Partial unique index (PostgreSQL syntax) rejects a
        # second INSERT for the same (dataset, node, branch) while an
        # earlier one is still queued/running, across ALL processes —
        # closing the race the process-local asyncio.Lock could not.
        (
            "0.8.0",
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_training_jobs_active_idempotency "
            "ON training_jobs (idempotency_key) "
            "WHERE status IN ('queued', 'running')",
        ),
```

Note: SQLite (used in dev/tests per this repo's existing migration entries using `ALTER TABLE ... ADD COLUMN`) supports partial unique indexes via `CREATE UNIQUE INDEX ... WHERE ...` since SQLite 3.8.0, so this single statement works on both PostgreSQL and SQLite — no dialect branching needed. Verify this assumption by running the test in Step 8 against the repo's actual test DB backend.

- [ ] **Step 7: Catch the uniqueness violation in `JobManager.create_job` and resolve to the existing job**

Edit `backend/ml_pipeline/_execution/jobs.py`:

```python
    @staticmethod
    async def create_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        job_type: Literal["training", "tuning", "preview"],
        dataset_id: str = "unknown",
        user_id: int | None = None,
        model_type: str = "unknown",
        graph: dict[str, Any] | None = None,
        branch_index: int = 0,
    ) -> str:
        """Creates a new job in the database (Async).

        Idempotent at the database level: if a concurrent request (in this
        process or another) already inserted an active job for the same
        (dataset_id, node_id, branch_index) between our find_active_job
        check and our INSERT, the unique partial index on `idempotency_key`
        rejects our INSERT with an IntegrityError — we catch it and return
        the winning job's id instead of raising.
        """
        from sqlalchemy.exc import IntegrityError

        try:
            if job_type == "training":
                return await BasicTrainingManager.create_training_job(
                    session, pipeline_id, node_id, dataset_id, user_id,
                    model_type, graph, branch_index=branch_index,
                )
            elif job_type == "tuning":
                return await AdvancedTuningManager.create_tuning_job(
                    session, pipeline_id, node_id, dataset_id, user_id,
                    model_type, graph, branch_index=branch_index,
                )
            elif job_type == "preview":
                return await BasicTrainingManager.create_training_job(
                    session, pipeline_id, node_id, dataset_id, user_id,
                    model_type, graph, is_preview=True, branch_index=branch_index,
                )
            else:
                raise ValueError(f"Unknown job_type: {job_type}")
        except IntegrityError:
            await session.rollback()
            existing = await JobManager.find_active_job(session, dataset_id, node_id, branch_index)
            if existing is not None:
                return existing
            raise
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_job_idempotency.py -v`
Expected: PASS — both concurrent `submit()` calls resolve to the same `job_id`.

- [ ] **Step 9: Run the full job-management test suite to check for regressions**

Run: `cd backend && python -m pytest tests/ -k "job" -v`
Expected: all tests PASS.

- [ ] **Step 10: Lint and type-check**

Run: `cd /Users/BH7043/Skyulf && ruff check backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/jobs.py && ruff format --check backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/jobs.py backend/tests/test_job_idempotency.py`
Run: `ty check backend`
Expected: 0 errors on all four commands.

- [ ] **Step 11: Commit**

```bash
git add backend/database/models.py backend/database/engine.py backend/ml_pipeline/_execution/jobs.py backend/tests/test_job_idempotency.py
git commit -m "fix(backend): add DB-level idempotency index to prevent cross-process duplicate job creation"
```

---

## Self-Review Notes (completed during plan authoring)

**Spec coverage check** — all 9 bug-hunt findings mapped to a task:
1. Cross-process duplicate job → Task 9
2. Lag Features y-misalignment → Task 1
3. Rolling Aggregate y-misalignment → Task 2
4. Cyclic canvas graphs → Task 7
5. Out-of-order job polling → Task 8
6. Feature Selection default no-op → Task 3
7. General Binning `uniform` no-op → Task 4
8. Upload UI 500MB vs 10GB mismatch → Task 6
9. FeatureMath mixed-offset datetime drop → Task 5

Tasks are ordered by the bug-hunt doc's own "prioritized top 8" list (Task 9 = finding #1, Tasks 1–2 = findings #2–3, etc.), with the DB-level fix (Task 9, most invasive/highest-risk migration) deliberately placed **last** so the 8 lower-risk, self-contained fixes land first and can each ship independently without blocking on the migration review.

**Placeholder scan** — no "TBD"/"handle appropriately"/"similar to Task N" found; every step has literal code or an exact command. Task 5's and Task 9's "check first" instructions are legitimate repo-inspection steps (confirming an assumption about existing test fixtures/base classes before writing code), not placeholders for the actual fix logic, which is fully specified.

**Type consistency** — `_apply_pandas`/`_apply_polars` in Tasks 1–2 consistently return `tuple[Any, Any]` matching the existing `BaseApplier.apply` → `apply_method` wrapper contract (unpacks 2-tuples). `JobManager.create_job`'s signature is unchanged across Task 9. `useMaxUploadSizeBytes()` return type (`number`) matches its one call site in Task 6.

## Cross-References

- Source finding doc: `initiatives/enterprise-readiness/2026-08-11-bug-hunt.md`
- Master fix list entry: `initiatives/enterprise-readiness/2026-08-11-master-fix-list.md`, Phase 12.
- This plan does **not** cover Phases 0–11 or 13–15 of the master fix list (auth/tenancy foundations, security/scale hardening, API contract hardening, i18n, code escape hatch, training visualization) — each of those is architecturally independent and should get its own `writing-plans` pass when picked up, per the Scope Check rule.
