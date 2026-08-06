# Native Polars Correlation-Threshold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the fit-time Polars-to-Pandas conversion for eligible
Pearson and Spearman correlation-threshold selection while preserving the
existing dual-engine API, artifact, errors, and automatic column selection.

**Architecture:** Keep an explicit Pandas compatibility helper containing the
current behavior. Raw and wrapped Polars frames resolve eligible columns
natively, normalize them to Float64, and calculate pairwise-complete
upper-triangle correlations in one Polars `select`. Kendall, callables,
unsupported dtypes, and unavailable native capability always use the retained
Pandas helper.

**Tech Stack:** Python 3.12+, Polars >=1.36, pandas 2.x, NumPy,
pytest/pytest-benchmark, Ruff, and Ty.

## Global Constraints

- Preserve public Pandas and raw/wrapped Polars support. Do not change the
  frontend `correlation_method` enum, backend payloads, apply path, or
  `CorrelationThresholdArtifact` shape.
- Retain `polars>=1.36.0`; do not change dependency manifests or install a new
  package for this work.
- Use native Polars only for `pearson` and `spearman`. Keep the Pandas route
  for Kendall, callables, unsupported selected dtypes, generic frame types,
  and unavailable native capability.
- Do not use a broad exception handler to decide fallback behavior. Determine
  eligibility before executing the native calculation.
- Preserve pairwise-complete Pandas semantics: for every pair, omit rows
  where either value is null or float NaN; constant or sparse pairs must not
  drop a column.
- Preserve current upper-triangle ordering and strict comparison semantics:
  a column drops only when an earlier column has `abs(correlation) >
  threshold`, never on equality.
- Treat Polars float NaN as missing during automatic numeric selection, just
  as the current conversion-to-Pandas route does.
- Add the approved concise source comment beside the Pandas fallback:
  `Retain this compatibility route until Polars supports Kendall and callable
  correlations.`
- Do not add a changelog entry: accepted configuration, public artifact, and
  user-visible behavior remain stable.
- Promote the native route only after the parity suite passes and raw/wrapped
  Polars benchmarks show at least 25% lower peak memory or 20% lower fit time
  without a Pandas regression.

---

## File Structure

- `skyulf-core/skyulf/utils.py` — native numeric-column selection; normalize
  float NaNs before binary and constant filtering.
- `skyulf-core/tests/test_utils.py` — raw/wrapped Polars regression coverage
  for NaN-aware automatic selection.
- `skyulf-core/skyulf/preprocessing/feature_selection/correlation.py` —
  explicit Pandas compatibility helper, raw/wrapped Polars normalizer,
  capability predicate, and native upper-triangle calculator.
- `skyulf-core/tests/test_feature_selection_gaps.py` — exact
  Pandas/raw-Polars/wrapped-Polars artifact, no-conversion, fallback, and
  threshold-boundary regressions.
- `skyulf-core/tests/test_benchmarks.py` — opt-in timing and isolated RSS
  measurements for the retained and native fit routes.
- `temp/skyulf-core-pandas-polars-audit-2026-08-05.md` — ignored evidence
  record updated only after real promotion benchmark results are available.

## Task 1: Align Native Numeric Selection With Pandas NaN Semantics

**Files:**
- Modify: `skyulf-core/skyulf/utils.py:240-252`
- Modify: `skyulf-core/tests/test_utils.py:443-482`

**Interfaces:**
- Consumes: a raw `pl.DataFrame` or `SkyulfPolarsWrapper` passed to
  `detect_numeric_columns(frame, exclude_binary=True, exclude_constant=True)`.
- Produces: the same auto-selected column list as
  `detect_numeric_columns(frame.to_pandas(), ...)` for float NaN, null,
  binary, and constant cases.

- [ ] **Step 1: Add the failing raw/wrapped Polars selector regression**

  Add `SkyulfPolarsWrapper` to the imports in
  `skyulf-core/tests/test_utils.py`, then add this test after the existing
  Polars constant test:

  ```python
  @pytest.mark.skipif(not _POLARS_AVAILABLE, reason="polars not installed")
  def test_detect_numeric_columns_polars_treats_nan_as_missing_like_pandas() -> None:
      """Native Polars selection must ignore float NaN before exclusion checks."""
      import polars as pl

      raw = pl.DataFrame(
          {
              "one_finite": [1.0, float("nan"), None],
              "binary": [0.0, 1.0, float("nan")],
              "constant": [7.0, 7.0, float("nan")],
              "varied": [1.0, 2.0, float("nan")],
          }
      )
      expected = detect_numeric_columns(raw.to_pandas())

      for frame in (raw, SkyulfPolarsWrapper(raw)):
          result = detect_numeric_columns(typing.cast(pd.DataFrame, frame))
          assert result == expected == ["varied"]
  ```

- [ ] **Step 2: Run the regression against the current implementation**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_utils.py::test_detect_numeric_columns_polars_treats_nan_as_missing_like_pandas -q
  ```

  Expected: FAIL because the current Polars path retains
  `one_finite`, `binary`, and `constant`; it only calls `drop_nulls()`.

- [ ] **Step 3: Normalize float NaNs before existing Polars exclusion checks**

  Replace the `valid = series.drop_nulls()` line in
  `_polars_column_excluded()` with:

  ```python
  valid = series.fill_nan(None).drop_nulls() if series.dtype.is_float() else series.drop_nulls()
  ```

  Keep the Boolean exclusion, binary check, and constant check unchanged.
  This is intentionally limited to floating dtypes because only they can
  contain Polars NaN values.

- [ ] **Step 4: Run focused utility coverage**

  Run:

  ```bash
  source .venv/bin/activate && pytest skyulf-core/tests/test_utils.py -q
  ```

  Expected: PASS, including existing Pandas, raw Polars, wrapper, binary,
  all-null, and constant coverage.

- [ ] **Step 5: Run required Python static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. Fix only diagnostics introduced by this task.

- [ ] **Step 6: Commit the parity repair**

  Run:

  ```bash
  git add skyulf-core/skyulf/utils.py skyulf-core/tests/test_utils.py
  git commit -m "fix(skyulf-core): align Polars numeric selection with Pandas NaNs" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing only the selector repair and its regression
  coverage.

## Task 2: Add Native Pearson and Spearman Fit Calculation

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/feature_selection/correlation.py:1-78`
- Modify: `skyulf-core/tests/test_feature_selection_gaps.py:1-115`

**Interfaces:**
- Consumes: `X: Any`, `config: dict[str, Any]`, raw `pl.DataFrame`, and
  `SkyulfPolarsWrapper`.
- Produces:
  - `_as_polars_frame(X: Any) -> pl.DataFrame | None`;
  - `_fit_correlation_threshold_pandas(X: Any, config: dict[str, Any]) ->
    CorrelationThresholdArtifact`;
  - `_native_polars_correlation_eligible(frame: pl.DataFrame,
    columns: list[str], method: Any, threshold: Any) -> bool`;
  - `_polars_correlation_columns_to_drop(frame: pl.DataFrame,
    columns: list[str], method: str, threshold: Real) -> list[str]`; and
  - an unchanged public `CorrelationThresholdCalculator.fit()` signature and
    artifact.

- [ ] **Step 1: Add an exact native-path contract test that fails on conversion**

  Add these imports to `skyulf-core/tests/test_feature_selection_gaps.py`:

  ```python
  import skyulf.preprocessing.feature_selection.correlation as correlation_module
  from skyulf.engines.polars_engine import SkyulfPolarsWrapper
  ```

  Add the fixture helper and test after
  `test_correlation_threshold_polars_engine_parity`:

  ```python
  def _correlation_parity_fixture() -> dict[str, list[object]]:
      """Return ordered values covering boolean, constant, null, and correlation cases."""
      return {
          "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
          "c": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
          "flag": [True, False, True, False, True, False],
          "constant": [7, 7, 7, 7, 7, 7],
          "mostly_null": [1.0, None, None, None, None, 6.0],
          "target": ["x", "x", "y", "y", "z", "z"],
      }


  @pytest.mark.parametrize("method", ["pearson", "spearman"])
  def test_correlation_threshold_native_polars_matches_pandas_without_conversion(
      monkeypatch: pytest.MonkeyPatch,
      method: str,
  ) -> None:
      """Eligible raw and wrapped Polars fits must stay native and match Pandas."""
      pl = pytest.importorskip("polars")
      values = _correlation_parity_fixture()
      config = {
          "columns": ["a", "b", "c", "flag", "constant", "mostly_null"],
          "threshold": 0.95,
          "correlation_method": method,
          "drop_columns": True,
      }
      calculator = CorrelationThresholdCalculator()
      expected = calculator.fit(pd.DataFrame(values), config)

      monkeypatch.setattr(
          correlation_module,
          "to_pandas",
          lambda _frame: pytest.fail("eligible Polars fit called to_pandas"),
      )
      raw = pl.DataFrame(values)
      for frame in (raw, SkyulfPolarsWrapper(raw)):
          assert calculator.fit(frame, config) == expected
  ```

  The expected Pearson artifact is the audit golden artifact with
  `columns_to_drop == ["b", "c", "mostly_null"]`; the Spearman route must
  match the current Pandas artifact as well.

- [ ] **Step 2: Run the new contract test and confirm the current route fails**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_feature_selection_gaps.py \
    -k "native_polars_matches_pandas_without_conversion" -q
  ```

  Expected: FAIL with `eligible Polars fit called to_pandas`, proving the
  current implementation converts both raw and wrapped Polars frames.

- [ ] **Step 3: Factor the existing Pandas behavior into an explicit helper**

  In `correlation.py`, add imports:

  ```python
  import inspect
  from numbers import Real

  import polars as pl

  from ...engines.polars_engine import SkyulfPolarsWrapper
  ```

  Add these constants and helpers above the apply functions:

  ```python
  _NATIVE_POLARS_METHODS = frozenset(("pearson", "spearman"))
  _POLARS_CORRELATION_DTYPES = frozenset(
      (
          pl.Boolean,
          pl.Float32,
          pl.Float64,
          pl.Int8,
          pl.Int16,
          pl.Int32,
          pl.Int64,
          pl.UInt8,
          pl.UInt16,
          pl.UInt32,
          pl.UInt64,
      )
  )


  def _as_polars_frame(X: Any) -> pl.DataFrame | None:
      """Return raw Polars data for native correlation fitting when available."""
      if isinstance(X, pl.DataFrame):
          return X
      if isinstance(X, SkyulfPolarsWrapper):
          return X._df
      return None


  def _polars_corr_accepts_method() -> bool:
      """Return whether the installed Polars correlation API accepts ``method``."""
      corr = getattr(pl, "corr", None)
      if not callable(corr):
          return False
      try:
          return "method" in inspect.signature(corr).parameters
      except (TypeError, ValueError):
          return False


  def _fit_correlation_threshold_pandas(
      X: Any,
      config: dict[str, Any],
  ) -> CorrelationThresholdArtifact:
      """Run the retained Pandas-compatible correlation fit path."""
      X_pd = to_pandas(X)
      threshold = config.get("threshold", 0.95)
      drop_columns = config.get("drop_columns", True)
      method = config.get("correlation_method", "pearson")
      cols = resolve_columns(X_pd, config, detect_numeric_columns)
      if len(cols) < 2:
          return cast(CorrelationThresholdArtifact, {})

      corr_matrix = X_pd[cols].corr(method=method).abs()
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
      to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
      return cast(
          CorrelationThresholdArtifact,
          {
              "type": "correlation_threshold",
              "columns_to_drop": to_drop,
              "threshold": threshold,
              "method": method,
              "drop_columns": drop_columns,
          },
      )
  ```

  Keep the existing comment explaining why `correlation_method`, rather than
  the facade's `method`, is read from config. Place the approved future-support
  comment immediately before the call to this helper from the public fit
  method.

- [ ] **Step 4: Add native eligibility and upper-triangle helpers**

  Add these helpers below the Pandas helper:

  ```python
  def _native_polars_correlation_eligible(
      frame: pl.DataFrame,
      columns: list[str],
      method: Any,
      threshold: Any,
  ) -> bool:
      """Return whether this fit can preserve its contract on the native path."""
      return (
          isinstance(method, str)
          and method in _NATIVE_POLARS_METHODS
          and isinstance(threshold, Real)
          and _polars_corr_accepts_method()
          and all(
              frame.get_column(column).dtype in _POLARS_CORRELATION_DTYPES
              for column in columns
          )
      )


  def _polars_correlation_columns_to_drop(
      frame: pl.DataFrame,
      columns: list[str],
      method: str,
      threshold: Real,
  ) -> list[str]:
      """Return upper-triangle columns whose pairwise-complete correlation exceeds threshold."""
      normalized = frame.select(
          [pl.col(column).cast(pl.Float64).alias(column) for column in columns]
      )
      expressions: list[pl.Expr] = []
      for right_index, right_column in enumerate(columns):
          for left_column in columns[:right_index]:
              left = pl.col(left_column)
              right = pl.col(right_column)
              complete = (
                  left.is_not_null()
                  & right.is_not_null()
                  & left.is_not_nan()
                  & right.is_not_nan()
              )
              expressions.append(
                  pl.corr(
                      left.filter(complete),
                      right.filter(complete),
                      method=method,
                  )
                  .abs()
                  .alias(f"__skyulf_correlation_{len(expressions)}")
              )

      values = normalized.select(expressions).row(0)
      to_drop: list[str] = []
      offset = 0
      for right_index, right_column in enumerate(columns):
          pair_values = values[offset : offset + right_index]
          offset += right_index
          if any(value is not None and value > threshold for value in pair_values):
              to_drop.append(right_column)
      return to_drop
  ```

  Ordinal aliases make user column names irrelevant to intermediate-result
  naming. The Float64 cast is safe because eligibility limits this route to
  numeric and Boolean dtypes; anything else retains the Pandas path.

- [ ] **Step 5: Dispatch native fits and preserve the current artifact**

  Replace the body of `CorrelationThresholdCalculator.fit()` with:

  ```python
  threshold = config.get("threshold", 0.95)
  drop_columns = config.get("drop_columns", True)
  method = config.get("correlation_method", "pearson")
  frame = _as_polars_frame(X)

  if frame is not None:
      columns = resolve_columns(frame, config, detect_numeric_columns)
      if len(columns) < 2:
          return cast(CorrelationThresholdArtifact, {})
      if _native_polars_correlation_eligible(frame, columns, method, threshold):
          return cast(
              CorrelationThresholdArtifact,
              {
                  "type": "correlation_threshold",
                  "columns_to_drop": _polars_correlation_columns_to_drop(
                      frame,
                      columns,
                      method,
                      threshold,
                  ),
                  "threshold": threshold,
                  "method": method,
                  "drop_columns": drop_columns,
              },
          )

  # Retain this compatibility route until Polars supports Kendall and callable correlations.
  return _fit_correlation_threshold_pandas(X, config)
  ```

  This intentionally performs native column resolution only for a raw or
  wrapped Polars frame. Generic inputs and public Pandas inputs retain the
  legacy implementation exactly.

- [ ] **Step 6: Run focused correlation coverage**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_feature_selection_gaps.py -k "correlation" -q
  ```

  Expected: PASS, including the existing apply tests and the new raw/wrapped,
  no-conversion Pearson and Spearman contract test.

- [ ] **Step 7: Run required Python static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. Use `ruff format` on changed Python files
  before rerunning the check if formatting reports a drift.

- [ ] **Step 8: Commit the native fit path**

  Run:

  ```bash
  git add skyulf-core/skyulf/preprocessing/feature_selection/correlation.py \
    skyulf-core/tests/test_feature_selection_gaps.py
  git commit -m "feat(skyulf-core): add native Polars correlation fitting" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing only the native fit path and its direct
  contract coverage.

## Task 3: Lock Fallbacks, Errors, and Threshold Boundaries

**Files:**
- Modify: `skyulf-core/tests/test_feature_selection_gaps.py:20-115`

**Interfaces:**
- Consumes: the Task 2 helpers and public
  `CorrelationThresholdCalculator.fit()`.
- Produces: regression coverage proving unsupported/native-ineligible inputs
  execute the explicit Pandas helper and retain its public behavior.

- [ ] **Step 1: Add compatibility regression tests**

  Add these tests after the Task 2 native contract test:

  ```python
  def test_correlation_threshold_threshold_equality_does_not_drop_polars_columns() -> None:
      """A correlation exactly equal to the threshold remains selected."""
      pl = pytest.importorskip("polars")
      config = {
          "columns": ["a", "b", "c"],
          "threshold": 1.0,
          "correlation_method": "pearson",
      }
      values = _correlation_parity_fixture()
      artifact = CorrelationThresholdCalculator().fit(pl.DataFrame(values), config)
      assert artifact["columns_to_drop"] == []


  @pytest.mark.parametrize("method", ["kendall", "not_a_method"])
  def test_correlation_threshold_native_ineligible_methods_use_pandas_errors(
      monkeypatch: pytest.MonkeyPatch,
      method: str,
  ) -> None:
      """Kendall and invalid methods must retain the established Pandas route."""
      pl = pytest.importorskip("polars")
      frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
      calculator = CorrelationThresholdCalculator()
      config = {"correlation_method": method}
      expected = calculator.fit(frame.to_pandas(), config) if method == "kendall" else None
      original_to_pandas = correlation_module.to_pandas
      calls: list[object] = []

      def recording_to_pandas(frame: object) -> pd.DataFrame:
          calls.append(frame)
          return original_to_pandas(frame)

      monkeypatch.setattr(correlation_module, "to_pandas", recording_to_pandas)

      if method == "kendall":
          assert calculator.fit(frame, config) == expected
      else:
          with pytest.raises(
              ValueError,
              match="method must be either 'pearson', 'spearman', 'kendall', or a callable",
          ):
              calculator.fit(frame, config)

      assert len(calls) == 1
      assert calls[0] is frame


  def test_correlation_threshold_callable_and_unsupported_dtype_use_pandas_fallback(
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """Callable methods and selected strings keep the legacy compatibility behavior."""
      pl = pytest.importorskip("polars")
      original_to_pandas = correlation_module.to_pandas
      calls: list[object] = []

      def recording_to_pandas(frame: object) -> pd.DataFrame:
          calls.append(frame)
          return original_to_pandas(frame)

      def force_correlation(_left: np.ndarray, _right: np.ndarray) -> float:
          return 1.0

      monkeypatch.setattr(correlation_module, "to_pandas", recording_to_pandas)
      calculator = CorrelationThresholdCalculator()
      callable_frame = pl.DataFrame(
          {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "c": [2.0, 3.0, 4.0]}
      )
      callable_config = {
          "columns": ["a", "b", "c"],
          "correlation_method": force_correlation,
          "threshold": 0.95,
      }
      assert calculator.fit(callable_frame, callable_config)["columns_to_drop"] == ["b", "c"]

      unsupported_frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
      with pytest.raises(ValueError, match="could not convert string to float"):
          calculator.fit(unsupported_frame, {"columns": ["a", "text"]})

      assert len(calls) == 2
      assert calls[0] is callable_frame
      assert calls[1] is unsupported_frame


  def test_correlation_threshold_unavailable_native_capability_uses_pandas_fallback(
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """A missing native capability delegates before any Polars calculation."""
      pl = pytest.importorskip("polars")
      frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
      config = {"correlation_method": "pearson"}
      calculator = CorrelationThresholdCalculator()
      expected = calculator.fit(frame.to_pandas(), config)
      original_to_pandas = correlation_module.to_pandas
      calls: list[object] = []

      def recording_to_pandas(value: object) -> pd.DataFrame:
          calls.append(value)
          return original_to_pandas(value)

      monkeypatch.setattr(correlation_module, "_polars_corr_accepts_method", lambda: False)
      monkeypatch.setattr(correlation_module, "to_pandas", recording_to_pandas)

      assert calculator.fit(frame, config) == expected
      assert len(calls) == 1
      assert calls[0] is frame
  ```

- [ ] **Step 2: Run the compatibility regression module**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_feature_selection_gaps.py -k "correlation" -q
  ```

  Expected: PASS. These tests protect legacy behavior, so they may pass on
  the pre-Task-2 baseline; their value is preventing a future native branch
  from accidentally swallowing a compatibility case.

- [ ] **Step 3: Correct only behavior exposed by the tests**

  If a regression fails, change the eligibility predicate rather than adding
  a catch around native execution:

  ```python
  return (
      isinstance(method, str)
      and method in _NATIVE_POLARS_METHODS
      and isinstance(threshold, Real)
      and _polars_corr_accepts_method()
      and all(
          frame.get_column(column).dtype in _POLARS_CORRELATION_DTYPES
          for column in columns
      )
  )
  ```

  Do not route `kendall`, a callable, a non-real threshold, or an unsupported
  selected dtype through `_polars_correlation_columns_to_drop()`.

- [ ] **Step 4: Run focused suites and required static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_feature_selection_gaps.py \
      skyulf-core/tests/test_utils.py -q && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0.

- [ ] **Step 5: Commit compatibility coverage**

  Run:

  ```bash
  git add skyulf-core/tests/test_feature_selection_gaps.py \
    skyulf-core/skyulf/preprocessing/feature_selection/correlation.py
  git commit -m "test(skyulf-core): cover correlation compatibility fallbacks" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing fallback tests and only any eligibility
  correction directly required by them.

## Task 4: Add Opt-In Promotion Benchmarks and Record the Decision

**Files:**
- Modify: `skyulf-core/tests/test_benchmarks.py:1-80`
- Modify: `temp/skyulf-core-pandas-polars-audit-2026-08-05.md:70-90` (ignored
  evidence record only after measurements)

**Interfaces:**
- Consumes: `_fit_correlation_threshold_pandas()` as the legacy raw/wrapped
  Polars baseline and public `CorrelationThresholdCalculator.fit()` as the
  native candidate.
- Produces: opt-in benchmark rows for legacy/native × raw/wrapped ×
  representative shape, plus printed isolated RSS deltas for the same cases.

- [ ] **Step 1: Add an opt-in correlated-frame benchmark fixture**

  Add these imports to `skyulf-core/tests/test_benchmarks.py`:

  ```python
  import os
  import sys

  from skyulf.engines.polars_engine import SkyulfPolarsWrapper
  from skyulf.preprocessing.feature_selection import correlation as correlation_module
  from skyulf.preprocessing.feature_selection.correlation import CorrelationThresholdCalculator
  ```

  Add these helpers below `_as_engine`:

  ```python
  _RUN_LARGE_CORRELATION_BENCHMARKS = (
      os.environ.get("SKYULF_RUN_LARGE_BENCHMARKS") == "1"
  )
  _LARGE_CORRELATION_CASE = pytest.mark.skipif(
      not _RUN_LARGE_CORRELATION_BENCHMARKS,
      reason="set SKYULF_RUN_LARGE_BENCHMARKS=1 to run large correlation benchmarks",
  )
  _CORRELATION_BENCHMARK_CASES = [
      pytest.param(100_000, 50, id="100k-x-50"),
      pytest.param(1_000_000, 20, marks=_LARGE_CORRELATION_CASE, id="1m-x-20"),
      pytest.param(50_000, 500, marks=_LARGE_CORRELATION_CASE, id="50k-x-500"),
  ]


  def _correlated_polars_frame(rows: int, columns: int):
      """Build a deterministic numeric Polars frame with missing and correlated values."""
      pl = pytest.importorskip("polars")
      rng = np.random.default_rng(20260806)
      values = rng.normal(size=(rows, columns))
      values[rng.random(values.shape) < 0.05] = np.nan
      values[:, 1] = values[:, 0] * 2.0 + 1.0
      return pl.DataFrame(values, schema=[f"feature_{index}" for index in range(columns)])


  def _correlation_benchmark_config(columns: int) -> dict[str, object]:
      """Return an explicit candidate list so every route measures identical work."""
      return {
          "columns": [f"feature_{index}" for index in range(columns)],
          "threshold": 0.95,
          "correlation_method": "pearson",
      }


  def _peak_rss_bytes() -> int:
      """Return the process maximum RSS in bytes on supported benchmark platforms."""
      if sys.platform == "win32":
          pytest.skip("isolated RSS measurement is not implemented on Windows")
      import resource

      peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
      return peak if sys.platform == "darwin" else peak * 1024
  ```

  The correlated assignment occurs after the missing-value mask so the
  correlated block has matching missingness and remains a known drop case.

- [ ] **Step 2: Add timing and isolated RSS tests**

  Add these tests below `test_pipeline_fit_benchmark`:

  ```python
  @pytest.mark.parametrize(("rows", "columns"), _CORRELATION_BENCHMARK_CASES)
  @pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
  @pytest.mark.parametrize("route", ["legacy", "native"])
  def test_correlation_threshold_fit_benchmark(benchmark, rows, columns, wrapped, route):
      """Benchmark equivalent legacy and native correlation-threshold fitting."""
      raw = _correlated_polars_frame(rows, columns)
      frame = SkyulfPolarsWrapper(raw) if wrapped else raw
      config = _correlation_benchmark_config(columns)

      if route == "legacy":
          artifact = benchmark(correlation_module._fit_correlation_threshold_pandas, frame, config)
      else:
          artifact = benchmark(CorrelationThresholdCalculator().fit, frame, config)

      assert "feature_1" in artifact["columns_to_drop"]


  @pytest.mark.skipif(
      os.environ.get("SKYULF_MEASURE_CORRELATION_RSS") != "1",
      reason="set SKYULF_MEASURE_CORRELATION_RSS=1 for isolated RSS output",
  )
  @pytest.mark.parametrize(("rows", "columns"), _CORRELATION_BENCHMARK_CASES)
  @pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
  @pytest.mark.parametrize("route", ["legacy", "native"])
  def test_correlation_threshold_fit_peak_rss(rows, columns, wrapped, route):
      """Print incremental process RSS for one separately invoked fit route."""
      raw = _correlated_polars_frame(rows, columns)
      frame = SkyulfPolarsWrapper(raw) if wrapped else raw
      config = _correlation_benchmark_config(columns)
      baseline = _peak_rss_bytes()

      if route == "legacy":
          artifact = correlation_module._fit_correlation_threshold_pandas(frame, config)
      else:
          artifact = CorrelationThresholdCalculator().fit(frame, config)

      delta = max(0, _peak_rss_bytes() - baseline)
      print(f"route={route} wrapped={wrapped} rows={rows} columns={columns} peak_rss_delta={delta}")
      assert "feature_1" in artifact["columns_to_drop"]
  ```

  Do not use `tracemalloc` for this gate: it does not measure the Rust/NumPy
  allocations that determine the benefit of avoiding conversion.

- [ ] **Step 3: Run the small timing benchmark**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_benchmarks.py --benchmark-only \
      -k "correlation_threshold_fit_benchmark and 100k-x-50" \
      --benchmark-min-rounds=1 -q
  ```

  Expected: benchmark rows for legacy/native and raw/wrapped input, each
  dropping `feature_1`. Record the mean timing for each matching pair.

- [ ] **Step 4: Run the large timing benchmark**

  Run:

  ```bash
  source .venv/bin/activate && \
    SKYULF_RUN_LARGE_BENCHMARKS=1 \
    pytest skyulf-core/tests/test_benchmarks.py --benchmark-only \
      -k "correlation_threshold_fit_benchmark" \
      --benchmark-min-rounds=1 -q
  ```

  Expected: timing rows for 100k x 50, 1M x 20, and 50k x 500 across both
  legacy/native routes and raw/wrapped inputs.

- [ ] **Step 5: Measure RSS in isolated pytest processes**

  Run each route/shape/engine case in a fresh pytest process so
  `ru_maxrss` has no prior route's peak. For example, run these four small
  cases first:

  ```bash
  source .venv/bin/activate
  SKYULF_MEASURE_CORRELATION_RSS=1 pytest skyulf-core/tests/test_benchmarks.py \
    --benchmark-only -k "peak_rss and 100k-x-50 and raw and legacy" -q -s
  SKYULF_MEASURE_CORRELATION_RSS=1 pytest skyulf-core/tests/test_benchmarks.py \
    --benchmark-only -k "peak_rss and 100k-x-50 and raw and native" -q -s
  SKYULF_MEASURE_CORRELATION_RSS=1 pytest skyulf-core/tests/test_benchmarks.py \
    --benchmark-only -k "peak_rss and 100k-x-50 and wrapped and legacy" -q -s
  SKYULF_MEASURE_CORRELATION_RSS=1 pytest skyulf-core/tests/test_benchmarks.py \
    --benchmark-only -k "peak_rss and 100k-x-50 and wrapped and native" -q -s
  ```

  Repeat the same command shape for `1m-x-20` and `50k-x-500` with
  `SKYULF_RUN_LARGE_BENCHMARKS=1`. Each output line supplies the exact
  baseline-subtracted peak RSS delta needed for comparison.

- [ ] **Step 6: Apply the promotion gate and record actual evidence**

  For each matching raw/wrapped route and shape, calculate:

  ```text
  time_reduction = 1 - native_mean_seconds / legacy_mean_seconds
  memory_reduction = 1 - native_peak_rss_delta / legacy_peak_rss_delta
  ```

  Accept the native path only if at least one representative case achieves
  `time_reduction >= 0.20` or `memory_reduction >= 0.25`, every contract test
  passes, and the Pandas fit route remains unchanged. Add the actual command,
  date, environment, route/shape values, and promotion decision to Candidate
  A in the ignored audit. If the gate fails, remove the native branch rather
  than weakening artifact or error compatibility.

- [ ] **Step 7: Run final targeted validation and static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    pytest skyulf-core/tests/test_feature_selection_gaps.py \
      skyulf-core/tests/test_utils.py -q && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. No frontend lint, typecheck, or build is
  required because no frontend source, configuration key, or enum changes.

- [ ] **Step 8: Commit benchmark coverage**

  Run:

  ```bash
  git add skyulf-core/tests/test_benchmarks.py
  git commit -m "test(skyulf-core): benchmark native correlation fitting" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing only opt-in benchmark and RSS measurement
  coverage. Do not add the ignored audit file to Git.
