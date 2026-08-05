# Polars-First Core Migration Wave 0/1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v0.7.4 silhouette-memory claim true, remove the first
unnecessary Polars-to-Pandas profiling conversions, and produce the complete
production-Core migration inventory that gates subsequent Polars-first waves.

**Architecture:** The silhouette scorer remains a scikit-learn/NumPy boundary
and gets a cap-aware label scan rather than a dataframe-engine conversion.
Profiling expectations gain native raw/wrapped Polars operations while the
Pandas and generic `to_pandas()` fallback remains public compatibility. A
source-cited Core inventory then separates proven Polars migrations from
NumPy/sklearn and Pandas compatibility boundaries.

**Tech Stack:** Python 3.12+, NumPy, pandas 2.x, Polars >=1.36, scikit-learn
>=1.4, pytest, Ruff, Ty, MkDocs.

## Global Constraints

- The implementation boundary is production `skyulf-core`; do not change
  backend/frontend contracts, package versions, dependencies, or generated assets.
- Preserve public Pandas and Polars input support, documented Pandas workflows,
  engine wrappers, catalog/split contracts, and public prediction/report types.
- Do not replace pandas imports mechanically. A retained use must be classified
  as public/third-party compatibility, a NumPy/scikit-learn boundary, or an
  evidence-gated candidate.
- In clustering evaluation, labels remain NumPy after `SklearnBridge`; do not
  create a Polars frame merely to count distinct labels.
- For more distinct predicted labels than `silhouette_sample_size`, raise a
  deterministic, clear `ValueError` before retaining state proportional to
  label cardinality.
- Preserve deterministic representative sampling, unique selected rows, the
  default cap of 10,000, `silhouette_sample_size`, and full-input
  Calinski-Harabasz/Davies-Bouldin scores for representable inputs.
- Preserve the existing invalid-cap, one-cluster, and exact-cap error behavior
  except for the intentional early error on formerly silent all-unique
  over-cap labels.
- Only the sampler correction belongs in unreleased v0.7.4. Do not amend
  commit `60a48fee`; use follow-up commits with the required Copilot trailer.
- Native Polars expectation behavior must match Pandas for nulls, float NaN,
  range boundaries, duplicate rows, missing columns, and exact
  `ExpectationError` messages.
- No frontend validation is required unless a Core public configuration or
  result-schema change is discovered; none is expected in these tasks.

---

## File Structure

- `skyulf-core/skyulf/modeling/_evaluation/metrics.py` — cap-aware,
  NumPy/Python-only clustering label analysis and reservoir sampling.
- `skyulf-core/tests/test_evaluation_clustering.py` — high-cardinality,
  bounded-memory, error-contract, and deterministic sampling regressions.
- `skyulf-core/skyulf/profiling/expect.py` — native raw/wrapped Polars
  expectation branches with unchanged Pandas compatibility fallback.
- `skyulf-core/tests/test_expect.py` — Pandas/raw-Polars/wrapper parity tests.
- `skyulf-core/tests/test_profiling_expect_gap.py` — unsupported-frame
  regression retained after native Polars routing.
- `temp/skyulf-core-pandas-polars-audit-2026-08-05.md` — ignored,
  source-cited production-Core inventory and migration decision record.
- `temp/skyulf-platform-evolution-roadmap-2026-08-05.md` — ignored roadmap
  citation/rebaseline verification after the sampler repair.

## Task 1: Bound High-Cardinality Silhouette Sampling

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_evaluation/metrics.py:31-76,266-310`
- Modify: `skyulf-core/tests/test_evaluation_clustering.py:1-287`

**Interfaces:**
- Consumes: `labels: np.ndarray`, `silhouette_sample_size: int`, and
  `random_state: int` from `calculate_clustering_metrics()`.
- Produces: `_collect_silhouette_representatives(labels, sample_size)` with
  at most `sample_size` retained representative indices, and an unchanged
  public `calculate_clustering_metrics()` signature/result schema for
  representable inputs.

- [ ] **Step 1: Add failing high-cardinality behavior and allocation tests**

  Add these tests to `skyulf-core/tests/test_evaluation_clustering.py` after
  the existing sampler memory test. Build the arrays before enabling
  `tracemalloc` so the assertion measures only evaluation allocations.

  ```python
  def test_calculate_clustering_metrics_rejects_high_cardinality_before_unbounded_counting() -> None:
      """Over-cap distinct labels must fail before building a full unique-label result."""
      sample_size = 10
      n_samples = 1_000_000
      X = np.zeros((n_samples, 1), dtype=np.int8)
      labels = np.arange(n_samples)

      gc.collect()
      tracemalloc.start()
      tracemalloc.reset_peak()
      try:
          with pytest.raises(
              ValueError,
              match="silhouette_sample_size=10 is too small for more than 10 clusters",
          ):
              calculate_clustering_metrics(X, labels, silhouette_sample_size=sample_size)
          _, peak_bytes = tracemalloc.get_traced_memory()
      finally:
          tracemalloc.stop()

      assert peak_bytes < 2 * 1024 * 1024


  def test_calculate_clustering_metrics_rejects_all_unique_labels_above_cap() -> None:
      """All-unique labels above the cap use the same explicit resource boundary."""
      X = np.zeros((11, 1), dtype=float)
      labels = np.arange(11)

      with pytest.raises(
          ValueError,
          match="silhouette_sample_size=10 is too small for more than 10 clusters",
      ):
          calculate_clustering_metrics(X, labels, silhouette_sample_size=10)
  ```

  Keep the existing equal-cap test and sparse-string deterministic tests. They
  lock the compatibility behavior that the new collection helper must retain.

- [ ] **Step 2: Run the new tests and confirm the current behavior fails**

  Run:

  ```bash
  source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py \
    -k "high_cardinality or all_unique_labels_above_cap" -q
  ```

  Expected: FAIL because the current all-unique path calls `pd.unique()` and
  returns `n_clusters` instead of raising. The allocation assertion may also
  exceed the 2 MiB boundary.

- [ ] **Step 3: Add a bounded representative collector**

  In `skyulf-core/skyulf/modeling/_evaluation/metrics.py`, add these private
  helpers near the silhouette constants. Keep the existing `Any` import and
  add no dataframe-engine dependency.

  ```python
  _NAN_CLUSTER_LABEL = object()


  def _cluster_label_key(label: Any) -> Any:
      """Return a stable dictionary key for a scalar predicted cluster label."""
      if isinstance(label, (float, np.floating)) and np.isnan(label):
          return _NAN_CLUSTER_LABEL
      return label


  def _collect_silhouette_representatives(
      labels: np.ndarray,
      *,
      sample_size: int,
  ) -> dict[Any, int]:
      """Retain first cluster occurrences without exceeding the scoring cap."""
      representatives: dict[Any, int] = {}
      for index, label in enumerate(labels):
          key = _cluster_label_key(label)
          if key in representatives:
              continue
          if len(representatives) == sample_size:
              raise ValueError(
                  f"silhouette_sample_size={sample_size} is too small for more than "
                  f"{sample_size} clusters; increase it above the number of clusters "
                  "when scoring datasets larger than the cap"
              )
          representatives[key] = int(index)
      return representatives
  ```

  This gives repeated floating-point NaN labels one stable representative,
  matching the old distinct-count intent without using pandas. It deliberately
  stops at the first cap-plus-one distinct label.

- [ ] **Step 4: Reuse the collector for counting and sampling**

  Replace `pd.unique(labels_np)` with the collector result in
  `calculate_clustering_metrics()`:

  ```python
  representative_by_label = _collect_silhouette_representatives(
      labels_np,
      sample_size=silhouette_sample_size,
  )
  n_unique = len(representative_by_label)
  metrics: dict[str, float] = {"n_clusters": float(n_unique)}
  ```

  Extend `_select_silhouette_sample_indices()` with an optional private
  `representative_by_label: dict[Any, int] | None = None` keyword. When no
  representatives are supplied, collect them with the helper so direct tests
  keep their current call shape. When supplied, reuse them rather than
  rescanning into a second dictionary:

  ```python
  if n_samples <= sample_size:
      return np.arange(n_samples, dtype=int)

  representatives = (
      representative_by_label
      if representative_by_label is not None
      else _collect_silhouette_representatives(labels, sample_size=sample_size)
  )
  n_clusters = len(representatives)
  if sample_size <= n_clusters:
      raise ValueError(
          f"silhouette_sample_size={sample_size} is too small for {n_clusters} clusters; "
          "increase it above the number of clusters when scoring datasets larger than the cap"
      )

  required_indices = list(representatives.values())
  required_index_set = set(required_indices)
  ```

  Leave the existing reservoir loop unchanged after this setup. Pass
  `representative_by_label=representative_by_label` from
  `calculate_clustering_metrics()` when it calls the selector. Do not change
  the public function signature or sample metric keys.

- [ ] **Step 5: Run focused clustering coverage**

  Run:

  ```bash
  source .venv/bin/activate && pytest skyulf-core/tests/test_evaluation_clustering.py -q
  ```

  Expected: PASS, including the existing deterministic sparse-string,
  equal-cap, bounded-memory, and new high-cardinality tests.

- [ ] **Step 6: Run required Python static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. Fix only diagnostics caused by this task.

- [ ] **Step 7: Commit the sampler repair**

  Run:

  ```bash
  git add skyulf-core/skyulf/modeling/_evaluation/metrics.py \
    skyulf-core/tests/test_evaluation_clustering.py
  git commit -m "fix(skyulf-core): bound high-cardinality silhouette sampling" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing only the sampler implementation and its
  clustering regression tests.

## Task 2: Add Native Polars Profiling Expectations

**Files:**
- Modify: `skyulf-core/skyulf/profiling/expect.py:1-122`
- Modify: `skyulf-core/tests/test_expect.py:1-71`
- Modify: `skyulf-core/tests/test_profiling_expect_gap.py:1-10`

**Interfaces:**
- Consumes: raw `pl.DataFrame`, `SkyulfPolarsWrapper`, `pd.DataFrame`, and
  existing objects exposing `to_pandas()`.
- Produces: unchanged public `expect_columns_exist`, `expect_no_nulls`,
  `expect_value_range`, and `expect_unique` signatures and messages, with
  direct Polars execution for raw/wrapped Polars frames.

- [ ] **Step 1: Add failing raw/wrapped Polars parity tests**

  Add a helper and parity tests to `skyulf-core/tests/test_expect.py`:

  ```python
  from skyulf.engines.polars_engine import SkyulfPolarsWrapper


  def _polars_variants(data: dict[str, object]) -> list[object]:
      """Return raw and wrapped Polars frames with equivalent contents."""
      pl = pytest.importorskip("polars")
      raw = pl.DataFrame(data)
      return [raw, SkyulfPolarsWrapper(raw)]


  def test_polars_expectations_match_pandas_null_nan_and_range_messages() -> None:
      """Raw and wrapped Polars frames preserve Pandas expectation semantics."""
      pandas_frame = pd.DataFrame({"value": [1.0, float("nan"), None, 3.0]})
      with pytest.raises(ExpectationError) as pandas_null_error:
          expect_no_nulls(pandas_frame)
      with pytest.raises(ExpectationError) as pandas_range_error:
          expect_value_range(pandas_frame, "value", maximum=2)

      for frame in _polars_variants({"value": [1.0, float("nan"), None, 3.0]}):
          with pytest.raises(ExpectationError) as polars_null_error:
              expect_no_nulls(frame)
          with pytest.raises(ExpectationError) as polars_range_error:
              expect_value_range(frame, "value", maximum=2)
          assert str(polars_null_error.value) == str(pandas_null_error.value)
          assert str(polars_range_error.value) == str(pandas_range_error.value)


  def test_polars_expect_unique_matches_pandas_for_raw_and_wrapped_frames() -> None:
      """Duplicate-row counts and messages match the Pandas path."""
      pandas_frame = pd.DataFrame({"left": [1, 1, 2], "right": ["a", "a", "b"]})
      with pytest.raises(ExpectationError) as pandas_error:
          expect_unique(pandas_frame, ["left", "right"])

      for frame in _polars_variants({"left": [1, 1, 2], "right": ["a", "a", "b"]}):
          with pytest.raises(ExpectationError) as polars_error:
              expect_unique(frame, ["left", "right"])
          assert str(polars_error.value) == str(pandas_error.value)


  def test_polars_columns_and_strict_bounds_match_pandas_messages() -> None:
      """Missing-column and exclusive-bound failures stay byte-for-byte compatible."""
      pandas_frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
      with pytest.raises(ExpectationError) as pandas_columns_error:
          expect_columns_exist(pandas_frame, ["missing"])
      with pytest.raises(ExpectationError) as pandas_bound_error:
          expect_value_range(pandas_frame, "value", minimum=1, inclusive=False)

      for frame in _polars_variants({"value": [1.0, 2.0, 3.0]}):
          with pytest.raises(ExpectationError) as polars_columns_error:
              expect_columns_exist(frame, ["missing"])
          with pytest.raises(ExpectationError) as polars_bound_error:
              expect_value_range(frame, "value", minimum=1, inclusive=False)
          assert str(polars_columns_error.value) == str(pandas_columns_error.value)
          assert str(polars_bound_error.value) == str(pandas_bound_error.value)


  def test_polars_expectations_do_not_convert_to_pandas(
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """Native Polars expectation paths must not route through to_pandas()."""
      import skyulf.profiling.expect as expectation_module

      def fail_to_pandas(*_args: object, **_kwargs: object) -> None:
          raise AssertionError("unexpected pandas conversion")

      monkeypatch.setattr(expectation_module, "_as_pandas", fail_to_pandas)
      for frame in _polars_variants({"value": [1.0, 2.0, 3.0]}):
          expect_columns_exist(frame, ["value"])
          expect_no_nulls(frame)
          expect_value_range(frame, "value", minimum=1, maximum=3)
          expect_unique(frame, ["value"])
  ```

  Keep `test_expect_no_nulls_raises_type_error_for_unsupported_frame` in
  `test_profiling_expect_gap.py` unchanged to protect the generic fallback
  contract.

- [ ] **Step 2: Run the new tests and confirm the current paths convert**

  Run:

  ```bash
  source .venv/bin/activate && pytest skyulf-core/tests/test_expect.py \
    skyulf-core/tests/test_profiling_expect_gap.py -q
  ```

  Expected: FAIL with `AssertionError: unexpected pandas conversion` from
  `test_polars_expectations_do_not_convert_to_pandas`, proving the current
  raw/wrapped Polars path still calls `_as_pandas()`.

- [ ] **Step 3: Add explicit raw/wrapped Polars routing**

  In `skyulf-core/skyulf/profiling/expect.py`, import Polars and the wrapper
  because both are required Core dependencies/contracts:

  ```python
  import polars as pl

  from ..engines.polars_engine import SkyulfPolarsWrapper
  ```

  Add a helper that returns the native frame without converting it:

  ```python
  def _as_polars(df: Any) -> pl.DataFrame | None:
      """Return a raw Polars frame when the input can stay native."""
      if isinstance(df, pl.DataFrame):
          return df
      if isinstance(df, SkyulfPolarsWrapper):
          return df._df
      return None
  ```

  Keep `_as_pandas()` exactly as the fallback for Pandas and arbitrary
  `to_pandas()` implementations. Change `_resolve_columns()` to accept `Any`
  and read `.columns`, so both frame engines use the same missing-column
  validation.

- [ ] **Step 4: Implement native checks without changing public messages**

  Add an internal null counter and route each public expectation through
  `_as_polars()` first:

  ```python
  def _polars_null_count(series: pl.Series) -> int:
      """Count null and floating NaN values with Pandas isnull semantics."""
      count = int(series.null_count())
      if series.dtype.is_float():
          count += int(series.is_nan().sum())
      return count
  ```

  Implement the native branches as follows:

  ```python
  # expect_columns_exist
  frame = _as_polars(df)
  columns_in_frame = list(frame.columns) if frame is not None else list(_as_pandas(df).columns)

  # expect_no_nulls
  frame = _as_polars(df)
  if frame is not None:
      cols = _resolve_columns(frame, columns)
      offenders = {
          column: count
          for column in cols
          if (count := _polars_null_count(frame.get_column(column))) > 0
      }

  # expect_value_range
  frame = _as_polars(df)
  if frame is not None:
      expect_columns_exist(frame, [column])
      series = frame.get_column(column)
      if series.dtype.is_float():
          series = series.fill_nan(None)
      series = series.drop_nulls()
      _check_lower_bound(series, column, minimum, inclusive)
      _check_upper_bound(series, column, maximum, inclusive)

  # expect_unique
  frame = _as_polars(df)
  if frame is not None:
      expect_columns_exist(frame, columns)
      dup_count = int(frame.select(list(columns)).is_duplicated().sum())
  ```

  Use the unchanged existing exception strings after calculating
  `missing`, `offenders`, `series`, or `dup_count`. Do not call
  `_as_pandas()` from a raw/wrapped Polars branch.

- [ ] **Step 5: Run focused expectation coverage**

  Run:

  ```bash
  source .venv/bin/activate && pytest skyulf-core/tests/test_expect.py \
    skyulf-core/tests/test_profiling_expect_gap.py -q
  ```

  Expected: PASS for Pandas, raw Polars, wrapped Polars, NaN/null, bounds,
  duplicate, missing-column, and unsupported-frame behavior.

- [ ] **Step 6: Run required Python static checks**

  Run:

  ```bash
  source .venv/bin/activate && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. Do not suppress a typing diagnostic with an
  unsafe cast; narrow raw/wrapped Polars inputs through `_as_polars()`.

- [ ] **Step 7: Commit the native expectation paths**

  Run:

  ```bash
  git add skyulf-core/skyulf/profiling/expect.py \
    skyulf-core/tests/test_expect.py \
    skyulf-core/tests/test_profiling_expect_gap.py
  git commit -m "feat(skyulf-core): add native Polars expectations" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Expected: one commit containing only the expectation implementation and
  parity coverage.

## Task 3: Produce the Complete Production-Core Pandas/Polars Inventory

**Files:**
- Create: `temp/skyulf-core-pandas-polars-audit-2026-08-05.md`
- Inspect: `skyulf-core/setup.py`
- Inspect: `skyulf-core/skyulf/**/*.py`

**Interfaces:**
- Consumes: the completed Wave 0/1 commits and the production Core source tree.
- Produces: an ignored, source-cited audit that assigns every production
  Pandas use and conversion to a migration category and a next decision.

- [ ] **Step 1: Generate a reproducible production-source inventory**

  Run these commands from the repository root:

  ```bash
  git grep -n -E 'pandas|pd\.|\.to_pandas\(' \
    -- skyulf-core/skyulf skyulf-core/setup.py
  git ls-files 'skyulf-core/skyulf/**/*.py' 'skyulf-core/skyulf/*.py' | sort
  ```

  Review every returned production source site. Do not include tests,
  notebooks, generated artifacts, or `temp/` files in the migration count.
  Record whether the site is an import/type-only use, an actual Pandas
  operation, a `to_pandas()` conversion, or a third-party boundary.

- [ ] **Step 2: Write the source-cited decision record**

  Create `temp/skyulf-core-pandas-polars-audit-2026-08-05.md` with this
  structure and fill every row with current source evidence:

  ```markdown
  # Core Pandas/Polars Migration Audit

  ## Scope and Method

  ## Inventory Summary

  | Category | Count | Decision rule |
  | --- | ---: | --- |

  ## Site Decisions

  | Source | Current operation/conversion | Category | Dual-engine risk | Evidence required | Smallest next decision |
  | --- | --- | --- | --- | --- | --- |

  ## Implemented Wave 0/1

  ## Wave 2 Benchmark Candidates

  ## Retained Compatibility Boundaries

  ## Documentation and Release Implications
  ```

  Use only these categories:

  - **Polars-native opportunity** — migrate after behavior parity is proved.
  - **NumPy/scikit-learn boundary** — keep NumPy conversion; Polars adds no
    value after that boundary.
  - **Required Pandas compatibility** — keep public/third-party behavior.
  - **Needs evidence** — require a parity matrix and benchmark before code
    changes.

  Include, at minimum, the completed sampler, `profiling/expect.py`,
  preprocessing correlation selection, clustering numeric filtering,
  profiling multivariate fallback, engine protocols/wrappers,
  sklearn/imblearn/SHAP boundaries, catalog/split contracts, and package
  dependency/documentation implications. Expand this list until every
  inventory source site has a row or is explicitly grouped with the exact
  source list.

- [ ] **Step 3: Define executable Wave 2 promotion gates in the audit**

  For each Wave 2 candidate, specify:

  ```markdown
  - Pandas/raw-Polars/wrapped-Polars parity fixture and exact asserted output.
  - Null/NaN, dtype, column-order, schema, and error-message cases.
  - Representative time and peak-memory benchmark inputs.
  - The observable benefit required to justify migration.
  - The rollback condition: retain the current boundary if parity or benefit fails.
  ```

  For preprocessing correlation selection, include selected-column parity.
  For clustering numeric filtering, include mixed-dtype labels/metrics parity.
  For profiling multivariate fallback, include null-heavy and all-null
  matrices plus fallback-frequency evidence.

- [ ] **Step 4: Validate the ignored artifact and inventory completeness**

  Run:

  ```bash
  git check-ignore -v temp/skyulf-core-pandas-polars-audit-2026-08-05.md
  if grep -nE 'TBD|TODO|FIXME' temp/skyulf-core-pandas-polars-audit-2026-08-05.md; then
    exit 1
  fi
  git diff --check
  ```

  Expected: the audit is ignored by `temp/`, contains no placeholder markers,
  and has no whitespace errors. Do not create an empty commit for this
  ignored deliverable.

## Task 4: Reconcile v0.7.4 Documentation Evidence

**Files:**
- Modify if line references changed:
  `temp/skyulf-platform-evolution-roadmap-2026-08-05.md`
- Inspect: `changelog/0.7.x.md`
- Inspect: `CHANGELOG.md`

**Interfaces:**
- Consumes: the Task 1 sampler implementation and its focused validation.
- Produces: a current, evidence-backed roadmap rebaseline while retaining the
  already-approved v0.7.4 release-note wording.

- [ ] **Step 1: Verify the bounded-memory claim against live code**

  Check that `calculate_clustering_metrics()` no longer invokes `pd.unique`
  and that the representative collector stops on the cap-plus-one label:

  ```bash
  if git grep -n 'pd.unique' -- skyulf-core/skyulf/modeling/_evaluation/metrics.py; then
    exit 1
  fi
  git grep -n '_collect_silhouette_representatives' \
    -- skyulf-core/skyulf/modeling/_evaluation/metrics.py
  ```

  Expected: the first command has no output; the second identifies the
  collector definition and its calculation/sampler callers.

- [ ] **Step 2: Reconcile the ignored roadmap**

  Update only stale `metrics.py` source line ranges in
  `temp/skyulf-platform-evolution-roadmap-2026-08-05.md`. Retain the
  statement that the sampler uses bounded intermediate label-selection state,
  but do not broaden it into a claim that all clustering evaluation memory is
  bounded.

  Do not edit `changelog/0.7.x.md` or `CHANGELOG.md`: after Task 1, their
  current v0.7.4 wording is truthful and already committed in `60a48fee`.

- [ ] **Step 3: Validate documentation and boundaries**

  Run:

  ```bash
  source .venv/bin/activate && mkdocs build
  git diff --check
  git check-ignore -v temp/skyulf-platform-evolution-roadmap-2026-08-05.md
  git diff -- changelog/0.7.x.md CHANGELOG.md
  ```

  Expected: MkDocs exits 0; the roadmap remains ignored; no whitespace errors
  occur; the tracked changelog diff is empty. Record existing MkDocs advisory
  or navigation messages separately from a build failure.

## Task 5: Run the Combined Core Gate and Final Review

**Files:**
- Inspect: Task 1 and Task 2 changed files, both ignored audit artifacts, and
  the v0.7.4 documentation commit.

**Interfaces:**
- Consumes: committed sampler and expectation changes plus both ignored
  evidence artifacts.
- Produces: a reviewable, validated Wave 0/1 range and clear follow-on
  decisions for Wave 2/3.

- [ ] **Step 1: Run combined focused regression coverage**

  Run:

  ```bash
  source .venv/bin/activate && pytest \
    skyulf-core/tests/test_evaluation_clustering.py \
    skyulf-core/tests/test_expect.py \
    skyulf-core/tests/test_profiling_expect_gap.py -q
  ```

  Expected: PASS.

- [ ] **Step 2: Run the required repository Python gates**

  Run:

  ```bash
  source .venv/bin/activate && \
    ruff check . && \
    ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py && \
    ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py
  ```

  Expected: all commands exit 0. Fix only diagnostics caused by this plan.

- [ ] **Step 3: Check commit and ignored-artifact boundaries**

  Run:

  ```bash
  git log --oneline --decorate -8
  git diff --check
  git status --short
  git check-ignore -v temp/skyulf-core-pandas-polars-audit-2026-08-05.md
  git check-ignore -v temp/skyulf-platform-evolution-roadmap-2026-08-05.md
  ```

  Expected: code commits contain only their respective implementation/tests,
  both `temp/` artifacts remain ignored/untracked, and no unintended tracked
  files are staged. Do not create an empty validation commit.

- [ ] **Step 4: Request final whole-range review**

  Build a review package from the commit immediately before Task 1 through
  the current `HEAD`. The reviewer must inspect both ignored artifacts
  directly, confirm that the v0.7.4 memory claim is now supported by the
  live implementation, verify Pandas/raw-Polars/wrapper parity evidence, and
  ensure the inventory does not overstate which Pandas paths are migratable.

  Address every Critical or Important review finding before scheduling a Wave
  2 implementation plan.

## Follow-On Planning Boundary

This plan intentionally stops after the complete inventory and the first
high-confidence Polars-native migration. Wave 2 and Wave 3 source changes are
not pre-authorized by import count: each audit entry advances only after its
specified parity matrix and benchmark show a real benefit. Create the next
implementation plan from the completed audit, grouping only candidates with
independent files and proven compatible semantics.
