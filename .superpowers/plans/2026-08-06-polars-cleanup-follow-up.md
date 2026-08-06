# Polars Cleanup Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove five verified low-risk Pandas conversion and encoder duplication findings while preserving all public APIs and outputs.

**Architecture:** Reuse the existing narrow-column conversion helpers at sklearn boundaries and add one engine-agnostic target-extraction helper for supervised encoders. Keep Pandas-specific text assembly where it is still required, but narrow Polars frames before crossing that boundary.

**Tech Stack:** Python 3.12, Polars, Pandas, NumPy, scikit-learn, pytest, Ruff, ty.

## Global Constraints

- Existing public function and calculator signatures remain unchanged.
- Pandas and Polars callers remain supported.
- Existing missing-column, missing-target, dtype, and sklearn exceptions remain visible.
- Do not introduce a broad dataframe coercion abstraction or unrelated refactor.
- Keep the v0.7.4 changelog update concise.

---

### Task 1: Remove NumPy-Compatible Pandas Hops

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py`
- Modify: `skyulf-core/skyulf/preprocessing/transformations/general.py`
- Test: `skyulf-core/tests/test_feature_generation_gaps.py`
- Test: `skyulf-core/tests/test_transformations_general.py`

**Interfaces:**
- Consumes: `select_then_to_numpy(X: Any, requested: Iterable[str]) -> tuple[np.ndarray, list[str]]`
- Produces: unchanged `PolynomialFeaturesApplier.apply()` and `GeneralTransformationCalculator.fit()` behavior.

- [ ] **Step 1: Add failing no-Pandas-hop tests**

Add to `skyulf-core/tests/test_feature_generation_gaps.py`:

```python
def test_polynomial_features_apply_polars_skips_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polars polynomial apply should feed numpy directly to sklearn."""
    import polars as pl

    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    artifact = PolynomialFeaturesCalculator().fit(
        train, {"columns": ["a", "b"], "degree": 2}
    )
    frame = pl.from_pandas(train)

    def fail_to_pandas(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("polynomial apply converted Polars input to Pandas")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", fail_to_pandas)
    output = PolynomialFeaturesApplier().apply(frame, artifact)

    assert output.get_column("poly_a_b").to_list() == [4.0, 10.0, 18.0]
```

Add to `skyulf-core/tests/test_transformations_general.py`:

```python
def test_fit_power_for_polars_column_skips_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polars PowerTransformer fit should consume a numpy column directly."""
    frame = pl.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, 16.0]})

    def fail_to_pandas(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("power fit converted a Polars Series to Pandas")

    monkeypatch.setattr(pl.Series, "to_pandas", fail_to_pandas)
    artifact = _fit_power_for_column(frame, "a", "yeo-johnson", is_polars=True)

    assert len(artifact["lambdas"]) == 1
```

- [ ] **Step 2: Run the tests and confirm the current implementation fails**

Run:

```bash
cd /Users/BH7043/Skyulf/skyulf-core
source /Users/BH7043/Skyulf/.venv/bin/activate
python -m pytest \
  tests/test_feature_generation_gaps.py::test_polynomial_features_apply_polars_skips_pandas \
  tests/test_transformations_general.py::test_fit_power_for_polars_column_skips_pandas -q
```

Expected: both tests fail with the explicit `AssertionError` messages.

- [ ] **Step 3: Pass narrow numpy arrays directly to sklearn**

In `polynomial.py`, make `_polynomial_compute()` accept a numpy-compatible
matrix and use `select_then_to_numpy()` in both apply paths:

```python
def _polynomial_compute(
    X_subset: Any, valid_cols: list[str], params: dict[str, Any]
) -> tuple[Any, list[str]] | None:
    """Run sklearn PolynomialFeatures + name normalisation; ``None`` means skip."""
```

```python
X_np, valid_cols = select_then_to_numpy(X, params.get("columns", []))
if not valid_cols:
    return X, _y
result = _polynomial_compute(X_np, valid_cols, params)
```

Use the same block in `_polynomial_apply_polars()` and
`_polynomial_apply_pandas()`; retain their existing engine-specific output
construction and Pandas index preservation.

In `general.py`, replace the Polars branch of `_fit_power_for_column()`:

```python
if is_polars:
    column_values = X[col].to_numpy()
    fit_values = column_values.reshape(-1, 1)
else:
    column_values = X[col]
    fit_values = X[[col]]

if method == "box-cox" and (column_values <= 0).any():
    logger.warning(
        f"Skipping Box-Cox for column {col} because it contains non-positive values."
    )
    return {}

pt = PowerTransformer(method=method, standardize=True)
pt.fit(fit_values)
```

- [ ] **Step 4: Run focused behavior and parity tests**

Run:

```bash
python -m pytest \
  tests/test_feature_generation_gaps.py \
  tests/test_transformations_general.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  skyulf-core/skyulf/preprocessing/feature_generation/polynomial.py \
  skyulf-core/skyulf/preprocessing/transformations/general.py \
  skyulf-core/tests/test_feature_generation_gaps.py \
  skyulf-core/tests/test_transformations_general.py
git commit -m "refactor(skyulf-core): skip Pandas in polynomial and power paths

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Narrow Text Frames Before Pandas Conversion

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/vectorization/_common.py`
- Test: `skyulf-core/tests/test_vectorization_gaps.py`

**Interfaces:**
- Consumes: `resolve_valid_columns(X: Any, columns: Iterable[str]) -> list[str]`
- Produces: `resolve_fit_text_columns(X, config) -> tuple[pd.DataFrame, list[str]] | None`, now returning a frame narrowed to the resolved text columns.

- [ ] **Step 1: Add a failing frame-narrowing test**

Add imports for `polars as pl` and
`resolve_fit_text_columns` to `test_vectorization_gaps.py`, then add:

```python
def test_resolve_fit_text_columns_narrows_before_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only resolved text columns should cross the Polars-to-Pandas boundary."""
    frame = pl.DataFrame(
        {
            "title": ["hello", "goodbye"],
            "body": ["world", "moon"],
            "unused_numeric": [1, 2],
        }
    )
    converted_columns: list[list[str]] = []
    original_to_pandas = pl.DataFrame.to_pandas

    def tracked_to_pandas(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        converted_columns.append(self.columns)
        return original_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", tracked_to_pandas)
    resolved = resolve_fit_text_columns(
        frame, {"columns": ["title", "body", "missing"]}
    )

    assert resolved is not None
    frame_pd, columns = resolved
    assert columns == ["title", "body"]
    assert frame_pd.columns.tolist() == ["title", "body"]
    assert converted_columns == [["title", "body"]]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest \
  tests/test_vectorization_gaps.py::test_resolve_fit_text_columns_narrows_before_pandas -q
```

Expected: failure because the current conversion sees all three input columns
and returns the full Pandas frame.

- [ ] **Step 3: Resolve columns natively, then convert only the subset**

Replace the body after the configured-column guard in
`resolve_fit_text_columns()`:

```python
valid_cols = resolve_valid_columns(X, cols)
if not valid_cols:
    return None

if hasattr(X, "to_pandas") and not isinstance(X, pd.DataFrame):
    X_pd = X.select(valid_cols).to_pandas()
else:
    X_pd = X[valid_cols]

return X_pd, valid_cols
```

Update its docstring to say the returned Pandas frame contains only resolved
text columns.

- [ ] **Step 4: Run Count/TF-IDF and vectorization tests**

Run:

```bash
python -m pytest \
  tests/test_vectorization_gaps.py \
  tests/test_text_vectorization.py -q
```

Expected: all tests pass and Count/TF-IDF artifacts remain unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  skyulf-core/skyulf/preprocessing/vectorization/_common.py \
  skyulf-core/tests/test_vectorization_gaps.py
git commit -m "refactor(skyulf-core): narrow text frames before Pandas conversion

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Consolidate Supervised Encoder Target Extraction

**Files:**
- Modify: `skyulf-core/skyulf/preprocessing/encoding/_common.py`
- Modify: `skyulf-core/skyulf/preprocessing/encoding/woe.py`
- Modify: `skyulf-core/skyulf/preprocessing/encoding/target.py`
- Test: `skyulf-core/tests/test_encoding_woe.py`
- Test: `skyulf-core/tests/test_encoding_target.py`

**Interfaces:**
- Produces: `_extract_target(X: Any, y: Any, target_col: str | None) -> Any`
- Consumes: `select_then_to_pandas(X: Any, requested: Iterable[str]) -> pd.DataFrame`
- Preserves: WOE and TargetEncoder calculator/artifact contracts.

- [ ] **Step 1: Add failing shared-helper tests**

Add `_extract_target` import from
`skyulf.preprocessing.encoding._common` to `test_encoding_target.py`, then add:

```python
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_shared_extract_target_supports_both_engines(engine: str) -> None:
    """The shared helper should resolve target_col without changing explicit y."""
    frame_pd = pd.DataFrame({"city": ["a", "b"], "target": [0, 1]})
    frame = pl.from_pandas(frame_pd) if engine == "polars" else frame_pd

    extracted = _extract_target(frame, None, "target")
    values = extracted.to_list() if engine == "polars" else extracted.tolist()
    assert values == [0, 1]

    explicit = np.array([1, 0])
    assert _extract_target(frame, explicit, "target") is explicit
    assert _extract_target(frame, None, "missing") is None
```

- [ ] **Step 2: Run the helper test and verify import failure**

Run:

```bash
python -m pytest \
  tests/test_encoding_target.py::test_shared_extract_target_supports_both_engines -q
```

Expected: collection/import fails because `_extract_target` does not exist.

- [ ] **Step 3: Add the shared helper**

Add to `encoding/_common.py`:

```python
def _extract_target(X: Any, y: Any, target_col: str | None) -> Any:
    """Return explicit y or extract target_col from a Pandas/Polars frame."""
    if y is not None or not target_col:
        return y
    if target_col not in X.columns:
        return y
    getter = getattr(X, "get_column", None)
    return getter(target_col) if getter else X[target_col]
```

- [ ] **Step 4: Replace duplicate WOE and TargetEncoder implementations**

In `woe.py`:

```python
from .._helpers import select_then_to_pandas
from ._common import (
    _exclude_target_column,
    _extract_target,
    detect_categorical_columns,
)
```

Delete `_extract_y`, `_woe_fit_polars`, and `_woe_fit_pandas`; replace them
with:

```python
def _woe_fit(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    """Fit WOE for either dataframe engine using one narrow Pandas boundary."""
    y = _extract_target(X, y, config.get("target_column"))
    if y is None:
        logger.warning("WOEEncoder requires a target variable (y). Skipping.")
        return {}
    cols = _exclude_target_column(
        resolve_columns(X, config, detect_categorical_columns),
        config,
        "WOEEncoder",
        y,
    )
    if not cols:
        return {}
    frame = select_then_to_pandas(X, cols)
    return _woe_fit_common(frame[cols], y, cols, config)
```

Pass `_woe_fit` as both `polars_func` and `pandas_func` in
`WOEEncoderCalculator.fit()`.

In `target.py`, import `_extract_target`, delete
`_maybe_extract_y_polars()`/`_maybe_extract_y_pandas()`, and replace all four
calls with:

```python
fit_y = _extract_target(X, y, config.get("target_column"))
```

Use `y = ...` in the two ordinary fit functions and `fit_y = ...` in the two
cross-fit training functions, preserving their existing return behavior.

- [ ] **Step 5: Run encoder tests**

Run:

```bash
python -m pytest \
  tests/test_encoding_woe.py \
  tests/test_encoding_target.py \
  tests/test_woe_and_calibration.py -q
```

Expected: all tests pass for explicit `y`, embedded `target_column`, missing
targets, Pandas, and Polars.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  skyulf-core/skyulf/preprocessing/encoding/_common.py \
  skyulf-core/skyulf/preprocessing/encoding/woe.py \
  skyulf-core/skyulf/preprocessing/encoding/target.py \
  skyulf-core/tests/test_encoding_woe.py \
  skyulf-core/tests/test_encoding_target.py
git commit -m "refactor(skyulf-core): share supervised encoder target extraction

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Document and Verify the Consolidated Cleanup

**Files:**
- Modify: `changelog/0.7.x.md`
- Modify: `.superpowers/sdd/progress.md`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: concise v0.7.4 release note and durable implementation record.

- [ ] **Step 1: Add concise documentation**

Add one v0.7.4 paragraph summarizing:

```text
Five final Polars-first cleanup findings were fixed: polynomial apply and
PowerTransformer fit now pass narrow numpy arrays directly to sklearn;
Count/TF-IDF fit narrows text columns before its required Pandas boundary;
WOE and TargetEncoder now share one engine-agnostic target extractor; and WOE
uses the shared narrow-then-Pandas helper. Public APIs and outputs are unchanged.
```

Add the detailed file/test record to `.superpowers/sdd/progress.md`, including
why Count/TF-IDF still retain a Pandas text-assembly boundary.

- [ ] **Step 2: Run formatting, lint, and type checks**

Run:

```bash
cd /Users/BH7043/Skyulf/skyulf-core
source /Users/BH7043/Skyulf/.venv/bin/activate
ruff check \
  skyulf/preprocessing/feature_generation/polynomial.py \
  skyulf/preprocessing/vectorization/_common.py \
  skyulf/preprocessing/transformations/general.py \
  skyulf/preprocessing/encoding/_common.py \
  skyulf/preprocessing/encoding/woe.py \
  skyulf/preprocessing/encoding/target.py \
  tests/test_feature_generation_gaps.py \
  tests/test_transformations_general.py \
  tests/test_vectorization_gaps.py \
  tests/test_encoding_woe.py \
  tests/test_encoding_target.py
ruff format --check \
  skyulf/preprocessing/feature_generation/polynomial.py \
  skyulf/preprocessing/vectorization/_common.py \
  skyulf/preprocessing/transformations/general.py \
  skyulf/preprocessing/encoding/_common.py \
  skyulf/preprocessing/encoding/woe.py \
  skyulf/preprocessing/encoding/target.py \
  tests/test_feature_generation_gaps.py \
  tests/test_transformations_general.py \
  tests/test_vectorization_gaps.py \
  tests/test_encoding_woe.py \
  tests/test_encoding_target.py
ty check \
  skyulf/preprocessing/feature_generation/polynomial.py \
  skyulf/preprocessing/vectorization/_common.py \
  skyulf/preprocessing/transformations/general.py \
  skyulf/preprocessing/encoding/_common.py \
  skyulf/preprocessing/encoding/woe.py \
  skyulf/preprocessing/encoding/target.py
```

Expected: all checks pass.

- [ ] **Step 3: Run the full Core suite**

Run:

```bash
python -m pytest -q
```

Expected baseline: `2918 passed, 69 skipped, 1 xfailed`, plus the newly added
tests; no existing tests regress.

- [ ] **Step 4: Review the complete diff**

Run:

```bash
cd /Users/BH7043/Skyulf
git --no-pager diff --check
git --no-pager diff HEAD~3 -- \
  skyulf-core/skyulf/preprocessing \
  skyulf-core/tests \
  changelog/0.7.x.md \
  .superpowers/sdd/progress.md
```

Expected: only the five approved findings, focused tests, and documentation
are present.

- [ ] **Step 5: Commit documentation**

```bash
git add changelog/0.7.x.md .superpowers/sdd/progress.md
git commit -m "docs: record final Polars cleanup follow-up

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 6: Request final code review**

Dispatch a read-only reviewer over the Task 1-4 commit range. Fix every
Critical or Important issue, rerun the relevant focused tests and full suite,
and document any rejected finding with concrete code/test evidence.
