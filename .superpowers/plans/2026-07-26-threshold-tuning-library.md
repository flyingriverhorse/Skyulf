# Threshold Tuning (Library Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable, metric-configurable decision-threshold-tuning API to
`skyulf-core` — standalone `optimize_thresholds()`/`apply_thresholds()`
functions plus a thin `SkyulfPipeline` convenience wrapper — so callers no
longer have to hand-roll a Nelder-Mead/grid search per competition.

**Architecture:** New `skyulf-core/skyulf/modeling/_evaluation/thresholds.py`
module implements the pure array-level search/apply logic (grid search for
binary, `scipy.optimize.minimize(method="Nelder-Mead")` for multiclass,
scaled-argmax decision rule). `SkyulfPipeline` gets two additions:
`optimize_thresholds(X_val, y_val, metric, ...)` (computes `predict_proba` on
caller-supplied validation data and delegates to the standalone function,
storing the result on `self._tuned_thresholds`) and a new
`use_tuned_thresholds: bool = False` parameter on the existing `predict()`.

**Tech Stack:** Python 3.11+, NumPy, pandas, scikit-learn, SciPy (already a
core dependency — no new dependency added), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-26-threshold-tuning-design.md` — this
  plan implements that spec's Phase 1 (library-only) scope exactly. Do not
  add DB/backend/frontend changes — those are explicitly out of scope here.
- `metric` is a fully caller-supplied callable `(y_true, y_pred) -> float` to
  **maximize**. The library ships no default metric.
- `classes` defaults to `sorted(np.unique(y_true))` when not given (matches
  existing `_evaluation` conventions).
- `strategy` auto-selects `"grid"` for exactly 2 classes, `"nelder-mead"` for
  3+ classes, when `strategy=None`.
- Multiclass decision rule is **scaled argmax**:
  `classes[argmax(y_proba / thresholds_array, axis=1)]`.
- `SkyulfPipeline.optimize_thresholds()` requires explicit caller-supplied
  `(X_val, y_val)` — never reuses the pipeline's internal train/test split.
- `predict(use_tuned_thresholds=True)` must raise `ValueError` if
  `optimize_thresholds()` was never called on that pipeline instance — no
  silent fallback.
- Follow repo lint/type/test gates for every touched file: `ruff check`,
  `ruff format --check`, `ty check`, `pytest` (see repo-wide coding
  instructions). Run these before each commit, not just at the end.

---

## File Structure

- **Create** `skyulf-core/skyulf/modeling/_evaluation/thresholds.py` — pure
  functions `optimize_thresholds()` and `apply_thresholds()`. No pipeline/
  pandas-DataFrame awareness beyond accepting array-likes; single
  responsibility (search + apply a decision rule).
- **Modify** `skyulf-core/skyulf/modeling/_evaluation/__init__.py` — export
  the two new functions, following the existing pattern for
  `calculate_classification_metrics` etc.
- **Modify** `skyulf-core/skyulf/modeling/__init__.py` — re-export at the
  `skyulf.modeling` top level, alongside `calculate_classification_metrics`.
- **Modify** `skyulf-core/skyulf/pipeline.py` — add
  `SkyulfPipeline.optimize_thresholds()` and add the
  `use_tuned_thresholds` parameter to the existing `predict()`; add
  `self._tuned_thresholds: dict[Any, float] | None = None` initialization
  alongside the existing `self._target_column` init.
- **Create** `skyulf-core/tests/test_evaluation_thresholds.py` — tests for
  the two standalone functions.
- **Create** `skyulf-core/tests/test_pipeline_threshold_tuning.py` — tests
  for the pipeline wrapper (mirrors the existing
  `test_pipeline_split_extraction.py` file, one dedicated file per
  convenience API, matching established repo convention).
- **Modify** `skyulf-core/README.md` — add a "Threshold tuning" usage section
  right after the existing `get_fitted_split()` paragraph (before the
  "**Naming:**" paragraph, still inside the pipeline overview section).

---

### Task 1: Standalone `optimize_thresholds()` and `apply_thresholds()` functions

**Files:**
- Create: `skyulf-core/skyulf/modeling/_evaluation/thresholds.py`
- Test: `skyulf-core/tests/test_evaluation_thresholds.py`

**Interfaces:**
- Consumes: `numpy`, `scipy.optimize.minimize` (both already core
  dependencies — confirm via `skyulf-core/setup.py`).
- Produces:
  - `optimize_thresholds(y_true, y_proba, metric, classes=None, strategy=None, grid_points=101) -> dict[Any, float]`
  - `apply_thresholds(y_proba, thresholds, classes=None) -> np.ndarray`
  - Both importable as `from skyulf.modeling._evaluation.thresholds import optimize_thresholds, apply_thresholds`.
  - Later tasks (2, 3, 4) import these two names — do not rename them.

- [ ] **Step 1: Write the failing tests for `apply_thresholds` (binary and multiclass)**

Create `skyulf-core/tests/test_evaluation_thresholds.py`:

```python
"""Tests for skyulf.modeling._evaluation.thresholds (optimize_thresholds/apply_thresholds)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from skyulf.modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds


def test_apply_thresholds_binary_basic():
    """Binary: predicts positive class when proba[:, 1] >= threshold."""
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
            [0.55, 0.45],
            [0.1, 0.9],
        ]
    )
    preds = apply_thresholds(y_proba, thresholds=0.5, classes=[0, 1])
    np.testing.assert_array_equal(preds, [0, 1, 0, 1])


def test_apply_thresholds_binary_dict_form():
    """Binary thresholds may also be passed as a single-entry dict."""
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    preds = apply_thresholds(y_proba, thresholds={1: 0.5}, classes=[0, 1])
    np.testing.assert_array_equal(preds, [0, 1])


def test_apply_thresholds_multiclass_equal_thresholds_matches_argmax():
    """Equal thresholds across all classes must reduce to plain argmax."""
    y_proba = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.1, 0.3],
            [0.1, 0.1, 0.8],
        ]
    )
    classes = ["a", "b", "c"]
    preds = apply_thresholds(
        y_proba, thresholds={"a": 0.3, "b": 0.3, "c": 0.3}, classes=classes
    )
    expected = np.array(classes)[np.argmax(y_proba, axis=1)]
    np.testing.assert_array_equal(preds, expected)


def test_apply_thresholds_multiclass_scaled_argmax():
    """Dividing by a class's own threshold shifts which class 'wins' the row."""
    y_proba = np.array([[0.4, 0.4, 0.2]])
    classes = [0, 1, 2]
    # Class 1 has a much smaller threshold, so 0.4/0.1 (=4.0) beats 0.4/1.0.
    preds = apply_thresholds(
        y_proba, thresholds={0: 1.0, 1: 0.1, 2: 1.0}, classes=classes
    )
    np.testing.assert_array_equal(preds, [1])


def test_apply_thresholds_raises_on_incomplete_coverage():
    """Missing a class's threshold in the dict is a caller error, not silently ignored."""
    y_proba = np.array([[0.2, 0.5, 0.3]])
    with pytest.raises(ValueError, match="threshold"):
        apply_thresholds(y_proba, thresholds={0: 0.5, 1: 0.5}, classes=[0, 1, 2])


def test_apply_thresholds_raises_on_non_2d_proba():
    """y_proba must be 2D (n_samples, n_classes)."""
    with pytest.raises(ValueError, match="2D"):
        apply_thresholds(np.array([0.1, 0.9]), thresholds=0.5, classes=[0, 1])
```

- [ ] **Step 2: Run the tests to verify they fail (module doesn't exist yet)**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'skyulf.modeling._evaluation.thresholds'`

- [ ] **Step 3: Implement `apply_thresholds()`**

Create `skyulf-core/skyulf/modeling/_evaluation/thresholds.py`:

```python
"""Decision-threshold tuning: search per-class thresholds against a
caller-supplied metric, and apply them to convert predicted probabilities
into class predictions.

Unlike ``predict()``'s default decision rule (argmax for multiclass, 0.5 for
binary), the thresholds this module searches for are tuned against whatever
metric the caller actually cares about (F1, MCC, balanced accuracy, a custom
business metric, ...) on held-out validation data.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize


def _resolve_classes(y_true: Any, classes: Any) -> np.ndarray:
    """Return an explicit class array, defaulting to sorted unique y_true labels."""
    if classes is not None:
        return np.asarray(classes)
    return np.unique(np.asarray(y_true))


def apply_thresholds(
    y_proba: Any,
    thresholds: dict[Any, float] | float,
    classes: Any = None,
) -> np.ndarray:
    """Convert predicted probabilities into class predictions using per-class thresholds.

    Binary (``thresholds`` is a single float, or a one-entry dict): predicts
    the positive (second) class when ``y_proba[:, 1] >= threshold``, else the
    first class.

    Multiclass (``thresholds`` is a dict covering every class): scaled
    argmax — ``classes[argmax(y_proba / thresholds, axis=1)]``. Equal
    thresholds across all classes reduce to plain argmax.

    Args:
        y_proba: Array-like of shape (n_samples, n_classes), predicted
            probabilities in the same column order as ``classes``.
        thresholds: A single float (binary), or a dict mapping every class
            label present in ``classes`` to its threshold.
        classes: Explicit class label order matching ``y_proba``'s columns.
            Required when ``y_proba`` has more than 2 columns and
            ``thresholds`` is a dict (to know column-to-class mapping).

    Returns:
        1D numpy array of predicted class labels, length n_samples.

    Raises:
        ValueError: If ``y_proba`` isn't 2D, or ``thresholds`` doesn't cover
            every class implied by ``y_proba``'s column count.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D (n_samples, n_classes); got shape {y_proba.shape}")

    n_classes = y_proba.shape[1]
    if classes is None:
        classes = np.arange(n_classes)
    classes = np.asarray(classes)
    if len(classes) != n_classes:
        raise ValueError(
            f"classes has {len(classes)} entries but y_proba has {n_classes} columns"
        )

    if n_classes == 2 and not isinstance(thresholds, dict):
        threshold = float(thresholds)
        return np.where(y_proba[:, 1] >= threshold, classes[1], classes[0])

    if not isinstance(thresholds, dict):
        raise ValueError(
            "thresholds must be a dict mapping each class to its threshold "
            "for multiclass input (or when passing a single-entry dict for binary)."
        )

    if n_classes == 2 and len(thresholds) == 1:
        (threshold,) = thresholds.values()
        threshold = float(threshold)
        return np.where(y_proba[:, 1] >= threshold, classes[1], classes[0])

    missing = [c for c in classes if c not in thresholds]
    if missing:
        raise ValueError(
            f"thresholds is missing entries for classes: {missing}. "
            "apply_thresholds() requires a threshold for every class."
        )

    thresholds_array = np.array([float(thresholds[c]) for c in classes])
    scaled = y_proba / thresholds_array
    return classes[np.argmax(scaled, axis=1)]
```

- [ ] **Step 4: Run the `apply_thresholds` tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v -k apply_thresholds`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add skyulf-core/skyulf/modeling/_evaluation/thresholds.py skyulf-core/tests/test_evaluation_thresholds.py
git commit -m "feat(skyulf-core): add apply_thresholds() decision-rule function

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 6: Write the failing tests for `optimize_thresholds` (binary grid search and multiclass Nelder-Mead)**

Append to `skyulf-core/tests/test_evaluation_thresholds.py`:

```python
def test_optimize_thresholds_binary_grid_recovers_known_optimum():
    """A synthetic binary case where the F1-optimal threshold is analytically known."""
    rng = np.random.default_rng(0)
    n = 400
    y_true = np.array([0] * 300 + [1] * 100)
    # Positive class scores cluster around 0.7, negative class around 0.3,
    # so F1 is maximized near threshold ~0.5 regardless of exact noise.
    proba_pos = np.concatenate(
        [rng.normal(0.3, 0.05, 300), rng.normal(0.7, 0.05, 100)]
    ).clip(0.01, 0.99)
    y_proba = np.column_stack([1 - proba_pos, proba_pos])

    thresholds = optimize_thresholds(
        y_true, y_proba, metric=f1_score, classes=[0, 1], strategy="grid", grid_points=101
    )
    assert set(thresholds.keys()) == {0, 1}
    tuned_pred = apply_thresholds(y_proba, thresholds, classes=[0, 1])
    default_pred = apply_thresholds(y_proba, thresholds=0.5, classes=[0, 1])
    assert f1_score(y_true, tuned_pred) >= f1_score(y_true, default_pred)


def test_optimize_thresholds_defaults_to_grid_for_binary():
    """strategy=None must auto-select grid search for exactly 2 classes."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    thresholds = optimize_thresholds(y_true, y_proba, metric=f1_score, classes=[0, 1])
    assert set(thresholds.keys()) == {0, 1}


def test_optimize_thresholds_multiclass_nelder_mead_improves_on_argmax():
    """Nelder-Mead-tuned multiclass thresholds must not do worse than plain argmax
    on balanced accuracy for an imbalanced synthetic dataset."""
    rng = np.random.default_rng(1)
    classes = np.array(["a", "b", "c"])
    n_per_class = [300, 50, 50]
    y_true_parts = []
    proba_parts = []
    for i, n in enumerate(n_per_class):
        y_true_parts.append(np.full(n, classes[i]))
        base = rng.dirichlet(alpha=[1, 1, 1], size=n)
        # Bias each row's own-class column upward so there's real signal.
        base[:, i] += 1.5
        base = base / base.sum(axis=1, keepdims=True)
        proba_parts.append(base)
    y_true = np.concatenate(y_true_parts)
    y_proba = np.concatenate(proba_parts)

    def balanced_acc(y_t, y_p):
        from sklearn.metrics import balanced_accuracy_score

        return balanced_accuracy_score(y_t, y_p)

    thresholds = optimize_thresholds(
        y_true, y_proba, metric=balanced_acc, classes=classes, strategy="nelder-mead"
    )
    assert set(thresholds.keys()) == set(classes)
    tuned_pred = apply_thresholds(y_proba, thresholds, classes=classes)
    argmax_pred = classes[np.argmax(y_proba, axis=1)]
    assert balanced_acc(y_true, tuned_pred) >= balanced_acc(y_true, argmax_pred) - 1e-9


def test_optimize_thresholds_defaults_to_nelder_mead_for_multiclass():
    """strategy=None must auto-select nelder-mead for 3+ classes."""
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    thresholds = optimize_thresholds(
        y_true, y_proba, metric=lambda a, b: f1_score(a, b, average="macro"),
        classes=["a", "b", "c"],
    )
    assert set(thresholds.keys()) == {"a", "b", "c"}


def test_optimize_thresholds_raises_on_unknown_strategy():
    y_true = np.array([0, 1])
    y_proba = np.array([[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="strategy"):
        optimize_thresholds(y_true, y_proba, metric=f1_score, strategy="bogus")
```

- [ ] **Step 7: Run the new tests to verify they fail (function doesn't exist yet)**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v -k optimize_thresholds`
Expected: FAIL with `ImportError: cannot import name 'optimize_thresholds'` (the
module created in Step 3 only defines `apply_thresholds` so far).

- [ ] **Step 8: Implement `optimize_thresholds()`**

Append to `skyulf-core/skyulf/modeling/_evaluation/thresholds.py`:

```python
def _grid_search_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: Callable[[Any, Any], float],
    classes: np.ndarray,
    grid_points: int,
) -> dict[Any, float]:
    """Grid search over (0, 1) exclusive for the best binary threshold."""
    candidates = np.linspace(0.0, 1.0, grid_points + 2)[1:-1]  # exclude 0 and 1
    best_threshold = 0.5
    best_score = -np.inf
    for t in candidates:
        pred = np.where(y_proba[:, 1] >= t, classes[1], classes[0])
        score = metric(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = t
    return {classes[0]: 1.0 - best_threshold, classes[1]: best_threshold}


def _nelder_mead_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: Callable[[Any, Any], float],
    classes: np.ndarray,
) -> dict[Any, float]:
    """Nelder-Mead search over per-class thresholds for the scaled-argmax rule.

    Optimizes in log-space (``x = log(threshold)``) so the raw optimizer
    variables can be any real number while the resulting thresholds stay
    strictly positive, matching apply_thresholds()'s division-based rule.
    """
    n_classes = len(classes)
    x0 = np.zeros(n_classes)  # log(1.0) == 0 for every class: starts at plain argmax

    def negative_score(x: np.ndarray) -> float:
        thresholds = {c: float(np.exp(xi)) for c, xi in zip(classes, x, strict=True)}
        pred = apply_thresholds(y_proba, thresholds, classes=classes)
        return -metric(y_true, pred)

    result = minimize(negative_score, x0, method="Nelder-Mead")
    return {c: float(np.exp(xi)) for c, xi in zip(classes, result.x, strict=True)}


def optimize_thresholds(
    y_true: Any,
    y_proba: Any,
    metric: Callable[[Any, Any], float],
    classes: Any = None,
    strategy: str | None = None,
    grid_points: int = 101,
) -> dict[Any, float]:
    """Search for per-class decision thresholds that maximize ``metric``.

    Args:
        y_true: 1D array-like of true labels.
        y_proba: Array-like of shape (n_samples, n_classes), predicted
            probabilities in the same column order as ``classes``.
        metric: Callable ``(y_true, y_pred) -> float`` to maximize. Fully
            caller-supplied — this function ships no default metric.
        classes: Explicit class label order matching ``y_proba``'s columns.
            Defaults to ``sorted(np.unique(y_true))``.
        strategy: ``"grid"`` or ``"nelder-mead"``. If ``None`` (default),
            auto-selects ``"grid"`` for exactly 2 classes and
            ``"nelder-mead"`` for 3+ classes.
        grid_points: Number of threshold candidates for the ``"grid"``
            strategy, evenly spaced over (0, 1) exclusive.

    Returns:
        Dict mapping each class label to its tuned threshold.

    Raises:
        ValueError: If ``strategy`` is not one of ``"grid"``/``"nelder-mead"``/``None``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = _resolve_classes(y_true, classes)

    if strategy is None:
        strategy = "grid" if len(classes) == 2 else "nelder-mead"
    if strategy not in ("grid", "nelder-mead"):
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'grid' or 'nelder-mead'")

    if strategy == "grid":
        return _grid_search_binary(y_true, y_proba, metric, classes, grid_points)
    return _nelder_mead_multiclass(y_true, y_proba, metric, classes)
```

- [ ] **Step 9: Run all threshold tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v`
Expected: PASS (all tests in the file, ~12 passed)

- [ ] **Step 10: Lint, format, and type-check the new file**

Run:
```bash
cd skyulf-core
ruff check skyulf/modeling/_evaluation/thresholds.py tests/test_evaluation_thresholds.py --fix
ruff format skyulf/modeling/_evaluation/thresholds.py tests/test_evaluation_thresholds.py
ty check skyulf/modeling/_evaluation/thresholds.py
```
Expected: No errors reported by any of the three commands.

- [ ] **Step 11: Commit**

```bash
git add skyulf-core/skyulf/modeling/_evaluation/thresholds.py skyulf-core/tests/test_evaluation_thresholds.py
git commit -m "feat(skyulf-core): add optimize_thresholds() grid/Nelder-Mead search

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Export `optimize_thresholds`/`apply_thresholds` from `_evaluation` and `skyulf.modeling`

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_evaluation/__init__.py`
- Modify: `skyulf-core/skyulf/modeling/__init__.py`
- Test: `skyulf-core/tests/test_evaluation_thresholds.py` (extend)

**Interfaces:**
- Consumes: `optimize_thresholds`, `apply_thresholds` from Task 1's
  `skyulf/modeling/_evaluation/thresholds.py`.
- Produces: `from skyulf.modeling import optimize_thresholds, apply_thresholds`
  and `from skyulf.modeling._evaluation import optimize_thresholds, apply_thresholds`
  both work. Task 3 (pipeline wrapper) imports from
  `skyulf.modeling._evaluation.thresholds` directly (internal import), so
  this task's exports are for external/public consumers only.

- [ ] **Step 1: Write the failing test for top-level exports**

Append to `skyulf-core/tests/test_evaluation_thresholds.py`:

```python
def test_optimize_thresholds_and_apply_thresholds_exported_from_evaluation_package():
    from skyulf.modeling._evaluation import apply_thresholds as ev_apply
    from skyulf.modeling._evaluation import optimize_thresholds as ev_optimize

    assert ev_optimize is optimize_thresholds
    assert ev_apply is apply_thresholds


def test_optimize_thresholds_and_apply_thresholds_exported_from_modeling_top_level():
    from skyulf.modeling import apply_thresholds as top_apply
    from skyulf.modeling import optimize_thresholds as top_optimize

    assert top_optimize is optimize_thresholds
    assert top_apply is apply_thresholds
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v -k exported`
Expected: FAIL with `ImportError` (names not yet exported from either package).

- [ ] **Step 3: Add exports to `_evaluation/__init__.py`**

In `skyulf-core/skyulf/modeling/_evaluation/__init__.py`, insert a new import
line right after the existing `from .regression import evaluate_regression_model`
import and before the `from .schemas import (...)` block:

```python
from .metrics import (
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
)
from .regression import evaluate_regression_model
from .schemas import (
    ClassificationEvaluation,
    ClusterCentroid,
    ClusteringEvaluation,
    ConfusionMatrixData,
    CurveData,
    CurvePoint,
    ModelEvaluationReport,
    RegressionEvaluation,
    ResidualsData,
)
from .thresholds import apply_thresholds, optimize_thresholds
```

And add both names to `__all__`:

```python
__all__ = [
    "evaluate_classification_model",
    "evaluate_regression_model",
    "evaluate_clustering_model",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_clustering_metrics",
    "downsample_curve",
    "sanitize_metrics",
    "optimize_thresholds",
    "apply_thresholds",
    "ModelEvaluationReport",
    "ClassificationEvaluation",
    "RegressionEvaluation",
    "ClusteringEvaluation",
    "ClusterCentroid",
    "ConfusionMatrixData",
    "CurveData",
    "CurvePoint",
    "ResidualsData",
]
```

- [ ] **Step 4: Add exports to `skyulf/modeling/__init__.py`**

Add `optimize_thresholds, apply_thresholds` to the existing
`from ._evaluation import (...)` block:

```python
from ._evaluation import (
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
    optimize_thresholds,
    apply_thresholds,
)
```

And add both to `__all__` (right after the existing
`"calculate_clustering_metrics",` line):

```python
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_clustering_metrics",
    "optimize_thresholds",
    "apply_thresholds",
    "compute_shap_explanation",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v -k exported`
Expected: PASS (2 passed)

- [ ] **Step 6: Lint, format, type-check**

Run:
```bash
cd skyulf-core
ruff check skyulf/modeling/_evaluation/__init__.py skyulf/modeling/__init__.py --fix
ruff format skyulf/modeling/_evaluation/__init__.py skyulf/modeling/__init__.py
ty check skyulf/modeling/_evaluation/__init__.py skyulf/modeling/__init__.py
```
Expected: No errors.

- [ ] **Step 7: Run the full threshold test file once more to confirm no regressions**

Run: `cd skyulf-core && python -m pytest tests/test_evaluation_thresholds.py -v`
Expected: PASS (all tests).

- [ ] **Step 8: Commit**

```bash
git add skyulf-core/skyulf/modeling/_evaluation/__init__.py skyulf-core/skyulf/modeling/__init__.py skyulf-core/tests/test_evaluation_thresholds.py
git commit -m "feat(skyulf-core): export optimize_thresholds/apply_thresholds publicly

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: `SkyulfPipeline.optimize_thresholds()` wrapper

**Files:**
- Modify: `skyulf-core/skyulf/pipeline.py`
- Test: `skyulf-core/tests/test_pipeline_threshold_tuning.py` (create)

**Interfaces:**
- Consumes: `optimize_thresholds` from
  `skyulf.modeling._evaluation.thresholds` (Task 1); `self.feature_engineer`,
  `self.model_estimator` (existing `SkyulfPipeline` attributes, see
  `pipeline.py:101-104`); `self.model_estimator.applier.predict_proba(df, model)`
  (existing method, returns `pd.DataFrame` with one column per class, column
  names are `str(c)` for each `c` in `model.classes_`, or `None` if
  unsupported — see `skyulf/modeling/base.py:151` and
  `skyulf/modeling/sklearn_wrapper.py:229`).
- Produces: `SkyulfPipeline.optimize_thresholds(X_val, y_val, metric, strategy=None, grid_points=101) -> dict[Any, float]`,
  and sets `self._tuned_thresholds`. Task 4 consumes
  `self._tuned_thresholds` and this same method's proba-computation logic
  (via a shared private helper, see Step 3 below).

- [ ] **Step 1: Write the failing tests**

Create `skyulf-core/tests/test_pipeline_threshold_tuning.py`:

```python
"""Tests for SkyulfPipeline.optimize_thresholds() and predict(use_tuned_thresholds=...)."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from skyulf.pipeline import SkyulfPipeline


def _binary_config(test_size=0.25, random_state=42):
    return {
        "preprocessing": [
            {
                "name": "imputer",
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"},
            },
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            },
        ],
        "modeling": {"type": "logistic_regression"},
    }


def test_optimize_thresholds_returns_dict_covering_both_classes(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    X_train, y_train, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")
    pipeline.fit(data, target_column="target")

    def metric(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")

    thresholds = pipeline.optimize_thresholds(X_val, y_val, metric=metric)
    assert set(thresholds.keys()) == set(np.unique(y_train))


def test_optimize_thresholds_stores_result_on_instance(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    _, _, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")

    assert pipeline._tuned_thresholds is None
    thresholds = pipeline.optimize_thresholds(
        X_val, y_val, metric=lambda a, b: f1_score(a, b, average="macro")
    )
    assert pipeline._tuned_thresholds == thresholds


def test_optimize_thresholds_raises_if_pipeline_not_fitted(sample_classification_data):
    pipeline = SkyulfPipeline(_binary_config())
    data = sample_classification_data.drop(columns=["category"])
    with pytest.raises(ValueError, match="fitted"):
        pipeline.optimize_thresholds(
            data.drop(columns=["target"]),
            data["target"],
            metric=lambda a, b: f1_score(a, b, average="macro"),
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_threshold_tuning.py -v`
Expected: FAIL with `AttributeError: 'SkyulfPipeline' object has no attribute 'optimize_thresholds'`

- [ ] **Step 3: Implement `SkyulfPipeline.optimize_thresholds()`**

First, check the top of `skyulf-core/skyulf/pipeline.py` for its existing
import block (where `extract_xy` is imported from `.modeling.base`, and
whether `numpy`/`Callable`/`Any` are already imported) and add whichever of
these are missing:

```python
from collections.abc import Callable
from typing import Any

import numpy as np

from .modeling._evaluation.thresholds import apply_thresholds, optimize_thresholds
```

Then, in `SkyulfPipeline.__init__` (near `pipeline.py:104`, right after the
existing `self._target_column: str | None = None` line), add:

```python
        self._tuned_thresholds: dict[Any, float] | None = None
```

Then add two new methods right after `get_fitted_split()` (before
`predict()`, i.e. before `pipeline.py:296`):

```python
    def _predict_proba_transformed(self, transformed_data: pd.DataFrame | SkyulfDataFrame) -> Any:
        """Run predict_proba on already-transformed data, raising if unsupported."""
        if self.model_estimator is None or self.model_estimator.model is None:
            raise ValueError("Pipeline not fitted or no model configured.")
        proba = self.model_estimator.applier.predict_proba(
            transformed_data, self.model_estimator.model
        )
        if proba is None:
            raise ValueError(
                "The configured model does not support predict_proba(); "
                "threshold tuning requires predicted class probabilities."
            )
        return proba

    def optimize_thresholds(
        self,
        X_val: pd.DataFrame | SkyulfDataFrame,
        y_val: pd.Series | Any,
        metric: Callable[[Any, Any], float],
        strategy: str | None = None,
        grid_points: int = 101,
    ) -> dict[Any, float]:
        """
        Search for per-class decision thresholds that maximize ``metric`` on
        caller-supplied validation data, and store the result for later use
        by ``predict(use_tuned_thresholds=True)``.

        Always uses the *explicit* ``(X_val, y_val)`` the caller passes in —
        never the pipeline's internal train/test split. Get a clean,
        independent holdout via ``get_fitted_split()`` (or your own split)
        before calling this, the same way you would for any other
        out-of-sample evaluation.

        Args:
            X_val: Validation features, *not* yet transformed (this method
                runs the pipeline's fitted preprocessing on it internally).
            y_val: Validation true labels.
            metric: Callable ``(y_true, y_pred) -> float`` to maximize.
            strategy: ``"grid"`` or ``"nelder-mead"``. If ``None``,
                auto-selects based on the number of classes (see
                ``skyulf.modeling.optimize_thresholds``).
            grid_points: Number of grid candidates for the ``"grid"``
                strategy.

        Returns:
            Dict mapping each class label to its tuned threshold. Also
            stored on ``self._tuned_thresholds`` for
            ``predict(use_tuned_thresholds=True)`` to use.

        Raises:
            ValueError: If the pipeline isn't fitted, or the underlying
                model doesn't support ``predict_proba``.
        """
        if self.model_estimator is None or self.model_estimator.model is None:
            raise ValueError(
                "Pipeline not fitted or no model configured. Call fit() before "
                "optimize_thresholds()."
            )

        transformed_val = self.feature_engineer.transform(X_val)
        proba_df = self._predict_proba_transformed(transformed_val)
        classes = np.asarray(self.model_estimator.model.classes_)
        y_proba = np.asarray(proba_df)[:, : len(classes)]

        thresholds = optimize_thresholds(
            y_val,
            y_proba,
            metric=metric,
            classes=classes,
            strategy=strategy,
            grid_points=grid_points,
        )
        self._tuned_thresholds = thresholds
        return thresholds
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_threshold_tuning.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full pipeline test suite to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline.py tests/test_pipeline_split_extraction.py tests/test_pipeline_integration_modeling.py -v`
Expected: All PASS, no regressions from the `__init__`/import changes.

- [ ] **Step 6: Lint, format, type-check**

Run:
```bash
cd skyulf-core
ruff check skyulf/pipeline.py tests/test_pipeline_threshold_tuning.py --fix
ruff format skyulf/pipeline.py tests/test_pipeline_threshold_tuning.py
ty check skyulf/pipeline.py
```
Expected: No errors. If `ty` complains about `self.model_estimator.model.classes_`
(since `model` is typed loosely), narrow with a local variable and an
explicit `getattr(model, "classes_", None)` check that raises `ValueError`
if `None`, rather than silencing the diagnostic.

- [ ] **Step 7: Commit**

```bash
git add skyulf-core/skyulf/pipeline.py skyulf-core/tests/test_pipeline_threshold_tuning.py
git commit -m "feat(skyulf-core): add SkyulfPipeline.optimize_thresholds() wrapper

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: `predict(use_tuned_thresholds=True)` opt-in flag

**Files:**
- Modify: `skyulf-core/skyulf/pipeline.py`
- Test: `skyulf-core/tests/test_pipeline_threshold_tuning.py` (extend)

**Interfaces:**
- Consumes: `self._tuned_thresholds` and `self._predict_proba_transformed()`
  (both from Task 3); `apply_thresholds` (from Task 1, already imported in
  Task 3's Step 3 import line).
- Produces: `SkyulfPipeline.predict(data, use_tuned_thresholds: bool = False)`.
  This is the final task of this plan — no downstream consumers within this
  plan.

- [ ] **Step 1: Write the failing tests**

Append to `skyulf-core/tests/test_pipeline_threshold_tuning.py`:

```python
def test_predict_use_tuned_thresholds_raises_before_tuning(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    X_test = data.drop(columns=["target"])

    with pytest.raises(ValueError, match="optimize_thresholds"):
        pipeline.predict(X_test, use_tuned_thresholds=True)


def test_predict_use_tuned_thresholds_applies_stored_thresholds(sample_classification_data):
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    _, _, X_val, y_val = pipeline.get_fitted_split(data, target_column="target")
    pipeline.optimize_thresholds(
        X_val, y_val, metric=lambda a, b: f1_score(a, b, average="macro")
    )

    X_test = data.drop(columns=["target"])
    tuned_preds = pipeline.predict(X_test, use_tuned_thresholds=True)
    assert len(tuned_preds) == len(X_test)
    assert set(np.unique(tuned_preds)).issubset(set(np.unique(data["target"])))


def test_predict_default_behavior_unchanged_when_flag_is_false(sample_classification_data):
    """Regression check: use_tuned_thresholds=False (the default) must behave
    exactly like predict() did before this feature existed."""
    data = sample_classification_data.drop(columns=["category"])
    pipeline = SkyulfPipeline(_binary_config())
    pipeline.fit(data, target_column="target")
    X_test = data.drop(columns=["target"])

    default_preds = pipeline.predict(X_test)
    explicit_false_preds = pipeline.predict(X_test, use_tuned_thresholds=False)
    np.testing.assert_array_equal(np.asarray(default_preds), np.asarray(explicit_false_preds))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_threshold_tuning.py -v -k "use_tuned_thresholds"`
Expected: FAIL with `TypeError: predict() got an unexpected keyword argument 'use_tuned_thresholds'`

- [ ] **Step 3: Implement the `predict()` change**

Replace the existing `predict()` method in `skyulf-core/skyulf/pipeline.py`
(currently at `pipeline.py:296-322`):

```python
    def predict(self, data: pd.DataFrame | SkyulfDataFrame) -> Any:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.

        Returns:
            Series of predictions.

        Raises:
            ValueError: If the input still contains the target column used during fit.
        """
        if self._target_column is not None and self._target_column in data.columns:
            raise ValueError(
                f"predict() input still contains the target column '{self._target_column}' "
                "used during fit(); drop it before calling predict()."
            )

        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if self.model_estimator and self.model_estimator.model is not None:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )
        else:
            raise ValueError("Pipeline not fitted or no model configured.")
```

with:

```python
    def predict(
        self,
        data: pd.DataFrame | SkyulfDataFrame,
        use_tuned_thresholds: bool = False,
    ) -> Any:
        """
        Generate predictions.

        Args:
            data: Input DataFrame.
            use_tuned_thresholds: If True, apply the decision thresholds
                stored by a prior ``optimize_thresholds()`` call instead of
                the model's default decision rule (argmax/0.5). Requires
                ``optimize_thresholds()`` to have been called on this
                pipeline instance first.

        Returns:
            Series (or array, when ``use_tuned_thresholds=True``) of
            predictions.

        Raises:
            ValueError: If the input still contains the target column used
                during fit(); if the pipeline isn't fitted; or if
                ``use_tuned_thresholds=True`` but ``optimize_thresholds()``
                was never called on this instance.
        """
        if self._target_column is not None and self._target_column in data.columns:
            raise ValueError(
                f"predict() input still contains the target column '{self._target_column}' "
                "used during fit(); drop it before calling predict()."
            )

        # 1. Feature Engineering (Transform only)
        transformed_data = self.feature_engineer.transform(data)

        # 2. Modeling
        if not (self.model_estimator and self.model_estimator.model is not None):
            raise ValueError("Pipeline not fitted or no model configured.")

        if not use_tuned_thresholds:
            return self.model_estimator.applier.predict(
                transformed_data, self.model_estimator.model
            )

        if self._tuned_thresholds is None:
            raise ValueError(
                "use_tuned_thresholds=True but optimize_thresholds() was never "
                "called on this pipeline instance. Call optimize_thresholds() first."
            )

        proba_df = self._predict_proba_transformed(transformed_data)
        classes = np.asarray(self.model_estimator.model.classes_)
        y_proba = np.asarray(proba_df)[:, : len(classes)]
        return apply_thresholds(y_proba, self._tuned_thresholds, classes=classes)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_threshold_tuning.py -v`
Expected: PASS (all 6 tests in the file)

- [ ] **Step 5: Run the broader pipeline + evaluation suites to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline.py tests/test_pipeline_split_extraction.py tests/test_pipeline_integration_modeling.py tests/test_pipeline_integration_multi_model.py tests/test_evaluation_thresholds.py -v`
Expected: All PASS.

- [ ] **Step 6: Lint, format, type-check**

Run:
```bash
cd skyulf-core
ruff check skyulf/pipeline.py tests/test_pipeline_threshold_tuning.py --fix
ruff format skyulf/pipeline.py tests/test_pipeline_threshold_tuning.py
ty check skyulf/pipeline.py
```
Expected: No errors.

- [ ] **Step 7: Commit**

```bash
git add skyulf-core/skyulf/pipeline.py skyulf-core/tests/test_pipeline_threshold_tuning.py
git commit -m "feat(skyulf-core): add predict(use_tuned_thresholds=True) opt-in

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: README documentation

**Files:**
- Modify: `skyulf-core/README.md`

**Interfaces:**
- Consumes: `SkyulfPipeline.optimize_thresholds()`, `predict(use_tuned_thresholds=True)`
  (Tasks 3–4), `optimize_thresholds()`/`apply_thresholds()` top-level exports
  (Task 2).
- Produces: Documentation only — no code consumers.

- [ ] **Step 1: Add the "Threshold tuning" section to README**

In `skyulf-core/README.md`, find this existing paragraph (ends the
`get_fitted_split()` section, right before the `**Naming:**` paragraph):

```
```python
X_train, y_train, X_test, y_test = pipeline.get_fitted_split(
    customers, target_column="purchased"
)
```

**Naming:** preprocessing names are `PascalCase` (`SimpleImputer`,
```

Insert a new subsection between the code block and `**Naming:**`:

```markdown
```python
X_train, y_train, X_test, y_test = pipeline.get_fitted_split(
    customers, target_column="purchased"
)
```

**Threshold tuning:** the default decision rule (argmax for multiclass, 0.5
for binary) is rarely optimal for imbalanced classes or a metric you actually
care about (F1, MCC, balanced accuracy, ...). `pipeline.optimize_thresholds()`
searches per-class thresholds against a metric you supply, evaluated on
validation data you supply explicitly (never the pipeline's internal split —
get a clean holdout via `get_fitted_split()` above, or your own):

```python
from sklearn.metrics import f1_score

X_train, y_train, X_val, y_val = pipeline.get_fitted_split(customers, target_column="purchased")
pipeline.fit(customers, target_column="purchased")

thresholds = pipeline.optimize_thresholds(
    X_val, y_val, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
)
# thresholds: {False: 0.62, True: 0.38} (binary) or one entry per class (multiclass)

tuned_predictions = pipeline.predict(new_customers, use_tuned_thresholds=True)
```

Binary search is a grid search over `(0, 1)`; multiclass search is
Nelder-Mead over a scaled-argmax rule
(`classes[argmax(proba / threshold_per_class)]`). Both are also available as
standalone array-level functions for use outside a pipeline:
`from skyulf.modeling import optimize_thresholds, apply_thresholds`.

**Naming:** preprocessing names are `PascalCase` (`SimpleImputer`,
```

- [ ] **Step 2: Verify the README renders sensibly (manual check, no automated test for docs)**

Run: `sed -n '155,205p' skyulf-core/README.md` and read the output to confirm
the new section flows correctly between `get_fitted_split()` and `**Naming:**`.

- [ ] **Step 3: Commit**

```bash
git add skyulf-core/README.md
git commit -m "docs(skyulf-core): document optimize_thresholds()/apply_thresholds()

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Full lint/type/test gate + changelog entry

**Files:**
- All files touched by Tasks 1–5 (final gate pass)
- Modify: `changelog/0.7.x.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: nothing new — this is a validation + documentation task.
- Produces: nothing new — final quality gate for this plan.

- [ ] **Step 1: Run the full skyulf-core test suite**

Run: `cd skyulf-core && python -m pytest -q`
Expected: All tests pass except the pre-existing, unrelated
`test_text_vectorization.py::TestSentenceEmbedder::test_embeddings_shape`
SSL/network failure (confirmed pre-existing via `git log` in a prior
session — not caused by this plan). If any other test fails, stop and fix
before proceeding.

- [ ] **Step 2: Run the full lint/format/type gate on every file touched by this plan**

Run:
```bash
cd skyulf-core
ruff check .
ruff format --check .
ty check skyulf tests
```
Expected: No errors. If pre-existing unrelated diagnostics appear in files
this plan didn't touch, confirm via `git blame`/`git log -S` that they
predate this plan's commits, and leave them alone.

- [ ] **Step 3: Add changelog entry**

View `changelog/0.7.x.md`'s most recent version section to match its exact
heading/list format, then add a new section above it (or bump to the next
patch version per the existing versioning convention in that file — check
`skyulf-core/setup.py`'s current version first to pick the right number).
Document: new `optimize_thresholds()`/`apply_thresholds()` functions in
`skyulf.modeling`, `SkyulfPipeline.optimize_thresholds()`, and
`predict(use_tuned_thresholds=True)`.

Add a matching one-line summary to the `0.7.x` row of the table in the
top-level `CHANGELOG.md`, following the exact pattern used for the previous
version entry.

- [ ] **Step 4: Commit**

```bash
git add changelog/0.7.x.md CHANGELOG.md
git commit -m "docs: add changelog entry for threshold-tuning feature

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 5: Final full-suite re-run to confirm the changelog commit didn't break anything**

Run: `cd skyulf-core && python -m pytest -q`
Expected: Same pass/fail result as Step 1 (changelog changes don't touch
code, so this should be a no-op re-confirmation).
