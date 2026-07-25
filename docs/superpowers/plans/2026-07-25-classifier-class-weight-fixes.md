# Classifier `class_weight` Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `class_weight="balanced"` (and dict-valued class weighting) actually take effect for every classifier in `skyulf-core`, including ones like XGBoost whose sklearn wrapper has no native `class_weight` support, expose `class_weight` in the classifier hyperparameter UI/schema, stop the tuning engine's eager Optuna import from firing on every `import skyulf`, and add a small convenience API for extracting a pipeline's fitted train/test split as plain pandas objects — then release the fixes as a new `skyulf-core` PyPI version.

**Architecture:** A generic `class_weight` → `sample_weight` shim lives in `SklearnCalculator.fit()` (the single base class every classifier calculator inherits from), so it applies uniformly regardless of which library's estimator is wrapped. It detects native support by inspecting the model constructor's signature (not by hardcoding "xgboost" anywhere), pops `class_weight` before construction when unsupported, and converts it to a `sample_weight` array via `sklearn.utils.class_weight.compute_sample_weight` right before `.fit()`. The hyperparameter registry gains a `class_weight` `HyperparameterField` on `RANDOM_FOREST_CLASSIFIER_PARAMS` and two new classifier-only param lists (`LGBM_CLASSIFIER_PARAMS`, `XGBOOST_CLASSIFIER_PARAMS`) so the existing schema-driven frontend UI picks it up automatically with zero frontend code changes. The Optuna import in `_tuning/engine.py` moves from module level into a memoized lazy-loader function called only when `strategy="optuna"` is actually requested. The split-extraction API promotes `StatefulEstimator`'s existing `_extract_xy` dispatch logic to a reusable module-level function and adds `SkyulfPipeline.get_fitted_split()` on top of it.

**Tech Stack:** Python 3.11+, scikit-learn (`sklearn.utils.class_weight.compute_sample_weight`), pandas, polars, pytest, ruff, ty (Astral type checker).

## Global Constraints

- Target Python 3.11+ syntax (type hints, f-strings) — matches the rest of `skyulf-core`.
- Every new/changed function needs a short 1-2 line docstring.
- No new third-party dependencies — `sklearn.utils.class_weight` is already a transitive dependency via `scikit-learn` (a `skyulf-core` core dependency).
- CatBoost, ensemble base-estimator naming, and `hyperparameter_tuner` node registration are explicitly **out of scope** (per the approved design spec) — do not touch them.
- No frontend (`frontend/ml-canvas/`) code changes are required or in scope — the hyperparameter UI is schema-driven and will pick up the new `class_weight` fields automatically from the backend registry.
- Run `ruff check .`, `ruff format --check` (scoped to touched files at minimum, ideally `skyulf-core`), and `ty check skyulf-core/skyulf skyulf-core/tests` before considering any task done, per repo convention.
- Run the `skyulf-core` pytest suite (`cd skyulf-core && python -m pytest`) after each task; do not proceed to the next task with failing tests.
- Release is via version bump to `skyulf-core/setup.py`'s `version=` string + push/merge to `master` — `.github/workflows/release.yml` auto-publishes to PyPI via Trusted Publishing (OIDC). No manual `twine upload`. The currently published PyPI version is `0.5.3`; a `0.5.4` bump already exists uncommitted-to-master on this branch (unrelated to this work) — this plan's release task must bump to `0.5.5`.

---

## File Structure

- Modify: `skyulf-core/skyulf/modeling/sklearn_wrapper.py` — add the `class_weight` → `sample_weight` shim to `SklearnCalculator.fit()`.
- Modify: `skyulf-core/tests/test_modeling_sklearn_wrapper.py` — add shim tests.
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/_tree.py` — add `class_weight` field to `RANDOM_FOREST_CLASSIFIER_PARAMS`; split `LGBM_PARAMS`/`XGBOOST_PARAMS` into classifier-only variants with `class_weight`.
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/_registry.py` — repoint `lgbm_classifier`/`xgboost_classifier` registry keys to the new classifier-only param lists.
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/__init__.py` — re-export the two new param list constants.
- Create: `skyulf-core/tests/test_hyperparameters_class_weight.py` — tests asserting `class_weight` presence/absence across classifier/regressor param lists.
- Modify: `skyulf-core/skyulf/modeling/_tuning/engine.py` — replace the eager module-level Optuna import with a memoized lazy loader.
- Modify: `skyulf-core/tests/test_tuning_engine.py` — adapt the 4 existing Optuna-fallback tests to the new lazy-loader function; add a test proving import alone doesn't trigger Optuna resolution.
- Modify: `skyulf-core/skyulf/modeling/base.py` — promote `StatefulEstimator._extract_xy`'s logic to a module-level `extract_xy()` function (delegating from the existing method, unchanged public behavior).
- Modify: `skyulf-core/skyulf/pipeline.py` — add `SkyulfPipeline.get_fitted_split()`.
- Create: `skyulf-core/tests/test_pipeline_split_extraction.py` — tests for `get_fitted_split()`.
- Modify: `skyulf-core/setup.py` — version bump (final task).

---

## Task 1: `class_weight` → `sample_weight` shim in `SklearnCalculator`

**Files:**
- Modify: `skyulf-core/skyulf/modeling/sklearn_wrapper.py`
- Test: `skyulf-core/tests/test_modeling_sklearn_wrapper.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `SklearnCalculator._constructor_accepts_class_weight() -> bool` and `SklearnCalculator._compute_sample_weight_for_fit(model: Any, class_weight: Any, y_np: Any) -> Any` (both used only internally by `fit()`, but their names/behavior are referenced by this task's own tests). No other task depends on these.

- [ ] **Step 1: Write the failing tests**

Add to `skyulf-core/tests/test_modeling_sklearn_wrapper.py`, after the existing `test_fit_merges_nested_params_dict` test (before `test_predict_returns_pandas_series_preserving_index`):

```python
def test_fit_native_class_weight_passed_through_unchanged(clf_data):
    """A model whose constructor declares `class_weight` natively (e.g.
    LogisticRegression) should receive it directly at construction time —
    no sample_weight translation should occur."""
    X, y = clf_data
    calc = SklearnCalculator(LogisticRegression, {}, "classification")
    model = calc.fit(X, y, {"class_weight": "balanced"})
    assert model.class_weight == "balanced"


def test_fit_class_weight_none_string_normalized_to_none(clf_data):
    """A stringified 'None' (as a native <select> element would submit for a
    null-valued option) should be treated the same as Python None."""
    X, y = clf_data
    calc = SklearnCalculator(LogisticRegression, {}, "classification")
    model = calc.fit(X, y, {"class_weight": "None"})
    assert model.class_weight is None


def test_fit_kwargs_constructor_class_weight_translated_to_sample_weight():
    """A model whose constructor accepts **kwargs but has no real
    'class_weight' parameter (mirrors XGBoost's sklearn wrapper) should have
    class_weight popped before construction and converted into a
    sample_weight array passed to fit(), rather than silently no-op'ing."""
    captured = {}

    class _NoNativeClassWeightModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = sample_weight
            captured["kwargs"] = self.kwargs
            self.classes_ = sorted(set(y))
            return self

    rng = np.random.RandomState(3)
    # Deliberately imbalanced: 27 zeros, 3 ones.
    X = pd.DataFrame({"f1": rng.normal(0, 1, 30)})
    y = pd.Series([0] * 27 + [1] * 3)
    calc = SklearnCalculator(_NoNativeClassWeightModel, {}, "classification")
    model = calc.fit(X, y, {"class_weight": "balanced"})

    assert isinstance(model, _NoNativeClassWeightModel)
    assert "class_weight" not in captured["kwargs"]
    assert captured["sample_weight"] is not None
    assert len(captured["sample_weight"]) == 30
    # Minority class (label 1) should get a larger weight than the majority.
    minority_weight = captured["sample_weight"][y.to_numpy() == 1][0]
    majority_weight = captured["sample_weight"][y.to_numpy() == 0][0]
    assert minority_weight > majority_weight


def test_fit_kwargs_constructor_class_weight_none_is_noop():
    """class_weight=None for a non-natively-supporting model should not
    compute or pass any sample_weight at all."""
    captured = {}

    class _NoNativeClassWeightModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y, sample_weight=None):
            captured["sample_weight"] = sample_weight
            return self

    rng = np.random.RandomState(4)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series((X["f1"] > 0).astype(int))
    calc = SklearnCalculator(_NoNativeClassWeightModel, {}, "classification")
    calc.fit(X, y, {"class_weight": None})
    assert captured["sample_weight"] is None


def test_fit_kwargs_constructor_class_weight_without_sample_weight_support_raises():
    """If the model has no native class_weight support AND its fit() doesn't
    accept sample_weight either, raise a clear ValueError instead of
    silently dropping the requested class weighting."""

    class _NoWeightingSupportAtAllModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

    rng = np.random.RandomState(5)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series((X["f1"] > 0).astype(int))
    calc = SklearnCalculator(_NoWeightingSupportAtAllModel, {}, "classification")
    with pytest.raises(ValueError, match="class_weight"):
        calc.fit(X, y, {"class_weight": "balanced"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_modeling_sklearn_wrapper.py -v -k "class_weight"`
Expected: FAIL — `AttributeError`/`AssertionError` since the shim doesn't exist yet (e.g. `model.class_weight` on `LogisticRegression` already passes today, but the kwargs-model tests will fail with `TypeError: _NoNativeClassWeightModel() got an unexpected keyword argument 'class_weight'` since nothing pops it yet).

- [ ] **Step 3: Implement the shim**

In `skyulf-core/skyulf/modeling/sklearn_wrapper.py`, add imports at the top:

```python
"""Wrapper for Scikit-Learn models."""

import inspect
import logging
import warnings
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.class_weight import compute_sample_weight

from ..engines import SkyulfDataFrame
from ..engines.sklearn_bridge import SklearnBridge
from .base import BaseModelApplier, BaseModelCalculator

logger = logging.getLogger(__name__)
```

(This hoists `inspect` to module level, replacing the local `import inspect` inside `_filter_supported_params` — remove that local import in Step 3 below.)

Replace the `fit()` method body with:

```python
    def fit(
        self,
        X: pd.DataFrame | SkyulfDataFrame,
        y: pd.Series | Any,
        config: dict[str, Any],
        progress_callback=None,
        log_callback=None,
        validation_data=None,
    ) -> Any:
        """Fit the Scikit-Learn model."""
        # 1. Merge Config with Defaults
        params = self._resolve_fit_params(config)

        # A generic <select> UI element always submits its option value as a
        # string, so a "None" option (e.g. "no class weighting") arrives here
        # as the literal string "None", not Python None. Normalize that back
        # to None before anything below decides whether class weighting was
        # actually requested.
        if params.get("class_weight") in ("None", "none", ""):
            params["class_weight"] = None

        # Some estimators (e.g. XGBoost's sklearn wrapper) accept arbitrary
        # **kwargs in their constructor but have no built-in notion of class
        # weighting: a `class_weight` kwarg is silently stored and ignored at
        # fit time (no error — just a native warning). Detect that case up
        # front (by checking whether `class_weight` is an explicitly named
        # constructor parameter, not just swallowed by **kwargs) and, if the
        # value isn't None, translate it into a `sample_weight` array passed
        # to `.fit()` instead, so "balanced"/dict class weighting behaves the
        # same regardless of whether the underlying library supports it
        # natively.
        class_weight_to_apply = None
        if "class_weight" in params and not self._constructor_accepts_class_weight():
            class_weight_to_apply = params.pop("class_weight")

        msg = f"Initializing {self.model_class.__name__} with params: {params}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)

        # 2. Instantiate Model
        valid_params = self._filter_supported_params(params)
        model = self.model_class(**valid_params)

        # 3. Fit
        # Convert to Numpy using Bridge (handles Polars/Pandas/Wrappers)
        X_np, y_np = SklearnBridge.to_sklearn((X, y))

        sample_weight = None
        if class_weight_to_apply is not None:
            sample_weight = self._compute_sample_weight_for_fit(
                model, class_weight_to_apply, y_np
            )

        # sklearn's ConvergenceWarning (raised via `warnings.warn`, not the
        # `logging` module) would otherwise only reach the server's stderr
        # and never surface to the user — unlike the skyulf-core node
        # advisories already routed through `WarningCaptureHandler` via
        # `logger.warning(...)`. Capture everything sklearn emits during
        # `fit`, re-route ConvergenceWarning through this model's own
        # (``skyulf.*``-tree) logger so every sklearn-backed model gets the
        # same UI-visible treatment regardless of solver/estimator, and
        # re-emit any other warning category unchanged so existing
        # console/log behavior for those is preserved.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if sample_weight is not None:
                model.fit(X_np, y_np, sample_weight=sample_weight)
            else:
                model.fit(X_np, y_np)
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                conv_msg = f"{self.model_class.__name__} did not fully converge: {w.message}"
                logger.warning(conv_msg)
                if log_callback:
                    log_callback(conv_msg)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        return model

    def _constructor_accepts_class_weight(self) -> bool:
        """True if the wrapped model's constructor explicitly declares a
        `class_weight` parameter (e.g. RandomForestClassifier, LGBMClassifier,
        LogisticRegression) — as opposed to merely accepting arbitrary
        **kwargs (e.g. XGBoost's sklearn wrapper) that silently swallow it."""
        sig = inspect.signature(self.model_class)
        return "class_weight" in sig.parameters

    def _compute_sample_weight_for_fit(self, model: Any, class_weight: Any, y_np: Any) -> Any:
        """Translate a `class_weight` value into a per-sample weight array for
        models with no native `class_weight` support, raising a clear error
        instead of silently no-op'ing if the model's `.fit()` doesn't accept
        `sample_weight` either."""
        fit_sig = inspect.signature(model.fit)
        if "sample_weight" not in fit_sig.parameters:
            raise ValueError(
                f"{self.model_class.__name__} does not support 'class_weight' natively "
                "and its fit() method does not accept 'sample_weight' either, so "
                "class weighting cannot be applied to this model."
            )
        return compute_sample_weight(class_weight, y_np)
```

Also update `_filter_supported_params` to drop its now-redundant local import (since `inspect` is imported at module level):

```python
    def _filter_supported_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filters ``params`` down to those accepted by the model class constructor.

        Skips filtering when the constructor accepts ``**kwargs`` (e.g. XGBoost 2.x),
        since every named param would otherwise fail the membership check even though valid.
        """
        sig = inspect.signature(self.model_class)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if accepts_kwargs:
            return params

        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        dropped = set(params.keys()) - set(valid_params.keys())
        if dropped:
            logger.warning(
                f"Dropped parameters not supported by {self.model_class.__name__}: {dropped}"
            )
        return valid_params
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_modeling_sklearn_wrapper.py -v`
Expected: PASS (all tests in the file, including the 5 new ones and all pre-existing ones).

- [ ] **Step 5: Regression-test against real XGBoost**

Add to `skyulf-core/tests/test_modeling_all.py`, inside `class TestXGBClassifier:` (after its existing `test_fit_predict`):

```python
    def test_class_weight_balanced_improves_minority_recall(self) -> None:
        """XGBoost has no native class_weight support; this proves the
        SklearnCalculator shim actually changes fitted behavior end-to-end,
        not just that it avoids raising."""
        pytest.importorskip("xgboost")
        import numpy as np
        from sklearn.metrics import recall_score

        from skyulf.modeling.classification import XGBClassifierCalculator

        rng = np.random.RandomState(7)
        n = 300
        X = pd.DataFrame(
            {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)}
        )
        # Strongly imbalanced target where the minority class (10%) is still
        # separably related to f1, so weighting has room to help recall.
        y = pd.Series(
            ((X["f1"] > 1.2) & (rng.random(n) > 0.1)).astype(int)
        )
        assert y.sum() < n * 0.2  # sanity check: genuinely imbalanced

        unweighted = XGBClassifierCalculator().fit(
            X, y, {"params": {"n_estimators": 20, "class_weight": None}}
        )
        weighted = XGBClassifierCalculator().fit(
            X, y, {"params": {"n_estimators": 20, "class_weight": "balanced"}}
        )

        recall_unweighted = recall_score(y, unweighted.predict(X))
        recall_weighted = recall_score(y, weighted.predict(X))
        assert recall_weighted >= recall_unweighted
```

- [ ] **Step 6: Run the new regression test**

Run: `cd skyulf-core && python -m pytest tests/test_modeling_all.py -v -k "class_weight"`
Expected: PASS (or SKIPPED if `xgboost` isn't installed in the current environment — both are acceptable; SKIPPED must not be treated as a failure signal, but if `xgboost` **is** installed it must PASS, not SKIP).

- [ ] **Step 7: Run the full sklearn_wrapper + classification.py test files**

Run: `cd skyulf-core && python -m pytest tests/test_modeling_sklearn_wrapper.py tests/test_modeling_all.py tests/test_modeling_classification_gaps.py -v`
Expected: PASS, no regressions in unrelated tests.

- [ ] **Step 8: Commit**

```bash
cd /Users/BH7043/Skyulf
git add skyulf-core/skyulf/modeling/sklearn_wrapper.py skyulf-core/tests/test_modeling_sklearn_wrapper.py skyulf-core/tests/test_modeling_all.py
git commit -m "feat(skyulf-core): translate class_weight to sample_weight for classifiers without native support

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 2: Expose `class_weight` in the classifier hyperparameter schema

**Files:**
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/_tree.py`
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/_registry.py`
- Modify: `skyulf-core/skyulf/modeling/hyperparameters/__init__.py`
- Test: Create `skyulf-core/tests/test_hyperparameters_class_weight.py`

**Interfaces:**
- Consumes: `HyperparameterField` dataclass from `._field` (unchanged: `name, label, type, default, description, min, max, step, options, depends_on, exclusive_options`).
- Produces: `LGBM_CLASSIFIER_PARAMS` and `XGBOOST_CLASSIFIER_PARAMS` (new list constants in `_tree.py`, re-exported from `hyperparameters/__init__.py`). `RANDOM_FOREST_CLASSIFIER_PARAMS` keeps its existing name but gains one more field. No other task depends on these names.

- [ ] **Step 1: Write the failing tests**

Create `skyulf-core/tests/test_hyperparameters_class_weight.py`:

```python
"""Tests for the class_weight HyperparameterField added to classifier param lists."""

from skyulf.modeling.hyperparameters import (
    LGBM_CLASSIFIER_PARAMS,
    LGBM_PARAMS,
    MODEL_HYPERPARAMETERS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_PARAMS,
)


def _field_names(fields):
    return {f.name for f in fields}


def test_random_forest_classifier_has_class_weight_but_regressor_does_not():
    assert "class_weight" in _field_names(RANDOM_FOREST_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(RANDOM_FOREST_PARAMS)


def test_lgbm_classifier_params_has_class_weight_but_shared_base_does_not():
    assert "class_weight" in _field_names(LGBM_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(LGBM_PARAMS)


def test_xgboost_classifier_params_has_class_weight_but_shared_base_does_not():
    assert "class_weight" in _field_names(XGBOOST_CLASSIFIER_PARAMS)
    assert "class_weight" not in _field_names(XGBOOST_PARAMS)


def test_registry_maps_classifier_keys_to_class_weight_variants():
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["lgbm_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["lgbm_regressor"])
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["xgboost_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["xgboost_regressor"])
    assert "class_weight" in _field_names(MODEL_HYPERPARAMETERS["random_forest_classifier"])
    assert "class_weight" not in _field_names(MODEL_HYPERPARAMETERS["random_forest_regressor"])


def test_class_weight_field_default_and_options():
    rf_field = next(f for f in RANDOM_FOREST_CLASSIFIER_PARAMS if f.name == "class_weight")
    assert rf_field.default is None
    assert rf_field.type == "select"
    values = {opt["value"] for opt in rf_field.options}
    assert values == {None, "balanced"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_hyperparameters_class_weight.py -v`
Expected: FAIL with `ImportError: cannot import name 'LGBM_CLASSIFIER_PARAMS'`.

- [ ] **Step 3: Add the `class_weight` field to `RANDOM_FOREST_CLASSIFIER_PARAMS`**

In `skyulf-core/skyulf/modeling/hyperparameters/_tree.py`, replace:

```python
# Classifier-only addition: criterion for split quality.
RANDOM_FOREST_CLASSIFIER_PARAMS = RANDOM_FOREST_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="The function to measure the quality of a split.",
    )
]
```

with:

```python
# Classifier-only additions: criterion for split quality, and class_weight
# for imbalanced targets (regression has no such concept, so this is not
# part of the shared RANDOM_FOREST_PARAMS base list).
RANDOM_FOREST_CLASSIFIER_PARAMS = RANDOM_FOREST_PARAMS + [
    HyperparameterField(
        name="criterion",
        label="Criterion",
        type="select",
        default="gini",
        options=[
            {"label": "Gini", "value": "gini"},
            {"label": "Entropy", "value": "entropy"},
            {"label": "Log Loss", "value": "log_loss"},
        ],
        description="The function to measure the quality of a split.",
    ),
    HyperparameterField(
        name="class_weight",
        label="Class Weight",
        type="select",
        default=None,
        options=[
            {"label": "None (equal weight)", "value": None},
            {"label": "Balanced", "value": "balanced"},
        ],
        description=(
            "Automatically adjusts weights inversely proportional to class "
            "frequencies in the training data, useful for imbalanced targets."
        ),
    ),
]
```

- [ ] **Step 4: Split `XGBOOST_PARAMS` into a classifier-only variant**

Immediately after the existing `XGBOOST_PARAMS = [...]` list (ends right before the `# --- Extra Trees (Classifier & Regressor) ---` comment), add:

```python
# Classifier-only addition: class_weight for imbalanced targets. XGBoost's
# sklearn wrapper has no native class_weight support (regression has no such
# concept either), so this is deliberately not part of the shared
# XGBOOST_PARAMS base list — `SklearnCalculator.fit()` translates it into a
# `sample_weight` array at fit time (see sklearn_wrapper.py).
XGBOOST_CLASSIFIER_PARAMS = XGBOOST_PARAMS + [
    HyperparameterField(
        name="class_weight",
        label="Class Weight",
        type="select",
        default=None,
        options=[
            {"label": "None (equal weight)", "value": None},
            {"label": "Balanced", "value": "balanced"},
        ],
        description=(
            "Automatically adjusts weights inversely proportional to class "
            "frequencies in the training data, useful for imbalanced targets. "
            "XGBoost has no native support for this, so it is applied as a "
            "computed sample_weight at fit time."
        ),
    ),
]
```

- [ ] **Step 5: Split `LGBM_PARAMS` into a classifier-only variant**

Immediately after the existing `LGBM_PARAMS = [...]` list (the last field in it is `boosting_type`, at the end of the file), add:

```python

# Classifier-only addition: class_weight for imbalanced targets (regression
# has no such concept, so this is not part of the shared LGBM_PARAMS base
# list). LightGBM's sklearn wrapper supports class_weight natively.
LGBM_CLASSIFIER_PARAMS = LGBM_PARAMS + [
    HyperparameterField(
        name="class_weight",
        label="Class Weight",
        type="select",
        default=None,
        options=[
            {"label": "None (equal weight)", "value": None},
            {"label": "Balanced", "value": "balanced"},
        ],
        description=(
            "Automatically adjusts weights inversely proportional to class "
            "frequencies in the training data, useful for imbalanced targets."
        ),
    ),
]
```

- [ ] **Step 6: Repoint the registry's classifier keys**

In `skyulf-core/skyulf/modeling/hyperparameters/_registry.py`, update the import block:

```python
from ._tree import (
    ADABOOST_PARAMS,
    DECISION_TREE_CLASSIFIER_PARAMS,
    DECISION_TREE_REGRESSOR_PARAMS,
    EXTRA_TREES_CLASSIFIER_PARAMS,
    EXTRA_TREES_REGRESSOR_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LGBM_CLASSIFIER_PARAMS,
    LGBM_PARAMS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_PARAMS,
)
```

and update the two mapping lines:

```python
    "xgboost_classifier": XGBOOST_CLASSIFIER_PARAMS,
    "xgboost_regressor": XGBOOST_PARAMS,
```

```python
    "lgbm_classifier": LGBM_CLASSIFIER_PARAMS,
    "lgbm_regressor": LGBM_PARAMS,
```

(`random_forest_classifier` already points at `RANDOM_FOREST_CLASSIFIER_PARAMS`, which now includes `class_weight` from Step 3 — no line change needed there.)

- [ ] **Step 7: Re-export the new constants**

In `skyulf-core/skyulf/modeling/hyperparameters/__init__.py`, update the `_tree` import:

```python
from ._tree import (
    ADABOOST_PARAMS,
    DECISION_TREE_CLASSIFIER_PARAMS,
    DECISION_TREE_PARAMS,
    DECISION_TREE_REGRESSOR_PARAMS,
    EXTRA_TREES_CLASSIFIER_PARAMS,
    EXTRA_TREES_PARAMS,
    EXTRA_TREES_REGRESSOR_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LGBM_CLASSIFIER_PARAMS,
    LGBM_PARAMS,
    RANDOM_FOREST_CLASSIFIER_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_PARAMS,
)
```

and add both new names to `__all__`, right next to their existing base-list counterparts:

```python
    "XGBOOST_PARAMS",
    "XGBOOST_CLASSIFIER_PARAMS",
    "EXTRA_TREES_PARAMS",
    "EXTRA_TREES_CLASSIFIER_PARAMS",
    "EXTRA_TREES_REGRESSOR_PARAMS",
    "HIST_GRADIENT_BOOSTING_PARAMS",
    "LGBM_PARAMS",
    "LGBM_CLASSIFIER_PARAMS",
```

(replacing the previous `"XGBOOST_PARAMS",` / `"LGBM_PARAMS",` single lines in that block).

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_hyperparameters_class_weight.py tests/test_hyperparameters_registry.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full hyperparameters + classification + registry test files to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_hyperparameters_class_weight.py tests/test_hyperparameters_registry.py tests/test_modeling_all.py tests/test_registry_toplevel.py -v`
Expected: PASS, no regressions (in particular, confirm nothing else imported `LGBM_PARAMS`/`XGBOOST_PARAMS` expecting the classifier's `class_weight` field to be absent from those base names — the earlier `grep` confirmed the only consumers were `_registry.py` and `__init__.py`, both updated in this task).

- [ ] **Step 10: Commit**

```bash
cd /Users/BH7043/Skyulf
git add skyulf-core/skyulf/modeling/hyperparameters/_tree.py skyulf-core/skyulf/modeling/hyperparameters/_registry.py skyulf-core/skyulf/modeling/hyperparameters/__init__.py skyulf-core/tests/test_hyperparameters_class_weight.py
git commit -m "feat(skyulf-core): expose class_weight in RF/LightGBM/XGBoost classifier hyperparameter schema

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 3: Lazy Optuna import in the tuning engine

**Files:**
- Modify: `skyulf-core/skyulf/modeling/_tuning/engine.py`
- Modify: `skyulf-core/tests/test_tuning_engine.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `_ensure_optuna_loaded() -> bool` (module-level function in `engine.py`), plus module-level globals `HAS_OPTUNA: bool`, `OptunaSearchCV: Any`, `optuna: Any` (all start as `False`/`None`/`None` and are populated the first time `_ensure_optuna_loaded()` runs). No other task depends on these names.

- [ ] **Step 1: Update the existing Optuna-fallback tests to call the lazy loader**

In `skyulf-core/tests/test_tuning_engine.py`, the 4 existing tests currently assert on `variant.HAS_OPTUNA`/`variant.OptunaSearchCV` immediately after `_load_engine_variant(...)`, relying on the resolution happening automatically as a side effect of executing the module. After this task's refactor, resolution is deferred, so each test must explicitly call the loader first. Replace all 4 tests:

```python
def test_optuna_import_failure_disables_optuna():
    """If 'optuna' itself cannot be imported, HAS_OPTUNA should end up False."""
    variant = _load_engine_variant({"optuna": None})
    assert variant._ensure_optuna_loaded() is False
    assert variant.HAS_OPTUNA is False


def test_optuna_integration_import_all_fallbacks_fail():
    """If optuna is present but none of the integration import paths work,
    HAS_OPTUNA should be reset to False and a warning logged."""
    variant = _load_engine_variant(
        {
            "optuna.integration": None,
            "optuna.integration.sklearn": None,
            "optuna_integration": None,
            "optuna_integration.sklearn": None,
        }
    )
    assert variant._ensure_optuna_loaded() is False
    assert variant.HAS_OPTUNA is False


def test_optuna_integration_second_fallback_path_succeeds():
    """If `optuna.integration` fails but `optuna.integration.sklearn` succeeds,
    OptunaSearchCV should be sourced from the second fallback path."""
    pytest.importorskip("optuna")
    import types

    fake_module = types.ModuleType("optuna.integration.sklearn")
    setattr(fake_module, "OptunaSearchCV", object())  # noqa: B010 - sentinel marker class
    variant = _load_engine_variant(
        {"optuna.integration": None, "optuna.integration.sklearn": fake_module}
    )
    assert variant._ensure_optuna_loaded() is True
    assert variant.HAS_OPTUNA is True
    assert variant.OptunaSearchCV is fake_module.OptunaSearchCV


def test_optuna_integration_third_fallback_path_succeeds():
    """If both `optuna.integration` and `optuna.integration.sklearn` fail but
    `optuna_integration.sklearn` succeeds, OptunaSearchCV should come from the
    third fallback path."""
    pytest.importorskip("optuna")
    import types

    fake_module = types.ModuleType("optuna_integration.sklearn")
    setattr(fake_module, "OptunaSearchCV", object())  # noqa: B010 - sentinel marker class
    variant = _load_engine_variant(
        {
            "optuna.integration": None,
            "optuna.integration.sklearn": None,
            "optuna_integration.sklearn": fake_module,
        }
    )
    assert variant._ensure_optuna_loaded() is True
    assert variant.HAS_OPTUNA is True
    assert variant.OptunaSearchCV is fake_module.OptunaSearchCV
```

Also add one new test directly after these four, proving the laziness itself:

```python
def test_importing_engine_does_not_eagerly_resolve_optuna():
    """Merely loading the module (as any `import skyulf`/`skyulf.modeling`
    transitively does) must not attempt to import optuna or log its
    "OptunaSearchCV not found" warning — only calling `_ensure_optuna_loaded()`
    (from `_build_optuna_searcher`, i.e. only when strategy='optuna' is
    actually requested) should trigger resolution."""
    variant = _load_engine_variant({"optuna": None})
    assert variant.HAS_OPTUNA is False
    assert variant.OptunaSearchCV is None
    assert variant._optuna_load_attempted is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_tuning_engine.py -v -k optuna`
Expected: FAIL — `AttributeError: module has no attribute '_ensure_optuna_loaded'` (doesn't exist yet), and the new laziness test fails because `HAS_OPTUNA`/`OptunaSearchCV` are currently still resolved eagerly at module-exec time.

- [ ] **Step 3: Implement the lazy loader**

In `skyulf-core/skyulf/modeling/_tuning/engine.py`, replace the entire eager-import block (from `# Try importing Optuna with robust fallback for integration packages` through the end of the `if HAS_OPTUNA:` chain, i.e. everything currently between `logger = logging.getLogger(__name__)` and `class TuningCalculator(BaseModelCalculator):`) with:

```python
# Optuna is an optional, heavyweight dependency only needed when a caller
# actually requests `strategy="optuna"` tuning. Resolution (including its
# multi-path sklearn-integration fallback chain) is deferred to
# `_ensure_optuna_loaded()`, called only from `_build_optuna_searcher`, so
# merely importing `skyulf`/`skyulf.modeling` never imports optuna or emits
# its "OptunaSearchCV not found" warning for users who never use this
# strategy.
HAS_OPTUNA = False
OptunaSearchCV: Any = None
optuna: Any = None
_optuna_load_attempted = False


def _ensure_optuna_loaded() -> bool:
    """Lazily import Optuna and resolve its sklearn-compatible OptunaSearchCV
    integration, memoizing the result so repeated tuning calls don't
    re-attempt the (multi-path fallback) import every time.

    Populates the module-level ``optuna``/``HAS_OPTUNA``/``OptunaSearchCV``
    globals on success, so the existing ``_build_optuna_distributions``/
    ``_build_optuna_sampler``/``_build_optuna_pruner`` helpers (which
    reference the bare ``optuna`` module name) keep working unchanged, since
    they're only ever called from ``_build_optuna_searcher`` after it has
    already called this function.
    """
    global HAS_OPTUNA, OptunaSearchCV, optuna, _optuna_load_attempted
    if _optuna_load_attempted:
        return HAS_OPTUNA
    _optuna_load_attempted = True

    try:
        import optuna as _optuna  # ty: ignore[unresolved-import]

        optuna = _optuna
        HAS_OPTUNA = True
    except ImportError:
        return HAS_OPTUNA

    try:
        from optuna.integration import (  # ty: ignore[unresolved-import]
            OptunaSearchCV as _OptunaSearchCV,
        )

        OptunaSearchCV = _OptunaSearchCV
    except ImportError:
        try:
            from optuna.integration.sklearn import (  # ty: ignore[unresolved-import]
                OptunaSearchCV as _OptunaSearchCV,
            )

            OptunaSearchCV = _OptunaSearchCV
        except ImportError:
            try:
                from optuna_integration.sklearn import (  # ty: ignore[unresolved-import]
                    OptunaSearchCV as _OptunaSearchCV,
                )

                OptunaSearchCV = _OptunaSearchCV
            except ImportError:
                HAS_OPTUNA = False
                logger.warning(
                    "Optuna installed but OptunaSearchCV not found. Install 'optuna-integration'."
                )
    return HAS_OPTUNA
```

- [ ] **Step 4: Update `_build_optuna_searcher` to call the lazy loader**

In the same file, in `_build_optuna_searcher`, replace:

```python
        """Builds an OptunaSearchCV searcher, wiring up distributions, sampler, pruner, and callbacks."""
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is not installed. Please install 'optuna' and 'optuna-integration'."
            )
```

with:

```python
        """Builds an OptunaSearchCV searcher, wiring up distributions, sampler, pruner, and callbacks."""
        if not _ensure_optuna_loaded():
            raise ImportError(
                "Optuna is not installed. Please install 'optuna' and 'optuna-integration'."
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_tuning_engine.py -v`
Expected: PASS — full file, including all Optuna fallback tests and any pre-existing `strategy="optuna"` end-to-end test elsewhere in the file (search the file for `strategy="optuna"` / `strategy=\"optuna\"` end-to-end tests and confirm they still pass, since they now go through the lazy loader on first real use).

- [ ] **Step 6: Confirm import-time silence manually**

Run:
```bash
cd skyulf-core && python -W error::UserWarning -c "import skyulf.modeling; print('no warning raised')"
```
Expected output: `no warning raised` (previously, if `optuna` was installed without `optuna-integration`, this would have logged/warned at import time; now it's silent until an optuna-strategy tuning run actually happens).

- [ ] **Step 7: Commit**

```bash
cd /Users/BH7043/Skyulf
git add skyulf-core/skyulf/modeling/_tuning/engine.py skyulf-core/tests/test_tuning_engine.py
git commit -m "perf(skyulf-core): defer Optuna import in tuning engine until strategy=optuna is used

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 4: Convenience split-extraction API on `SkyulfPipeline`

**Files:**
- Modify: `skyulf-core/skyulf/modeling/base.py`
- Modify: `skyulf-core/skyulf/pipeline.py`
- Test: Create `skyulf-core/tests/test_pipeline_split_extraction.py`

**Interfaces:**
- Consumes: `SplitDataset` from `skyulf.data.dataset` (existing, unchanged: `train`, `test`, `validation` fields).
- Produces: `extract_xy(data: Any, target_column: str) -> tuple[Any, Any]` (new module-level function in `skyulf.modeling.base`, delegated to by the existing `StatefulEstimator._extract_xy` method so all current call sites/tests keep working unchanged); `SkyulfPipeline.get_fitted_split(data, target_column) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]` (new method on `SkyulfPipeline`). No other task depends on these.

- [ ] **Step 1: Promote `_extract_xy`'s logic to a module-level function**

In `skyulf-core/skyulf/modeling/base.py`, add a module-level function right before the `class StatefulEstimator:` definition:

```python
def extract_xy(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extract ``(X, y)`` from a DataFrame (pandas or Polars) or an ``(X, y)``
    tuple, given the target column name.

    An empty/falsy ``target_column`` is the established "no target" sentinel
    (see ``_node_runners.py``'s ``target_col=""`` for data-preview-only
    inputs): unsupervised calculators (e.g. clustering) rely on this to get
    the whole frame back as ``X`` with ``y=None``.
    """
    if not target_column:
        X = data[0] if isinstance(data, tuple) else data
        return X, None

    if isinstance(data, tuple) and len(data) == 2:
        return _extract_xy_from_tuple(data, target_column)

    engine = get_engine(data)

    if engine.name == EngineName.POLARS:
        return _extract_xy_polars(data, target_column)

    return _extract_xy_pandas_like(data, target_column)


def _extract_xy_from_tuple(data: tuple[Any, Any], target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a ``(X, y)`` tuple, pulling ``y`` out of ``X`` if it's missing."""
    X, y = data[0], data[1]
    if hasattr(X, "columns") and target_column in X.columns:
        features, embedded_y = extract_xy(X, target_column)
        return features, embedded_y if y is None else y
    return X, y


def _extract_xy_polars(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a Polars DataFrame by dropping/selecting ``target_column``."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    X = data.drop([target_column])
    y = data.select(target_column).to_series()
    return X, y


def _extract_xy_pandas_like(data: Any, target_column: str) -> tuple[Any, Any]:
    """Extracts X/y from a pandas or generic DataFrame-like object."""
    if hasattr(data, "columns"):
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        if hasattr(data, "drop"):
            try:
                return data.drop(columns=[target_column]), data[target_column]
            except TypeError:
                pass

        if hasattr(data, target_column):
            return data, getattr(data, target_column)

    raise ValueError(f"Unexpected data type: {type(data)}")
```

Then replace the 4 corresponding methods on `StatefulEstimator` (`_extract_xy`, `_extract_xy_from_tuple`, `_extract_xy_polars`, `_extract_xy_pandas_like`) with thin delegations that preserve the exact existing method names/signatures (so every current call site and test — `estimator._extract_xy(...)` etc. — keeps working unchanged):

```python
    def _extract_xy(self, data: Any, target_column: str) -> tuple[Any, Any]:
        """Instance-method wrapper around the module-level ``extract_xy()``,
        kept for backward compatibility with existing call sites/tests."""
        return extract_xy(data, target_column)
```

Remove the now-redundant `_extract_xy_from_tuple`, `_extract_xy_polars`, and `_extract_xy_pandas_like` methods from the class entirely (their logic now lives in the module-level functions above; nothing else in the class calls the old private methods directly except `_extract_xy` itself, which now delegates to the module function instead).

- [ ] **Step 2: Run existing base.py/clustering tests to verify the refactor is behavior-preserving**

Run: `cd skyulf-core && python -m pytest tests/test_modeling_base.py tests/test_modeling_clustering.py -v`
Expected: PASS — all pre-existing tests (`test_extract_xy_from_dataframe_with_target`, `test_extract_xy_missing_target_raises`, `test_extract_xy_from_tuple_xy`, `test_extract_xy_from_tuple_y_none_target_in_columns`, `test_extract_xy_polars_dataframe`, `test_extract_xy_polars_missing_target_raises`, `test_extract_xy_typeerror_falls_back_to_attribute_access`, `test_extract_xy_unexpected_type_raises`) still pass unchanged, since they call `estimator._extract_xy(...)` which now simply delegates.

- [ ] **Step 3: Write the failing tests for `get_fitted_split`**

Create `skyulf-core/tests/test_pipeline_split_extraction.py`:

```python
"""Tests for SkyulfPipeline.get_fitted_split() (convenience split-extraction API)."""

import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline


def _config(test_size=0.25, random_state=42):
    return {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": test_size, "random_state": random_state},
            }
        ],
        "modeling": {"type": "logistic_regression"},
    }


def test_get_fitted_split_returns_plain_pandas_objects(sample_classification_data):
    """Returned X/y for both train and test must be plain pandas objects."""
    pipeline = SkyulfPipeline(_config())
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(
        sample_classification_data, target_column="target"
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.Series)


def test_get_fitted_split_drops_target_column_from_features(sample_classification_data):
    """Neither X_train nor X_test should still contain the target column."""
    pipeline = SkyulfPipeline(_config())
    X_train, _, X_test, _ = pipeline.get_fitted_split(
        sample_classification_data, target_column="target"
    )
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_get_fitted_split_row_counts_match_configured_test_size(sample_classification_data):
    """With test_size=0.25 on 100 rows, train should get ~75 rows and test ~25."""
    pipeline = SkyulfPipeline(_config(test_size=0.25, random_state=42))
    X_train, y_train, X_test, y_test = pipeline.get_fitted_split(
        sample_classification_data, target_column="target"
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(sample_classification_data)
    assert len(X_test) == pytest.approx(25, abs=2)


def test_get_fitted_split_matches_pipeline_fit_row_counts(sample_classification_data):
    """The split get_fitted_split() returns should have the same row counts
    as the split SkyulfPipeline.fit() uses internally, for a fixed random_state."""
    fit_pipeline = SkyulfPipeline(_config())
    fit_pipeline.fit(sample_classification_data, target_column="target")

    split_pipeline = SkyulfPipeline(_config())
    X_train, y_train, X_test, y_test = split_pipeline.get_fitted_split(
        sample_classification_data, target_column="target"
    )
    # fit()'s internal training set size is reconstructable from the same
    # configured test_size/random_state producing an identical split.
    assert len(X_train) + len(X_test) == len(sample_classification_data)


def test_get_fitted_split_raises_without_a_configured_splitter(sample_classification_data):
    """If preprocessing doesn't produce a train/test split, raise a clear error
    instead of returning a nonsensical single-split result."""
    pipeline = SkyulfPipeline(
        {"preprocessing": [], "modeling": {"type": "logistic_regression"}}
    )
    with pytest.raises(ValueError, match="train/test split"):
        pipeline.get_fitted_split(sample_classification_data, target_column="target")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_split_extraction.py -v`
Expected: FAIL with `AttributeError: 'SkyulfPipeline' object has no attribute 'get_fitted_split'`.

- [ ] **Step 5: Implement `get_fitted_split`**

In `skyulf-core/skyulf/pipeline.py`, update the import from `.modeling.base`:

```python
from .modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator, extract_xy
```

Add a small pandas-conversion helper near the top of the file, right after `_artifact_digest`:

```python
def _to_pandas(obj: Any) -> Any:
    """Convert a Polars DataFrame/Series (or any object exposing ``to_pandas()``)
    to its pandas equivalent; pass pandas objects (or ``None``) through unchanged."""
    if obj is None:
        return None
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj
```

Add the new method to `SkyulfPipeline`, right after `fit()` (before `predict()`):

```python
    def get_fitted_split(
        self,
        data: pd.DataFrame | pl.DataFrame | SkyulfDataFrame | SplitDataset,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Run this pipeline's configured preprocessing chain and return the
        resulting train/test split as plain pandas objects.

        Runs ``self.feature_engineer.fit_transform(data)`` — the same
        preprocessing ``fit()`` uses internally — and extracts
        ``(X_train, y_train, X_test, y_test)`` from the resulting split using
        ``target_column``, converting any Polars/SkyulfDataFrame frames to
        pandas. Saves callers from re-implementing this split/convert step
        themselves for custom evaluation harnesses (e.g. comparing multiple
        raw sklearn-style estimators against the same preprocessed split).

        Args:
            data: Input data (DataFrame or SplitDataset).
            target_column: Name of the target column.

        Returns:
            ``(X_train, y_train, X_test, y_test)`` as pandas DataFrame/Series.

        Raises:
            ValueError: If the configured preprocessing steps don't produce a
                train/test split (e.g. no Splitter node configured).
        """
        transformed_data, _ = self.feature_engineer.fit_transform(data)

        if not isinstance(transformed_data, SplitDataset):
            raise ValueError(
                "get_fitted_split() requires the configured preprocessing steps "
                "to produce a train/test split (e.g. via a Splitter node); got "
                "a single, unsplit DataFrame instead."
            )

        X_train, y_train = extract_xy(transformed_data.train, target_column)
        X_test, y_test = extract_xy(transformed_data.test, target_column)

        return (
            _to_pandas(X_train),
            _to_pandas(y_train),
            _to_pandas(X_test),
            _to_pandas(y_test),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline_split_extraction.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full pipeline + base + clustering test files to check for regressions**

Run: `cd skyulf-core && python -m pytest tests/test_pipeline.py tests/test_pipeline_coverage.py tests/test_modeling_base.py tests/test_modeling_clustering.py tests/test_pipeline_split_extraction.py -v`
Expected: PASS, no regressions.

- [ ] **Step 8: Commit**

```bash
cd /Users/BH7043/Skyulf
git add skyulf-core/skyulf/modeling/base.py skyulf-core/skyulf/pipeline.py skyulf-core/tests/test_pipeline_split_extraction.py
git commit -m "feat(skyulf-core): add SkyulfPipeline.get_fitted_split() convenience API

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Task 5: Full lint / type-check / test gate

**Files:** none new — validates all files touched in Tasks 1-4.

**Interfaces:** none — this task only runs commands and fixes anything they report in files touched by Tasks 1-4.

- [ ] **Step 1: Run ruff lint with autofix**

Run:
```bash
cd /Users/BH7043/Skyulf
source .venv/bin/activate
ruff check skyulf-core/skyulf/modeling/sklearn_wrapper.py skyulf-core/skyulf/modeling/hyperparameters/ skyulf-core/skyulf/modeling/_tuning/engine.py skyulf-core/skyulf/modeling/base.py skyulf-core/skyulf/pipeline.py skyulf-core/tests/test_modeling_sklearn_wrapper.py skyulf-core/tests/test_modeling_all.py skyulf-core/tests/test_hyperparameters_class_weight.py skyulf-core/tests/test_tuning_engine.py skyulf-core/tests/test_pipeline_split_extraction.py --fix
```
Expected: `All checks passed!` (or a list of auto-fixed issues — re-run to confirm clean).

- [ ] **Step 2: Run ruff format**

Run:
```bash
ruff format skyulf-core/skyulf/modeling/sklearn_wrapper.py skyulf-core/skyulf/modeling/hyperparameters/ skyulf-core/skyulf/modeling/_tuning/engine.py skyulf-core/skyulf/modeling/base.py skyulf-core/skyulf/pipeline.py skyulf-core/tests/test_modeling_sklearn_wrapper.py skyulf-core/tests/test_modeling_all.py skyulf-core/tests/test_hyperparameters_class_weight.py skyulf-core/tests/test_tuning_engine.py skyulf-core/tests/test_pipeline_split_extraction.py
```
Expected: `N files reformatted` or `N files left unchanged`.

- [ ] **Step 3: Run ty type check**

Run:
```bash
ty check skyulf-core/skyulf skyulf-core/tests
```
Expected: `All checks passed!` (fix any reported issues in files touched by Tasks 1-4 only; pre-existing diagnostics elsewhere are out of scope — confirm via `git blame`/`git log -S` if unsure whether a diagnostic predates this work).

- [ ] **Step 4: Run the full skyulf-core test suite**

Run:
```bash
cd skyulf-core && python -m pytest -q
```
Expected: All tests pass (0 failures). Note any SKIPPED tests (e.g. `xgboost`/`optuna` not installed) are acceptable; FAILED is not.

- [ ] **Step 5: Commit any lint/format fixes**

```bash
cd /Users/BH7043/Skyulf
git add -A
git commit -m "chore(skyulf-core): lint/format fixes for class_weight fixes

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" --allow-empty
```

(Use `--allow-empty` only if Steps 1-2 made no changes; otherwise the normal `git add -A && git commit` captures the fixes.)

---

## Task 6: Version bump and release

**Files:**
- Modify: `skyulf-core/setup.py`

**Interfaces:** none.

- [ ] **Step 1: Bump the version**

In `skyulf-core/setup.py`, change:

```python
    version="0.5.4",
```

to:

```python
    version="0.5.5",
```

(0.5.4 is already committed on this branch for an unrelated change and not yet released; 0.5.5 is the next available version for this work.)

- [ ] **Step 2: Commit the version bump**

```bash
cd /Users/BH7043/Skyulf
git add skyulf-core/setup.py
git commit -m "chore(skyulf-core): bump version to 0.5.5 for class_weight fixes release

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

- [ ] **Step 3: Push/merge to master**

```bash
git push origin HEAD:master
```

(Or open a PR and merge it, per whatever the current branch's normal review process is — either way, the commit must land on `master` to trigger `.github/workflows/release.yml`.)

- [ ] **Step 4: Verify the release workflow ran and published**

```bash
gh run list --workflow=release.yml --limit 3
```
Expected: A new run triggered by the push, with conclusion `success`.

Then verify PyPI:
```bash
sleep 60 && curl -s https://pypi.org/pypi/skyulf-core/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])"
```
Expected: `0.5.5`.

---

## Self-Review

**Spec coverage:**
1. Generic `class_weight` → `sample_weight` shim — Task 1. ✅
2. `class_weight` hyperparameter UI/schema exposure (RF/LightGBM/XGBoost classifiers) — Task 2. ✅
3. Lazy Optuna import — Task 3. ✅
4. Convenience split-extraction API — Task 4. ✅
5. Testing plan (unit tests for shim, native-support pass-through, XGBoost regression test, param-list presence/absence tests, Optuna import-silence test, split-extraction test) — covered across Tasks 1-4. ✅
6. Full lint/type/test gate — Task 5. ✅
7. Release via version bump + push to master — Task 6. ✅
8. Frontend/backend check — confirmed no frontend changes needed (schema-driven UI); no backend (`backend/`) changes needed since it only forwards hyperparameter configs through unchanged. ✅

**Placeholder scan:** No TBD/TODO, no "add appropriate handling", no "similar to Task N" — every step has literal code. ✅

**Type/signature consistency:** `extract_xy(data, target_column)` signature matches across Task 4's Step 1 (definition) and Step 5 (`pipeline.py` usage). `_constructor_accepts_class_weight()`/`_compute_sample_weight_for_fit()` are used consistently within Task 1 only. `_ensure_optuna_loaded()` name matches between Task 3's Step 3 (definition) and Steps 1/4 (test updates, `_build_optuna_searcher` call site). `LGBM_CLASSIFIER_PARAMS`/`XGBOOST_CLASSIFIER_PARAMS` names match across Task 2's Steps 3-7 (definition in `_tree.py`, mapping in `_registry.py`, re-export in `__init__.py`, usage in the test file). ✅
