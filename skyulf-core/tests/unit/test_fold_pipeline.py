"""Unit tests for ``FoldPreprocessingStep`` — the Pipeline step that gives
``halving_*``/``optuna`` tuning their per-fold refit (F-15 follow-up)."""

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skyulf.modeling._tuning.fold_pipeline import FoldPreprocessingStep


class RowDroppingPreprocessor:
    """fit_transform drops the first row; transform keeps every row (F-18)."""

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X.iloc[1:], y.iloc[1:]

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


class CountingPreprocessor:
    """Counts fit_transform calls; deep-copy isolation probe."""

    def __init__(self) -> None:
        self.fits = 0

    def fit_transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        self.fits += 1
        return X, y

    def transform(self, X: Any, y: Any) -> tuple[Any, Any]:
        return X, y


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y = pd.Series([0, 1] * 10, name="target")
    return X, y


def test_fit_transform_returns_refit_output_directly() -> None:
    """Row-count-changing steps shape what the model trains on, while
    transform keeps every held-out row."""
    X, y = _xy()
    step = FoldPreprocessingStep(RowDroppingPreprocessor())

    trained_on = step.fit_transform(X, y)
    assert len(trained_on) == len(X) - 1

    held_out = step.transform(X)
    assert len(held_out) == len(X)


def test_fit_deep_copies_the_preprocessor() -> None:
    """Searcher clones share one constructor preprocessor; each fit must work
    on its own copy so parallel candidates never share fitted state."""
    X, y = _xy()
    original = CountingPreprocessor()
    step = FoldPreprocessingStep(original)

    step.fit(X, y)
    assert original.fits == 0  # the worker was a deep copy
    assert step.preprocessor_.fits == 1

    step.fit(X, y)  # second fit starts from a fresh copy again
    assert original.fits == 0
    assert step.preprocessor_.fits == 1


def test_step_is_sklearn_cloneable() -> None:
    step = FoldPreprocessingStep(CountingPreprocessor())
    cloned = clone(step)
    assert cloned is not step
    assert isinstance(cloned, FoldPreprocessingStep)


def test_searcher_cv_refits_every_fold_and_leaves_original_unfitted() -> None:
    """Inside a real searcher's CV loop: one worker fit per fold, the shared
    constructor preprocessor never fitted."""
    X, y = _xy()
    original = CountingPreprocessor()
    pipe = Pipeline(
        [
            ("preprocessing", FoldPreprocessingStep(original)),
            ("model", LogisticRegression()),
        ]
    )

    search = GridSearchCV(pipe, {"model__C": [1.0]}, cv=3)
    search.fit(X, y)

    assert original.fits == 0
    assert search.best_params_ == {"model__C": 1.0}
