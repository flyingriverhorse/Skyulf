"""Keep searcher-held validation targets aligned with fold preprocessing."""

import copy
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline

from ...data.coverage import transform_evaluation
from .fold_pipeline import FoldAwareModelStep


def _fold_step(estimator: Any) -> FoldAwareModelStep | None:
    """Recognize only the single-step fold wrapper whose data contract Skyulf owns."""
    if isinstance(estimator, Pipeline) and len(estimator.steps) == 1:
        name, step = estimator.steps[0]
        if name == "model":
            estimator = step
    return estimator if isinstance(estimator, FoldAwareModelStep) else None


def prepare_fold_evaluation(
    estimator: Any, X: Any, y: Any, *, allow_empty: bool = False
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Return a disposable scoring view and one aligned eligible evaluation pair."""
    step = _fold_step(estimator)
    if step is None or step.preprocessor_ is None:
        X, y, coverage = transform_evaluation(None, X, y, allow_empty=allow_empty)
        return estimator, X, y, coverage
    X, y = step._ensure_frames(X, y)
    X_t, y_t, coverage = transform_evaluation(step.preprocessor_, X, y, allow_empty=allow_empty)
    if step.label_map_ is not None:
        # Some custom preprocessors encode only during fit; preserve labels
        # already in the public class space instead of replacing them with NaN.
        y_t = np.asarray([step.label_map_.get(label, label) for label in np.asarray(y_t)])

    # Preserve response methods/classes without applying preprocessing again.
    # Copies also leave the real fitted pipeline intact if the scorer raises.
    scoring_step = copy.copy(step)
    scoring_step.preprocessor_ = None
    scoring_estimator = scoring_step
    if isinstance(estimator, Pipeline):
        scoring_estimator = copy.copy(estimator)
        scoring_estimator.steps = [("model", scoring_step)]
    return scoring_estimator, X_t, y_t, coverage


def _score_fold(
    scorer: Callable, estimator: Any, X: Any, y: Any, *, include_coverage: bool = False
) -> Any:
    """Score eligible pairs and optionally transport small diagnostics through sklearn workers."""
    view, X_t, y_t, coverage = prepare_fold_evaluation(
        estimator, X, y, allow_empty=include_coverage
    )
    # Halving workers must return the empty population alongside the failed score
    # so the parent can explain the failure without fitting or scoring again.
    score = scorer(view, X_t, y_t) if len(X_t) else float("nan")
    if include_coverage:
        return {"score": score, **{key: coverage[key] for key in _COVERAGE_KEYS}}
    return score


_COVERAGE_KEYS = ("input_rows", "scored_rows", "excluded_rows")


def wrap_fold_scorer(estimator: Any, scoring: Any, *, include_coverage: bool = False) -> Any:
    """Adapt ordinary searcher scoring when a Skyulf fold step transforms the data.

    The scorer may request labels, probabilities or decision scores repeatedly;
    all responses use the same already transformed rows. Unwrapped estimators
    and generic sklearn pipelines retain their original scorer contract.
    """
    step = _fold_step(estimator)
    if not include_coverage and (step is None or step.preprocessor is None):
        return scoring
    return partial(
        _score_fold, check_scoring(estimator, scoring=scoring), include_coverage=include_coverage
    )
