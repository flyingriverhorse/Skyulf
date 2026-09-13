"""Keep searcher-held validation targets aligned with fold preprocessing."""

import copy
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import check_scoring
from sklearn.pipeline import Pipeline

from .fold_pipeline import FoldAwareModelStep


def _fold_step(estimator: Any) -> FoldAwareModelStep | None:
    """Recognize only the single-step fold wrapper whose data contract Skyulf owns."""
    if isinstance(estimator, Pipeline) and len(estimator.steps) == 1:
        name, step = estimator.steps[0]
        if name == "model":
            estimator = step
    return estimator if isinstance(estimator, FoldAwareModelStep) else None


def _score_fold(scorer: Callable, estimator: Any, X: Any, y: Any) -> float:
    """Transform X/y once and score a disposable view using the original class labels."""
    step = _fold_step(estimator)
    if step is None or step.preprocessor_ is None:
        return scorer(estimator, X, y)
    X, y = step._ensure_frames(X, y)
    X_t, y_t = step.preprocessor_.transform(X, y)
    if step.label_map_ is not None:
        y_t = pd.Series(np.asarray(y_t)).map(step.label_map_).to_numpy()

    # Preserve response methods/classes without applying preprocessing again.
    # Copies also leave the real fitted pipeline intact if the scorer raises.
    scoring_step = copy.copy(step)
    scoring_step.preprocessor_ = None
    scoring_estimator = scoring_step
    if isinstance(estimator, Pipeline):
        scoring_estimator = copy.copy(estimator)
        scoring_estimator.steps = [("model", scoring_step)]
    return scorer(scoring_estimator, X_t, y_t)


def wrap_fold_scorer(estimator: Any, scoring: Any) -> Any:
    """Adapt ordinary searcher scoring when a Skyulf fold step transforms the data.

    The scorer may request labels, probabilities or decision scores repeatedly;
    all responses use the same already transformed rows. Unwrapped estimators
    and generic sklearn pipelines retain their original scorer contract.
    """
    step = _fold_step(estimator)
    if step is None or step.preprocessor is None:
        return scoring
    return partial(_score_fold, check_scoring(estimator, scoring=scoring))
