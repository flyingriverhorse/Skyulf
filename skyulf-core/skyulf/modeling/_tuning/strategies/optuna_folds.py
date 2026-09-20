"""Fit isolated tuning folds and score native boosting iterations safely."""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ....engines.sklearn_bridge import SklearnBridge
from ..._class_weights import sample_weight_for_fit
from ..fold_pipeline import FoldAwareModelStep

IterationReport = Callable[[float, int], None]


@dataclass
class _PreparedFold:
    """Own this fold's estimator, transformed data, and optional label view."""

    model: Any
    X_train: Any
    y_train: Any
    X_valid: Any
    y_valid: Any
    scoring_step: FoldAwareModelStep | None = None
    sample_weight: Any = None

    def score(self, scorer: Callable, model: Any) -> float:
        """Evaluate the caller's signed scorer in the original target space."""
        scoring_model = model
        if self.scoring_step is not None:
            scoring_model = copy.copy(self.scoring_step)
            scoring_model.model_ = model
        score = float(scorer(scoring_model, self.X_valid, self.y_valid))
        if not np.isfinite(score):
            raise ValueError("The validation scorer returned a non-finite score.")
        return score


def _fold_step(estimator: Any) -> FoldAwareModelStep | None:
    """Recognize only the wrapper whose fit and target contracts Skyulf owns."""
    if isinstance(estimator, FoldAwareModelStep):
        return estimator
    if isinstance(estimator, Pipeline) and len(estimator.steps) == 1:
        name, step = estimator.steps[0]
        if name == "model" and isinstance(step, FoldAwareModelStep):
            return step
    return None


def _prepare_fold(
    estimator: Any, X_train: Any, y_train: Any, X_valid: Any, y_valid: Any
) -> _PreparedFold:
    """Fit preprocessing on training rows and transform validation X and y together."""
    estimator = clone(estimator)
    step = _fold_step(estimator)
    if step is None:
        return _PreparedFold(estimator, X_train, y_train, X_valid, y_valid)
    model = step.estimator
    worker = step.preprocessor
    if worker is not None:
        X_train, y_train = step._ensure_frames(X_train, y_train)
        X_valid, y_valid = step._ensure_frames(X_valid, y_valid)
    original_y = y_train
    if worker is not None:
        X_train, y_train = worker.fit_transform(X_train, y_train)
        X_valid, y_valid = worker.transform(X_valid, y_valid)
    SklearnBridge.validate_features(X_train)
    SklearnBridge.validate_features(X_valid)
    step.label_map_ = step._build_label_map(original_y, y_train, model)
    step.preprocessor_ = None  # Scorers receive data already transformed as a pair.
    if step.label_map_ is not None:
        y_valid = pd.Series(np.asarray(y_valid)).map(step.label_map_).to_numpy()
    sample_weight = sample_weight_for_fit(model, step.class_weight, y_train)
    return _PreparedFold(model, X_train, y_train, X_valid, y_valid, step, sample_weight)


def _native_library(model: Any) -> str | None:
    """Identify native estimator subclasses without importing optional libraries."""
    modules = {base.__module__ for base in type(model).__mro__}
    if "xgboost.sklearn" in modules:
        return "xgboost"
    if "lightgbm.sklearn" in modules:
        return "lightgbm"
    return None


def _score_booster(fold: _PreparedFold, scorer: Callable, booster: Any, library: str) -> float:
    """Expose the live booster through a disposable native sklearn scoring view.

    Native fit initializes classifier labels and objectives before callbacks,
    but assigns the completed booster only after training returns. A shallow
    view preserves those semantics without mutating the fitting estimator.
    """
    model = copy.copy(fold.model)
    model._Booster = booster
    if library == "lightgbm":
        model.fitted_ = True
        model._n_features = booster.num_feature()
    return fold.score(scorer, model)


def _xgboost_callback(fold: _PreparedFold, scorer: Callable, report: IterationReport) -> Any:
    """Lazily build a public XGBoost callback that propagates pruning decisions."""
    from xgboost.callback import TrainingCallback  # noqa: PLC0415 - optional dependency

    class ScoringCallback(TrainingCallback):
        """Score after a native boosting round without changing its objective."""

        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            """Report the requested sklearn score and let TrialPruned escape fit."""
            report(_score_booster(fold, scorer, model, "xgboost"), epoch)
            return False

    return ScoringCallback()


def _lightgbm_callback(fold: _PreparedFold, scorer: Callable, report: IterationReport) -> Callable:
    """Build a public LightGBM callback executed after each boosting round."""

    def callback(env: Any) -> None:
        """Preserve native classification responses while reporting signed scores."""
        report(_score_booster(fold, scorer, env.model, "lightgbm"), env.iteration)

    return callback


def _native_fit_kwargs(
    fold: _PreparedFold, scorer: Callable, report: IterationReport | None, library: str
) -> dict[str, Any]:
    """Add pruning alongside caller callbacks, retaining native fit defaults."""
    callbacks = list(fold.model.get_params(deep=False).get("callbacks") or [])
    if library == "xgboost":
        if report is None:
            return {}
        callbacks.append(_xgboost_callback(fold, scorer, report))
        fold.model.set_params(callbacks=callbacks)
        return {"eval_set": [(fold.X_valid, _native_validation_y(fold))], "verbose": False}
    if report is not None:
        callbacks.append(_lightgbm_callback(fold, scorer, report))
    # LightGBM accepts callbacks in fit rather than its native booster parameters.
    fold.model._other_params.pop("callbacks", None)
    fit_kwargs = {"callbacks": callbacks}
    if report is not None:
        fit_kwargs["eval_set"] = [(fold.X_valid, _native_validation_y(fold))]
    return fit_kwargs


def _native_validation_y(fold: _PreparedFold) -> Any:
    """Recover encoded targets for native eval sets from the scoring label space."""
    step = fold.scoring_step
    if step is None or step.label_map_ is None:
        return fold.y_valid
    encoding = {original: encoded for encoded, original in step.label_map_.items()}
    return pd.Series(np.asarray(fold.y_valid)).map(encoding).to_numpy()


def fit_and_score_fold(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    scorer: Callable,
    report: IterationReport | None = None,
) -> float:
    """Fit an independent fold and return its finite signed validation score.

    Known Skyulf wrappers refit preprocessing on this training fold and keep
    validation features and targets aligned through row or target changes.
    XGBoost and LightGBM additionally report zero-based boosting iterations
    when requested. Arbitrary estimators and sklearn pipelines retain their
    ordinary full fitting behavior. Fit, scorer, and pruning errors propagate.
    """
    fold = _prepare_fold(estimator, X_train, y_train, X_valid, y_valid)
    fit_kwargs = {}
    library = _native_library(fold.model)
    if library is not None:
        fit_kwargs = _native_fit_kwargs(fold, scorer, report, library)
    if fold.sample_weight is not None:
        fit_kwargs["sample_weight"] = fold.sample_weight
    fold.model.fit(fold.X_train, fold.y_train, **fit_kwargs)
    return fold.score(scorer, fold.model)
