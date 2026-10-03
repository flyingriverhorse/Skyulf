"""Select binary decision thresholds using only inner out-of-fold predictions."""

from typing import Any

import numpy as np
from sklearn.utils.metaestimators import available_if

from .._class_weights import split_class_weight_params
from .._cv_weights import preflight_weights, take_weights, weight_kwargs
from .._evaluation.thresholds import apply_thresholds, optimize_thresholds
from .fold_pipeline import FoldAwareModelStep, _fitted_model_has
from .grid_random import _slice_fold_rows
from .metrics import resolve_metric, resolve_scorer
from .params import instantiate_model, seed_params
from .refit import resolve_threshold_metric
from .schemas import TuningConfig, TuningResult


class _ThresholdModelStep(FoldAwareModelStep):
    """Keep probability responses intact while applying selected hard decisions."""

    # sklearn 1.4/1.5 recognize classifiers through this legacy marker.
    _estimator_type = "classifier"
    decision_thresholds: dict[Any, float]

    @property
    def classes_(self) -> Any:
        """Expose the canonical raw label order required by probability scorers."""
        return np.sort(super().classes_)

    def predict_proba(self, X: Any) -> Any:
        """Reorder probability columns when preprocessing reversed the class order."""
        original = np.asarray(super().classes_)
        return np.asarray(super().predict_proba(X))[:, np.argsort(original)]

    @available_if(_fitted_model_has("decision_function"))
    def decision_function(self, X: Any) -> Any:
        """Keep decision scores oriented toward the canonical raw positive class."""
        original = np.asarray(super().classes_)
        scores = super().decision_function(X)
        return -scores if original[0] > original[1] else scores

    def predict(self, X: Any) -> Any:
        """Apply training-only thresholds in the original target label space."""
        return apply_thresholds(
            self.predict_proba(X), self.decision_thresholds, classes=self.classes_
        )


def _fit_step(
    tuner: Any,
    X: Any,
    y: Any,
    config: TuningConfig,
    best_params: dict[str, Any],
    preprocessing: Any,
    sample_weight: Any = None,
) -> _ThresholdModelStep:
    """Refit the selected recipe, including preprocessing and fold-local class weights."""
    calculator = tuner.model_calculator
    params = {**calculator.default_params, **seed_params(config), **best_params}
    constructor, class_weight = split_class_weight_params(calculator.model_class, params)
    estimator = instantiate_model(calculator.model_class, constructor)
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Nested threshold selection requires a model exposing predict_proba.")
    step = _ThresholdModelStep(
        estimator=estimator, preprocessor=preprocessing, class_weight=class_weight
    )
    step.fit(X, y, **weight_kwargs(sample_weight))
    return step


def _binary_classes(tuner: Any, y: Any) -> np.ndarray:
    """Reject unsupported targets before collecting threshold evidence."""
    classes = np.unique(np.asarray(y))
    if tuner.problem_type != "classification" or len(classes) != 2:
        raise ValueError("Nested threshold selection supports binary classification only.")
    return classes


def _probabilities(step: Any, X: Any, classes: np.ndarray) -> np.ndarray:
    """Validate class coverage and probability shape, then align columns to raw labels."""
    fitted_classes = np.asarray(step.classes_)
    if len(fitted_classes) != 2 or set(fitted_classes) != set(classes):
        raise ValueError("Nested threshold selection found a label/class mismatch in a fold.")
    proba = np.asarray(step.predict_proba(X), dtype=float)
    if proba.shape != (len(X), 2):
        raise ValueError("Nested threshold probabilities must retain held-out row alignment.")
    if not np.all(np.isfinite(proba)):
        raise ValueError("Nested threshold probabilities must be finite.")
    if np.any(proba < 0) or np.any(proba > 1) or not np.allclose(proba.sum(axis=1), 1):
        raise ValueError("Nested threshold probabilities must be normalized between zero and one.")
    order = [fitted_classes.tolist().index(label) for label in classes]
    return proba[:, order]


def _validate_partition(train: Any, test: Any, covered: np.ndarray) -> None:
    """Reject duplicate OOF observations and overlap with the fitting partition."""
    if len(train) == 0 or len(test) == 0 or np.intersect1d(train, test).size:
        raise ValueError("Nested threshold folds require disjoint nonempty train/test rows.")
    if np.unique(test).size != len(test) or np.any(covered[test]):
        raise ValueError("Nested threshold OOF rows must be evaluated exactly once.")


def select_nested_threshold(
    tuner: Any,
    X: Any,
    y: Any,
    config: TuningConfig,
    best_params: dict[str, Any],
    cv: Any,
    preprocessing: Any = None,
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Select a threshold from fresh inner-fold predictions of the winning parameters.

    Temporal warmup rows may be uncovered. They are excluded from optimization;
    evidence records the actual coverage without storing row-level predictions.
    """
    preflight_weights(sample_weight, cv, X, y)
    classes = _binary_classes(tuner, y)
    partitions = list(cv.split(X, y) if hasattr(cv, "split") else cv)
    covered = np.zeros(len(y), dtype=bool)
    probabilities = np.empty((len(y), 2), dtype=float)
    for train, test in partitions:
        _validate_partition(train, test, covered)
        step = _fit_step(
            tuner,
            _slice_fold_rows(X, train),
            _slice_fold_rows(y, train),
            config,
            best_params,
            preprocessing,
            **weight_kwargs(take_weights(sample_weight, train)),
        )
        probabilities[test] = _probabilities(step, _slice_fold_rows(X, test), classes)
        covered[test] = True
    if not np.any(covered) or len(np.unique(np.asarray(y)[covered])) != 2:
        raise ValueError("Nested threshold selection requires OOF observations of both classes.")
    metric, metric_name = resolve_threshold_metric(config.metric, None, pos_label=classes[1])

    def finite_metric(y_true: Any, y_pred: Any) -> float:
        """Prevent invalid candidate scores from silently keeping the default cutoff."""
        value = float(metric(y_true, y_pred))
        if not np.isfinite(value):
            raise ValueError("Nested threshold selection requires finite candidate scores.")
        return value

    thresholds = optimize_thresholds(
        np.asarray(y)[covered], probabilities[covered], finite_metric, classes=classes
    )
    return {
        "decision_thresholds": {label: float(thresholds[label]) for label in classes.tolist()},
        "decision_threshold_metric": metric_name,
        "selection": "inner_oof",
        "oof_rows": int(covered.sum()),
        "training_rows": len(y),
        "inner_folds": len(partitions),
        "positive_class": classes.tolist()[1],
    }


def score_nested_threshold(
    tuner: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    config: TuningConfig,
    best_params: dict[str, Any],
    selection: dict[str, Any],
    preprocessing: Any = None,
    sample_weight: Any = None,
) -> float:
    """Refit on outer training rows and evaluate the selected threshold on untouched rows."""
    classes = _binary_classes(tuner, y_train)
    if not set(np.unique(np.asarray(y_test))).issubset(set(classes)):
        raise ValueError("Nested threshold outer labels do not match training classes.")
    step = _fit_step(tuner, X_train, y_train, config, best_params, preprocessing, sample_weight)
    _probabilities(step, X_test, classes)
    step.decision_thresholds = selection["decision_thresholds"]
    metric = resolve_metric(config, y_train, tuner.problem_type)
    score = float(resolve_scorer(metric, y_train, tuner.problem_type)(step, X_test, y_test))
    if not np.isfinite(score):
        raise ValueError("Nested threshold outer evaluation must produce a finite score.")
    return score


def remap_nested_thresholds(result: TuningResult, model: Any, y_raw: Any, y_refit: Any) -> None:
    """Map raw-label selection thresholds onto the final model's encoded class axis."""
    thresholds = result.decision_thresholds
    if thresholds is None:
        raise ValueError("Nested threshold selection did not produce final decision thresholds.")
    label_map = FoldAwareModelStep._build_label_map(y_raw, y_refit, model)
    classes = np.asarray(model.classes_).tolist()
    raw_classes = [label_map.get(label, label) for label in classes] if label_map else classes
    if len(raw_classes) != 2 or set(raw_classes) != set(thresholds):
        raise ValueError("Nested threshold final model has a label/class mismatch.")
    result.decision_thresholds = {
        encoded: thresholds[raw] for encoded, raw in zip(classes, raw_classes, strict=True)
    }
