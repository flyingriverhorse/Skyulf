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
        raise ValueError(f"classes has {len(classes)} entries but y_proba has {n_classes} columns")

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


# Initial-simplex step sizes (log-threshold units) tried by
# `_nelder_mead_multiclass`. `apply_thresholds`'s scaled-argmax rule makes the
# objective piecewise-constant (a plateau) around any point, and scipy's
# default initial simplex for a zero-valued starting coordinate perturbs by a
# minuscule ~0.00025 log-units (~0.025% threshold change) -- almost never
# enough to cross a real decision boundary. Left at scipy's default, the
# optimizer sees a flat plateau in every direction and immediately
# "converges" at the untouched starting point (plain argmax), silently never
# tuning anything. These larger steps (~exp(0.75)=2.1x up to exp(3.0)=20x
# threshold swings) give Nelder-Mead a real gradient to follow.
_INITIAL_SIMPLEX_STEPS: tuple[float, ...] = (0.75, 1.5, 3.0)


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

    Runs from several initial simplex sizes (see `_INITIAL_SIMPLEX_STEPS`)
    and keeps whichever run scores best, always compared against the
    untouched starting point -- so the result is guaranteed never worse than
    plain argmax, and trying multiple simplex sizes reduces the risk of the
    search getting stuck on the flat plateau surrounding any single scale.
    """
    n_classes = len(classes)
    x0 = np.zeros(n_classes)  # log(1.0) == 0 for every class: starts at plain argmax

    def negative_score(x: np.ndarray) -> float:
        thresholds = {c: float(np.exp(xi)) for c, xi in zip(classes, x, strict=True)}
        pred = apply_thresholds(y_proba, thresholds, classes=classes)
        return -metric(y_true, pred)

    best_x = x0
    best_score = negative_score(x0)
    identity = np.eye(n_classes)
    for step in _INITIAL_SIMPLEX_STEPS:
        initial_simplex = np.vstack([x0, *(x0 + step * identity[i] for i in range(n_classes))])
        result = minimize(
            negative_score,
            x0,
            method="Nelder-Mead",
            options={"initial_simplex": initial_simplex},
        )
        if result.fun < best_score:
            best_score = result.fun
            best_x = result.x

    return {c: float(np.exp(xi)) for c, xi in zip(classes, best_x, strict=True)}


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
