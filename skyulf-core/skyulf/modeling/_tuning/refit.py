"""Best-model refit and decision-threshold tuning.

Leaf module (F-18 split of ``engine.py``). The wrapped model calculator is
passed in and read lazily (``getattr``) so the refit stays compatible with
calculators that only expose part of the SklearnCalculator surface.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ...engines.sklearn_bridge import SklearnBridge
from .._evaluation.thresholds import optimize_thresholds
from ..base import BaseModelCalculator
from .params import instantiate_model
from .schemas import TuningConfig, TuningResult

logger = logging.getLogger(__name__)


def refit_best_model(
    model_calculator: BaseModelCalculator,
    tuning_result: TuningResult,
    tuning_config: TuningConfig,
    X_np: Any,
    y_np: Any,
    log_callback: Callable[[str], None] | None,
    iteration_callback: Callable[..., None] | None = None,
) -> Any:
    """Build and fit the final model on the full dataset using the tuned best params."""
    best_params = tuning_result.best_params
    final_params = {**model_calculator.default_params, **best_params}

    # The caller's seed must win over the calculator's baked-in default,
    # but not over a seed the search itself selected.
    if "random_state" not in best_params and tuning_config.random_state is not None:
        final_params["random_state"] = tuning_config.random_state

    if log_callback:
        log_callback(f"Refitting best model with params: {final_params}")

    # Mypy doesn't know that model_calculator has model_class because it's typed as BaseModelCalculator
    # We can cast it or ignore it.
    model_cls = getattr(model_calculator, "model_class", None)
    if not model_cls:
        raise ValueError("Model calculator does not have a model_class attribute")

    # Build the final model. ``instantiate_model`` filters constructor args
    # to the signature (when there is no **kwargs) and routes nested
    # ``a__b`` keys — e.g. an ensemble's tuned base-model params — through
    # ``set_params`` so they are not silently dropped.
    model = instantiate_model(model_cls, final_params)
    # Boosting base calculators (XGBoost/LightGBM) attach an eval set +
    # iteration callback here so the final refit streams per-round
    # progress like any other boosting fit; every other model keeps a
    # plain fit.
    boosting_hook = getattr(model_calculator, "_boosting_fit_kwargs", None)
    extra_fit_kwargs = (
        boosting_hook(model, X_np, y_np, iteration_callback) if boosting_hook is not None else {}
    )
    detach_callbacks = extra_fit_kwargs.pop("_detach_callbacks", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=".*valid feature names.*")
        model.fit(X_np, y_np, **extra_fit_kwargs)
    if detach_callbacks:
        model.callbacks = None
    for w in caught:
        if issubclass(w.category, ConvergenceWarning):
            conv_msg = (
                f"Final refit of {model_cls.__name__} with the best params did not "
                f"fully converge: {w.message}"
            )
            logger.warning(conv_msg)
            if log_callback:
                log_callback(conv_msg)
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    return model


# Hard-label (y_true, y_pred) classification metrics that a threshold
# search can maximise. Probability-only metrics (roc_auc*, log_loss,
# pr_auc*, g_score) cannot be computed from hard labels, so threshold
# search falls back to balanced_accuracy for those.
def resolve_threshold_metric(
    metric_name: str,
    log_callback: Callable[[str], None] | None,
    pos_label: Any = None,
) -> tuple[Callable[[Any, Any], float], str]:
    """Map the user's tuning metric to a hard-label callable for the threshold search.

    Returns ``(callable, used_metric_name)``. Probability-only metrics fall
    back to ``balanced_accuracy`` (with a log), since a threshold sweep only
    ever produces hard class labels.

    ``pos_label`` (the model's positive class) is pinned into the binary
    f1/precision/recall callables when given: their sklearn defaults assume
    ``pos_label=1`` and raise for label spaces without a 1 (string labels).
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    pos_kwargs = {"pos_label": pos_label} if pos_label is not None else {}
    hard_label: dict[str, Callable[[Any, Any], float]] = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0, **pos_kwargs),
        "f1_weighted": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0, **pos_kwargs),
        "precision_weighted": lambda yt, yp: precision_score(
            yt, yp, average="weighted", zero_division=0
        ),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0, **pos_kwargs),
        "recall_weighted": lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0),
        "matthews_corrcoef": matthews_corrcoef,
    }
    if metric_name in hard_label:
        return hard_label[metric_name], metric_name

    msg = (
        f"Decision-threshold tuning: metric '{metric_name}' needs predicted "
        "probabilities, but a threshold sweep only produces hard labels — "
        "tuning the threshold against 'balanced_accuracy' instead."
    )
    logger.info(msg)
    if log_callback:
        log_callback(msg)
    return balanced_accuracy_score, "balanced_accuracy"


def tune_decision_thresholds(
    model_calculator: BaseModelCalculator,
    model: Any,
    tuning_result: TuningResult,
    tuning_config: TuningConfig,
    validation_data: tuple[Any, Any] | None,
    log_callback: Callable[[str], None] | None,
) -> None:
    """F-13: search a decision threshold on the validation split for a binary
    classifier and store it on ``tuning_result``.

    Gates: classification problem, a model exposing ``predict_proba`` and
    ``classes_``, an exactly-binary target, and a provided validation split.
    Any gate that fails — or any error during the search — logs and leaves
    ``tuning_result.decision_thresholds`` as ``None`` so ``predict()`` keeps
    the model's default decision rule; threshold tuning is best-effort and
    must never abort an otherwise successful tuning run.
    """

    def _skip(reason: str) -> None:
        msg = f"Decision-threshold tuning skipped: {reason}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    if getattr(model_calculator, "problem_type", None) != "classification":
        _skip("this is not a classification model.")
        return
    if validation_data is None:
        _skip("no validation split was provided.")
        return
    if not hasattr(model, "predict_proba"):
        _skip("the model does not expose predict_proba.")
        return
    classes = getattr(model, "classes_", None)
    if classes is None:
        _skip("the fitted model does not expose class labels (classes_).")
        return
    classes = np.asarray(classes)
    if len(classes) != 2:
        _skip(f"only binary classification is supported (found {len(classes)} classes).")
        return

    try:
        X_val, y_val = validation_data
        X_val_np, y_val_np = SklearnBridge.to_sklearn((X_val, y_val))
        y_proba = np.asarray(model.predict_proba(X_val_np))

        metric_callable, metric_name = resolve_threshold_metric(
            tuning_config.metric, log_callback, pos_label=classes[1]
        )
        thresholds = optimize_thresholds(
            y_val_np,
            y_proba,
            metric=metric_callable,
            classes=classes,
            strategy="grid",
        )

        tuning_result.decision_thresholds = thresholds
        tuning_result.decision_threshold_metric = metric_name

        positive_threshold = thresholds.get(classes[1])
        msg = (
            f"Decision-threshold tuning: selected threshold "
            f"{positive_threshold:.4f} for class '{classes[1]}' "
            f"(maximising '{metric_name}' on the validation split)."
        )
        logger.info(msg)
        if log_callback:
            log_callback(msg)
    except Exception as e:  # noqa: BLE001 - threshold tuning is best-effort; a failure must not abort a successful tuning run
        msg = f"Decision-threshold tuning failed; predict() keeps the default decision rule. ({e})"
        logger.warning(msg)
        if log_callback:
            log_callback(msg)
