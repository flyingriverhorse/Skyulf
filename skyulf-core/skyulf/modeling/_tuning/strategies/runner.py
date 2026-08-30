"""Shared machinery for searcher-based strategies (halving / optuna).

Leaf module (F-18 split of ``engine.py``): fits a built searcher, extracts
the best result and per-trial records, and translates known sklearn/optuna
failure messages into actionable errors.
"""

import contextlib
import logging
import warnings
from collections.abc import Callable
from typing import Any, cast

from joblib import parallel_backend

from ..schemas import TuningConfig
from .optuna import _ensure_optuna_loaded

logger = logging.getLogger(__name__)


def execute_search(
    searcher: Any,
    X_arr: Any,
    y_arr: Any,
    config: TuningConfig,
    log_callback: Callable[[str], None] | None = None,
) -> list[str]:
    """Fits the searcher, translating known sklearn/optuna failure messages into
    actionable ``ValueError``s and re-raising anything else unchanged.

    Returns the per-trial failure messages captured for optuna runs (one
    entry per failed trial). Optuna logs these at WARNING level on its own
    logger — without capturing them the only visible symptom is the generic
    "no trials completed" error, with the real cause stuck in stderr.
    """
    captured: list[str] = []
    optuna_logger: logging.Logger | None = None
    handler: logging.Handler | None = None
    if config.strategy == "optuna" and _ensure_optuna_loaded():

        class _TrialFailureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                message = record.getMessage()
                if "failed" in message:
                    captured.append(message)
                    if log_callback:
                        with contextlib.suppress(Exception):
                            log_callback(message)

        optuna_logger = logging.getLogger("optuna")
        handler = _TrialFailureHandler(level=logging.WARNING)
        optuna_logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Failed to report cross validation scores for TerminatorCallback",
            )
            # LightGBM 4.x sets feature_names_in_ even for numpy input; during
            # halving/optuna internal CV sklearn's validate_data emits this warning
            # on every fold's score() call. Suppress it here — the root cause is
            # already fixed in the LGBM calculator's fit() override.
            warnings.filterwarnings(
                "ignore",
                message=".*valid feature names.*",
            )
            if config.parallel_backend:
                with parallel_backend(config.parallel_backend):
                    searcher.fit(X_arr, y_arr)
            else:
                searcher.fit(X_arr, y_arr)
    except Exception as e:
        logger.exception("Hyperparameter tuning failed")
        error_msg = str(e)
        if "No trials are completed yet" in error_msg:
            raise ValueError(
                "Hyperparameter tuning failed: No trials completed successfully. "
                "This usually means the model failed to train with the provided hyperparameter combinations. "
                "Please check your search space and data."
            ) from e

        if "n_samples" in error_msg and "resample" in error_msg and "Got 0" in error_msg:
            raise ValueError(
                "Hyperparameter tuning with Halving strategy failed because the dataset is too small "
                "for the configured halving parameters. Please try using 'Random Search' or 'Grid Search' instead, "
                "or increase your dataset size."
            ) from e

        raise e
    finally:
        if optuna_logger is not None and handler is not None:
            optuna_logger.removeHandler(handler)
    return captured


def extract_best_result(searcher: Any, first_trial_error: str | None = None) -> tuple[Any, float]:
    """Reads ``best_params_``/``best_score_`` off a fitted searcher, translating the
    "no completed trials" ``ValueError`` into a clearer, actionable message that
    carries the first captured per-trial error when available.
    """
    try:
        # Accessing best_params_ raises ValueError if no trials completed successfully
        best_params = searcher.best_params_
        best_score = searcher.best_score_
    except ValueError as e:
        if "No trials are completed yet" in str(e):
            detail = f" First trial error: {first_trial_error}" if first_trial_error else ""
            raise ValueError(
                "Hyperparameter tuning failed: All trials failed. "
                "This often happens if the model produces NaN scores "
                "(e.g., due to unscaled data for linear models/SVMs, exploding gradients, "
                "or mismatched parameters). "
                "Try adding a 'Scale' node before this model or checking for NaN/Infinity in your data."
                + detail
            ) from e
        raise e
    return best_params, best_score


def collect_trials(searcher: Any, config: TuningConfig) -> list[dict[str, Any]]:
    """Extracts per-trial params/scores from a fitted searcher (Optuna study or cv_results_)."""
    trials: list[dict[str, Any]] = []
    # Special handling for Optuna
    if config.strategy == "optuna" and hasattr(searcher, "study_"):
        # Only include completed trials
        trials.extend(
            {"params": trial.params, "score": trial.value}
            for trial in cast(Any, searcher).study_.trials
            if trial.state.name == "COMPLETE"
        )
    elif hasattr(searcher, "cv_results_"):
        results = searcher.cv_results_
        if "params" in results:
            n_candidates = len(results["params"])
            trials.extend(
                {
                    "params": results["params"][i],
                    "score": results["mean_test_score"][i],
                }
                for i in range(n_candidates)
            )
    return trials


def strip_model_prefix(params: Any) -> Any:
    """Removes the internal ``model__estimator__`` pipeline prefix from
    extracted params (see ``tune``'s wrapped Pipeline path) so callers see
    the original search-space keys."""
    if not isinstance(params, dict):
        return params
    return {
        key.removeprefix("model__").removeprefix("estimator__"): value
        for key, value in params.items()
    }


def log_final_completion(
    log_callback: Callable[[str], None] | None,
    config: TuningConfig,
    trials: list[dict[str, Any]],
    best_score: float,
    best_params: Any,
) -> None:
    """Emits the completion log for searcher-based strategies that don't emit
    per-trial callbacks (halving_grid / halving_random / optuna).
    """
    if log_callback and config.strategy in [
        "halving_grid",
        "halving_random",
        "optuna",
    ]:
        log_callback(
            f"Tuning Completed ({config.strategy}). "
            f"Trials evaluated: {len(trials)}. Best Score: {best_score:.4f}"
        )
        log_callback(f"Best Params: {best_params}")
