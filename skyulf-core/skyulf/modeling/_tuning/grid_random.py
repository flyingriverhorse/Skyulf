"""The grid/random search strategy: candidate generation and the per-fold CV loop.

Leaf module (F-18 split of ``engine.py``). The wrapped model calculator is
passed through whole (never destructured) so that attribute failures — e.g.
a calculator without ``default_params`` — still surface inside each fold's
try/except and degrade to a ``-inf`` fold score instead of aborting the
search.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from ..base import BaseModelCalculator
from .metrics import resolve_scorer
from .params import clean_search_space, instantiate_model, seed_params
from .schemas import TuningConfig, TuningResult

if TYPE_CHECKING:
    from ..fold_preprocessing import FoldPreprocessor


def generate_search_candidates(config: TuningConfig) -> list[dict[str, Any]]:
    """Generates the list of hyperparameter candidates for grid or random search."""
    param_space = clean_search_space(config.search_space)
    if config.strategy == "grid":
        return list(ParameterGrid(param_space))
    # Random Search
    return list(
        ParameterSampler(
            param_space,
            n_iter=config.n_trials,
            random_state=config.random_state,
        )
    )


def evaluate_candidate_cv(
    candidate_idx: int,
    params: dict[str, Any],
    model_class: Any,
    cv: Any,
    X_for_search: Any,
    y_for_search: Any,
    metric: str,
    log_callback: Callable[[str], None] | None,
    preprocessing: "FoldPreprocessor | None" = None,
    fold_errors: list[str] | None = None,
    seed_params_overlay: dict[str, Any] | None = None,
    *,
    model_calculator: BaseModelCalculator,
) -> float:
    """Cross-validates one grid/random-search candidate and returns its mean fold score.

    Fold failures are logged and penalized with ``-inf`` instead of raised, so a single
    bad hyperparameter combination doesn't abort the whole search.
    """
    fold_scores = []

    # Ensure numpy
    X_any = cast(Any, X_for_search)
    y_any = cast(Any, y_for_search)
    X_arr = X_any.to_numpy() if hasattr(X_any, "to_numpy") else X_any
    y_arr = y_any.to_numpy() if hasattr(y_any, "to_numpy") else y_any

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr, y_arr)):
        score = fit_and_score_candidate_fold(
            candidate_idx=candidate_idx,
            fold_idx=fold_idx,
            params=params,
            model_class=model_class,
            cv=cv,
            X_any=X_any,
            y_any=y_any,
            X_arr=X_arr,
            y_arr=y_arr,
            train_idx=train_idx,
            val_idx=val_idx,
            metric=metric,
            log_callback=log_callback,
            preprocessing=preprocessing,
            fold_errors=fold_errors,
            seed_params_overlay=seed_params_overlay,
            model_calculator=model_calculator,
        )
        fold_scores.append(score)

    # Filter out failed folds for mean calculation if possible, or penalize
    valid_scores = [s for s in fold_scores if s != -float("inf")]
    return float(np.mean(valid_scores)) if valid_scores else -float("inf")


def fit_and_score_candidate_fold(
    candidate_idx: int,
    fold_idx: int,
    params: dict[str, Any],
    model_class: Any,
    cv: Any,
    X_any: Any,
    y_any: Any,
    X_arr: Any,
    y_arr: Any,
    train_idx: Any,
    val_idx: Any,
    metric: str,
    log_callback: Callable[[str], None] | None,
    preprocessing: "FoldPreprocessor | None" = None,
    fold_errors: list[str] | None = None,
    seed_params_overlay: dict[str, Any] | None = None,
    *,
    model_calculator: BaseModelCalculator,
) -> float:
    """Fits one candidate on a single CV fold and returns its score, or ``-inf`` on failure.

    Errors (e.g. incompatible params) are caught and logged rather than raised, so a single
    bad fold doesn't abort the whole candidate evaluation. When ``fold_errors`` is provided,
    every failure message is appended so the search can surface them if every trial fails.
    """
    # Split
    X_train_fold = X_any.iloc[train_idx] if hasattr(X_any, "iloc") else X_any[train_idx]
    y_train_fold = y_any.iloc[train_idx] if hasattr(y_any, "iloc") else y_any[train_idx]
    X_val_fold = X_any.iloc[val_idx] if hasattr(X_any, "iloc") else X_any[val_idx]
    y_val_fold = y_any.iloc[val_idx] if hasattr(y_any, "iloc") else y_any[val_idx]

    # Instantiate and Fit
    # Note: We must handle potential errors (e.g. incompatible params)
    try:
        # F-15: refit preprocessing inside the fold so its statistics
        # never see this fold's held-out rows (inside the try so a
        # preprocessing failure is contained like a model-fit failure).
        if preprocessing is not None:
            X_train_fold, y_train_fold = preprocessing.fit_transform(X_train_fold, y_train_fold)
            X_val_fold, y_val_fold = preprocessing.transform(X_val_fold, y_val_fold)

        model = instantiate_model(
            model_class,
            {**model_calculator.default_params, **(seed_params_overlay or {}), **params},
        )
        model.fit(X_train_fold, y_train_fold)

        # Score — resolved against the fold's (post-transform) labels so
        # binary scorers get a valid pos_label even for string targets.
        scorer = resolve_scorer(
            metric, y_train_fold, getattr(model_calculator, "problem_type", None)
        )
        score = scorer(model, X_val_fold, y_val_fold)

        if log_callback:
            n_splits = cv.get_n_splits(X_arr, y_arr)
            log_callback(
                f"  [Candidate {candidate_idx + 1}] CV Fold {fold_idx + 1}/{n_splits} Score: {score:.4f}"
            )
        return score
    except Exception as e:  # noqa: BLE001 - per-fold failures are collected for reporting, must not abort tuning
        if fold_errors is not None:
            fold_errors.append(str(e))
        if log_callback:
            n_splits = cv.get_n_splits(X_arr, y_arr)
            log_callback(
                f"  [Candidate {candidate_idx + 1}] CV Fold {fold_idx + 1}/{n_splits} Failed: {str(e)}"
            )
        return -float("inf")


def evaluate_search_candidates(
    candidates: list[dict[str, Any]],
    X_for_search: Any,
    y_for_search: Any,
    model_class: Any,
    cv: Any,
    metric: str,
    progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
    log_callback: Callable[[str], None] | None,
    preprocessing: "FoldPreprocessor | None" = None,
    fold_errors: list[str] | None = None,
    seed_params_overlay: dict[str, Any] | None = None,
    *,
    model_calculator: BaseModelCalculator,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    """Evaluates every candidate via CV, emitting progress/log callbacks, and tracks the best.

    Returns the collected trials, the best score, and the best params (or ``None`` if all failed).
    """
    total_candidates = len(candidates)
    trials: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_params = None

    for i, params in enumerate(candidates):
        if log_callback:
            log_callback(f"Evaluating Candidate {i + 1}/{total_candidates}: {params}")

        # Use custom cross-validation loop to enable per-fold logging and progress tracking.
        # We instantiate the model with the current candidate parameters and evaluate it
        # using the configured CV strategy.
        mean_score = evaluate_candidate_cv(
            i,
            params,
            model_class,
            cv,
            X_for_search,
            y_for_search,
            metric,
            log_callback,
            preprocessing,
            fold_errors,
            seed_params_overlay,
            model_calculator=model_calculator,
        )

        if log_callback:
            log_callback(f"Candidate {i + 1} Mean Score: {mean_score:.4f}")

        if progress_callback:
            progress_callback(i + 1, total_candidates, mean_score, params)

        trials.append({"params": params, "score": mean_score})

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return trials, best_score, best_params


def run_grid_or_random_search(
    X_for_search: Any,
    y_for_search: Any,
    config: TuningConfig,
    model_class: Any,
    cv: Any,
    metric: str,
    progress_callback: Callable[[int, int, float | None, dict | None], None] | None,
    log_callback: Callable[[str], None] | None,
    preprocessing: "FoldPreprocessor | None" = None,
    *,
    model_calculator: BaseModelCalculator,
) -> TuningResult:
    """Runs a custom grid/random search loop (instead of sklearn's searchers) so
    per-candidate and per-fold progress/log callbacks can be emitted during tuning.
    """
    if log_callback:
        log_callback(f"Starting {config.strategy} search with custom loop for detailed logging...")

    # 1. Generate Candidates
    candidates = generate_search_candidates(config)
    total_candidates = len(candidates)
    if log_callback:
        log_callback(f"Total candidates to evaluate: {total_candidates}")

    # 2. Iterate Candidates
    fold_errors: list[str] = []
    trials, best_score, best_params = evaluate_search_candidates(
        candidates,
        X_for_search,
        y_for_search,
        model_class,
        cv,
        metric,
        progress_callback,
        log_callback,
        preprocessing,
        fold_errors,
        seed_params(config),
        model_calculator=model_calculator,
    )

    if log_callback:
        log_callback(f"Tuning Completed. Best Score: {best_score:.4f}")
        log_callback(f"Best Params: {best_params}")

    if best_params is None:
        detail = ""
        if fold_errors:
            detail = f" First trial error: {fold_errors[0]}"
            if len(fold_errors) > 1:
                detail += f" ({len(fold_errors) - 1} more fold failures suppressed)"
        raise ValueError(
            "Hyperparameter tuning failed: All trials failed. "
            "This usually means the model failed to train with the provided hyperparameter combinations. "
            f"Please check your search space and data.{detail}"
        )

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        n_trials=total_candidates,
        trials=trials,
        scoring_metric=metric,
    )
