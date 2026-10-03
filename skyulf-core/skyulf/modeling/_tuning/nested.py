"""Evaluate independent inner searches on untouched outer folds."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .._cv_weights import preflight_weights, take_weights, weight_kwargs
from .cv_policy import effective_cv_type, policy_description, policy_splitter
from .grid_random import _slice_fold_rows, fit_and_score_candidate_fold
from .metrics import resolve_metric
from .nested_threshold import score_nested_threshold, select_nested_threshold
from .params import seed_params
from .schemas import TuningConfig, TuningResult
from .splitters import nested_inner_folds


def _nested_configs(config: TuningConfig, problem_type: str) -> tuple[TuningConfig, TuningConfig]:
    """Reject contradictory policies and build ordinary outer and inner splitters."""
    if type(config.cv_folds) is not int or config.cv_folds < 2:
        raise ValueError("Nested CV requires cv_folds to be an integer of at least 2.")
    if config.tune_threshold and problem_type != "classification":
        raise ValueError("Nested threshold tuning requires binary classification.")
    if problem_type not in ("classification", "regression"):
        raise ValueError("Nested tuning requires a classification or regression model.")
    method = effective_cv_type(config, problem_type)
    outer = replace(config, cv_type=method, tune_threshold=False)
    return outer, replace(outer, cv_folds=nested_inner_folds(config))


def _require_fold_labels(y: Any, config: TuningConfig, problem_type: str) -> None:
    """Require every class in every stratified test fold before starting searches."""
    if problem_type == "classification":
        counts = pd.Series(np.asarray(y)).value_counts()
        if len(counts) < 2 or counts.min() < config.cv_folds:
            raise ValueError(
                "Nested CV needs at least one row per class in every inner/outer fold."
            )
    if len(y) < config.cv_folds:
        raise ValueError("Nested CV has fewer training rows than folds.")


def _outer_score(
    tuner: Any,
    X: Any,
    y: Any,
    train: Any,
    test: Any,
    cv: Any,
    config: TuningConfig,
    result: TuningResult,
    fold: int,
    preprocessing: Any,
    sample_weight: Any = None,
) -> float:
    """Refit the selected recipe on outer training rows and reject incomplete evaluations."""
    errors: list[str] = []
    score = fit_and_score_candidate_fold(
        candidate_idx=0,
        fold_idx=fold,
        params=result.best_params,
        model_class=tuner.model_calculator.model_class,
        cv=cv,
        X_any=X,
        y_any=y,
        X_arr=np.asarray(X),
        y_arr=np.asarray(y),
        train_idx=train,
        val_idx=test,
        metric=result.scoring_metric or resolve_metric(config, y, tuner.problem_type),
        log_callback=None,
        preprocessing=deepcopy(preprocessing),
        fold_errors=errors,
        seed_params_overlay=seed_params(config),
        model_calculator=tuner.model_calculator,
        **weight_kwargs(sample_weight),
    )
    if not np.isfinite(score):
        detail = errors[0] if errors else "nonfinite outer score"
        raise ValueError(f"Nested CV outer fold {fold + 1} failed: {detail}")
    return float(score)


def _nested_partitions(
    X: Any,
    y: Any,
    config: TuningConfig,
    inner_config: TuningConfig,
    problem_type: str,
    metadata: dict[str, np.ndarray],
) -> tuple[Any, list[Any]]:
    """Validate every inner and outer split before the first candidate is fitted."""
    outer = policy_splitter(config, problem_type, y, metadata)
    inner = []
    for train, _test in outer.split(X, y):
        _require_fold_labels(_slice_fold_rows(y, train), inner_config, problem_type)
        inner_metadata = {key: values[train] for key, values in metadata.items()}
        inner.append(
            policy_splitter(inner_config, problem_type, _slice_fold_rows(y, train), inner_metadata)
        )
    return outer, inner


def _evaluate_selected(
    tuner: Any,
    X: Any,
    y: Any,
    train: Any,
    test: Any,
    outer: Any,
    inner: Any,
    config: TuningConfig,
    result: TuningResult,
    index: int,
    preprocessing: Any,
    sample_weight: Any = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate selected parameters, with a training-only threshold when requested."""
    if not config.tune_threshold:
        return _outer_score(
            tuner, X, y, train, test, outer, config, result, index, preprocessing, sample_weight
        ), {}
    train_x, train_y = _slice_fold_rows(X, train), _slice_fold_rows(y, train)
    selection = select_nested_threshold(
        tuner,
        train_x,
        train_y,
        config,
        result.best_params,
        inner,
        preprocessing,
        **weight_kwargs(take_weights(sample_weight, train)),
    )
    score = score_nested_threshold(
        tuner,
        train_x,
        train_y,
        _slice_fold_rows(X, test),
        _slice_fold_rows(y, test),
        config,
        result.best_params,
        selection,
        preprocessing,
        **weight_kwargs(take_weights(sample_weight, train)),
    )
    return score, {"threshold_selection": selection}


def _final_threshold(
    tuner: Any,
    X: Any,
    y: Any,
    config: TuningConfig,
    final: TuningResult,
    cv: Any,
    preprocessing: Any,
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Select the deployable cutoff independently of every outer-fold winner."""
    if not config.tune_threshold:
        return {}
    selection = select_nested_threshold(
        tuner, X, y, config, final.best_params, cv, preprocessing, **weight_kwargs(sample_weight)
    )
    final.decision_thresholds = selection["decision_thresholds"]
    final.decision_threshold_metric = selection["decision_threshold_metric"]
    return {"threshold_selection": selection}


def run_nested_search(
    tuner: Any,
    X: Any,
    y: Any,
    config: TuningConfig,
    *,
    preprocessing: Any = None,
    progress_callback: Callable[..., Any] | None = None,
    log_callback: Callable[[str], None] | None = None,
    split_metadata: dict[str, np.ndarray] | None = None,
    sample_weight: Any = None,
) -> TuningResult:
    """Search separately within outer folds, then run a distinct final training search.

    External validation data is intentionally absent from this interface. Search
    budgets apply independently to each inner search and the final search. Outer
    scores never participate in final hyperparameter selection.
    """
    outer_config, inner_config = _nested_configs(config, tuner.problem_type)
    _require_fold_labels(y, outer_config, tuner.problem_type)
    metadata = split_metadata or {}
    outer, inner_splitters = _nested_partitions(
        X, y, outer_config, inner_config, tuner.problem_type, metadata
    )
    partitions = list(outer.split(np.asarray(X), np.asarray(y)))

    _preflight_nested_weights(sample_weight, outer, inner_splitters, X, y)
    final_cv = policy_splitter(inner_config, tuner.problem_type, y, metadata)
    preflight_weights(sample_weight, final_cv, X, y)
    folds = []
    for index, (train, test) in enumerate(partitions):
        if log_callback:
            log_callback(
                f"Nested CV outer fold {index + 1}/{len(partitions)}: starting inner search."
            )
        train_x, train_y = _slice_fold_rows(X, train), _slice_fold_rows(y, train)
        result = tuner.tune(
            train_x,
            train_y,
            deepcopy(inner_config),
            log_callback=log_callback,
            preprocessing=deepcopy(preprocessing),
            preprocessing_frames=(train_x, train_y) if preprocessing is not None else None,
            split_metadata={key: values[train] for key, values in metadata.items()},
            cv_override=inner_splitters[index],
            **weight_kwargs(take_weights(sample_weight, train)),
        )
        score, threshold = _evaluate_selected(
            tuner,
            X,
            y,
            train,
            test,
            outer,
            inner_splitters[index],
            config,
            result,
            index,
            preprocessing,
            sample_weight,
        )
        folds.append(
            {
                "fold": index + 1,
                "train_rows": len(train),
                "test_rows": len(test),
                "best_params": result.best_params,
                "inner_best_score": result.best_score,
                "outer_score": score,
                "n_trials": result.n_trials,
                "split": outer.evidence[index],
                "inner_splits": inner_splitters[index].evidence,
                **threshold,
            }
        )
        if log_callback:
            log_callback(f"Nested CV outer fold {index + 1}: held-out score {score:.6g}.")

    if log_callback:
        log_callback("Nested CV complete; starting separate final search on all training rows.")
    final_cv = policy_splitter(inner_config, tuner.problem_type, y, metadata)
    final = tuner.tune(
        X,
        y,
        deepcopy(inner_config),
        progress_callback=progress_callback,
        log_callback=log_callback,
        preprocessing=preprocessing,
        preprocessing_frames=(X, y) if preprocessing is not None else None,
        split_metadata=metadata,
        cv_override=final_cv,
        **weight_kwargs(sample_weight),
    )
    final.nested_cv = _nested_report(
        folds, config, inner_config, final, final_cv, tuner.problem_type
    ) | _final_threshold(tuner, X, y, config, final, final_cv, preprocessing, sample_weight)
    return final


def _nested_report(
    folds: list[dict[str, Any]],
    config: TuningConfig,
    inner_config: TuningConfig,
    final: TuningResult,
    final_cv: Any,
    problem_type: str,
) -> dict[str, Any]:
    """Keep outer evaluation evidence separate from final-search selection results."""
    scores = [fold["outer_score"] for fold in folds]
    return {
        "status": "nested_cv",
        "method": "nested_cv",
        "outer_folds": config.cv_folds,
        "inner_folds": inner_config.cv_folds,
        "scoring_metric": final.scoring_metric,
        "score_direction": "higher_is_better; sklearn negative loss remains negative",
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "folds": folds,
        "total_trials": sum(f["n_trials"] for f in folds) + final.n_trials,
        "final_search_trials": final.n_trials,
        "split_policy": policy_description(config, problem_type),
        "final_splits": final_cv.evidence,
        "aggregated_metrics": {
            final.scoring_metric: {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
        },
        "cv_config": {
            "method": "nested_cv",
            "n_folds": config.cv_folds,
            "inner_folds": inner_config.cv_folds,
        },
    }


def _preflight_nested_weights(weights: Any, outer: Any, inner: list[Any], X: Any, y: Any) -> None:
    """Check outer and inner training weights before the first nested candidate fits."""
    if weights is None:
        return
    preflight_weights(weights, outer, X, y)
    for index, (train, _) in enumerate(outer.split(X, y)):
        preflight_weights(
            take_weights(weights, train),
            inner[index],
            _slice_fold_rows(X, train),
            _slice_fold_rows(y, train),
        )
