"""Comparable candidate evidence on shared, bounded training-only folds."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from skyulf.integrations.databricks._local_frames import frame_bytes

from ...inference.local_pipeline import LocalPipelineArtifact
from ...modeling._sample_weights import validate_sample_weight
from ...modeling._tuning.cv_policy import (
    FrozenSplit,
    fold_evidence,
    policy_description,
    policy_splitter,
    prepare_policy_data,
    validate_class_membership,
)
from ...modeling._tuning.engine import TuningCalculator
from ...modeling._tuning.grid_random import fit_and_score_candidate_fold
from ...modeling._tuning.metrics import resolve_metric
from ...modeling._tuning.schemas import TuningConfig
from ...modeling._tuning.splitters import build_shuffle_split_cv, nested_inner_folds
from ...preprocessing.fold_adapter import (
    FeatureEngineerFoldAdapter,
    merged_branch_step_unsafe_reason,
)
from ...registry import NodeRegistry
from .local_cv import CV_FIELDS, LocalCVSpec
from .local_search import base_model_config, validate_metric
from .local_search_results import tuning_evidence

_MINIMIZE = {"mae", "mse", "rmse", "log_loss"}
_BINARY_METRICS = {"f1", "precision", "recall", "roc_auc", "pr_auc"}


def competition_metric(metric: str, task: str) -> str:
    """Map an admitted heldout objective to the identical native Core search metric.

    Unsupported metrics fail before source access or fitting. In particular,
    Core's current search admission lacks MAPE and weighted precision/recall.
    """
    if not isinstance(metric, str) or not metric.startswith("heldout_"):
        raise ValueError("competition metric must name a supported heldout_ metric.")
    native = metric.removeprefix("heldout_")
    if native == "roc_auc_weighted":
        native = "roc_auc_ovr_weighted"
    try:
        validate_metric(native, task)
    except ValueError as exc:
        raise ValueError(f"Unsupported competition metric {metric} for {task}.") from exc
    if task not in {"regression", "classification"}:
        raise ValueError("competition metric requires regression or classification.")
    return native


def validate_competition_preprocessing(pipeline: dict[str, Any]) -> None:
    """Require candidate transforms to preserve shared validation membership.

    Row-changing nodes, including resampling, are not admitted in candidate
    recipes because stored nested reports do not prove validation survivors.
    Shared row filtering belongs in pre_split, before candidate folds are made.
    """
    for step in pipeline.get("preprocessing", []):
        reason = merged_branch_step_unsafe_reason(step)
        if reason is not None:
            raise ValueError(
                f"Competition preprocessing {step.get('transformer')} {reason}; "
                "candidate preprocessing must preserve row membership. Move row filtering "
                "to shared pre_split; candidate resampling and sorting are not supported."
            )


def _policy(cv: LocalCVSpec, metric: str, event_column: str | None) -> TuningConfig:
    """Translate the shared workflow controls without introducing splitter defaults."""
    fields = {name: getattr(cv, field) for name, field in CV_FIELDS.items()}
    return TuningConfig(
        **fields, metric=metric, cv_time_column=event_column if cv.temporal else None
    )


def _validate_input(
    frame: Any,
    artifact: LocalPipelineArtifact,
    cv: LocalCVSpec,
    target_column: str,
    max_rows: int,
    max_bytes: int,
) -> None:
    """Reject disabled CV, missing targets, and unbounded data before any fold fit."""
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("Competition training data must be a pandas or Polars DataFrame.")
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("Competition requires a fitted LocalPipelineArtifact.")
    if not cv.enabled:
        raise ValueError("Competition requires enabled CV.")
    if target_column not in frame.columns or target_column in artifact.manifest.input_columns:
        raise ValueError("Competition requires a separate target column.")
    _validate_bounds(frame, max_rows, max_bytes)


def _validate_bounds(frame: Any, max_rows: int, max_bytes: int) -> None:
    """Check explicit training memory limits before inspecting fold membership."""
    for name, limit in (("max_rows", max_rows), ("max_bytes", max_bytes)):
        if type(limit) is not int or limit <= 0:
            raise ValueError(f"Competition {name} must be a positive integer.")
    if len(frame) > max_rows or frame_bytes(frame) > max_bytes:
        raise ValueError("Competition training data exceeds max_rows or max_bytes.")


def _split_plan(policy: TuningConfig, task: str, y: Any, metadata: dict) -> FrozenSplit:
    """Use Core policies, including the actual repeated 20-percent shuffle split."""
    if policy.cv_type != "shuffle_split":
        return policy_splitter(policy, task, y, metadata)
    labels = np.asarray(y)
    parts = list(build_shuffle_split_cv(policy).split(np.arange(len(labels)), labels))
    for train, test in parts:
        if min(len(train), len(test)) < 2:
            raise ValueError("Each competition fold requires at least two rows per partition.")
        validate_class_membership(labels, train, test, task)
    return FrozenSplit(parts, [fold_evidence(train, test, metadata) for train, test in parts])


def _membership_digest(frame: Any, policy: TuningConfig, plan: FrozenSplit) -> str:
    """Bind positional membership to the original training order, including time sorting."""
    order = np.arange(len(frame))
    if policy_description(policy, "regression")["method"] == "time_series_split":
        times = pd.to_datetime(np.asarray(frame[policy.cv_time_column]), utc=True)
        order = np.argsort(times, kind="stable")
    membership = [
        {"train": order[train].tolist(), "test": order[test].tolist()}
        for train, test in plan.partitions
    ]
    payload = {"row_count": len(frame), "folds": membership}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _candidate_recipe(artifact: LocalPipelineArtifact) -> tuple[Any, dict, dict | None]:
    """Resolve structural ensemble parameters and selected search parameters without state reuse."""
    pipeline: dict[str, Any] = dict(artifact.pipeline.config)
    selected = deepcopy(base_model_config(pipeline))
    evidence = tuning_evidence(artifact)
    calculator = NodeRegistry.get_calculator(selected["type"])()
    calculator.prepare_tuning_params(selected)
    params = deepcopy(selected.get("params", {}))
    for name in calculator.STRUCTURAL_TUNING_KEYS:
        params.pop(name, None)
    params.pop("tune_base_models", None)
    if evidence is not None:
        params.update(evidence["best_params"])
        params.setdefault("random_state", pipeline["modeling"].get("random_state", 42))
    return calculator, params, evidence


def _ordinary_scores(
    X: Any,
    y: Any,
    calculator: Any,
    params: dict,
    plan: FrozenSplit,
    scorer: str,
    adapter: FeatureEngineerFoldAdapter,
    sample_weight: Any = None,
) -> list[float]:
    """Refit every fold independently and fail the candidate if even one score is missing."""
    scores = []
    for index, (train, test) in enumerate(plan.partitions):
        errors: list[str] = []
        score = fit_and_score_candidate_fold(
            candidate_idx=0,
            fold_idx=index,
            params=deepcopy(params),
            model_class=calculator.model_class,
            cv=plan,
            X_any=X,
            y_any=y,
            X_arr=np.asarray(X),
            y_arr=np.asarray(y),
            train_idx=train,
            val_idx=test,
            metric=scorer,
            log_callback=None,
            preprocessing=deepcopy(adapter),
            fold_errors=errors,
            model_calculator=calculator,
            sample_weight=sample_weight,
        )
        if not math.isfinite(score):
            detail = errors[0] if errors else "nonfinite score"
            raise ValueError(f"Competition fold {index + 1} failed: {detail}")
        scores.append(float(score))
    return scores


def _nested_scores(
    report: Any, policy: TuningConfig, task: str, scorer: str, plan: FrozenSplit
) -> list[float]:
    """Accept complete genuine outer reports only, checking score units and split policy."""
    if not isinstance(report, dict) or report.get("status") != "nested_cv":
        raise ValueError("Competition requires genuine nested outer-fold evidence.")
    if report.get("scoring_metric") != scorer:
        raise ValueError("Nested competition scoring metric does not match the shared objective.")
    _validate_nested_policy(report, policy, task)
    folds = report.get("folds")
    if not isinstance(folds, list) or len(folds) != len(plan.partitions):
        raise ValueError("Nested competition requires every outer fold.")
    scores = []
    for index, fold in enumerate(folds):
        if (
            not isinstance(fold, dict)
            or fold.get("fold") != index + 1
            or fold.get("split") != plan.evidence[index]
        ):
            raise ValueError("Nested competition fold membership does not match shared CV.")
        scores.append(_finite_score(fold.get("outer_score")))
    return scores


def _validate_nested_policy(report: dict, policy: TuningConfig, task: str) -> None:
    """Require both levels of the stored nested policy to match the shared request."""
    if report.get("split_policy") != policy_description(policy, task):
        raise ValueError("Nested competition split policy does not match shared CV.")
    if report.get("outer_folds") != policy.cv_folds or report.get(
        "inner_folds"
    ) != nested_inner_folds(policy):
        raise ValueError("Nested competition inner/outer fold count does not match shared CV.")


def _finite_score(value: Any) -> float:
    """Reject missing, boolean and nonfinite scores instead of averaging surviving folds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Competition requires a finite score for every fold.")
    return float(value)


def _score_summary(values: list[float]) -> dict[str, float]:
    """Reject numerical overflow in aggregates as well as in individual fold scores."""
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = float(np.mean(values)), float(np.std(values))
    return {"mean": _finite_score(mean), "std": _finite_score(std)}


def _nested_report(
    X: Any,
    y: Any,
    calculator: Any,
    params: dict,
    policy: TuningConfig,
    adapter: FeatureEngineerFoldAdapter,
    evidence: dict | None,
    sample_weight: Any = None,
) -> dict:
    """Reuse nested search evidence or evaluate a fixed recipe as a singleton search."""
    if evidence is not None:
        report = evidence.get("nested_cv")
        if report is None:
            raise ValueError("Competition requires genuine nested outer-fold evidence.")
        return report
    singleton = replace(policy, strategy="grid", search_space={k: [v] for k, v in params.items()})
    result = TuningCalculator(calculator).tune(
        X,
        y,
        singleton,
        preprocessing=adapter,
        preprocessing_frames=(X, y),
        sample_weight=sample_weight,
    )
    if result.nested_cv is None:
        raise ValueError("Fixed competition candidate produced no nested outer report.")
    return result.nested_cv


def _validate_objective(policy: TuningConfig, task: str, y: Any, modeling: dict) -> str:
    """Keep binary metric names and configured search objectives semantically exact."""
    if task == "classification" and policy.metric in _BINARY_METRICS and len(np.unique(y)) != 2:
        raise ValueError(
            "Binary competition metric requires two classes; choose a weighted metric."
        )
    scorer = resolve_metric(policy, np.asarray(y), task)
    if modeling.get("type") == "hyperparameter_tuner":
        configured = replace(policy, metric=modeling.get("metric", ""))
        if resolve_metric(configured, np.asarray(y), task) != scorer:
            raise ValueError("Candidate search metric does not match the competition metric.")
    if modeling.get("tune_threshold") and policy.cv_type != "nested_cv":
        raise ValueError("Competition threshold tuning requires nested_cv.")
    return scorer


def evaluate_competition_candidate(
    frame: pd.DataFrame | pl.DataFrame,
    artifact: LocalPipelineArtifact,
    cv: LocalCVSpec,
    *,
    target_column: str,
    metric: str,
    max_rows: int,
    max_bytes: int,
    event_column: str | None = None,
    cv_results: dict[str, Any] | None = None,
    sample_weight: Any = None,
) -> dict[str, Any]:
    """Evaluate one fitted recipe on bounded shared training rows, never a final holdout.

    Ordinary searches are post-selection diagnostics: their selected parameters
    were chosen using these training rows, so scores are not unbiased estimates.
    Nested searches use actual stored outer scores, never final-search best_score.
    Fixed nested candidates run a singleton search with the requested objective.
    The legacy cv_results argument is accepted for callers, but cannot override
    artifact-backed evidence or supply differently defined fixed-model metrics.
    """
    _validate_input(frame, artifact, cv, target_column, max_rows, max_bytes)
    task = artifact.manifest.task
    native_metric = competition_metric(metric, task)
    policy = _policy(cv, native_metric, event_column)
    pipeline: dict[str, Any] = dict(artifact.pipeline.config)
    validate_competition_preprocessing(pipeline)
    cv.validate_pipeline(pipeline, target_column=target_column, event_column=event_column)
    y = frame[target_column]
    scorer = _validate_objective(policy, task, y, pipeline["modeling"])
    X = (
        frame.drop(target_column)
        if isinstance(frame, pl.DataFrame)
        else frame.drop(columns=[target_column])
    )
    adapter = FeatureEngineerFoldAdapter(pipeline.get("preprocessing", []), target_column)
    sample_weight = validate_sample_weight(sample_weight, len(frame))
    prepared_X, prepared_y, metadata, positions = prepare_policy_data(
        X, y, policy, task, adapter, return_positions=True
    )
    ordered_weight = None if sample_weight is None else sample_weight[positions]
    plan = _split_plan(policy, task, prepared_y, metadata)
    calculator, params, evidence = _candidate_recipe(artifact)
    if cv.method == "nested_cv":
        report = _nested_report(X, y, calculator, params, policy, adapter, evidence, sample_weight)
        scores = _nested_scores(report, policy, task, scorer, plan)
        mode = "nested_cv"
    else:
        scores = _ordinary_scores(
            prepared_X, prepared_y, calculator, params, plan, scorer, adapter, ordered_weight
        )
        mode = "post_selection_cv" if evidence is not None else "fixed_cv"
    minimize = native_metric in _MINIMIZE
    values = [-score if minimize else score for score in scores]
    return {
        "metric": metric,
        "scoring_metric": scorer,
        "direction": "minimize" if minimize else "maximize",
        **_score_summary(values),
        "fold_scores": values,
        "fold_membership_sha256": _membership_digest(frame, policy, plan),
        "evaluation_mode": mode,
        "unbiased_estimate": mode != "post_selection_cv",
        "split_policy": policy_description(policy, task),
        "folds": plan.evidence,
    }
