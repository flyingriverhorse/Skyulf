"""Read-only comparison of pinned registered fitted pipeline versions."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import pandas as pd
import polars as pl

from ....inference.fitted_pipeline import FittedPipelineArtifact
from ....inference.pipeline_evaluation import evaluate_holdout
from ..registration.registry import (
    ResolvedModel,
    load_registered_pipeline,
)

_REGRESSION = {
    "heldout_mae",
    "heldout_mse",
    "heldout_rmse",
    "heldout_r2",
    "heldout_mape",
    "heldout_explained_variance",
}
_CLASSIFICATION = {
    "heldout_accuracy",
    "heldout_balanced_accuracy",
    "heldout_precision_weighted",
    "heldout_recall_weighted",
    "heldout_f1_weighted",
    "heldout_matthews_corrcoef",
    "heldout_precision",
    "heldout_recall",
    "heldout_f1",
    "heldout_g_score",
    "heldout_log_loss",
    "heldout_roc_auc",
    "heldout_pr_auc",
    "heldout_roc_auc_ovr_weighted",
    "heldout_roc_auc_weighted",
    "heldout_roc_auc_ovr",
    "heldout_roc_auc_ovo",
    "heldout_roc_auc_ovo_weighted",
    "heldout_pr_auc_weighted",
}
_MINIMIZE = {
    "heldout_mae",
    "heldout_mse",
    "heldout_rmse",
    "heldout_mape",
    "heldout_log_loss",
}
_MAXIMIZE = (_REGRESSION | _CLASSIFICATION) - _MINIMIZE


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """Record one candidate/champion decision without modifying the registry."""

    dataset_id: str
    row_count: int
    code_version: str
    model_name: str
    candidate_version: str
    candidate_digest: str
    champion_version: str | None
    champion_digest: str | None
    metric: str
    metric_direction: str
    min_improvement: float
    quality_threshold: float | None
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    improvement: float | None
    eligible: bool
    reason: str
    quality_gates: dict[str, float] | None = None


def _validate_threshold(metric: str, value: float, field: str) -> None:
    """Reject nonnumeric bounds and values outside the metric's mathematical domain."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    if metric in _MINIMIZE and value < 0:
        raise ValueError(f"{field} must be nonnegative for error/loss metrics.")
    if metric in _CLASSIFICATION - {"heldout_log_loss"}:
        lower = -1 if metric == "heldout_matthews_corrcoef" else 0
        if not lower <= value <= 1:
            raise ValueError(f"{field} for {metric} must be between {lower} and 1.")
    if metric in {"heldout_r2", "heldout_explained_variance"} and value > 1:
        raise ValueError(f"{field} for {metric} must be at most 1.")


def validate_quality_policy(
    metric: str,
    quality_threshold: float | None,
    quality_gates: dict[str, float] | None = None,
    *,
    task: str | None = None,
) -> None:
    """Validate one selection metric and optional additional absolute quality bounds."""
    supported = _REGRESSION | _CLASSIFICATION
    if task is not None:
        if task not in {"regression", "classification"}:
            raise ValueError("Quality gates require regression or classification.")
        supported = _REGRESSION if task == "regression" else _CLASSIFICATION
    if not isinstance(metric, str) or metric not in supported:
        raise ValueError("metric must be a supported heldout metric compatible with the task.")
    if quality_threshold is not None:
        _validate_threshold(metric, quality_threshold, "quality_threshold")
    _validate_additional_gates(quality_gates, metric, supported)


def evaluate_quality_gates(
    metrics: dict[str, float],
    metric: str,
    quality_threshold: float | None,
    quality_gates: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Explain every absolute gate, including unavailable probability/class metrics."""
    thresholds = {} if quality_threshold is None else {metric: quality_threshold}
    thresholds.update(sorted((quality_gates or {}).items()))
    results = []
    for name, threshold in thresholds.items():
        value, direction, available, passed = _quality_gate_outcome(metrics, name, threshold)
        results.append(
            {
                "metric": name,
                "direction": direction,
                "threshold": threshold,
                "value": value if available else None,
                "passed": bool(passed),
                "reason": "metric_unavailable_or_non_finite"
                if not available
                else "passed"
                if passed
                else "threshold_not_met",
            }
        )
    return results


def quality_gate_results(report: ModelComparisonReport) -> list[dict[str, Any]]:
    """Derive gate evidence from the saved policy and observed candidate metrics."""
    return evaluate_quality_gates(
        report.candidate_metrics, report.metric, report.quality_threshold, report.quality_gates
    )


def quality_gates_pass(report: ModelComparisonReport) -> bool:
    """Require every configured absolute gate to pass, including the first champion's."""
    return all(gate["passed"] for gate in quality_gate_results(report))


def comparison_payload(report: ModelComparisonReport) -> dict[str, Any]:
    """Preserve historical single-gate evidence bytes when no guardrails are configured."""
    payload = asdict(report)
    if not report.quality_gates:
        payload.pop("quality_gates")
    return payload


def comparison_digest(report: ModelComparisonReport) -> str:
    """Hash the same canonical comparison for training, approval and registry events."""
    return hashlib.sha256(
        json.dumps(comparison_payload(report), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _validate_model_reference(reference: ResolvedModel) -> None:
    """Require a concrete registered version and digest before loading its artifact."""
    if not isinstance(reference, ResolvedModel):
        raise TypeError("candidate and champion must be resolved model references.")
    if (
        type(reference.name) is not str
        or not reference.name
        or type(reference.version) is not str
        or not reference.version.isascii()
        or not reference.version.isdigit()
        or int(reference.version) <= 0
        or reference.model_uri != f"models:/{reference.name}/{reference.version}"
    ):
        raise ValueError("Each model must have a concrete version and artifact digest.")
    _validate_reference_digest(reference)


def _validate_model_pair(candidate: ResolvedModel, champion: ResolvedModel | None) -> None:
    """Keep both concrete versions within one registered model's identity."""
    _validate_model_reference(candidate)
    if champion is None:
        return
    _validate_model_reference(champion)
    if candidate.name != champion.name or candidate.version == champion.version:
        raise ValueError("Candidate and champion must be distinct versions of one model.")


def _validate_holdout(
    heldout: pd.DataFrame | pl.DataFrame, target_column: str, dataset_id: str
) -> None:
    """Require a named labeled evaluation set with enough rows to compute metrics."""
    if not isinstance(heldout, pd.DataFrame | pl.DataFrame):
        raise TypeError("heldout must be a pandas or Polars DataFrame.")
    if type(dataset_id) is not str or not dataset_id.strip():
        raise ValueError("dataset_id must identify the pinned evaluation set.")
    if type(target_column) is not str or target_column not in heldout.columns:
        raise ValueError("target_column must exist in the evaluation set.")
    if len(heldout) < 2:
        raise ValueError("heldout must contain at least two labeled rows.")


def _validate_holdout_budget(
    heldout: pd.DataFrame | pl.DataFrame, max_rows: int, max_bytes: int
) -> None:
    """Bound the existing frame before registry access or prediction."""
    if type(max_rows) is not int or max_rows < 2 or len(heldout) > max_rows:
        raise ValueError("heldout exceeds max_rows or the row limit is invalid.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")
    frame_bytes = (
        int(heldout.memory_usage(index=True, deep=True).sum())
        if isinstance(heldout, pd.DataFrame)
        else int(heldout.estimated_size())
    )
    if frame_bytes > max_bytes:
        raise ValueError("heldout exceeds max_bytes.")


def _validate_request(
    candidate: ResolvedModel,
    champion: ResolvedModel | None,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    dataset_id: str,
    metric: str,
    min_improvement: float,
    quality_threshold: float | None,
    max_rows: int,
    max_bytes: int,
    quality_gates: dict[str, float] | None = None,
) -> None:
    """Reject ambiguous identities, policies and oversized data before loading."""
    _validate_model_pair(candidate, champion)
    _validate_holdout(heldout, target_column, dataset_id)
    validate_quality_policy(metric, quality_threshold, quality_gates)
    if (
        type(min_improvement) not in (int, float)
        or not math.isfinite(min_improvement)
        or min_improvement < 0
    ):
        raise ValueError("min_improvement must be a finite nonnegative number.")
    _validate_holdout_budget(heldout, max_rows, max_bytes)


def _checked_artifact(
    reference: ResolvedModel, *, tracking_uri: str | None, registry_uri: str | None
) -> FittedPipelineArtifact:
    """Load one concrete artifact and confirm its registry digest."""
    artifact = load_registered_pipeline(
        reference, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if artifact.manifest.pipeline_sha256 != reference.digest:
        raise ValueError("Registered model digest differs from the loaded pipeline.")
    return artifact


def _validate_artifact_pair(
    candidate: FittedPipelineArtifact, champion: FittedPipelineArtifact | None
) -> None:
    """Require comparable tasks, class labels and target normalization recipes."""
    if champion is None:
        return
    if candidate.manifest.task != champion.manifest.task or (
        candidate.manifest.task == "classification"
        and candidate.manifest.classes != champion.manifest.classes
    ):
        raise ValueError("Candidate and champion task or class contracts differ.")
    if candidate.pipeline.config.get("pre_split_target_contract", []) != (
        champion.pipeline.config.get("pre_split_target_contract", [])
    ):
        raise ValueError("Candidate and champion target normalization contracts differ.")


def _comparison_metrics(
    candidate: FittedPipelineArtifact,
    champion: FittedPipelineArtifact | None,
    heldout: pd.DataFrame | pl.DataFrame,
    target_column: str,
) -> tuple[dict[str, float], dict[str, float] | None]:
    """Evaluate the same labeled rows with both saved pipelines, without fitting."""
    candidate_metrics = evaluate_holdout(candidate, heldout, target_column=target_column)
    if champion is None:
        return candidate_metrics, None
    return candidate_metrics, evaluate_holdout(champion, heldout, target_column=target_column)


def _selected_metric(metrics: dict[str, float], metric: str) -> float:
    """Reject an unavailable selection metric even when no champion exists yet."""
    value = metrics.get(metric)
    if value is None or not math.isfinite(value):
        raise ValueError("Selected metric is unavailable or non-finite on the holdout.")
    return value


def _metric_improvement(
    candidate_metrics: dict[str, float], champion_metrics: dict[str, float] | None, metric: str
) -> float | None:
    """Measure an absolute signed improvement in the selection metric's units."""
    candidate_value = _selected_metric(candidate_metrics, metric)
    if champion_metrics is None:
        return None
    champion_value = _selected_metric(champion_metrics, metric)
    if metric in _MINIMIZE:
        return champion_value - candidate_value
    return candidate_value - champion_value


def _comparison_decision(
    improvement: float | None, quality_passed: bool, min_improvement: float
) -> tuple[bool, str]:
    """Explain promotion eligibility while preserving explicit first-champion approval."""
    if improvement is None:
        return False, "no_champion"
    if not quality_passed:
        return False, "quality_gate_failed"
    if improvement > 0 and improvement >= min_improvement:
        return True, "candidate_improved"
    return False, "insufficient_improvement"


def compare_registered_pipeline_models(
    candidate: ResolvedModel,
    champion: ResolvedModel | None,
    heldout: pd.DataFrame | pl.DataFrame,
    *,
    target_column: str,
    dataset_id: str,
    metric: str,
    min_improvement: float,
    max_rows: int,
    max_bytes: int,
    quality_threshold: float | None = None,
    quality_gates: dict[str, float] | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> ModelComparisonReport:
    """Score two pinned versions on identical labeled rows without moving aliases.

    The caller must pin and describe the source snapshot/split in dataset_id.
    The report does not prove that identity or approve the model for production.
    A tie cannot pass even when min_improvement is zero. Without a champion,
    the first candidate is measured but never promoted implicitly.
    """
    _validate_request(
        candidate,
        champion,
        heldout,
        target_column=target_column,
        dataset_id=dataset_id,
        metric=metric,
        min_improvement=min_improvement,
        quality_threshold=quality_threshold,
        quality_gates=quality_gates,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )
    candidate_artifact = _checked_artifact(
        candidate, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    champion_artifact = (
        _checked_artifact(champion, tracking_uri=tracking_uri, registry_uri=registry_uri)
        if champion is not None
        else None
    )
    task = candidate_artifact.manifest.task
    validate_quality_policy(metric, quality_threshold, quality_gates, task=task)
    _validate_artifact_pair(candidate_artifact, champion_artifact)
    candidate_metrics, champion_metrics = _comparison_metrics(
        candidate_artifact, champion_artifact, heldout, target_column
    )
    improvement = _metric_improvement(candidate_metrics, champion_metrics, metric)
    quality_passed = all(
        gate["passed"]
        for gate in evaluate_quality_gates(
            candidate_metrics, metric, quality_threshold, quality_gates
        )
    )
    eligible, reason = _comparison_decision(improvement, quality_passed, min_improvement)
    return ModelComparisonReport(
        dataset_id=dataset_id,
        row_count=len(heldout),
        code_version=version("skyulf-core"),
        model_name=candidate.name,
        candidate_version=candidate.version,
        candidate_digest=candidate.digest or "",
        champion_version=champion.version if champion is not None else None,
        champion_digest=champion.digest if champion is not None else None,
        metric=metric,
        metric_direction="minimize" if metric in _MINIMIZE else "maximize",
        min_improvement=float(min_improvement),
        quality_threshold=float(quality_threshold) if quality_threshold is not None else None,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        improvement=improvement,
        eligible=eligible,
        reason=reason,
        quality_gates=dict(quality_gates) if quality_gates else None,
    )


def _validate_additional_gates(
    quality_gates: dict[str, float] | None, metric: str, supported: set[str]
) -> None:
    """Validate additional task-compatible thresholds in their configured order."""
    if quality_gates is None:
        return
    if not isinstance(quality_gates, dict):
        raise ValueError("quality_gates must map additional heldout metrics to thresholds.")
    for name, threshold in quality_gates.items():
        if not isinstance(name, str) or name not in supported or name == metric:
            raise ValueError("quality_gates needs distinct, task-compatible additional metrics.")
        _validate_threshold(name, threshold, f"quality_gates[{name}]")


def _quality_gate_outcome(
    metrics: dict[str, float], name: str, threshold: float
) -> tuple[float | None, str, bool, bool]:
    """Compute availability and threshold outcome for one quality metric."""
    value = metrics.get(name)
    direction = "minimize" if name in _MINIMIZE else "maximize"
    available = value is not None and math.isfinite(value)
    passed = available and (value <= threshold if direction == "minimize" else value >= threshold)
    return value, direction, available, passed


def _validate_reference_digest(reference: ResolvedModel) -> None:
    """Require artifact identity after the concrete registry identity is validated."""
    if not isinstance(reference.digest, str) or not reference.digest:
        raise ValueError("Each model must have a concrete version and artifact digest.")
