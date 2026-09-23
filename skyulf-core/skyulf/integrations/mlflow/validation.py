"""Read-only comparison of pinned registered local pipeline versions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version

import pandas as pd
import polars as pl

from ...inference.local_evaluation import evaluate_local_holdout
from ...inference.local_pipeline import LocalPipelineArtifact
from .registry import ResolvedModel, load_registered_local_pipeline

_MINIMIZE = {"heldout_mae", "heldout_rmse"}
_MAXIMIZE = {"heldout_r2", "heldout_accuracy", "heldout_f1_weighted", "heldout_f1"}
_REGRESSION = _MINIMIZE | {"heldout_r2"}
_CLASSIFICATION = _MAXIMIZE - {"heldout_r2"}


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
) -> None:
    """Reject ambiguous identities, policies and oversized data before loading."""
    if not isinstance(candidate, ResolvedModel) or (
        champion is not None and not isinstance(champion, ResolvedModel)
    ):
        raise TypeError("candidate and champion must be resolved model references.")
    for reference in (candidate, champion):
        if reference is None:
            continue
        if (
            type(reference.name) is not str
            or not reference.name
            or type(reference.version) is not str
            or not reference.version.isascii()
            or not reference.version.isdigit()
            or int(reference.version) <= 0
            or reference.model_uri != f"models:/{reference.name}/{reference.version}"
            or not isinstance(reference.digest, str)
            or not reference.digest
        ):
            raise ValueError("Each model must have a concrete version and artifact digest.")
    if champion is not None and (
        candidate.name != champion.name or candidate.version == champion.version
    ):
        raise ValueError("Candidate and champion must be distinct versions of one model.")
    if not isinstance(heldout, pd.DataFrame | pl.DataFrame):
        raise TypeError("heldout must be a pandas or Polars DataFrame.")
    if type(dataset_id) is not str or not dataset_id.strip():
        raise ValueError("dataset_id must identify the pinned evaluation set.")
    if type(target_column) is not str or target_column not in heldout.columns:
        raise ValueError("target_column must exist in the evaluation set.")
    if type(metric) is not str or metric not in _MINIMIZE | _MAXIMIZE:
        raise ValueError("metric must be a supported heldout metric.")
    if (
        type(min_improvement) not in (int, float)
        or not math.isfinite(min_improvement)
        or min_improvement < 0
    ):
        raise ValueError("min_improvement must be a finite nonnegative number.")
    if quality_threshold is not None and (
        type(quality_threshold) not in (int, float) or not math.isfinite(quality_threshold)
    ):
        raise ValueError("quality_threshold must be a finite number or None.")
    if len(heldout) < 2:
        raise ValueError("heldout must contain at least two labeled rows.")
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


def _checked_artifact(
    reference: ResolvedModel, *, tracking_uri: str | None, registry_uri: str | None
) -> LocalPipelineArtifact:
    """Load one concrete artifact and confirm its registry digest."""
    artifact = load_registered_local_pipeline(
        reference, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    if artifact.manifest.pipeline_sha256 != reference.digest:
        raise ValueError("Registered model digest differs from the loaded pipeline.")
    return artifact


def compare_registered_local_models(
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
    if metric not in (_REGRESSION if task == "regression" else _CLASSIFICATION):
        raise ValueError("metric is incompatible with the candidate task.")
    if champion_artifact is not None and (
        task != champion_artifact.manifest.task
        or (
            task == "classification"
            and candidate_artifact.manifest.classes != champion_artifact.manifest.classes
        )
    ):
        raise ValueError("Candidate and champion task or class contracts differ.")
    candidate_metrics = evaluate_local_holdout(
        candidate_artifact, heldout, target_column=target_column
    )
    champion_metrics = (
        evaluate_local_holdout(champion_artifact, heldout, target_column=target_column)
        if champion_artifact is not None
        else None
    )
    candidate_value = candidate_metrics.get(metric)
    champion_value = champion_metrics.get(metric) if champion_metrics is not None else None
    if (
        candidate_value is None
        or not math.isfinite(candidate_value)
        or (
            champion_metrics is not None
            and (champion_value is None or not math.isfinite(champion_value))
        )
    ):
        raise ValueError("Selected metric is unavailable or non-finite on the holdout.")
    direction = "minimize" if metric in _MINIMIZE else "maximize"
    improvement = (
        None
        if champion_value is None
        else (
            champion_value - candidate_value
            if direction == "minimize"
            else candidate_value - champion_value
        )
    )
    quality_passed = quality_threshold is None or (
        candidate_value <= quality_threshold
        if direction == "minimize"
        else candidate_value >= quality_threshold
    )
    eligible = (
        improvement is not None
        and improvement > 0
        and improvement >= min_improvement
        and quality_passed
    )
    reason = (
        "no_champion"
        if champion is None
        else "quality_gate_failed"
        if not quality_passed
        else "candidate_improved"
        if eligible
        else "insufficient_improvement"
    )
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
        metric_direction=direction,
        min_improvement=float(min_improvement),
        quality_threshold=float(quality_threshold) if quality_threshold is not None else None,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        improvement=improvement,
        eligible=eligible,
        reason=reason,
    )
