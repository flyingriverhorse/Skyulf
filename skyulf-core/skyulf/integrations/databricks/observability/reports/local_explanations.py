"""Optional, bounded SHAP explanations for fitted local training artifacts."""

import json
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from .....inference.local_pipeline import LocalPipelineArtifact
from .....modeling._explainability.shap_explanation import compute_shap_explanation
from .explanation_report import render_explanation_report

_DEFAULTS = {"max_samples": 100, "max_features": 30, "max_display_samples": 10}
_LIMITS = {"max_samples": (1, 200), "max_features": (1, 50), "max_display_samples": (0, 50)}


def validate_explanation_config(pipeline: dict[str, Any]) -> None:
    """Reject unsupported methods and budgets before any explanation work."""
    if type(pipeline) is not dict:
        raise ValueError("explainability requires a pipeline dictionary.")
    settings = pipeline.get("explainability")
    if settings is None:
        return
    if type(settings) is not dict or set(settings) - {"method", *_LIMITS}:
        raise ValueError("explainability must contain only supported settings.")
    if settings.get("method") != "shap":
        raise ValueError("explainability.method must be 'shap'.")
    for name, (lower, upper) in _LIMITS.items():
        value = settings.get(name, _DEFAULTS[name])
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"explainability.{name} must be an integer from {lower} to {upper}.")
    sample_limit = settings.get("max_samples", _DEFAULTS["max_samples"])
    display_limit = settings.get("max_display_samples", min(10, sample_limit))
    if display_limit > sample_limit:
        raise ValueError("explainability.max_display_samples exceeds max_samples.")


def _settings(pipeline: dict[str, Any]) -> dict[str, int] | None:
    """Return validated effective limits for an opted-in pipeline."""
    validate_explanation_config(pipeline)
    if pipeline.get("explainability") is None:
        return None
    supplied = pipeline["explainability"]
    limits = {name: supplied.get(name, default) for name, default in _DEFAULTS.items()}
    limits["max_display_samples"] = supplied.get(
        "max_display_samples", min(10, limits["max_samples"])
    )
    return limits


def _input_unavailable_reason(
    artifact: LocalPipelineArtifact,
    training_frame: pd.DataFrame | pl.DataFrame,
    limits: dict[str, int],
) -> str | None:
    """Check source shape and feature budget before sampling or preprocessing."""
    manifest = artifact.manifest
    if len(manifest.feature_columns) > limits["max_features"]:
        return "feature_limit_exceeded"
    if not len(training_frame):
        return "empty_training_frame"
    if set(manifest.input_columns) - set(training_frame.columns):
        return "missing_input_columns"
    return None


def _sample_training_inputs(
    artifact: LocalPipelineArtifact,
    training_frame: pd.DataFrame | pl.DataFrame,
    max_samples: int,
) -> pd.DataFrame | pl.DataFrame:
    """Select reproducible input-only rows and match the fitted preprocessing engine."""
    manifest = artifact.manifest
    count = min(len(training_frame), max_samples)
    positions = np.sort(np.random.default_rng(42).choice(len(training_frame), count, replace=False))
    projected = (
        training_frame.select(manifest.input_columns)
        if isinstance(training_frame, pl.DataFrame)
        else training_frame.loc[:, list(manifest.input_columns)]
    )
    sampled = (
        projected[positions.tolist()]
        if isinstance(projected, pl.DataFrame)
        else projected.iloc[positions]
    )
    if manifest.fitted_engine == "pandas" and isinstance(sampled, pl.DataFrame):
        return sampled.to_pandas()
    if manifest.fitted_engine == "polars" and isinstance(sampled, pd.DataFrame):
        return pl.from_pandas(sampled)
    return sampled


def _transformed_unavailable_reason(
    artifact: LocalPipelineArtifact,
    transformed: pd.DataFrame | pl.DataFrame,
    count: int,
    max_features: int,
) -> str | None:
    """Require row-preserving inference features in their saved model order."""
    if len(transformed) != count:
        return "row_count_changed"
    if tuple(transformed.columns) != artifact.manifest.feature_columns:
        return "feature_schema_mismatch"
    if len(transformed.columns) > max_features:
        return "feature_limit_exceeded"
    return None


def _explanation_result(explanation: Any, evidence: dict[str, Any]) -> dict[str, Any]:
    """Accept only finite JSON explanations matching the saved feature schema."""
    if explanation is None:
        return {**evidence, "reason": "shap_unavailable"}
    try:
        if explanation.get("feature_names") != evidence["feature_names"]:
            return {**evidence, "reason": "shap_feature_schema_mismatch"}
        json.dumps(explanation, allow_nan=False)
    except (TypeError, ValueError, AttributeError):
        return {**evidence, "reason": "invalid_shap_result"}
    return {**evidence, "status": "completed", "shap": explanation}


def _explain_sample(
    artifact: LocalPipelineArtifact,
    sampled: pd.DataFrame | pl.DataFrame,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Reuse fitted preprocessing and explain the fitted estimator without refitting."""
    try:
        transformed = artifact.pipeline.feature_engineer.transform(sampled, preserve_rows=True)
    except ValueError as exc:
        return {**evidence, "reason": "inference_transform_unavailable", "detail": str(exc)}
    limits = evidence["limits"]
    count = evidence["sample_count"]
    reason = _transformed_unavailable_reason(artifact, transformed, count, limits["max_features"])
    if reason:
        return {**evidence, "reason": reason}
    features = transformed.to_pandas() if isinstance(transformed, pl.DataFrame) else transformed
    estimator = artifact.pipeline.model_estimator
    if estimator is None or estimator.model is None:
        return {**evidence, "reason": "model_unavailable"}
    model = estimator._unwrap_tuned_model()
    explanation = compute_shap_explanation(
        model,
        features,
        max_samples=count,
        max_display_samples=min(count, limits["max_display_samples"]),
    )
    return _explanation_result(explanation, evidence)


def explain_training_artifact(
    artifact: LocalPipelineArtifact, training_frame: pd.DataFrame | pl.DataFrame
) -> dict[str, Any] | None:
    """Explain bounded training rows with fitted inference preprocessing only."""
    if not isinstance(artifact, LocalPipelineArtifact):
        raise TypeError("Expected a LocalPipelineArtifact.")
    limits = _settings(cast(dict[str, Any], artifact.pipeline.config))
    if limits is None:
        return None
    if not isinstance(training_frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("training_frame must be a pandas or Polars DataFrame.")
    evidence: dict[str, Any] = {
        "method": "shap",
        "status": "unavailable",
        "limits": limits,
        "sample_count": 0,
        "feature_count": len(artifact.manifest.feature_columns),
        "feature_names": list(artifact.manifest.feature_columns),
    }
    reason = _input_unavailable_reason(artifact, training_frame, limits)
    if reason:
        return {**evidence, "reason": reason}
    sampled = _sample_training_inputs(artifact, training_frame, limits["max_samples"])
    evidence["sample_count"] = len(sampled)
    return _explain_sample(artifact, sampled, evidence)


def log_training_explanations(
    run: Any, artifact: LocalPipelineArtifact, training_frame: Any
) -> None:
    """Persist bounded evidence and a portable report under the exact fitted model run."""
    evidence = explain_training_artifact(artifact, training_frame)
    if evidence is None:
        return
    evidence |= {
        "run_id": run.run_id,
        "experiment_id": run.client.get_run(run.run_id).info.experiment_id,
        "model_uri": f"runs:/{run.run_id}/model",
    }
    try:
        report = render_explanation_report(evidence)
    except Exception as exc:  # noqa: BLE001 - optional visualization cannot invalidate a fitted model
        logging.getLogger(__name__).warning("SHAP report rendering failed: %s", exc)
        evidence["report_status"] = "unavailable"
        evidence["report_reason"] = f"Chart rendering unavailable: {type(exc).__name__}: {exc}"
        report = render_explanation_report(
            {"status": "unavailable", "reason": evidence["report_reason"]}
        )
    else:
        evidence["report_status"] = evidence["status"]
    run.client.log_dict(run.run_id, evidence, "explanations.json")
    run.client.log_text(run.run_id, report, "explanations.html")
