"""Bounded explanations use only fitted local training features."""

from typing import Any, cast

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import FittedPipelineArtifact, load_pipeline, save_pipeline
from skyulf.integrations.databricks.observability.reports.explanations import (
    explain_training_artifact,
    validate_explanation_config,
)
from skyulf.pipeline import SkyulfPipeline


def _artifact(
    tmp_path, engine: str, settings: dict | None
) -> tuple[FittedPipelineArtifact, pd.DataFrame]:
    """Build a genuine fitted artifact so transformed feature order is observable."""
    frame = pd.DataFrame(
        {
            "city": ["Riga", "Vilnius", "Riga", "Tallinn", "Vilnius", "Tallinn"],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [11.0, 24.0, 13.0, 36.0, 27.0, 38.0],
            "event_time": pd.date_range("2025-01-01", periods=6),
        }
    )
    config = {
        "preprocessing": [
            {
                "name": "encode_city",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"},
            }
        ],
        "modeling": {"type": "linear_regression"},
    }
    pipeline = SkyulfPipeline(config)
    features = frame.drop(columns="event_time")
    native = pl.from_pandas(features) if engine == "polars" else features
    if isinstance(native, pl.DataFrame):
        train, test = native.slice(0, 5), native.slice(5, 1)
    else:
        train, test = native.iloc[:5], native.iloc[5:]
    pipeline.fit(SplitDataset(train=train, test=test), target_column="target")
    cast(dict[str, Any], pipeline.config)["explainability"] = settings
    path = tmp_path / "artifact"
    save_pipeline(pipeline, path)
    return load_pipeline(path), frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("input_engine", ["pandas", "polars"])
def test_explanation_projects_training_features_and_reuses_fitted_transform(
    tmp_path, monkeypatch, engine: str, input_engine: str
) -> None:
    """Target and temporal metadata must never reach fitted FE or SHAP."""
    artifact, frame = _artifact(tmp_path, engine, {"method": "shap", "max_samples": 3})
    observed = {}

    def fake_shap(model, X, *, max_samples, max_display_samples):
        """Capture the exact model input without requiring optional SHAP."""
        observed["model"] = model
        observed["X"] = X.copy()
        observed["limits"] = (max_samples, max_display_samples)
        return {
            "feature_names": list(X.columns),
            "mean_abs_importance": dict.fromkeys(X.columns, 0.1),
            "samples": [],
            "interactions": None,
        }

    monkeypatch.setattr(
        "skyulf.integrations.databricks.observability.reports.explanations.compute_shap_explanation",
        fake_shap,
    )
    native = pl.from_pandas(frame) if input_engine == "polars" else frame
    result = explain_training_artifact(artifact, native)

    assert result is not None
    assert result["status"] == "completed"
    assert result["sample_count"] == 3
    assert result["feature_count"] == len(artifact.manifest.feature_columns)
    assert tuple(observed["X"].columns) == artifact.manifest.feature_columns
    assert "target" not in observed["X"].columns
    assert "event_time" not in observed["X"].columns
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    assert observed["model"] is estimator._unwrap_tuned_model()
    assert observed["limits"] == (3, 3)


def test_absent_explanation_does_not_compute(tmp_path, monkeypatch) -> None:
    """Default training must avoid optional SHAP work entirely."""
    artifact, frame = _artifact(tmp_path, "pandas", None)
    monkeypatch.setattr(
        "skyulf.integrations.databricks.observability.reports.explanations.compute_shap_explanation",
        lambda *a, **k: pytest.fail("SHAP must stay off"),
    )
    assert explain_training_artifact(artifact, frame) is None


def test_real_shap_explains_a_saved_training_artifact(tmp_path) -> None:
    """The optional SHAP backend must explain the reloaded model and fitted features."""
    pytest.importorskip("shap")
    artifact, frame = _artifact(
        tmp_path, "pandas", {"method": "shap", "max_samples": 3, "max_display_samples": 2}
    )
    result = explain_training_artifact(artifact, frame)
    assert result is not None
    assert result["status"] == "completed"
    assert result["sample_count"] == 3
    assert result["shap"]["feature_names"] == list(artifact.manifest.feature_columns)
    assert len(result["shap"]["samples"]) == 2


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"method": "other"},
        {"method": "shap", "max_samples": True},
        {"method": "shap", "max_samples": 201},
        {"method": "shap", "max_features": 0},
        {"method": "shap", "max_display_samples": 51},
        {"method": "shap", "max_samples": 2, "max_display_samples": 3},
        {"method": "shap", "unknown": 1},
    ],
)
def test_invalid_explanation_configuration_is_rejected(settings: dict) -> None:
    """Invalid work budgets must fail before artifact or data access."""
    with pytest.raises(ValueError, match="explainability"):
        validate_explanation_config({"explainability": settings})


def test_unavailable_shap_is_explicit(tmp_path, monkeypatch) -> None:
    """A missing or unsupported SHAP backend cannot look completed."""
    artifact, frame = _artifact(tmp_path, "pandas", {"method": "shap"})
    monkeypatch.setattr(
        "skyulf.integrations.databricks.observability.reports.explanations.compute_shap_explanation",
        lambda *a, **k: None,
    )
    result = explain_training_artifact(artifact, frame)
    assert result is not None
    assert result["status"] == "unavailable"
    assert result["reason"] == "shap_unavailable"


def test_feature_cap_prevents_transform_and_shap(tmp_path, monkeypatch) -> None:
    """A low feature budget must stop work before costly preprocessing or SHAP."""
    artifact, frame = _artifact(tmp_path, "pandas", {"method": "shap", "max_features": 1})
    monkeypatch.setattr(
        artifact.pipeline.feature_engineer,
        "transform",
        lambda *a, **k: pytest.fail("transform must stay off"),
    )
    result = explain_training_artifact(artifact, frame)
    assert result is not None
    assert result["status"] == "unavailable"
    assert result["reason"] == "feature_limit_exceeded"
    assert result["sample_count"] == 0


def test_inference_row_change_is_visible(tmp_path, monkeypatch) -> None:
    """An inference transform that drops rows cannot produce misleading SHAP values."""
    artifact, frame = _artifact(tmp_path, "pandas", {"method": "shap"})
    monkeypatch.setattr(
        artifact.pipeline.feature_engineer,
        "transform",
        lambda data, **kwargs: data.iloc[:-1],
    )
    result = explain_training_artifact(artifact, frame)
    assert result is not None
    assert result["status"] == "unavailable"
    assert result["reason"] == "row_count_changed"


@pytest.mark.parametrize("empty", [True, False])
def test_unusable_input_stops_before_sampling(tmp_path, monkeypatch, empty) -> None:
    """Empty or incomplete source rows must retain their explicit unavailable reason."""
    artifact, frame = _artifact(tmp_path, "pandas", {"method": "shap"})
    frame = frame.head(0) if empty else frame.drop(columns="amount")
    monkeypatch.setattr(
        artifact.pipeline.feature_engineer,
        "transform",
        lambda *a, **k: pytest.fail("invalid input must not be transformed"),
    )
    result = explain_training_artifact(artifact, frame)
    assert result is not None
    assert result["reason"] == ("empty_training_frame" if empty else "missing_input_columns")
    assert result["sample_count"] == 0


@pytest.mark.parametrize("failure", ["transform", "schema", "model", "shap_schema", "shap_json"])
def test_explanation_failures_preserve_evidence(tmp_path, monkeypatch, failure) -> None:
    """Unavailable explanations must keep sample counts and specific failure reasons."""
    artifact, frame = _artifact(tmp_path, "pandas", {"method": "shap", "max_samples": 3})

    def fail_transform(*args, **kwargs):
        """Represent a fitted preprocessing operation unavailable during inference."""
        raise ValueError("inference blocked")

    if failure == "transform":
        monkeypatch.setattr(artifact.pipeline.feature_engineer, "transform", fail_transform)
    elif failure == "schema":
        monkeypatch.setattr(
            artifact.pipeline.feature_engineer, "transform", lambda data, **kwargs: data
        )
    elif failure == "model":
        monkeypatch.setattr(artifact.pipeline, "model_estimator", None)
    else:
        explanation = {
            "feature_names": list(artifact.manifest.feature_columns),
            "mean_abs_importance": {"amount": float("nan")},
        }
        if failure == "shap_schema":
            explanation["feature_names"] = ["wrong"]
        monkeypatch.setattr(
            "skyulf.integrations.databricks.observability.reports.explanations.compute_shap_explanation",
            lambda *a, **k: explanation,
        )
    result = explain_training_artifact(artifact, frame)
    reasons = {
        "transform": "inference_transform_unavailable",
        "schema": "feature_schema_mismatch",
        "model": "model_unavailable",
        "shap_schema": "shap_feature_schema_mismatch",
        "shap_json": "invalid_shap_result",
    }
    assert result is not None
    assert result["status"] == "unavailable"
    assert result["sample_count"] == 3
    assert result["reason"] == reasons[failure]
    if failure == "transform":
        assert result["detail"] == "inference blocked"
