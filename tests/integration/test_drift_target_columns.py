"""Regression coverage for target labels leaking into feature drift reports."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi import UploadFile
from starlette.requests import Request

from backend.ml_pipeline._execution.engine._artifacts import ArtifactsMixin
from backend.monitoring.router import calculate_drift


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_shape", ["series", "frame"])
def test_split_reference_preserves_target_name(engine, target_shape):
    """Splitter snapshots must not turn a named target into an empty schema column."""
    if engine == "pandas":
        features = pd.DataFrame({"length": [1.0, 2.0, 3.0]})
        target = pd.Series(["a", "b", "c"], name="Species")
        if target_shape == "frame":
            target = target.to_frame()
    else:
        features = pl.DataFrame({"length": [1.0, 2.0, 3.0]})
        target = pl.Series("Species", ["a", "b", "c"])
        if target_shape == "frame":
            target = target.to_frame()

    reference = ArtifactsMixin()._normalize_train_frame((features, target), "")

    assert list(reference.columns) == ["length", "Species"]
    assert reference["Species"].to_list() == ["a", "b", "c"]
    assert list(features.columns) == ["length"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_unnamed_target_does_not_create_empty_reference_column(engine):
    """An unavailable target name must not manufacture a missing feature during drift."""
    features = (
        pd.DataFrame({"length": [1.0, 2.0]})
        if engine == "pandas"
        else pl.DataFrame({"length": [1.0, 2.0]})
    )

    reference = ArtifactsMixin()._normalize_train_frame((features, np.array([0, 1])), "")

    assert list(reference.columns) == ["length"]


@pytest.mark.parametrize("reference_target", ["", "Species"])
@pytest.mark.parametrize("include_target", [True, False])
@pytest.mark.asyncio
async def test_drift_ignores_target_in_legacy_and_named_references(
    reference_target, include_target
):
    """Identical features stay stable whether the upload includes labels or omits them."""
    ref = pl.DataFrame({"length": [1.0, 2.0, 3.0], reference_target: ["a", "b", "c"]})
    current = pl.DataFrame({"length": [1.0, 2.0, 3.0]})
    if include_target:
        current = current.with_columns(pl.Series("Species", ["c", "c", "c"]))

    result = await _calculate_report(ref, current)

    assert result.drifted_columns_count == 0
    assert result.severity == "none"
    assert result.missing_columns == []
    assert result.new_columns == []
    assert list(result.column_drifts) == ["length"]
    assert result.column_drifts["length"]["drift_detected"] is False


@pytest.mark.asyncio
async def test_drift_keeps_real_missing_and_new_features():
    """Excluding the target must preserve alerts for actual feature schema changes."""
    ref = pl.DataFrame({"length": [1.0, 2.0], "width": [3.0, 4.0], "": ["a", "b"]})
    current = pl.DataFrame({"length": [1.0, 2.0], "extra": [5.0, 6.0], "Species": ["a", "b"]})

    result = await _calculate_report(ref, current)

    assert result.drifted_columns_count == 2
    assert result.severity == "critical"
    assert result.missing_columns == ["width"]
    assert result.new_columns == ["extra"]


async def _calculate_report(reference, current):
    """Exercise the real upload parser and calculator with isolated storage and metadata."""
    with (
        patch("backend.monitoring.router.ArtifactFactory"),
        patch("backend.monitoring.router._find_reference_key", return_value="reference"),
        patch("backend.monitoring.router._load_reference_dataframe", return_value=reference),
        patch(
            "backend.monitoring.router._fetch_drift_job_rows",
            new=AsyncMock(return_value={"job-1": SimpleNamespace()}),
        ),
        patch("backend.monitoring.router._extract_drift_target_column", return_value="Species"),
        patch(
            "backend.monitoring.router._find_deployment_context",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch(
            "backend.monitoring.router._get_or_create_threshold_version",
            new=AsyncMock(return_value=SimpleNamespace(version=1)),
        ),
        patch("backend.monitoring.router._save_drift_alert", new=AsyncMock(return_value=None)),
        patch(
            "backend.monitoring.router._load_feature_importances", new=AsyncMock(return_value=None)
        ),
    ):
        return await calculate_drift(
            request=Request({"type": "http", "path": "/monitoring/drift/calculate"}),
            job_id="job-1",
            file=UploadFile(file=io.BytesIO(current.write_csv().encode()), filename="current.csv"),
            dataset_name="iris",
            threshold_psi=None,
            threshold_ks=None,
            threshold_wasserstein=None,
            threshold_kl=None,
            db=AsyncMock(),
        )
