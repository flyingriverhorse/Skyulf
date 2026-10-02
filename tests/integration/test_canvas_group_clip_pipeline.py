"""Run Canvas GroupImputer and ClipValues nodes through the real backend engine."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.imputation.group import GroupImputerCalculator
from skyulf.preprocessing.outliers.clip_values import ClipValuesCalculator


@pytest.fixture(params=["pandas", "polars"])
def frame_engine(request, monkeypatch):
    """Each graph must use the requested engine at both ingestion and Core execution."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", request.param)
    return request.param


def _rows() -> pd.DataFrame:
    """Two industries with one missing employee count each and one extreme value."""
    return pd.DataFrame(
        {
            "industry": ["bank", "bank", "bank", "shop", "shop", "shop"],
            "employees": [10.0, 30.0, None, 2.0, 4.0, None],
            "revenue": [5.0, 7.0, 9.0, -50.0, 1.0, 900.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


def _pandas(value) -> pd.DataFrame:
    """Compare values without depending on the engine's frame type."""
    value = value.to_native() if hasattr(value, "to_native") else value
    return value.to_pandas() if isinstance(value, pl.DataFrame) else value


def _run(tmp_path: Path) -> LocalArtifactStore:
    """Execute load -> X/y split -> group fill -> clip exactly as Canvas serializes it."""
    source = tmp_path / "companies.csv"
    _rows().to_csv(source, index=False)
    config = PipelineConfig(
        pipeline_id="canvas-group-clip",
        nodes=[
            NodeConfig("source", "data_loader", params={"source": "csv", "path": str(source)}),
            NodeConfig(
                "xy", "feature_target_split", params={"target_column": "target"}, inputs=["source"]
            ),
            NodeConfig(
                "fill",
                "GroupImputer",
                params={"columns": ["employees"], "group_by": "industry", "strategy": "mean"},
                inputs=["xy"],
            ),
            NodeConfig(
                "clip",
                "ClipValues",
                params={"bounds": {"revenue": {"lower": 0, "upper": 100}}},
                inputs=["fill"],
            ),
        ],
    )
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, catalog=FileSystemCatalog(str(tmp_path))).run(
        deepcopy(config), inspect_all=True
    )
    assert result.status == "success", result.node_results
    return store


def test_canvas_group_fill_and_clip_keep_every_row(tmp_path, frame_engine):
    """Missing values take their own group's mean and extremes are capped without dropping rows."""
    store = _run(tmp_path)
    features, labels = extract_xy(store.load("clip"), "target")
    actual = _pandas(features).reset_index(drop=True)
    assert actual["employees"].tolist() == [10.0, 30.0, 20.0, 2.0, 4.0, 3.0]
    assert actual["revenue"].tolist() == [5.0, 7.0, 9.0, 0.0, 1.0, 100.0]
    assert pd.Series(labels).tolist() == [0, 1, 0, 1, 0, 1]


def test_saved_group_fill_and_clip_replay_on_new_rows(tmp_path, frame_engine):
    """Inference reuses the stored group means and bounds and never refits on the new batch."""
    store = _run(tmp_path)
    reloaded = LocalArtifactStore(str(Path(store.base_path)))
    fill = reloaded.load("exec_fill_pipeline")
    clip = reloaded.load("exec_clip_pipeline")
    fresh_rows = {
        "industry": ["bank", "shop", "farm"],
        "employees": [None, None, None],
        "revenue": [500.0, -1.0, 50.0],
    }
    fresh = pl.DataFrame(fresh_rows) if frame_engine == "polars" else pd.DataFrame(fresh_rows)
    with (
        patch.object(GroupImputerCalculator, "fit", side_effect=AssertionError("refit")),
        patch.object(ClipValuesCalculator, "fit", side_effect=AssertionError("refit")),
    ):
        actual = _pandas(clip.transform(fill.transform(fresh)))
    # "farm" was never seen in training, so it falls back to the overall mean of 11.5.
    assert actual["employees"].tolist() == [20.0, 3.0, 11.5]
    assert actual["revenue"].tolist() == [100.0, 0.0, 50.0]
