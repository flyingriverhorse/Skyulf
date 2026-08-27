"""Unit tests for the training node's non-numeric fail-fast guard.

``_assert_numeric_training_frame`` preflights the eagerly-merged training
frame (which equals the per-fold refit output by the F-15 eager==refit
parity invariant) and raises an actionable ``ValueError`` naming leftover
non-numeric columns, instead of letting every tuning fold/trial die inside
sklearn and surface only as "All trials failed".
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.constants import StepType


def _runner() -> PipelineEngine:
    # The guard and its helper _resolve_train_frame touch no instance state.
    return PipelineEngine.__new__(PipelineEngine)


def _node() -> NodeConfig:
    return NodeConfig(
        node_id="node_training",
        step_type=StepType.TRAINING,
        inputs=["node_features"],
        params={"target_column": "target"},
    )


def test_pandas_object_column_raises_actionable_message():
    frame = pd.DataFrame({"city": ["a", "b", "c"], "f1": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    with pytest.raises(ValueError, match="non-numeric column\\(s\\)") as excinfo:
        _runner()._assert_numeric_training_frame(_node(), frame, "target", {})
    message = str(excinfo.value)
    assert "city" in message
    assert "merge order" in message
    assert "node_training" in message


def test_polars_string_column_raises():
    frame = pl.DataFrame({"city": ["a", "b"], "f1": [1.0, 2.0], "target": [0, 1]})
    with pytest.raises(ValueError, match="city"):
        _runner()._assert_numeric_training_frame(_node(), frame, "target", {})


def test_all_numeric_frame_passes():
    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3, 4],
            "flag": [True, False],
            "target": [0, 1],
        }
    )
    _runner()._assert_numeric_training_frame(_node(), frame, "target", {})


def test_string_target_column_is_excluded():
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "Species": ["a", "b", "c"]})
    _runner()._assert_numeric_training_frame(_node(), frame, "Species", {})


def test_category_dtype_is_flagged():
    frame = pd.DataFrame({"cat": pd.Categorical(["x", "y"]), "f1": [1.0, 2.0], "target": [0, 1]})
    with pytest.raises(ValueError, match="cat"):
        _runner()._assert_numeric_training_frame(_node(), frame, "target", {})


def test_explicit_time_series_time_column_is_excluded():
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "f1": [1.0, 2.0],
            "target": [0, 1],
        }
    )
    params = {"cv_type": "time_series_split", "cv_time_column": "ts"}
    _runner()._assert_numeric_training_frame(_node(), frame, "target", params)


def test_auto_detected_time_column_is_excluded():
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "f1": [1.0, 2.0],
            "target": [0, 1],
        }
    )
    params = {"cv_type": "time_series_split"}
    _runner()._assert_numeric_training_frame(_node(), frame, "target", params)


def test_datetime_column_flagged_outside_time_series_cv():
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "f1": [1.0, 2.0],
            "target": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="ts"):
        _runner()._assert_numeric_training_frame(_node(), frame, "target", {})


def test_offending_column_list_is_truncated():
    frame = pd.DataFrame({f"s{i}": ["a", "b"] for i in range(8)} | {"target": [0, 1]})
    with pytest.raises(ValueError, match=r"\+2 more"):
        _runner()._assert_numeric_training_frame(_node(), frame, "target", {})


def test_split_dataset_input_resolves_to_train_frame():
    from skyulf.data.dataset import SplitDataset

    train = pd.DataFrame({"city": ["a", "b"], "target": [0, 1]})
    data = SplitDataset(train=train, test=pd.DataFrame(), validation=None)
    with pytest.raises(ValueError, match="city"):
        _runner()._assert_numeric_training_frame(_node(), data, "target", {})


def test_frameless_payload_is_ignored():
    payload = (np.array([[1.0, 2.0]]), np.array([0]))
    _runner()._assert_numeric_training_frame(_node(), payload, "target", {})
