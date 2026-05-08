"""E3 — Snapshot tests for pipeline config JSON shapes.

Captures the wire format of PipelineConfigModel and related schemas so
that any unintentional structural change (renamed field, added required
key, removed node) shows up as a diff in CI.

Update snapshots intentionally with:
    pytest tests/test_pipeline_config_snapshots.py --snapshot-update
"""

from syrupy.assertion import SnapshotAssertion

from backend.ml_pipeline._internal._schemas import (
    NodeConfigModel,
    PipelineConfigModel,
)
from backend.ml_pipeline.api import _build_node_registry
from backend.ml_pipeline.constants import StepType


# ---------------------------------------------------------------------------
# Helper: build representative PipelineConfigModel instances
# ---------------------------------------------------------------------------


def _linear_pipeline() -> PipelineConfigModel:
    """A simple linear 3-node pipeline: data_loader → standard_scaler → random_forest."""
    return PipelineConfigModel(
        pipeline_id="snap_linear",
        nodes=[
            NodeConfigModel(
                node_id="n1",
                step_type=StepType.DATA_LOADER,
                params={"source_id": "42"},
            ),
            NodeConfigModel(
                node_id="n2",
                step_type="StandardScaler",
                params={},
                inputs=["n1"],
            ),
            NodeConfigModel(
                node_id="n3",
                step_type="random_forest_classifier",
                params={"n_estimators": 100, "target_column": "label"},
                inputs=["n2"],
            ),
        ],
        metadata={"label": "snapshot linear"},
    )


def _preprocessing_pipeline() -> PipelineConfigModel:
    """Preprocessing-only pipeline: data_loader → imputer → winsorize → feature_target_split."""
    return PipelineConfigModel(
        pipeline_id="snap_prep",
        nodes=[
            NodeConfigModel(
                node_id="ds",
                step_type=StepType.DATA_LOADER,
                params={"source_id": "7"},
            ),
            NodeConfigModel(
                node_id="imp",
                step_type="simple_imputer",
                params={"strategy": "median"},
                inputs=["ds"],
            ),
            NodeConfigModel(
                node_id="win",
                step_type="winsorize",
                params={"lower": 0.05, "upper": 0.95},
                inputs=["imp"],
            ),
            NodeConfigModel(
                node_id="spl",
                step_type="feature_target_split",
                params={"target_column": "price"},
                inputs=["win"],
            ),
        ],
    )


def _parallel_branches_pipeline() -> PipelineConfigModel:
    """Two-branch pipeline sharing a common data_loader / scaler."""
    return PipelineConfigModel(
        pipeline_id="snap_parallel",
        nodes=[
            NodeConfigModel(
                node_id="ds",
                step_type=StepType.DATA_LOADER,
                params={"source_id": "3"},
            ),
            NodeConfigModel(
                node_id="sc",
                step_type="StandardScaler",
                params={},
                inputs=["ds"],
            ),
            # Branch A
            NodeConfigModel(
                node_id="spl_a",
                step_type="TrainTestSplitter",
                params={"test_size": 0.2, "target_column": "y"},
                inputs=["sc"],
            ),
            NodeConfigModel(
                node_id="rf",
                step_type="random_forest_classifier",
                params={"n_estimators": 50, "target_column": "y"},
                inputs=["spl_a"],
            ),
            # Branch B
            NodeConfigModel(
                node_id="spl_b",
                step_type="TrainTestSplitter",
                params={"test_size": 0.2, "target_column": "y"},
                inputs=["sc"],
            ),
            NodeConfigModel(
                node_id="lr",
                step_type="logistic_regression",
                params={"C": 1.0, "target_column": "y"},
                inputs=["spl_b"],
            ),
        ],
        metadata={"job_type": "basic_training"},
    )


# ---------------------------------------------------------------------------
# Snapshot tests — pipeline config model_dump()
# ---------------------------------------------------------------------------


def test_linear_pipeline_snapshot(snapshot: SnapshotAssertion) -> None:
    """Linear pipeline dict must match stored snapshot."""
    assert _linear_pipeline().model_dump() == snapshot


def test_preprocessing_pipeline_snapshot(snapshot: SnapshotAssertion) -> None:
    """Preprocessing-only pipeline dict must match stored snapshot."""
    assert _preprocessing_pipeline().model_dump() == snapshot


def test_parallel_branches_pipeline_snapshot(snapshot: SnapshotAssertion) -> None:
    """Parallel-branches pipeline dict must match stored snapshot."""
    assert _parallel_branches_pipeline().model_dump() == snapshot


# ---------------------------------------------------------------------------
# Snapshot test — PipelineConfigModel JSON schema
# ---------------------------------------------------------------------------


def test_pipeline_config_model_json_schema(snapshot: SnapshotAssertion) -> None:
    """Pydantic JSON schema of PipelineConfigModel must match stored snapshot.

    Any change to required fields, field types, or model structure will
    surface here so it can be reviewed intentionally.
    """
    assert PipelineConfigModel.model_json_schema() == snapshot


# ---------------------------------------------------------------------------
# Snapshot test — node registry stable core items
# ---------------------------------------------------------------------------


def test_node_registry_data_loader_item(snapshot: SnapshotAssertion) -> None:
    """The data_loader registry entry must stay structurally stable."""
    registry = {item.id: item.model_dump() for item in _build_node_registry()}
    assert registry[StepType.DATA_LOADER] == snapshot


def test_node_registry_standard_scaler_item(snapshot: SnapshotAssertion) -> None:
    """The StandardScaler registry entry must stay structurally stable."""
    registry = {item.id: item.model_dump() for item in _build_node_registry()}
    assert registry["StandardScaler"] == snapshot


def test_node_registry_random_forest_classifier_item(snapshot: SnapshotAssertion) -> None:
    """The random_forest_classifier registry entry must stay structurally stable."""
    registry = {item.id: item.model_dump() for item in _build_node_registry()}
    assert registry["random_forest_classifier"] == snapshot
