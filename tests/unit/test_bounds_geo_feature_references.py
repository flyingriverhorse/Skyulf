"""Canvas feature-only controls must not reference labels separated into y."""

import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.core.schema import SkyulfSchema


@pytest.mark.parametrize("split_type", ["feature_target_split", "TrainTestSplitter", "Split"])
@pytest.mark.parametrize(
    "step_type, field", [("ManualBounds", "bounds"), ("GeoDistance", "lat1_col")]
)
def test_feature_operations_reject_a_separated_target(split_type, step_type, field):
    """A target retained in schema metadata is unavailable to operations that act only on X."""
    params = {"bounds": {"target": {"upper": 10}}} if field == "bounds" else {field: "target"}
    pipeline = PipelineConfig(
        pipeline_id="separated-label",
        nodes=[
            NodeConfig("load", "data_loader"),
            NodeConfig("split", split_type, params={"target_column": "target"}, inputs=["load"]),
            NodeConfig(
                "pass", "ManualBounds", params={"bounds": {"age": {"lower": 18}}}, inputs=["split"]
            ),
            NodeConfig("operation", step_type, params=params, inputs=["pass"]),
        ],
    )
    schema = SkyulfSchema.from_columns(["age", "target"], {"age": "float64", "target": "float64"})
    predicted = predict_schemas(pipeline, {"load": schema})
    assert find_broken_references(pipeline, predicted) == [
        {"node_id": "operation", "field": field, "column": "target", "upstream_node_id": "pass"}
    ]


def test_bounds_can_filter_a_column_before_it_becomes_a_target():
    """A downstream split must not make a currently present raw column unavailable upstream."""
    pipeline = PipelineConfig(
        pipeline_id="raw-label",
        nodes=[
            NodeConfig("load", "data_loader"),
            NodeConfig(
                "bounds",
                "ManualBounds",
                params={"bounds": {"target": {"upper": 10}}},
                inputs=["load"],
            ),
            NodeConfig(
                "split", "TrainTestSplitter", params={"target_column": "target"}, inputs=["bounds"]
            ),
        ],
    )
    predicted = predict_schemas(pipeline, {"load": SkyulfSchema.from_columns(["target"])})
    assert find_broken_references(pipeline, predicted) == []
