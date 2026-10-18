"""Backend preview uses the same column rules as actual pipeline training."""

import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.core.schema import SkyulfSchema


@pytest.mark.parametrize(
    "rules",
    [
        {"columns_regex": "^measure_"},
        {"dtype": "numeric", "exclude_columns": ["id"]},
    ],
)
def test_preview_promotes_only_rule_selected_columns(rules):
    """Unselected integer identifiers should remain integer in downstream schema pickers."""
    seed = SkyulfSchema.from_columns(["measure_x", "id"], {"measure_x": "int64", "id": "int64"})
    config = PipelineConfig(
        pipeline_id="rules",
        nodes=[
            NodeConfig("data", "data_loader"),
            NodeConfig("scale", "StandardScaler", params=rules, inputs=["data"]),
        ],
    )
    predicted = predict_schemas(config, {"data": seed})
    assert predicted["scale"] == seed.with_dtype("measure_x", "float64")


def test_ambiguous_object_dtype_keeps_preview_unknown():
    """An object column may contain numeric Decimal values, so preview cannot classify it blindly."""
    seed = SkyulfSchema.from_columns(["x"], {"x": "object"})
    config = PipelineConfig(
        pipeline_id="rules",
        nodes=[
            NodeConfig("data", "data_loader"),
            NodeConfig("scale", "StandardScaler", params={"dtype": "numeric"}, inputs=["data"]),
        ],
    )
    assert predict_schemas(config, {"data": seed})["scale"] is None
