"""Canvas group parameters survive backend validation and produce the learned core behavior."""

import pandas as pd
import polars as pl
import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._schemas import PipelineConfigModel
from skyulf.core.schema import SkyulfSchema
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_group_options_wire_roundtrip_and_schema(engine):
    """Ordered fallback keys and deviation methods must reach the registered calculators unchanged."""
    fill = {
        "columns": ["value"],
        "group_by": ["region", "industry"],
        "strategy": "mean",
        "fallback": "hierarchical",
    }
    deviation = {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "deviation",
                "input_columns": ["region"],
                "secondary_columns": ["value"],
                "output_column": "centered",
            }
        ]
    }
    wire = PipelineConfigModel.model_validate(
        {
            "pipeline_id": "groups",
            "nodes": [
                {"node_id": "source", "step_type": "data_loader"},
                {
                    "node_id": "fill",
                    "step_type": "GroupImputer",
                    "inputs": ["source"],
                    "params": fill,
                },
                {
                    "node_id": "deviation",
                    "step_type": "FeatureMath",
                    "inputs": ["fill"],
                    "params": deviation,
                },
            ],
        }
    )
    restored = PipelineConfigModel.model_validate_json(wire.model_dump_json())
    config = PipelineConfig(
        restored.pipeline_id, [NodeConfig(**node.model_dump()) for node in restored.nodes]
    )
    frame = pd.DataFrame(
        {"region": ["a", "a", "a"], "industry": ["x", "x", "y"], "value": [2.0, 4.0, None]}
    )
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    predicted = predict_schemas(config, {"source": SkyulfSchema.from_dataframe(frame)})
    assert restored.nodes[1].params == fill
    assert restored.nodes[2].params == deviation
    assert predicted["fill"] is not None and predicted["fill"].dtypes["value"] == "float64"
    for node in config.nodes[1:]:
        state = NodeRegistry.get_calculator(node.step_type)().fit(frame, node.params)
        frame = NodeRegistry.get_applier(node.step_type)().apply(frame, state)
    assert frame["value"].to_list() == [2.0, 4.0, 3.0]
    assert frame["centered"].to_list() == [-1.0, 1.0, 0.0]
