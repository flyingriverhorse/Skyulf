"""Canvas date settings reach backend preview and the native calculator unchanged."""

import pandas as pd
import polars as pl
import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._schemas import PipelineConfigModel
from skyulf.core.schema import SkyulfSchema
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_date_canvas_wire_schema_matches_local_calendar_output(engine):
    """Timezone and cycle selections must survive API validation and agree with backend schema preview."""
    params = {
        "columns": ["date"],
        "features": ["second", "hour_sin", "hour_cos"],
        "timezone": "Europe/Istanbul",
        "drop_original": True,
    }
    wire = PipelineConfigModel.model_validate(
        {
            "pipeline_id": "calendar",
            "nodes": [
                {"node_id": "source", "step_type": "data_loader"},
                {
                    "node_id": "calendar",
                    "step_type": "DateFeatures",
                    "inputs": ["source"],
                    "params": params,
                },
            ],
        }
    )
    restored = PipelineConfigModel.model_validate_json(wire.model_dump_json())
    config = PipelineConfig(
        restored.pipeline_id, [NodeConfig(**node.model_dump()) for node in restored.nodes]
    )
    frame = pd.DataFrame({"date": ["2024-01-01T00:00:17Z"]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    predicted = predict_schemas(config, {"source": SkyulfSchema.from_dataframe(frame)})["calendar"]
    assert restored.nodes[1].params == params
    assert predicted is not None
    state = NodeRegistry.get_calculator("DateFeatures")().fit(frame, config.nodes[1].params)
    output = NodeRegistry.get_applier("DateFeatures")().apply(frame, state)
    assert predicted.column_list() == list(output.columns)
    assert predicted.dtypes == {
        "date_second": "Int64",
        "date_hour_sin": "Float64",
        "date_hour_cos": "Float64",
    }
    assert output["date_second"].to_list() == [17]
    assert output["date_hour_sin"].to_list()[0] == pytest.approx(2**-0.5)
