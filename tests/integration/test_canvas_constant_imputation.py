"""Canvas constant strings execute through the real backend and persist typed core fills."""

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._schemas import PipelineConfigModel
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_canvas_text_constant_executes_and_saved_backend_step_replays_typed_fills(
    engine, tmp_path, monkeypatch
):
    """The exact string payload from Canvas must fill numeric and text columns without changing their kinds."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    source = tmp_path / "source.csv"
    pd.DataFrame({"value": [2.0, None, 6.0], "text": ["a", None, "b"]}).to_csv(source, index=False)
    wire = PipelineConfigModel.model_validate(
        {
            "pipeline_id": "constant",
            "nodes": [
                {
                    "node_id": "source",
                    "step_type": "data_loader",
                    "params": {"source": "csv", "path": str(source)},
                },
                {
                    "node_id": "fill",
                    "step_type": "SimpleImputer",
                    "params": {
                        "columns": ["value", "text"],
                        "strategy": "constant",
                        "fill_value": "0",
                    },
                    "inputs": ["source"],
                },
            ],
        }
    )
    restored = PipelineConfigModel.model_validate_json(wire.model_dump_json())
    config = PipelineConfig(
        restored.pipeline_id, [NodeConfig(**node.model_dump()) for node in restored.nodes]
    )
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, catalog=FileSystemCatalog(str(tmp_path))).run(
        config, inspect_all=True
    )
    assert result.status == "success", result.node_results
    saved = LocalArtifactStore(str(store.base_path)).load("exec_fill_pipeline")
    query = pd.DataFrame({"value": [None, 8.0], "text": [None, "kept"]})
    output = saved.transform(pl.from_pandas(query) if engine == "polars" else query)
    output = output.to_native() if hasattr(output, "to_native") else output
    assert output["value"].to_list() == [0.0, 8.0]
    assert output["text"].to_list() == ["0", "kept"]
