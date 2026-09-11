"""Feature operation validation must propagate through the production node runner."""

from pathlib import Path

import pandas as pd
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("operation_type", ["polynomial", "not_an_operation"])
def test_unsupported_operation_fails_its_pipeline_node(
    operation_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview must expose an unsupported operation as a failed node, not a successful no-op."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas")
    csv_path = tmp_path / "numbers.csv"
    pd.DataFrame({"x": [2.0, 3.0]}).to_csv(csv_path, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    config = PipelineConfig(
        pipeline_id="unsupported-feature-operation",
        nodes=[
            NodeConfig("load", "data_loader", params={"source": "csv", "path": str(csv_path)}),
            NodeConfig(
                "features",
                "FeatureGenerationNode",
                inputs=["load"],
                params={"operations": [{"operation_type": operation_type, "input_columns": ["x"]}]},
            ),
        ],
    )

    result = engine.run(config)

    assert result.status == "failed"
    assert result.node_results["load"].status == "success"
    failed_node = result.node_results["features"]
    assert failed_node.status == "failed"
    assert failed_node.output_artifact_id is None
    assert operation_type in (failed_node.error or "")
    if operation_type == "polynomial":
        assert "PolynomialFeatures" in (failed_node.error or "")
