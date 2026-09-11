"""Empty-vocabulary warnings must reach the engine response with node identity."""

from pathlib import Path

import pandas as pd
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore


@pytest.mark.parametrize("node_type", ["count_vectorizer", "tfidf_vectorizer"])
def test_empty_vocabulary_preserves_data_and_reaches_node_warnings(
    node_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview must return a usable result and a node-tagged warning for an empty vocabulary."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas")
    source = pd.DataFrame({"text": ["the and", "is of"], "value": [3, 7], "target": [0, 1]})
    csv_path = tmp_path / "text.csv"
    source.to_csv(csv_path, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    config = PipelineConfig(
        pipeline_id="empty-vocabulary",
        nodes=[
            NodeConfig("load", "data_loader", params={"source": "csv", "path": str(csv_path)}),
            NodeConfig(
                "vector",
                node_type,
                inputs=["load"],
                params={"columns": ["text"], "stop_words": "english", "drop_original": True},
            ),
        ],
    )

    result = engine.run(config)

    assert result.status == "success"
    pd.testing.assert_frame_equal(store.load("vector"), store.load("load"))
    warnings = [warning for warning in result.node_warnings if warning["node_id"] == "vector"]
    assert len(warnings) == 1
    assert warnings[0]["node_type"] == node_type
    assert warnings[0]["level"] == "warning"
    assert "empty vocabulary" in warnings[0]["message"].lower()
    assert "unchanged" in warnings[0]["message"].lower()

    fitted_pipeline = store.load("exec_vector_pipeline")
    later = pd.DataFrame({"text": ["new usable vocabulary"], "value": [4], "target": [1]})
    pd.testing.assert_frame_equal(fitted_pipeline.transform(later), later)
