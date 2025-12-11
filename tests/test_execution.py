import pytest
import pandas as pd
import os
from core.ml_pipeline.execution.engine import PipelineEngine
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig
from core.ml_pipeline.artifacts.local import LocalArtifactStore

@pytest.fixture
def pipeline_data_csv(tmp_path):
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6] * 10,
        "f2": [10, 20, 30, 40, 50, 60] * 10,
        "target": [0, 0, 0, 1, 1, 1] * 10
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_pipeline_execution_flow(pipeline_data_csv, tmp_path):
    # 1. Setup Artifact Store
    artifact_dir = tmp_path / "artifacts"
    artifact_store = LocalArtifactStore(str(artifact_dir))
    
    # 2. Define Pipeline
    config = PipelineConfig(
        pipeline_id="test_pipeline_001",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type="data_loader",
                params={"source": "csv", "path": pipeline_data_csv}
            ),
            NodeConfig(
                node_id="node_features",
                step_type="feature_engineering",
                inputs=["node_data"],
                params={
                    "steps": [
                        {"name": "split", "transformer": "TrainTestSplitter", "params": {"target_column": "target"}}
                    ]
                }
            ),
            NodeConfig(
                node_id="node_training",
                step_type="model_training",
                inputs=["node_features"],
                params={
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "hyperparameters": {"C": 1.0},
                    "evaluate": True
                }
            )
        ]
    )
    
    # 3. Run Engine
    engine = PipelineEngine(artifact_store)
    result = engine.run(config)
    
    # 4. Verify
    assert result.status == "success"
    assert len(result.node_results) == 3
    
    # Check Data Node
    assert result.node_results["node_data"].status == "success"
    assert artifact_store.exists("node_data")
    
    # Check Feature Node
    assert result.node_results["node_features"].status == "success"
    assert artifact_store.exists("node_features")
    
    # Check Training Node
    train_res = result.node_results["node_training"]
    assert train_res.status == "success"
    assert "test_accuracy" in train_res.metrics
    assert artifact_store.exists("node_training")

def test_pipeline_tuning_flow(pipeline_data_csv, tmp_path):
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts_tuning"))
    
    config = PipelineConfig(
        pipeline_id="test_pipeline_tuning",
        nodes=[
            NodeConfig(
                node_id="node_data",
                step_type="data_loader",
                params={"source": "csv", "path": pipeline_data_csv}
            ),
            NodeConfig(
                node_id="node_features",
                step_type="feature_engineering",
                inputs=["node_data"],
                params={
                    "steps": [
                        {"name": "split", "transformer": "TrainTestSplitter", "params": {"target_column": "target"}}
                    ]
                }
            ),
            NodeConfig(
                node_id="node_tuning",
                step_type="model_tuning",
                inputs=["node_features"],
                params={
                    "target_column": "target",
                    "algorithm": "logistic_regression",
                    "tuning_config": {
                        "strategy": "grid",
                        "metric": "accuracy",
                        "cv_folds": 2,
                        "search_space": {"C": [0.1, 1.0]}
                    }
                }
            )
        ]
    )
    
    engine = PipelineEngine(artifact_store)
    result = engine.run(config)
    
    assert result.status == "success"
    assert result.node_results["node_tuning"].status == "success"
    assert "best_score" in result.node_results["node_tuning"].metrics
