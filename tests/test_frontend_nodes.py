import os

import numpy as np
import pandas as pd
import pytest

from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.registry import NodeRegistry

# Get all node IDs to ensure we cover them
ALL_NODES = {node.id: node for node in NodeRegistry.get_all_nodes()}


@pytest.fixture
def sample_data(tmp_path):
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
            "B": ["a", "b", "a", "b", "c", "a", "b", "c", "a", "b"],
            "C": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.mark.asyncio
async def test_all_transformers(sample_data, tmp_path):
    """
    Test that all transformer nodes can be initialized and run.
    We skip models and splitters here.
    """
    # List of nodes that are transformers (take df, return df)
    # We exclude data_loader, models, and splitters for now
    excluded = [
        "data_loader",
        "logistic_regression",
        "random_forest_classifier",
        "ridge_regression",
        "random_forest_regressor",
        "TrainTestSplitter",
        "feature_target_split",
    ]

    transformers = [nid for nid in ALL_NODES.keys() if nid not in excluded]

    for node_id in transformers:
        print(f"Testing node: {node_id}")

        # Construct a minimal pipeline
        pipeline_config = {
            "pipeline_id": f"test_{node_id}",
            "nodes": [
                {
                    "node_id": "loader",
                    "step_type": "data_loader",
                    "params": {"source_id": "csv", "path": sample_data},
                    "inputs": [],
                },
                {
                    "node_id": "test_node",
                    "step_type": node_id,  # In our system, step_type often maps to the registry ID for transformers
                    "params": {},  # Use defaults
                    "inputs": ["loader"],
                },
            ],
        }

        # Special parameter handling for some nodes
        if node_id == "CustomBinning":
            pipeline_config["nodes"][1]["params"]["bins"] = [0, 5, 10]
        elif node_id == "FeatureGenerationNode":
            pipeline_config["nodes"][1]["params"]["expression"] = "A + C"
            pipeline_config["nodes"][1]["params"]["new_column"] = "generated"
        elif node_id == "ValueReplacement":
            pipeline_config["nodes"][1]["params"]["to_replace"] = {"a": "x"}
        elif node_id == "Casting":
            pipeline_config["nodes"][1]["params"]["target_type"] = "string"
            pipeline_config["nodes"][1]["params"]["columns"] = ["A"]
        elif node_id == "feature_target_split":
            pipeline_config["nodes"][1]["params"]["target_column"] = "target"
        elif node_id == "TargetEncoder":
            # TargetEncoder needs a target. In a real pipeline, it might need X and y.
            # If it's just a transformer in our system, it might expect 'target' column in df
            # or separate input.
            # For now, let's assume it handles the dataframe if target is present.
            pipeline_config["nodes"][1]["params"]["target_column"] = "target"

        # Execute
        artifact_store = LocalArtifactStore(base_path=str(tmp_path))
        engine = PipelineEngine(artifact_store=artifact_store)

        # Convert dict to Pydantic model
        nodes = [NodeConfig(**node) for node in pipeline_config["nodes"]]
        config = PipelineConfig(
            pipeline_id=pipeline_config["pipeline_id"],
            nodes=nodes,
            metadata=pipeline_config.get("metadata", {}),
        )

        try:
            result = engine.run(config, job_id="test_job")
            assert result.status == "success"
            # Check if node output exists in artifacts
            # The engine doesn't return the data directly, it saves it.
            # But for this test, we might need to inspect the artifact store or the result object
            # The result object contains node_results which has output_artifacts

            node_res = result.node_results.get("test_node")
            assert node_res is not None
            assert node_res.status == "success"

        except Exception as e:
            pytest.fail(f"Node {node_id} failed: {str(e)}")


@pytest.mark.asyncio
async def test_models(sample_data, tmp_path):
    models = [
        "logistic_regression",
        "random_forest_classifier",
        "ridge_regression",
        "random_forest_regressor",
    ]

    for node_id in models:
        print(f"Testing model: {node_id}")

        pipeline_config = {
            "pipeline_id": f"test_{node_id}",
            "nodes": [
                {
                    "node_id": "loader",
                    "step_type": "data_loader",
                    "params": {"source_id": "csv", "path": sample_data},
                    "inputs": [],
                },
                {
                    "node_id": "imputer",
                    "step_type": "SimpleImputer",
                    "params": {"strategy": "mean"},
                    "inputs": ["loader"],
                },
                {
                    "node_id": "encoder",
                    "step_type": "OrdinalEncoder",
                    "params": {"columns": ["B"]},
                    "inputs": ["imputer"],
                },
                {
                    "node_id": "splitter",
                    "step_type": "feature_target_split",
                    "params": {"target_column": "target"},
                    "inputs": ["encoder"],
                },
                {
                    "node_id": "model",
                    "step_type": "model_training",
                    "params": {"algorithm": node_id, "target_column": "target"},
                    "inputs": ["splitter"],
                },
            ],
        }

        artifact_store = LocalArtifactStore(base_path=str(tmp_path))
        engine = PipelineEngine(artifact_store=artifact_store)
        nodes = [NodeConfig(**node) for node in pipeline_config["nodes"]]
        config = PipelineConfig(
            pipeline_id=pipeline_config["pipeline_id"],
            nodes=nodes,
            metadata=pipeline_config.get("metadata", {}),
        )

        try:
            result = engine.run(config, job_id="test_job")
            assert result.status == "success"
            node_res = result.node_results.get("model")
            assert node_res is not None
            assert node_res.status == "success"
        except Exception as e:
            pytest.fail(f"Model {node_id} failed: {str(e)}")


@pytest.mark.asyncio
async def test_splitters(sample_data, tmp_path):
    splitters = ["TrainTestSplitter"]

    for node_id in splitters:
        print(f"Testing splitter: {node_id}")

        pipeline_config = {
            "pipeline_id": f"test_{node_id}",
            "nodes": [
                {
                    "node_id": "loader",
                    "step_type": "data_loader",
                    "params": {"source_id": "csv", "path": sample_data},
                    "inputs": [],
                },
                {
                    "node_id": "splitter",
                    "step_type": "feature_target_split",
                    "params": {"target_column": "target"},
                    "inputs": ["loader"],
                },
                {
                    "node_id": "t_splitter",
                    "step_type": node_id,
                    "params": {"test_size": 0.2},
                    "inputs": ["splitter"],
                },
            ],
        }

        artifact_store = LocalArtifactStore(base_path=str(tmp_path))
        engine = PipelineEngine(artifact_store=artifact_store)
        nodes = [NodeConfig(**node) for node in pipeline_config["nodes"]]
        config = PipelineConfig(
            pipeline_id=pipeline_config["pipeline_id"],
            nodes=nodes,
            metadata=pipeline_config.get("metadata", {}),
        )

        try:
            result = engine.run(config, job_id="test_job")
            assert result.status == "success"
            node_res = result.node_results.get("t_splitter")
            assert node_res is not None
            assert node_res.status == "success"
        except Exception as e:
            pytest.fail(f"Splitter {node_id} failed: {str(e)}")
