"""Tests for Phase 2 parallel pipeline branch partitioning."""

import pytest

from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.execution.graph_utils import (
    partition_parallel_pipeline,
    _collect_ancestors,
)


def _make_node(node_id: str, step_type: str = "StandardScaler",
               inputs: list[str] | None = None,
               params: dict | None = None) -> NodeConfig:
    return NodeConfig(
        node_id=node_id,
        step_type=step_type,
        params=params or {},
        inputs=inputs or [],
    )


def _make_config(nodes: list[NodeConfig], pipeline_id: str = "test") -> PipelineConfig:
    return PipelineConfig(pipeline_id=pipeline_id, nodes=nodes)


# ---------- Helper Tests ----------


class TestCollectAncestors:
    def test_single_chain(self):
        nodes = [
            _make_node("A"),
            _make_node("B", inputs=["A"]),
            _make_node("C", inputs=["B"]),
        ]
        node_map = {n.node_id: n for n in nodes}
        ancestors = _collect_ancestors("C", node_map)
        assert ancestors == ["A", "B", "C"]

    def test_diamond(self):
        nodes = [
            _make_node("A"),
            _make_node("B", inputs=["A"]),
            _make_node("C", inputs=["A"]),
            _make_node("D", inputs=["B", "C"]),
        ]
        node_map = {n.node_id: n for n in nodes}
        ancestors = _collect_ancestors("D", node_map)
        assert set(ancestors) == {"A", "B", "C", "D"}
        # A must come before B,C; B,C must come before D
        assert ancestors.index("A") < ancestors.index("D")


# ---------- No-Partition Cases ----------


class TestNoPartition:
    def test_single_training_node_no_parallel(self):
        """Single training node without parallel mode → unchanged."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("scale", inputs=["ds"]),
            _make_node("train", "basic_training", inputs=["scale"],
                       params={"target_column": "y"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)
        assert len(result) == 1
        assert result[0] is config  # Same object, untouched

    def test_no_training_nodes(self):
        """Pipeline with no training nodes → unchanged."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("preview", "data_preview", inputs=["ds"]),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)
        assert len(result) == 1

    def test_single_training_parallel_but_one_input(self):
        """Parallel mode with only 1 input → nothing to split."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("train", "basic_training", inputs=["ds"],
                       params={"execution_mode": "parallel", "target_column": "y"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)
        assert len(result) == 1


# ---------- Case 1: Multiple Training Nodes ----------


class TestMultipleTerminals:
    def test_two_training_nodes_from_same_dataset(self):
        """Two training nodes → 2 sub-pipelines."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("scale", inputs=["ds"]),
            _make_node("encode", inputs=["ds"]),
            _make_node("trainA", "basic_training", inputs=["scale"],
                       params={"target_column": "y", "model_type": "rf"}),
            _make_node("trainB", "basic_training", inputs=["encode"],
                       params={"target_column": "y", "model_type": "xgb"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)

        assert len(result) == 2

        # Branch 0: ds → scale → trainA
        ids_0 = {n.node_id for n in result[0].nodes}
        assert ids_0 == {"ds", "scale", "trainA"}

        # Branch 1: ds → encode → trainB
        ids_1 = {n.node_id for n in result[1].nodes}
        assert ids_1 == {"ds", "encode", "trainB"}

    def test_shared_prefix_included_in_both(self):
        """Shared preprocessing nodes appear in both branches."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("clean", inputs=["ds"]),
            _make_node("scale", inputs=["clean"]),
            _make_node("pca", inputs=["clean"]),
            _make_node("trainA", "basic_training", inputs=["scale"],
                       params={"target_column": "y"}),
            _make_node("trainB", "advanced_tuning", inputs=["pca"],
                       params={"target_column": "y"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)

        assert len(result) == 2
        # Both branches include ds and clean
        for sub in result:
            ids = {n.node_id for n in sub.nodes}
            assert "ds" in ids
            assert "clean" in ids

    def test_pipeline_ids_include_branch_index(self):
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("trainA", "basic_training", inputs=["ds"],
                       params={"target_column": "y"}),
            _make_node("trainB", "basic_training", inputs=["ds"],
                       params={"target_column": "y"}),
        ]
        config = _make_config(nodes, pipeline_id="my_pipe")
        result = partition_parallel_pipeline(config)
        assert result[0].pipeline_id == "my_pipe__branch_0"
        assert result[1].pipeline_id == "my_pipe__branch_1"


# ---------- Case 2: Single Terminal, Parallel Mode ----------


class TestParallelMode:
    def test_parallel_splits_into_branches(self):
        """Single training node with execution_mode=parallel and 2 inputs."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("branchA", inputs=["ds"]),
            _make_node("branchB", inputs=["ds"]),
            _make_node("train", "basic_training", inputs=["branchA", "branchB"],
                       params={"execution_mode": "parallel", "target_column": "y"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)

        assert len(result) == 2

        # Branch 0: ds → branchA → train (single input)
        ids_0 = {n.node_id for n in result[0].nodes}
        assert ids_0 == {"ds", "branchA", "train"}
        train_0 = [n for n in result[0].nodes if n.node_id == "train"][0]
        assert train_0.inputs == ["branchA"]
        # execution_mode should be stripped so it doesn't re-trigger partitioning
        assert "execution_mode" not in train_0.params

        # Branch 1: ds → branchB → train (single input)
        ids_1 = {n.node_id for n in result[1].nodes}
        assert ids_1 == {"ds", "branchB", "train"}
        train_1 = [n for n in result[1].nodes if n.node_id == "train"][0]
        assert train_1.inputs == ["branchB"]

    def test_parallel_with_long_branches(self):
        """Full pipeline branches: each with encode+scale → parallel training."""
        nodes = [
            _make_node("ds", "data_loader"),
            # Branch A: drop cols → split → encode → scale
            _make_node("dropA", inputs=["ds"]),
            _make_node("splitA", inputs=["dropA"]),
            _make_node("encA", inputs=["splitA"]),
            _make_node("scaleA", inputs=["encA"]),
            # Branch B: impute → split → encode
            _make_node("impB", inputs=["ds"]),
            _make_node("splitB", inputs=["impB"]),
            _make_node("encB", inputs=["splitB"]),
            # Training
            _make_node("train", "basic_training", inputs=["scaleA", "encB"],
                       params={"execution_mode": "parallel", "target_column": "y"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)

        assert len(result) == 2

        ids_0 = {n.node_id for n in result[0].nodes}
        assert ids_0 == {"ds", "dropA", "splitA", "encA", "scaleA", "train"}

        ids_1 = {n.node_id for n in result[1].nodes}
        assert ids_1 == {"ds", "impB", "splitB", "encB", "train"}

    def test_metadata_contains_parent_id(self):
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("a", inputs=["ds"]),
            _make_node("b", inputs=["ds"]),
            _make_node("train", "basic_training", inputs=["a", "b"],
                       params={"execution_mode": "parallel", "target_column": "y"}),
        ]
        config = _make_config(nodes, pipeline_id="parent_123")
        result = partition_parallel_pipeline(config)
        for sub in result:
            assert sub.metadata["parent_pipeline_id"] == "parent_123"
            assert "branch_index" in sub.metadata

    def test_multi_terminal_with_parallel_mode_hybrid(self):
        """Two training nodes. One is normal (1 input), the other has
        execution_mode=parallel with 2 inputs.  Should produce 3 sub-pipelines:
        1 for the normal terminal + 2 for the parallel terminal's inputs."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("scaler", inputs=["ds"]),
            _make_node("pca", inputs=["scaler"]),
            _make_node("encoder", inputs=["scaler"]),
            # Normal training node — gets 1 sub-pipeline
            _make_node("train_rf", "basic_training", inputs=["pca"],
                       params={"model_type": "random_forest"}),
            # Parallel training node with 2 inputs — gets 2 sub-pipelines
            _make_node("train_xgb", "advanced_tuning", inputs=["pca", "encoder"],
                       params={"model_type": "xgboost",
                               "execution_mode": "parallel"}),
        ]
        config = _make_config(nodes, pipeline_id="hybrid_test")
        result = partition_parallel_pipeline(config)

        assert len(result) == 3

        # First sub-pipeline: normal train_rf with its ancestors
        rf_subs = [s for s in result if any(
            n.node_id == "train_rf" for n in s.nodes
        )]
        assert len(rf_subs) == 1

        # Two sub-pipelines for train_xgb (one per input)
        xgb_subs = [s for s in result if any(
            n.node_id == "train_xgb" for n in s.nodes
        )]
        assert len(xgb_subs) == 2

        # Each xgb sub should have only 1 input on the terminal node
        for sub in xgb_subs:
            train_nodes = [n for n in sub.nodes if n.node_id == "train_xgb"]
            assert len(train_nodes) == 1
            assert len(train_nodes[0].inputs) == 1

        # execution_mode should be stripped from the split terminal's params
        for sub in xgb_subs:
            train_node = next(n for n in sub.nodes if n.node_id == "train_xgb")
            assert "execution_mode" not in train_node.params

    def test_multi_terminal_no_parallel_stays_flat(self):
        """Two training nodes, neither has parallel mode. Should produce
        exactly 2 sub-pipelines (no further splitting)."""
        nodes = [
            _make_node("ds", "data_loader"),
            _make_node("a", inputs=["ds"]),
            _make_node("b", inputs=["ds"]),
            _make_node("train1", "basic_training", inputs=["a", "b"],
                       params={"model_type": "rf"}),
            _make_node("train2", "basic_training", inputs=["b"],
                       params={"model_type": "xgb"}),
        ]
        config = _make_config(nodes)
        result = partition_parallel_pipeline(config)
        assert len(result) == 2
