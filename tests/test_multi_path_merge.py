"""Tests for multi-path merge execution in the pipeline engine."""

import pandas as pd
import pytest

from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.constants import StepType
from backend.data.catalog import FileSystemCatalog
from skyulf.data.dataset import SplitDataset


class _FakeModel:
    """Module-level fake model for pickle compatibility."""
    def predict(self, X):
        pass


@pytest.fixture
def artifact_store(tmp_path):
    return LocalArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def catalog():
    return FileSystemCatalog()


@pytest.fixture
def engine(artifact_store, catalog):
    return PipelineEngine(artifact_store, catalog=catalog)


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6] * 10,
        "f2": [10, 20, 30, 40, 50, 60] * 10,
        "cat": ["a", "b", "a", "b", "a", "b"] * 10,
        "target": [0, 0, 0, 1, 1, 1] * 10,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- Unit Tests: _to_dataframe ---

class TestToDataframe:
    def test_from_dataframe(self, engine):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = engine._to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 2

    def test_from_split_dataset(self, engine):
        train = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        ds = SplitDataset(train=train, test=pd.DataFrame(), validation=None)
        result = engine._to_dataframe(ds)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_from_xy_tuple(self, engine):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        result = engine._to_dataframe((X, y), target_col="target")
        assert "target" in result.columns
        assert len(result) == 2

    def test_from_split_dataset_with_xy_tuple(self, engine):
        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        y = pd.Series([0, 1])
        ds = SplitDataset(train=(X, y), test=pd.DataFrame(), validation=None)
        result = engine._to_dataframe(ds, target_col="target")
        assert "target" in result.columns
        assert "a" in result.columns

    def test_invalid_type_raises(self, engine):
        with pytest.raises(TypeError, match="Cannot convert"):
            engine._to_dataframe("not a dataframe")


# --- Unit Tests: _resolve_all_inputs ---

class TestResolveAllInputs:
    def test_single_input(self, engine, artifact_store):
        df = pd.DataFrame({"a": [1, 2]})
        artifact_store.save("upstream_1", df)

        node = NodeConfig(
            node_id="test_node",
            step_type=StepType.BASIC_TRAINING,
            inputs=["upstream_1"],
        )
        # Need topo order
        engine._topo_order = {"upstream_1": 0, "test_node": 1}

        result = engine._resolve_all_inputs(node)
        assert len(result) == 1
        assert isinstance(result[0], pd.DataFrame)

    def test_multiple_inputs_ordered(self, engine, artifact_store):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        artifact_store.save("node_a", df1)
        artifact_store.save("node_b", df2)

        node = NodeConfig(
            node_id="test_node",
            step_type=StepType.BASIC_TRAINING,
            inputs=["node_b", "node_a"],  # Out of topo order
        )
        engine._topo_order = {"node_a": 0, "node_b": 1, "test_node": 2}

        result = engine._resolve_all_inputs(node)
        assert len(result) == 2
        # Should be sorted by topo order: node_a first, then node_b
        assert list(result[0].columns) == ["a"]
        assert list(result[1].columns) == ["b"]

    def test_no_inputs_raises(self, engine):
        node = NodeConfig(
            node_id="test_node",
            step_type=StepType.BASIC_TRAINING,
            inputs=[],
        )
        with pytest.raises(ValueError, match="has no inputs"):
            engine._resolve_all_inputs(node)


# --- Unit Tests: _merge_inputs ---

class TestMergeInputs:
    def test_single_input_passthrough(self, engine, artifact_store):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        artifact_store.save("up_1", df)

        node = NodeConfig(
            node_id="merge_test",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1"],
        )
        engine._topo_order = {"up_1": 0, "merge_test": 1}

        result = engine._merge_inputs(node, target_col="target")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_column_wise_merge(self, engine, artifact_store):
        """Same row count, different columns → column concat."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"b": [10, 20, 30], "c": [100, 200, 300]})
        artifact_store.save("scaler", df1)
        artifact_store.save("encoder", df2)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["scaler", "encoder"],
        )
        engine._topo_order = {"scaler": 0, "encoder": 1, "train": 2}

        result = engine._merge_inputs(node, target_col="target")
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"a", "target", "b", "c"}
        assert len(result) == 3

    def test_column_wise_dedup(self, engine, artifact_store):
        """Same row count, overlapping columns → deduplicates."""
        df1 = pd.DataFrame({"a": [1, 2], "shared": [10, 20], "target": [0, 1]})
        df2 = pd.DataFrame({"b": [3, 4], "shared": [10, 20]})
        artifact_store.save("up_1", df1)
        artifact_store.save("up_2", df2)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "train": 2}

        result = engine._merge_inputs(node, target_col="target")
        assert list(result.columns).count("shared") == 1  # No duplicate
        assert "b" in result.columns
        assert len(result) == 2

    def test_row_wise_merge(self, engine, artifact_store):
        """Different row counts, same columns → row concat."""
        df1 = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        df2 = pd.DataFrame({"a": [3, 4, 5], "target": [1, 0, 1]})
        artifact_store.save("up_1", df1)
        artifact_store.save("up_2", df2)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "train": 2}

        result = engine._merge_inputs(node, target_col="target")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ["a", "target"]

    def test_row_wise_drops_non_shared_columns(self, engine, artifact_store):
        """Different rows, partially overlapping columns → keep only shared."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
        df2 = pd.DataFrame({"a": [5, 6, 7], "target": [1, 0, 1]})
        artifact_store.save("up_1", df1)
        artifact_store.save("up_2", df2)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "train": 2}

        result = engine._merge_inputs(node, target_col="target")
        assert len(result) == 5
        assert "b" not in result.columns  # Not shared
        assert "a" in result.columns

    def test_model_artifact_raises(self, engine, artifact_store):
        """Merging a model object should raise a clear error."""
        artifact_store.save("up_1", pd.DataFrame({"a": [1]}))
        artifact_store.save("up_2", _FakeModel())

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "train": 2}

        with pytest.raises(ValueError, match="Model object"):
            engine._merge_inputs(node, target_col="target")

    def test_no_common_columns_raises(self, engine, artifact_store):
        """Different rows + no shared columns → error."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4, 5]})
        artifact_store.save("up_1", df1)
        artifact_store.save("up_2", df2)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "train": 2}

        with pytest.raises(ValueError, match="no common columns"):
            engine._merge_inputs(node, target_col="target")

    def test_three_inputs_column_wise(self, engine, artifact_store):
        """Three upstream nodes, same rows, disjoint columns."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        df3 = pd.DataFrame({"c": [7, 8, 9], "target": [0, 1, 0]})
        artifact_store.save("up_1", df1)
        artifact_store.save("up_2", df2)
        artifact_store.save("up_3", df3)

        node = NodeConfig(
            node_id="train",
            step_type=StepType.BASIC_TRAINING,
            inputs=["up_1", "up_2", "up_3"],
        )
        engine._topo_order = {"up_1": 0, "up_2": 1, "up_3": 2, "train": 3}

        result = engine._merge_inputs(node, target_col="target")
        assert set(result.columns) == {"a", "b", "c", "target"}
        assert len(result) == 3


# --- Integration Test: Full Pipeline with Multi-Input Training ---

class TestMultiPathPipeline:
    def test_forked_pipeline_basic_training(self, sample_csv, tmp_path):
        """
        Dataset → [Scaler branch (f1,f2), Encoder branch (cat)] → Training
        Both branches feed into the same training node.
        The scaler outputs all columns (scaled f1 & f2 + cat + target).
        The encoder outputs all columns (encoded cat + f1 + f2 + target).
        Column-wise merge deduplicates overlapping columns.
        """
        artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
        catalog = FileSystemCatalog()

        # Build a simpler pipeline: two numeric transforms → training
        # This avoids the issue of raw string columns surviving merge.
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6] * 10,
            "f2": [10, 20, 30, 40, 50, 60] * 10,
            "target": [0, 0, 0, 1, 1, 1] * 10,
        })
        csv_path = tmp_path / "numeric_data.csv"
        df.to_csv(csv_path, index=False)

        config = PipelineConfig(
            pipeline_id="multi_path_test",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": str(csv_path)},
                ),
                NodeConfig(
                    node_id="scaler",
                    step_type="StandardScaler",
                    inputs=["data"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler2",
                    step_type="MinMaxScaler",
                    inputs=["data"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="training",
                    step_type=StepType.BASIC_TRAINING,
                    inputs=["scaler", "scaler2"],
                    params={
                        "target_column": "target",
                        "algorithm": "logistic_regression",
                        "hyperparameters": {"C": 1.0},
                        "evaluate": True,
                        "cv_enabled": False,
                    },
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", (
            f"Pipeline failed: "
            f"{[(nid, nr.error) for nid, nr in result.node_results.items() if nr.status == 'failed']}"
        )
        assert result.node_results["training"].status == "success"
        assert artifact_store.exists("training")

    def test_single_input_still_works(self, tmp_path):
        """Ensure single-input pipelines are not broken by the merge logic."""
        artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
        catalog = FileSystemCatalog()

        df = pd.DataFrame({
            "f1": [1, 2, 3, 4, 5, 6] * 10,
            "f2": [10, 20, 30, 40, 50, 60] * 10,
            "target": [0, 0, 0, 1, 1, 1] * 10,
        })
        csv_path = tmp_path / "numeric_data.csv"
        df.to_csv(csv_path, index=False)

        config = PipelineConfig(
            pipeline_id="single_path_test",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": str(csv_path)},
                ),
                NodeConfig(
                    node_id="split",
                    step_type="TrainTestSplitter",
                    inputs=["data"],
                    params={"target_column": "target"},
                ),
                NodeConfig(
                    node_id="training",
                    step_type=StepType.BASIC_TRAINING,
                    inputs=["split"],
                    params={
                        "target_column": "target",
                        "algorithm": "logistic_regression",
                        "hyperparameters": {"C": 1.0},
                        "evaluate": True,
                        "cv_enabled": False,
                    },
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success"
        assert result.node_results["training"].status == "success"
