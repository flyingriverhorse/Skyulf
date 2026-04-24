"""Regression tests for the merge fix.

Verifies that multi-input merging now works for **all** node types — not just
training/tuning. Covers:

* Preview node merging two preprocessing branches from a Dataset.
* Preview node merging two branches from a TrainTestSplitter (SplitDataset).
* A preprocessing node (transformer) that has two parents.
* Training that merges SplitDatasets and preserves the test split for
  evaluation (this used to be silently dropped).
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from skyulf.data.dataset import SplitDataset

# --- Fixtures -------------------------------------------------------------


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
def numeric_csv(tmp_path):
    """Small balanced numeric dataset usable for both preview and training."""
    df = pd.DataFrame(
        {
            "f1": list(range(1, 31)),
            "f2": [v * 10 for v in range(1, 31)],
            "target": [0, 1] * 15,
        }
    )
    path = tmp_path / "numeric.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- Unit tests for the new merge surface ---------------------------------


class TestSplitDatasetMerge:
    """`_merge_inputs` must keep test/validation portions when all inputs are
    SplitDatasets — previously they were silently flattened into train-only."""

    def _make_node(self, engine, store, inputs):
        for nid, art in inputs:
            store.save(nid, art)
        node = NodeConfig(
            node_id="downstream",
            step_type="StandardScaler",
            inputs=[nid for nid, _ in inputs],
        )
        engine._topo_order = {nid: i for i, (nid, _) in enumerate(inputs)}
        engine._topo_order["downstream"] = len(inputs)
        return node

    def test_two_split_datasets_merge_train_and_test(self, engine, artifact_store):
        train_a = pd.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
        test_a = pd.DataFrame({"f1": [10, 11], "target": [1, 0]})
        train_b = pd.DataFrame({"f2": [4, 5, 6]})
        test_b = pd.DataFrame({"f2": [40, 41]})

        node = self._make_node(
            engine,
            artifact_store,
            [
                ("branch_a", SplitDataset(train=train_a, test=test_a, validation=None)),
                ("branch_b", SplitDataset(train=train_b, test=test_b, validation=None)),
            ],
        )

        result = engine._merge_inputs(node, target_col="target")

        assert isinstance(
            result, SplitDataset
        ), "Merging two SplitDatasets must return a SplitDataset, not a flat frame."
        assert set(result.train.columns) == {"f1", "f2", "target"}
        assert set(result.test.columns) == {"f1", "f2", "target"}
        assert len(result.train) == 3
        assert len(result.test) == 2

    def test_split_with_validation_is_preserved(self, engine, artifact_store):
        sd_a = SplitDataset(
            train=pd.DataFrame({"a": [1, 2]}),
            test=pd.DataFrame({"a": [3]}),
            validation=pd.DataFrame({"a": [4, 5]}),
        )
        sd_b = SplitDataset(
            train=pd.DataFrame({"b": [10, 20]}),
            test=pd.DataFrame({"b": [30]}),
            validation=pd.DataFrame({"b": [40, 50]}),
        )
        node = self._make_node(engine, artifact_store, [("a", sd_a), ("b", sd_b)])

        result = engine._merge_inputs(node)

        assert isinstance(result, SplitDataset)
        assert result.validation is not None
        assert set(result.validation.columns) == {"a", "b"}
        assert len(result.validation) == 2

    def test_mixed_split_and_dataframe_falls_back_to_frame(self, engine, artifact_store):
        sd = SplitDataset(
            train=pd.DataFrame({"a": [1, 2]}),
            test=pd.DataFrame({"a": [3]}),
            validation=None,
        )
        df = pd.DataFrame({"b": [9, 8]})
        node = self._make_node(engine, artifact_store, [("sd", sd), ("df", df)])

        result = engine._merge_inputs(node)

        # Mixed inputs collapse to a DataFrame and emit a warning in the log.
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"a", "b"}

    def test_xy_tuple_shape_preserved_when_merging(self, engine, artifact_store):
        """``TrainTestSplitter`` outputs ``(X, y)`` tuples. Merging two
        SplitDatasets that both wrap tuples must return a SplitDataset whose
        train/test are still tuples — otherwise the frontend collapses
        ``train_X``/``train_y`` into a single ``train`` tab."""
        y_train = pd.Series([0, 1, 0], name="target")
        y_test = pd.Series([1, 0], name="target")

        sd_a = SplitDataset(
            train=(pd.DataFrame({"f1": [1, 2, 3]}), y_train),
            test=(pd.DataFrame({"f1": [10, 11]}), y_test),
            validation=None,
        )
        sd_b = SplitDataset(
            train=(pd.DataFrame({"f2": [4, 5, 6]}), y_train),
            test=(pd.DataFrame({"f2": [40, 41]}), y_test),
            validation=None,
        )
        node = self._make_node(engine, artifact_store, [("a", sd_a), ("b", sd_b)])

        result = engine._merge_inputs(node)

        assert isinstance(result, SplitDataset)
        assert (
            isinstance(result.train, tuple) and len(result.train) == 2
        ), "Merging tuple-shaped SplitDatasets must keep the (X, y) tuple."
        assert isinstance(result.test, tuple) and len(result.test) == 2
        x_train, y_train_out = result.train
        x_test, y_test_out = result.test
        assert set(x_train.columns) == {"f1", "f2"}
        assert set(x_test.columns) == {"f1", "f2"}
        # y is identical across branches (same Splitter origin) → first kept.
        assert list(y_train_out) == [0, 1, 0]
        assert list(y_test_out) == [1, 0]


# --- Integration tests: Preview merges (the user's reported bug) -----------


class TestPreviewMerge:
    """Run Preview must merge multi-input nodes. Before the fix, a preview
    node with two parents silently kept only the first edge."""

    def test_preview_merges_two_branches_from_dataset(self, numeric_csv, artifact_store, catalog):
        """Dataset → StandardScaler(f1)
        ↘ MinMaxScaler(f2)
                ↘ Preview (must merge both)"""
        config = PipelineConfig(
            pipeline_id="preview_dataset_merge",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": numeric_csv},
                ),
                NodeConfig(
                    node_id="scaler_a",
                    step_type="StandardScaler",
                    inputs=["data"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler_b",
                    step_type="MinMaxScaler",
                    inputs=["data"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="preview",
                    step_type="data_preview",
                    inputs=["scaler_a", "scaler_b"],
                    params={},
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", _failure_summary(result)

        # Preview saves a metadata dict; the merged frame is summarised
        # under data_summary['full'].
        preview_info = artifact_store.load("preview")
        assert preview_info["operation_mode"] == "fit_transform"
        full = preview_info["data_summary"]["full"]
        # Both upstream branches contributed columns to the preview.
        assert {"f1", "f2", "target"}.issubset(set(full["columns"]))
        # No row loss across the merge.
        assert full["shape"][0] == 30

    def test_preview_merges_two_branches_from_split(self, numeric_csv, artifact_store, catalog):
        """Dataset → Splitter → StandardScaler
        ↘ MinMaxScaler
                  ↘ Preview"""
        config = PipelineConfig(
            pipeline_id="preview_split_merge",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": numeric_csv},
                ),
                NodeConfig(
                    node_id="split",
                    step_type="TrainTestSplitter",
                    inputs=["data"],
                    params={
                        "target_column": "target",
                        "test_size": 0.3,
                        "random_state": 0,
                        "stratify": True,
                    },
                ),
                NodeConfig(
                    node_id="scaler_a",
                    step_type="StandardScaler",
                    inputs=["split"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler_b",
                    step_type="MinMaxScaler",
                    inputs=["split"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="preview",
                    step_type="data_preview",
                    inputs=["scaler_a", "scaler_b"],
                    params={},
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", _failure_summary(result)

        preview_info = artifact_store.load("preview")
        # Both branches operate on a SplitDataset → merge keeps train AND test.
        # With ``target_column`` configured on the splitter, train/test are
        # ``(X, y)`` tuples; the preview surfaces X-only column lists.
        summary = preview_info["data_summary"]
        assert (
            "train" in summary and "test" in summary
        ), "Test split was dropped during merge; preview only sees train."
        # X frames carry the feature columns from both branches; the target
        # lives on the y side of the tuple, not in X.
        assert {"f1", "f2"}.issubset(set(summary["train"]["columns"]))
        assert {"f1", "f2"}.issubset(set(summary["test"]["columns"]))
        assert "target" not in summary["train"]["columns"]
        # No data loss across the two branches.
        assert summary["train"]["shape"][0] + summary["test"]["shape"][0] == 30


class TestPreprocessingMerge:
    """A preprocessing transformer (not a model) with two parents must merge
    rather than silently keep the first input."""

    def test_transformer_merges_two_parents(self, numeric_csv, artifact_store, catalog):
        """Dataset → StandardScaler(f1)
        ↘ MinMaxScaler(f2)
                ↘ Deduplicate (multi-input transformer)"""
        config = PipelineConfig(
            pipeline_id="transformer_merge",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": numeric_csv},
                ),
                NodeConfig(
                    node_id="scaler_a",
                    step_type="StandardScaler",
                    inputs=["data"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler_b",
                    step_type="MinMaxScaler",
                    inputs=["data"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="dedup",
                    step_type="Deduplicate",
                    inputs=["scaler_a", "scaler_b"],
                    params={},
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", _failure_summary(result)

        out = artifact_store.load("dedup")
        assert isinstance(out, pd.DataFrame)
        # Both upstream column sets must reach the downstream node.
        assert {"f1", "f2", "target"}.issubset(set(out.columns))


# --- Integration tests: Training merge (test split must survive) -----------


class TestTrainingMerge:
    def test_training_after_split_merge_keeps_test_metrics(
        self, numeric_csv, artifact_store, catalog
    ):
        """Dataset → Splitter → StandardScaler
                                  ↘ MinMaxScaler
                                            ↘ Training (merge SplitDatasets)

        Before the fix the test halves were dropped at merge time, so
        ``test_*`` metrics never appeared in the result. This is the
        regression check.
        """
        config = PipelineConfig(
            pipeline_id="training_split_merge",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": numeric_csv},
                ),
                NodeConfig(
                    node_id="split",
                    step_type="TrainTestSplitter",
                    inputs=["data"],
                    params={
                        "target_column": "target",
                        "test_size": 0.3,
                        "random_state": 0,
                        "stratify": True,
                    },
                ),
                NodeConfig(
                    node_id="scaler_a",
                    step_type="StandardScaler",
                    inputs=["split"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler_b",
                    step_type="MinMaxScaler",
                    inputs=["split"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="training",
                    step_type=StepType.BASIC_TRAINING,
                    inputs=["scaler_a", "scaler_b"],
                    params={
                        "target_column": "target",
                        "algorithm": "logistic_regression",
                        "hyperparameters": {"C": 1.0, "max_iter": 200},
                        "evaluate": True,
                        "cv_enabled": False,
                    },
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", _failure_summary(result)
        metrics = result.node_results["training"].metrics
        # The held-out test split survived the merge → test metrics exist.
        test_keys = [k for k in metrics if k.startswith("test_")]
        assert test_keys, (
            "Expected at least one test_* metric to survive multi-branch "
            f"SplitDataset merge, got: {sorted(metrics)}"
        )

    def test_training_merges_two_dataset_branches(self, numeric_csv, artifact_store, catalog):
        """Sanity check: two parallel preprocessing chains from raw dataset
        still feed into a training node (existing behaviour, not regressed)."""
        config = PipelineConfig(
            pipeline_id="training_dataset_merge",
            nodes=[
                NodeConfig(
                    node_id="data",
                    step_type=StepType.DATA_LOADER,
                    params={"path": numeric_csv},
                ),
                NodeConfig(
                    node_id="scaler_a",
                    step_type="StandardScaler",
                    inputs=["data"],
                    params={"columns": ["f1"]},
                ),
                NodeConfig(
                    node_id="scaler_b",
                    step_type="MinMaxScaler",
                    inputs=["data"],
                    params={"columns": ["f2"]},
                ),
                NodeConfig(
                    node_id="training",
                    step_type=StepType.BASIC_TRAINING,
                    inputs=["scaler_a", "scaler_b"],
                    params={
                        "target_column": "target",
                        "algorithm": "logistic_regression",
                        "hyperparameters": {"C": 1.0, "max_iter": 200},
                        "evaluate": True,
                        "cv_enabled": False,
                    },
                ),
            ],
        )

        engine = PipelineEngine(artifact_store, catalog=catalog)
        result = engine.run(config)

        assert result.status == "success", _failure_summary(result)
        assert artifact_store.exists("training")


class TestMergeStrategyOverride:
    """``_merge_strategy`` in node params switches column-overlap semantics."""

    def _merge(self, engine, frames, strategy):
        node_id = "merge_node"
        engine._node_configs = {
            node_id: NodeConfig(
                node_id=node_id,
                step_type="StandardScaler",
                inputs=[f"in_{i}" for i in range(len(frames))],
                params={"_merge_strategy": strategy} if strategy else {},
            )
        }
        return engine._merge_frames(frames, node_id)

    def test_last_wins_is_default(self, engine):
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"x": [10, 20, 30]})
        out = self._merge(engine, [a, b], strategy=None)
        assert list(out["x"]) == [10, 20, 30], "default must be last_wins"

    def test_first_wins_keeps_earlier_input(self, engine):
        a = pd.DataFrame({"x": [1, 2, 3]})
        b = pd.DataFrame({"x": [10, 20, 30]})
        out = self._merge(engine, [a, b], strategy="first_wins")
        assert list(out["x"]) == [1, 2, 3], "first_wins must keep the earlier input"

    def test_unknown_strategy_falls_back_to_last_wins(self, engine):
        a = pd.DataFrame({"x": [1]})
        b = pd.DataFrame({"x": [99]})
        out = self._merge(engine, [a, b], strategy="nonsense")
        assert list(out["x"]) == [99]


# --- helpers ---------------------------------------------------------------


def _failure_summary(result) -> str:
    return (
        "; ".join(
            f"{nid}: {nr.error}" for nid, nr in result.node_results.items() if nr.status == "failed"
        )
        or "no failures recorded"
    )
