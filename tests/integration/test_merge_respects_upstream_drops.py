"""A merge must not resurrect columns an upstream Drop Columns node removed.

Reproduces a real pipeline that silently trained on a dropped column: a dataset
fans out into a no-op Transformation branch and a Drop Columns([Id]) branch, and
both feed the same Missing Indicator node. The column-wise merge restored ``Id``
from the untouched branch, so the model was fit on it and inference then demanded
an ``Id`` value the user believed had been dropped.
"""

import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.data.dataset import SplitDataset


@pytest.fixture
def engine(tmp_path):
    return PipelineEngine(
        LocalArtifactStore(str(tmp_path / "artifacts")), catalog=FileSystemCatalog()
    )


def _wire_drop_and_passthrough(engine, merge_node_id="merge", dropped=("Id",)):
    """Build dataset -> {no-op branch, drop branch} -> merge and return the merge node."""
    nodes = {
        "dataset": NodeConfig(node_id="dataset", step_type="data_loader", params={}, inputs=[]),
        "passthrough": NodeConfig(
            node_id="passthrough",
            step_type="GeneralTransformation",
            params={"transformations": []},
            inputs=["dataset"],
        ),
        "drop": NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": list(dropped), "missing_threshold": 0},
            inputs=["dataset"],
        ),
        merge_node_id: NodeConfig(
            node_id=merge_node_id,
            step_type="MissingIndicator",
            params={},
            inputs=["drop", "passthrough"],
        ),
    }
    engine._node_configs = nodes
    return nodes[merge_node_id]


def test_merge_does_not_restore_column_dropped_upstream(engine, monkeypatch):
    merge_node = _wire_drop_and_passthrough(engine)
    dropped_branch = pd.DataFrame({"SepalLengthCm": [1.0, 2.0], "Species": ["a", "b"]})
    full_branch = pd.DataFrame({"Id": [1, 2], "SepalLengthCm": [1.0, 2.0], "Species": ["a", "b"]})
    monkeypatch.setattr(engine, "_resolve_all_inputs", lambda node: [dropped_branch, full_branch])

    merged = engine._merge_inputs(merge_node)

    assert "Id" not in merged.columns
    assert list(merged.columns) == ["SepalLengthCm", "Species"]


def test_merge_keeps_columns_no_upstream_node_dropped(engine, monkeypatch):
    merge_node = _wire_drop_and_passthrough(engine, dropped=("Id",))
    left = pd.DataFrame({"SepalLengthCm": [1.0, 2.0]})
    right = pd.DataFrame({"PetalLengthCm": [3.0, 4.0]})
    monkeypatch.setattr(engine, "_resolve_all_inputs", lambda node: [left, right])

    merged = engine._merge_inputs(merge_node)

    assert list(merged.columns) == ["SepalLengthCm", "PetalLengthCm"]


def test_split_dataset_merge_also_drops_upstream_dropped_column(engine, monkeypatch):
    merge_node = _wire_drop_and_passthrough(engine)
    dropped_branch = SplitDataset(
        train=pd.DataFrame({"SepalLengthCm": [1.0, 2.0]}),
        test=pd.DataFrame({"SepalLengthCm": [3.0]}),
        validation=None,
    )
    full_branch = SplitDataset(
        train=pd.DataFrame({"Id": [1, 2], "SepalLengthCm": [1.0, 2.0]}),
        test=pd.DataFrame({"Id": [3], "SepalLengthCm": [3.0]}),
        validation=None,
    )
    monkeypatch.setattr(engine, "_resolve_all_inputs", lambda node: [dropped_branch, full_branch])

    merged = engine._merge_inputs(merge_node)

    assert "Id" not in merged.train.columns
    assert "Id" not in merged.test.columns


def test_single_input_is_untouched_by_drop_enforcement(engine, monkeypatch):
    merge_node = _wire_drop_and_passthrough(engine)
    only = pd.DataFrame({"Id": [1, 2], "SepalLengthCm": [1.0, 2.0]})
    monkeypatch.setattr(engine, "_resolve_all_inputs", lambda node: [only])

    assert engine._merge_inputs(merge_node) is only
