"""Executable examples for column-level precedence in the canvas guide."""

import pandas as pd
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.data.dataset import SplitDataset


@pytest.fixture
def merge_engine(tmp_path, monkeypatch):
    """Use real graph resolution and temporary artifacts without external services."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas")
    return PipelineEngine(
        LocalArtifactStore(str(tmp_path / "merge-artifacts")),
        catalog=FileSystemCatalog(),
    )


def _merge(engine, baseline, branch_a, branch_b, strategy):
    """Exercise the production fan-in path with two sibling input artifacts."""
    node = NodeConfig(
        node_id="merged",
        step_type="data_preview",
        inputs=["a", "b"],
        params={} if strategy is None else {"_merge_strategy": strategy},
    )
    engine._node_configs = {
        "source": NodeConfig(node_id="source", step_type="data_loader"),
        "a": NodeConfig(node_id="a", step_type="MissingIndicator", inputs=["source"]),
        "b": NodeConfig(node_id="b", step_type="MissingIndicator", inputs=["source"]),
        "merged": node,
    }
    for node_id, artifact in (("source", baseline), ("a", branch_a), ("b", branch_b)):
        engine.artifact_store.save(node_id, artifact)
    return engine._merge_inputs(node, target_col="target")


def _payload(kind, frames):
    """Wrap literal branch outputs in the same shapes used by the backend."""
    if kind == "frame":
        return frames["train"]
    targets = {"train": [0, 1], "test": [1], "validation": [0]}
    parts = {
        part: (frame, pd.Series(targets[part], name="target")) if kind == "split_xy" else frame
        for part, frame in frames.items()
    }
    return SplitDataset(**parts)


@pytest.mark.parametrize("kind", ["frame", "split_frame", "split_xy"])
@pytest.mark.parametrize(
    ("strategy", "expected_ages"),
    [
        ("first_wins", {"train": [20, 40], "test": [60], "validation": [80]}),
        ("last_wins", {"train": [0.4, 0.8], "test": [1.2], "validation": [1.6]}),
        (None, {"train": [0.4, 0.8], "test": [1.2], "validation": [1.6]}),
    ],
)
def test_guide_example_keeps_both_distinct_features(merge_engine, kind, strategy, expected_ages):
    """Picking an age version must not discard the other branch's unique feature."""
    baseline = {
        "train": pd.DataFrame({"age": [10, 30]}),
        "test": pd.DataFrame({"age": [50]}),
        "validation": pd.DataFrame({"age": [70]}),
    }
    branch_a = {
        "train": pd.DataFrame({"age": [20, 40], "income": [900, 1100]}),
        "test": pd.DataFrame({"age": [60], "income": [1500]}),
        "validation": pd.DataFrame({"age": [80], "income": [2000]}),
    }
    branch_b = {
        "train": pd.DataFrame({"age": [0.4, 0.8], "city_code": [2, 3]}),
        "test": pd.DataFrame({"age": [1.2], "city_code": [4]}),
        "validation": pd.DataFrame({"age": [1.6], "city_code": [5]}),
    }
    result = _merge(
        merge_engine,
        _payload(kind, baseline),
        _payload(kind, branch_a),
        _payload(kind, branch_b),
        strategy,
    )
    incomes = {"train": [900, 1100], "test": [1500], "validation": [2000]}
    cities = {"train": [2, 3], "test": [4], "validation": [5]}
    targets = {"train": [0, 1], "test": [1], "validation": [0]}
    if kind != "frame":
        assert isinstance(result, SplitDataset)
    for part in ["train"] if kind == "frame" else ["train", "test", "validation"]:
        actual = result if kind == "frame" else getattr(result, part)
        if kind == "split_xy":
            assert isinstance(actual, tuple)
            actual, target = actual
            assert target.tolist() == targets[part]
        expected = pd.DataFrame(
            {"age": expected_ages[part], "income": incomes[part], "city_code": cities[part]}
        )
        pd.testing.assert_frame_equal(actual, expected, check_like=True)


@pytest.mark.parametrize(
    ("strategy", "age", "income"),
    [("first_wins", 20, 900), ("last_wins", 0.4, 300)],
)
def test_strategy_applies_to_every_contested_column(merge_engine, strategy, age, income):
    """Precedence is a node-wide policy, not a special case for the age column."""
    result = _merge(
        merge_engine,
        pd.DataFrame({"age": [10], "income": [100]}),
        pd.DataFrame({"age": [20], "income": [900], "only_a": [1]}),
        pd.DataFrame({"age": [0.4], "income": [300], "only_b": [2]}),
        strategy,
    )
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"age": [age], "income": [income], "only_a": [1], "only_b": [2]}),
        check_like=True,
    )


@pytest.mark.parametrize(
    ("split", "strategy", "expected_age"),
    [
        (False, "first_wins", [20, 40]),
        (False, "last_wins", [20, 40]),
        (True, "first_wins", [20, 40]),
        (True, "last_wins", [10, 30]),
    ],
)
def test_single_modifier_ownership_requires_frame_baseline(
    merge_engine, split, strategy, expected_age
):
    """A split baseline cannot protect an earlier modifier from a later bypass branch."""
    baseline = pd.DataFrame({"age": [10, 30]})
    branch_a = pd.DataFrame({"age": [20, 40], "income": [900, 1100]})
    branch_b = pd.DataFrame({"age": [10, 30], "city_code": [2, 3]})
    if split:
        baseline = SplitDataset(train=baseline, test=baseline.copy(), validation=None)
        branch_a = SplitDataset(train=branch_a, test=branch_a.copy(), validation=None)
        branch_b = SplitDataset(train=branch_b, test=branch_b.copy(), validation=None)
    result = _merge(merge_engine, baseline, branch_a, branch_b, strategy)
    actual = result.train if split else result
    pd.testing.assert_frame_equal(
        actual,
        pd.DataFrame({"age": expected_age, "income": [900, 1100], "city_code": [2, 3]}),
        check_like=True,
    )
