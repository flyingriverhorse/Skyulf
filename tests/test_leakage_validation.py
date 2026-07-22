"""Focused unit tests for
``backend.ml_pipeline._execution._leakage_validation``.

These build small, synthetic ``NodeConfig`` graphs directly (no dataset,
no execution) to exercise the pre-execution leakage guard in isolation —
this is the fastest, most reliable way to *trigger* the
"Data leakage risk" ``ValueError`` deliberately, without needing a real
dataset or a full pipeline run.
"""

import pytest

from backend.ml_pipeline._execution._leakage_validation import (
    validate_no_preprocessing_before_split,
)
from backend.ml_pipeline._execution.schemas import NodeConfig


def _node(node_id: str, step_type: str, inputs: list[str]) -> NodeConfig:
    """Builds a minimal ``NodeConfig`` for graph-shape-only tests."""
    return NodeConfig(node_id=node_id, step_type=step_type, params={}, inputs=inputs)


def test_raises_when_scaler_precedes_splitter():
    """A StandardScaler wired before the TrainTestSplitter must be blocked."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("scale", "StandardScaler", ["load"]),
        _node("split", "TrainTestSplitter", ["scale"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="Data leakage risk"):
        validate_no_preprocessing_before_split(nodes)


def test_allows_scaler_after_splitter():
    """The same node, moved after the split, is the correct/safe order."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("split", "TrainTestSplitter", ["load"]),
        _node("scale", "StandardScaler", ["split"]),
        _node("model", "LogisticRegression", ["scale"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_noop_when_no_splitter_in_graph():
    """Inference-only pipelines (no train/test boundary) are never flagged."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("impute", "SimpleImputer", ["load"]),
        _node("scale", "StandardScaler", ["impute"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_feature_target_split_does_not_trigger_leakage():
    """FeatureTargetSplitter only separates X/y - no train/test boundary."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("encode", "OneHotEncoder", ["load"]),
        _node("split_xy", "feature_target_split", ["encode"]),
        _node("model", "LogisticRegression", ["split_xy"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_stateless_nodes_before_splitter_are_allowed():
    """Rule-based/stateless nodes (fixed bounds, hashing) never leak."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("bounds", "ManualBounds", ["load"]),
        _node("hash", "HashEncoder", ["bounds"]),
        _node("split", "TrainTestSplitter", ["hash"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    validate_no_preprocessing_before_split(nodes)  # must not raise


def test_raises_for_indirect_ancestor_through_branching_graph():
    """The leaking node need not be directly wired to the splitter -
    any path that reaches a splitter downstream counts."""
    nodes = [
        _node("load", "DataLoader", []),
        _node("impute", "SimpleImputer", ["load"]),
        _node("clean", "ValueReplacement", ["impute"]),
        _node("split", "TrainTestSplitter", ["clean"]),
        _node("model", "LogisticRegression", ["split"]),
    ]
    with pytest.raises(ValueError, match="'impute'"):
        validate_no_preprocessing_before_split(nodes)
