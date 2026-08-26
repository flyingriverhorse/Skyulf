"""Focused unit tests for ``backend.ml_pipeline._execution._cycle_validation``.

These build small, synthetic ``NodeConfig`` graphs directly (no dataset,
no execution) to exercise the pre-execution cycle guard in isolation —
cycles otherwise die late inside the node loop with a cryptic
"Artifact not found" error.
"""

import pytest

from backend.ml_pipeline._execution._cycle_validation import (
    PipelineCycleError,
    validate_no_cycles,
)
from backend.ml_pipeline._execution.schemas import NodeConfig


def _node(node_id: str, inputs: list[str]) -> NodeConfig:
    """Builds a minimal ``NodeConfig`` for graph-shape-only tests."""
    return NodeConfig(node_id=node_id, step_type="Noop", inputs=inputs)


def test_linear_chain_passes():
    nodes = [
        _node("load", []),
        _node("scale", ["load"]),
        _node("split", ["scale"]),
        _node("model", ["split"]),
    ]
    validate_no_cycles(nodes)  # must not raise


def test_diamond_passes():
    nodes = [
        _node("load", []),
        _node("left", ["load"]),
        _node("right", ["load"]),
        _node("merge", ["left", "right"]),
        _node("model", ["merge"]),
    ]
    validate_no_cycles(nodes)  # must not raise


def test_disconnected_subgraphs_pass():
    nodes = [
        _node("load_a", []),
        _node("model_a", ["load_a"]),
        _node("load_b", []),
        _node("model_b", ["load_b"]),
    ]
    validate_no_cycles(nodes)  # must not raise


def test_self_loop_raises():
    nodes = [_node("load", []), _node("model", ["model"])]
    with pytest.raises(PipelineCycleError, match="model"):
        validate_no_cycles(nodes)


def test_two_node_loop_raises():
    nodes = [_node("a", ["b"]), _node("b", ["a"])]
    with pytest.raises(PipelineCycleError):
        validate_no_cycles(nodes)


def test_longer_loop_raises_and_names_every_loop_node():
    nodes = [
        _node("load", []),
        _node("a", ["c"]),
        _node("b", ["a"]),
        _node("c", ["b"]),
    ]
    with pytest.raises(PipelineCycleError) as excinfo:
        validate_no_cycles(nodes)
    message = str(excinfo.value)
    for loop_node in ("a", "b", "c"):
        assert loop_node in message
    assert "load" not in message


def test_downstream_innocents_are_not_named():
    """Nodes fed by the cycle cannot run either, but they are not part of
    the loop — the message must point only at the loop itself."""
    nodes = [
        _node("a", ["b"]),
        _node("b", ["a"]),
        _node("downstream", ["b"]),
    ]
    with pytest.raises(PipelineCycleError) as excinfo:
        validate_no_cycles(nodes)
    message = str(excinfo.value)
    assert "a" in message and "b" in message
    assert "downstream" not in message


def test_cycle_error_is_value_error():
    """Callers catching ``ValueError`` (job/service layers) keep working."""
    assert issubclass(PipelineCycleError, ValueError)
