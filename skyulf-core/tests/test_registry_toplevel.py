"""Tests for skyulf.registry (NodeRegistry — top-level, not engines.registry)."""

import logging

import pytest

from skyulf.core.meta.decorators import node_meta
from skyulf.registry import NodeRegistry


class _DummyApplier:
    """Minimal stand-in applier class for registry tests."""


@node_meta(id="dummy_node", name="Dummy", category="Test", description="A dummy node.", params={})
class _DummyCalculatorWithMeta:
    """Calculator carrying __node_meta__ via the decorator."""


class _DummyCalculatorNoMeta:
    """Calculator with no __node_meta__ attribute at all."""


def test_register_and_get_calculator_round_trip():
    """A registered calculator should be retrievable by its registered name."""
    NodeRegistry.register("dummy_node_rt", _DummyApplier)(_DummyCalculatorWithMeta)
    assert NodeRegistry.get_calculator("dummy_node_rt") is _DummyCalculatorWithMeta


def test_register_and_get_applier_round_trip():
    """A registered applier should be retrievable by its registered name."""
    NodeRegistry.register("dummy_node_rt2", _DummyApplier)(_DummyCalculatorWithMeta)
    assert NodeRegistry.get_applier("dummy_node_rt2") is _DummyApplier


def test_register_extracts_metadata_from_node_meta_decorator():
    """When no explicit metadata dict is passed, __node_meta__ should populate the registry."""
    NodeRegistry.register("dummy_node_meta", _DummyApplier)(_DummyCalculatorWithMeta)
    meta = NodeRegistry.get_all_metadata()["dummy_node_meta"]
    assert meta["id"] == "dummy_node"
    assert meta["name"] == "Dummy"
    assert meta["category"] == "Test"


def test_register_with_explicit_metadata_overrides_node_meta():
    """Explicitly passed metadata should take priority over __node_meta__."""
    explicit = {
        "id": "explicit",
        "name": "Explicit",
        "category": "X",
        "description": "d",
        "params": {},
    }
    NodeRegistry.register("dummy_node_explicit", _DummyApplier, metadata=explicit)(
        _DummyCalculatorWithMeta
    )
    assert NodeRegistry.get_all_metadata()["dummy_node_explicit"] == explicit


def test_register_without_metadata_or_node_meta_has_no_metadata_entry():
    """A calculator with neither explicit metadata nor __node_meta__ should not appear."""
    NodeRegistry.register("dummy_node_no_meta", _DummyApplier)(_DummyCalculatorNoMeta)
    assert "dummy_node_no_meta" not in NodeRegistry.get_all_metadata()


def test_register_overwrites_existing_and_logs_warning(caplog):
    """Re-registering an existing node name should overwrite it and log a warning."""
    NodeRegistry.register("dummy_node_overwrite", _DummyApplier)(_DummyCalculatorWithMeta)
    with caplog.at_level(logging.WARNING, logger="skyulf.registry"):
        NodeRegistry.register("dummy_node_overwrite", _DummyApplier)(_DummyCalculatorNoMeta)
    assert NodeRegistry.get_calculator("dummy_node_overwrite") is _DummyCalculatorNoMeta
    assert any("re-registered" in record.message for record in caplog.records)


def test_get_calculator_unknown_name_raises_value_error():
    """Requesting an unregistered calculator name should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        NodeRegistry.get_calculator("totally_unregistered_node_xyz")


def test_get_applier_unknown_name_raises_value_error():
    """Requesting an unregistered applier name should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        NodeRegistry.get_applier("totally_unregistered_node_xyz")


def test_get_all_metadata_returns_snapshot_not_live_reference():
    """get_all_metadata() should return a copy, so mutating it doesn't affect the registry."""
    NodeRegistry.register("dummy_node_snapshot", _DummyApplier)(_DummyCalculatorWithMeta)
    snapshot = NodeRegistry.get_all_metadata()
    snapshot["dummy_node_snapshot"] = {"tampered": True}
    assert NodeRegistry.get_all_metadata()["dummy_node_snapshot"] != {"tampered": True}


def test_register_is_thread_safe_under_concurrent_registration():
    """Regression test: concurrent registration must not interleave a
    calculator/applier pair from different registration calls (which would
    silently produce a mismatched calculator/applier for the same name).
    Registers many distinct node names concurrently and checks every one
    ends up with a consistent (name -> its own applier) mapping.
    """
    import threading

    n_nodes = 50
    appliers = {f"concurrent_node_{i}": type(f"Applier{i}", (), {}) for i in range(n_nodes)}
    calculators = {f"concurrent_node_{i}": type(f"Calc{i}", (), {}) for i in range(n_nodes)}

    def _register(name: str) -> None:
        NodeRegistry.register(name, appliers[name])(calculators[name])

    threads = [threading.Thread(target=_register, args=(name,)) for name in appliers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for name in appliers:
        assert NodeRegistry.get_calculator(name) is calculators[name]
        assert NodeRegistry.get_applier(name) is appliers[name]
