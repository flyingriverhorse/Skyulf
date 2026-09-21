"""Explicit per-operation support must not turn into automatic local fallback."""

from typing import Any

import pytest

from skyulf.registry import NodeRegistry


@pytest.fixture
def isolated_registry(monkeypatch):
    """Restore registry dictionaries after tests register local probe nodes."""
    for name in ("_calculators", "_appliers", "_metadata"):
        monkeypatch.setattr(NodeRegistry, name, dict(getattr(NodeRegistry, name)))


def test_all_existing_nodes_reject_undeclared_spark_support():
    """An engine name alone must not advertise distributed execution for any node."""
    import skyulf.preprocessing  # noqa: F401 - populate real node registrations
    from skyulf.core.capabilities import UnsupportedExecutionError, require_capability

    nodes = NodeRegistry.get_all_metadata()
    assert nodes
    for node in nodes:
        for operation in ("fit", "apply"):
            with pytest.raises(UnsupportedExecutionError) as exc:
                require_capability(node, operation, "spark", config={})
            assert (exc.value.node_type, exc.value.operation, exc.value.engine) == (
                node,
                operation,
                "spark",
            )
            assert exc.value.reason


def test_support_is_operation_and_config_specific(isolated_registry):
    """Supporting one imputation strategy must not enable another or its fit operation."""
    from skyulf.core.capabilities import (
        ExecutionCapability,
        UnsupportedExecutionError,
        require_capability,
    )

    @NodeRegistry.register(
        "capability_probe",
        object,
        execution_capabilities=(
            ExecutionCapability(
                engine="spark",
                operation="apply",
                execution_kind="native",
                row_effect="preserve",
                context="row",
                codec_version=1,
                config_match=(("strategy", "mean"),),
            ),
        ),
    )
    class Probe:
        """A registered declaration without any executable estimator methods."""

    assert NodeRegistry.get_calculator("capability_probe") is Probe
    assert (
        require_capability("capability_probe", "apply", "spark", config={"strategy": "mean"})
        is None
    )
    for operation, config in [
        ("fit", {"strategy": "mean"}),
        ("apply", {}),
        ("apply", {"strategy": "median"}),
    ]:
        with pytest.raises(UnsupportedExecutionError):
            require_capability("capability_probe", operation, "spark", config=config)


def test_alias_shares_calculator_capabilities(isolated_registry):
    """Registering the same calculator under an alias must preserve its capabilities."""
    from skyulf.core.capabilities import ExecutionCapability, require_capability

    class Probe:
        """Share the exact same implementation under two registry names."""

    capability = ExecutionCapability(
        engine="spark",
        operation="apply",
        execution_kind="native",
        row_effect="preserve",
        context="row",
    )
    NodeRegistry.register("canonical_probe", object, execution_capabilities=(capability,))(Probe)
    NodeRegistry.register("alias_probe", object)(Probe)
    assert require_capability("canonical_probe", "apply", "spark", config={}) is None
    assert require_capability("alias_probe", "apply", "spark", config={}) is None


@pytest.mark.parametrize(
    "change",
    [
        {"engine": "cloud"},
        {"operation": "predict"},
        {"execution_kind": "automatic"},
        {"row_effect": "unknown"},
        {"context": "unknown"},
        {"codec_version": 0},
        {"codec_version": True},
        {"config_match": (("strategy", []),)},
        {"config_match": (("strategy", "mean"), ("strategy", "median"))},
    ],
)
def test_invalid_declarations_fail_before_registration(change):
    """Malformed capability metadata must not become an execution permission."""
    from skyulf.core.capabilities import ExecutionCapability

    kwargs: dict[str, Any] = {
        "engine": "spark",
        "operation": "apply",
        "execution_kind": "native",
        "row_effect": "preserve",
        "context": "row",
    }
    kwargs.update(change)
    with pytest.raises(ValueError):
        ExecutionCapability(**kwargs)


def test_unknown_node_has_structured_rejection():
    """An unknown node should report the requested operation without running anything."""
    from skyulf.core.capabilities import UnsupportedExecutionError, require_capability

    with pytest.raises(UnsupportedExecutionError) as exc:
        require_capability("missing_probe", "apply", "spark", config={})
    assert exc.value.node_type == "missing_probe"
    assert "not found" in exc.value.reason


def test_subclass_must_declare_its_own_support(isolated_registry):
    """Changing an implementation through inheritance must not inherit execution permission."""
    from skyulf.core.capabilities import (
        ExecutionCapability,
        UnsupportedExecutionError,
        require_capability,
    )

    @NodeRegistry.register(
        "parent_probe",
        object,
        execution_capabilities=(
            ExecutionCapability(
                engine="spark",
                operation="apply",
                execution_kind="native",
                row_effect="preserve",
                context="row",
            ),
        ),
    )
    class Parent:
        """Declare support for one implementation."""

    @NodeRegistry.register("child_probe", object)
    class Child(Parent):
        """Represent a replacement implementation with no verified declaration."""

    assert NodeRegistry.get_calculator("child_probe") is Child
    with pytest.raises(UnsupportedExecutionError):
        require_capability("child_probe", "apply", "spark", config={})


@pytest.mark.parametrize("config", [{}, {"count": True}, {"count": 1.0}, {"count": "1"}])
def test_selectors_do_not_coerce_or_supply_missing_values(config):
    """A numeric selector must not accidentally authorize a boolean or absent setting."""
    from skyulf.core.capabilities import ExecutionCapability

    capability = ExecutionCapability(
        engine="spark",
        operation="apply",
        execution_kind="native",
        row_effect="preserve",
        context="row",
        config_match=(("count", 1),),
    )
    assert capability.matches_config({"count": 1})
    assert not capability.matches_config(config)


@pytest.mark.parametrize("declarations", [[], (object(),)])
def test_registry_rejects_invalid_declaration_container(declarations, isolated_registry):
    """Malformed support metadata must not partially register a calculator."""
    with pytest.raises(ValueError, match="execution_capabilities"):
        NodeRegistry.register("invalid_probe", object, execution_capabilities=declarations)
    with pytest.raises(ValueError, match="not found"):
        NodeRegistry.get_calculator("invalid_probe")


def test_contract_imports_leave_optional_runtimes_unloaded():
    """Base installations must query execution support without importing optional runtimes."""
    import subprocess
    import sys

    code = """
import sys
from skyulf.core.execution import ExecutionOptions
from skyulf.core.capabilities import require_capability, UnsupportedExecutionError
ExecutionOptions(engine="spark")
try:
    require_capability("SimpleImputer", "fit", "spark", config={})
except UnsupportedExecutionError:
    pass
else:
    raise AssertionError("undeclared Spark support was accepted")
assert "pyspark" not in sys.modules
assert "mlflow" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
