"""Every built-in node declares whether Spark scoring and serving endpoints admit it."""

import pytest

import skyulf.modeling  # noqa: F401 - registers the model nodes
import skyulf.preprocessing  # noqa: F401 - registers the preprocessing nodes
from skyulf import registry_admission as admission
from skyulf.inference import (
    _partition_builtins,
    _partition_models,
    _partition_trees,
    partition_safety,
)
from skyulf.registry import NodeRegistry

# The deprecated "Split" alias is a registered node in its own right and warns on lookup.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _builtin_nodes() -> list[str]:
    """Skip nodes other tests register, so only shipped nodes must declare admission."""
    return sorted(
        name
        for name, calculator in NodeRegistry._calculators.items()
        if calculator.__module__.startswith("skyulf.")
    )


def _declared(reason: str) -> set[str]:
    """Return the registered built-in nodes declared with this exact reason."""
    return {name for name in _builtin_nodes() if admission.spark_admission(name) == reason}


def test_every_builtin_node_declares_spark_admission():
    """A new node must not reach Spark/endpoint review without an admit-or-reason decision."""
    missing = [name for name in _builtin_nodes() if admission.spark_admission(name) is None]
    assert missing == [], f"Declare these nodes in skyulf/registry_admission.py: {missing}"


def test_declarations_state_admitted_or_a_reason():
    """Each entry reads 'admitted: <how>' or 'not admitted: <reason>', never a bare flag."""
    for name, declaration in admission.DECLARATIONS.items():
        status, _, reason = declaration.partition(": ")
        assert status in ("admitted", "not admitted"), name
        assert reason.strip(), name


def test_capability_admitted_nodes_have_no_conflicting_entry():
    """A table entry must not contradict an admitting execution capability."""
    admitted = _declared(admission.ADMITTED_BY_CAPABILITY)
    assert admitted
    assert sorted(admitted & set(admission.DECLARATIONS)) == []


def test_declared_admissions_match_the_partition_review():
    """The table stays truthful: each admitted kind equals what partition_safety reviews."""
    nodes = set(_builtin_nodes())
    model_appliers = set(partition_safety._MODELS.values())
    reviewed_models = {name for name in nodes if NodeRegistry._appliers[name] in model_appliers} | {
        contract[0] for contract in _partition_trees._CONTRACTS.values()
    }
    reviewed_models.add("xgboost_regressor")

    assert _declared(admission.ADMITTED_BY_CAPABILITY) == set(partition_safety._APPLIERS)
    assert _declared(admission.ADMITTED_MODEL) == reviewed_models & nodes
    assert _declared(admission.PROBED_MODEL) == {
        item[0] for item in _partition_models.CONTRACTS.values()
    }
    assert _declared(admission.INFERENCE_SKIP) == set(partition_safety._SKIPS)
    assert _declared(admission.PROBED) == set(partition_safety._PROBED)
    exclusions = {"IQR", "EllipticEnvelope", "ManualBounds", "ZScore", "PowerTransformer"}
    assert _declared(admission.PROBED_EXCLUSION) == exclusions
    assert (
        _declared(admission.PROBED_BUILTIN) == set(_partition_builtins.PROBED_BUILTINS) - exclusions
    )
    assert _declared(admission.SPLIT_MARKER) == partition_safety._UNRECORDED & nodes


def test_unregistered_node_lookup_raises():
    """Unknown IDs fail loudly instead of reading as undeclared."""
    with pytest.raises(ValueError, match="not found in registry"):
        admission.spark_admission("no_such_node")
