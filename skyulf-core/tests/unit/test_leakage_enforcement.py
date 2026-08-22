"""Tests for the leakage-enforcement contract (F-16, F-17).

Phase 1: the data-dependent node list is registry-derived from a required
``learns_from_data`` field on ``@node_meta`` (single source of truth, cannot
drift, fails closed on unknown nodes).

Phase 2: pipelines with no train/test splitter get an explicit diagnostic
instead of silence, and definite violations raise by default (``on_leakage``).
"""

import pytest

import skyulf
from skyulf.core.meta.decorators import node_meta
from skyulf.leakage import (
    data_dependent_transformers,
    train_test_splitters,
    validate_leakage_safety,
)
from skyulf.registry import NodeRegistry


def _config(steps: list[dict]) -> dict:
    return {"preprocessing": steps, "modeling": {}}


# ---------------------------------------------------------------------------
# Phase 1 — required learns_from_data field, registry-derived lists
# ---------------------------------------------------------------------------


def test_node_meta_requires_learns_from_data():
    """Omitting learns_from_data must be a decoration (registration) error."""
    with pytest.raises(TypeError):

        @node_meta(id="x", name="X", category="Test", description="d")
        class _NoFlag:
            pass


def test_every_registered_node_declares_learns_from_data():
    """The registry snapshot must carry a bool learns_from_data for every node."""
    metadata = NodeRegistry.get_all_metadata()
    assert metadata, "registry unexpectedly empty"
    for node_id, meta in metadata.items():
        assert "learns_from_data" in meta, f"{node_id} missing learns_from_data"
        assert isinstance(meta["learns_from_data"], bool), f"{node_id} flag not bool"


@pytest.mark.parametrize(
    "node_id",
    [
        "MissingIndicator",
        "DropMissingColumns",
        "HashEncoder",
        "Deduplicate",
        "Oversampling",
        "Undersampling",
    ],
)
def test_previously_excluded_stateful_nodes_are_now_data_dependent(node_id):
    """G3 reclassification: these learn from the data they are fitted on."""
    assert node_id in data_dependent_transformers()


@pytest.mark.parametrize(
    "node_id",
    ["CustomBinning", "ManualBounds", "DropMissingRows", "DateFeatures", "GeoDistance"],
)
def test_rule_based_nodes_remain_safe(node_id):
    """Fixed-map / per-row rule nodes keep learns_from_data=False."""
    assert node_id not in data_dependent_transformers()


def test_splitters_are_registry_derived():
    """Both registered train/test splitter names derive from node metadata."""
    splitters = train_test_splitters()
    assert "TrainTestSplitter" in splitters
    assert "Split" in splitters
    assert "feature_target_split" not in splitters
    assert "StandardScaler" not in splitters


def test_top_level_export_still_available():
    assert skyulf.validate_leakage_safety is validate_leakage_safety


# ---------------------------------------------------------------------------
# Phase 1 — fail-closed on unknown nodes
# ---------------------------------------------------------------------------


def test_unknown_transformer_before_split_fails_closed():
    """An unrecognised transformer must be treated as data-dependent."""
    config = _config(
        [
            {"name": "mystery", "transformer": "MysteryTransformer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ]
    )
    with pytest.raises(ValueError, match="MysteryTransformer"):
        validate_leakage_safety(config)


# ---------------------------------------------------------------------------
# Phase 2 — no-splitter diagnostic, on_leakage modes
# ---------------------------------------------------------------------------


def test_no_splitter_emits_explicit_diagnostic_instead_of_silence():
    """G1: silence was the bug — a missing split must be surfaced."""
    config = _config([{"name": "scale", "transformer": "StandardScaler", "params": {}}])

    warnings = validate_leakage_safety(config, on_leakage="warn")

    assert len(warnings) == 1
    assert "No train/test split" in warnings[0]


def test_no_splitter_diagnostic_never_raises():
    """The no-splitter verdict is advisory, even under on_leakage='raise'."""
    config = _config([{"name": "scale", "transformer": "StandardScaler", "params": {}}])
    warnings = validate_leakage_safety(config)
    assert len(warnings) == 1


def test_no_splitter_diagnostic_silent_under_ignore():
    config = _config([{"name": "scale", "transformer": "StandardScaler", "params": {}}])
    assert validate_leakage_safety(config, on_leakage="ignore") == []


def test_default_on_leakage_raise_blocks_learned_step_before_split():
    """G7: definite violations raise by default."""
    config = _config(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ]
    )
    with pytest.raises(ValueError, match="SimpleImputer"):
        validate_leakage_safety(config)


def test_on_leakage_warn_returns_warnings_list():
    config = _config(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ]
    )
    warnings = validate_leakage_safety(config, on_leakage="warn")
    assert len(warnings) == 1
    assert "Step 0 ('SimpleImputer')" in warnings[0]


def test_on_leakage_ignore_returns_empty_list():
    config = _config(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {}},
            {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
        ]
    )
    assert validate_leakage_safety(config, on_leakage="ignore") == []


def test_invalid_on_leakage_value_rejected():
    with pytest.raises(ValueError, match="on_leakage"):
        validate_leakage_safety(_config([]), on_leakage="explode")


def test_reclassified_nodes_flagged_before_split():
    """F-16: the previously exempted stateful nodes are now gated."""
    for node_id in ("MissingIndicator", "DropMissingColumns", "Deduplicate", "HashEncoder"):
        config = _config(
            [
                {"name": "step", "transformer": node_id, "params": {}},
                {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
            ]
        )
        with pytest.raises(ValueError, match=node_id):
            validate_leakage_safety(config)


def test_splitter_first_still_clean():
    config = _config(
        [
            {"name": "split", "transformer": "Split", "params": {}},
            {"name": "scale", "transformer": "StandardScaler", "params": {}},
        ]
    )
    assert validate_leakage_safety(config) == []
