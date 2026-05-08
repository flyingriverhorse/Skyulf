"""Tests for the static column-reference validator (C7 Phase D)."""

from __future__ import annotations

import pandas as pd

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.preprocessing import SkyulfSchema


def _seed() -> SkyulfSchema:
    return SkyulfSchema.from_dataframe(pd.DataFrame({"a": [1.0], "b": [2.0], "c": ["x"]}))


def _config(node: NodeConfig) -> PipelineConfig:
    return PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            node,
        ],
    )


def test_drops_broken_columns_param() -> None:
    """A `columns` reference to a non-existent column is flagged."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["a", "ghost"]},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    assert len(broken) == 1
    assert broken[0]["node_id"] == "drop"
    assert broken[0]["column"] == "ghost"
    assert broken[0]["field"] == "columns"
    assert broken[0]["upstream_node_id"] == "loader"


def test_valid_references_emit_nothing() -> None:
    """All known columns -> no flags."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["a", "b"]},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})

    assert find_broken_references(config, predicted) == []


def test_unknown_upstream_schema_skips_validation() -> None:
    """When the upstream schema is None we cannot validate -> skip silently."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["ghost"]},
            inputs=["loader"],
        )
    )
    # No initial seed -> loader schema is None.
    predicted = predict_schemas(config)

    assert find_broken_references(config, predicted) == []


def test_target_field_string_reference() -> None:
    """The `target` (string) field is also validated."""
    config = _config(
        NodeConfig(
            node_id="trainer",
            step_type="basic_training",
            params={"target": "ghost"},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    assert len(broken) == 1
    assert broken[0]["field"] == "target"
    assert broken[0]["column"] == "ghost"


def test_column_types_dict_keys_validated() -> None:
    """`column_types` dict keys are checked too."""
    config = _config(
        NodeConfig(
            node_id="cast",
            step_type="Casting",
            params={"column_types": {"a": "int", "ghost": "float"}},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    columns_flagged = {b["column"] for b in broken}
    assert "ghost" in columns_flagged
    assert "a" not in columns_flagged
