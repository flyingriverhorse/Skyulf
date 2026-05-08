"""Tests for the pre-execution schema graph (C7 Phase B)."""

from __future__ import annotations

import pandas as pd

from backend.ml_pipeline._execution._schema_graph import (
    predict_schemas,
    schema_to_dict,
    schemas_to_dict,
)
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.preprocessing import SkyulfSchema


def _seed_schema() -> SkyulfSchema:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        }
    )
    return SkyulfSchema.from_dataframe(df)


def test_unseeded_data_loader_yields_none_chain() -> None:
    """No catalog seeding -> loader prediction is None and propagates."""
    config = PipelineConfig(
        pipeline_id="p1",
        nodes=[
            NodeConfig(node_id="n1", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type="DropMissingColumns",
                params={"columns": ["a"]},
                inputs=["n1"],
            ),
        ],
    )

    predicted = predict_schemas(config)

    assert predicted == {"n1": None, "n2": None}


def test_seeded_loader_propagates_through_passthrough() -> None:
    """A seeded loader feeds a passthrough Calculator -> downstream schema known."""
    seed = _seed_schema()
    config = PipelineConfig(
        pipeline_id="p2",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="impute",
                step_type="SimpleImputer",
                params={"strategy": "mean"},
                inputs=["loader"],
            ),
        ],
    )

    predicted = predict_schemas(config, initial_schemas={"loader": seed})

    assert predicted["loader"] == seed
    assert predicted["impute"] == seed  # SimpleImputer is passthrough


def test_drop_missing_columns_removes_column() -> None:
    """Config-driven DropMissingColumns drops the requested column."""
    seed = _seed_schema()
    config = PipelineConfig(
        pipeline_id="p3",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="drop",
                step_type="DropMissingColumns",
                params={"columns": ["b"]},
                inputs=["loader"],
            ),
        ],
    )

    predicted = predict_schemas(config, initial_schemas={"loader": seed})

    out = predicted["drop"]
    assert out is not None
    assert "b" not in out.columns
    assert "a" in out.columns and "c" in out.columns


def test_data_dependent_node_returns_none() -> None:
    """OneHotEncoder cannot be predicted from config alone -> None."""
    seed = _seed_schema()
    config = PipelineConfig(
        pipeline_id="p4",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="ohe",
                step_type="OneHotEncoder",
                params={"columns": ["c"]},
                inputs=["loader"],
            ),
            NodeConfig(
                node_id="impute",
                step_type="SimpleImputer",
                params={},
                inputs=["ohe"],
            ),
        ],
    )

    predicted = predict_schemas(config, initial_schemas={"loader": seed})

    assert predicted["ohe"] is None
    # Downstream must propagate the unknown
    assert predicted["impute"] is None


def test_unknown_step_type_returns_none() -> None:
    """A node whose step_type isn't in the registry yields None (not a crash)."""
    seed = _seed_schema()
    config = PipelineConfig(
        pipeline_id="p5",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="bogus",
                step_type="NotARealNode",
                params={},
                inputs=["loader"],
            ),
        ],
    )

    predicted = predict_schemas(config, initial_schemas={"loader": seed})

    assert predicted["bogus"] is None


def test_training_node_is_passthrough() -> None:
    """Training/preview leaves carry through the input schema."""
    seed = _seed_schema()
    config = PipelineConfig(
        pipeline_id="p6",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            NodeConfig(
                node_id="train",
                step_type="basic_training",
                params={},
                inputs=["loader"],
            ),
        ],
    )

    predicted = predict_schemas(config, initial_schemas={"loader": seed})

    assert predicted["train"] == seed


def test_schemas_to_dict_serialization() -> None:
    """Helper produces JSON-friendly dicts (and None passes through)."""
    seed = _seed_schema()
    out = schemas_to_dict({"a": seed, "b": None})
    assert out["b"] is None
    assert out["a"] is not None
    assert set(out["a"]["columns"]) == {"a", "b", "c"}
    assert "a" in out["a"]["dtypes"]


def test_schema_to_dict_none() -> None:
    assert schema_to_dict(None) is None
