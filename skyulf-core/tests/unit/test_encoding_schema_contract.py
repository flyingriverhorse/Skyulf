"""Encoder previews must describe known numeric output without guessing value-dependent selection."""

from decimal import Decimal

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,dtype",
    [
        ("LabelEncoder", "int64"),
        ("OrdinalEncoder", "float32"),
        ("HashEncoder", "int64"),
        ("WOEEncoder", "float64"),
        ("TargetEncoder", "float64"),
    ],
)
def test_explicit_encoder_schema_matches_real_output(node, dtype, engine):
    """Numeric pickers need the encoded dtype while untouched columns retain their original metadata."""
    frame = pd.DataFrame({"cat": ["a", "a", "b", "b"], "keep": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": ["cat", "missing"], "target_column": "y"}
    if node == "TargetEncoder":
        config["target_type"] = "binary"
    calculator = NodeRegistry.get_calculator(node)()
    schema = SkyulfSchema.from_dataframe(frame)
    preview = calculator.infer_output_schema(schema, config)
    state = calculator.fit(frame, config)
    output = NodeRegistry.get_applier(node)().apply(frame, state)
    actual = SkyulfSchema.from_dataframe(output)
    assert preview is not None
    assert preview.columns == actual.columns == schema.columns
    assert preview.dtypes["cat"] == actual.dtypes["cat"].lower() == dtype
    assert {k: v for k, v in preview.dtypes.items() if k != "cat"} == {
        k: v for k, v in schema.dtypes.items() if k != "cat"
    }


@pytest.mark.parametrize("node", ["OrdinalEncoder", "HashEncoder", "WOEEncoder", "TargetEncoder"])
@pytest.mark.parametrize("selection", [{}, {"columns": None}])
def test_automatic_categorical_selection_stays_opaque(node, selection):
    """Object dtype cannot distinguish genuine categories from Decimal values excluded during fit."""
    frame = pd.DataFrame({"cat": ["a", "b"], "amount": [Decimal("1.1"), Decimal("2.2")]})
    schema = SkyulfSchema.from_dataframe(frame)
    config = {**selection, "target_type": "binary"}
    assert NodeRegistry.get_calculator(node)().infer_output_schema(schema, config) is None


@pytest.mark.parametrize("node,dtype", [("LabelEncoder", "int64"), ("OrdinalEncoder", "float32")])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_selection_preserves_embedded_target_encoding(node, dtype, engine):
    """Label and Ordinal empty selections may still encode the embedded target rather than no-op."""
    frame = pd.DataFrame({"cat": ["a", "b"], "label": ["no", "yes"]})
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    config = {"columns": [], "target_column": "label"}
    calculator = NodeRegistry.get_calculator(node)()
    original = SkyulfSchema.from_dataframe(frame)
    preview = calculator.infer_output_schema(original, config)
    state = calculator.fit(frame, config)
    actual = SkyulfSchema.from_dataframe(NodeRegistry.get_applier(node)().apply(frame, state))
    assert preview is not None
    assert preview.dtypes["label"] == actual.dtypes["label"].lower() == dtype
    assert preview.dtypes["cat"] == original.dtypes["cat"]


@pytest.mark.parametrize("node", ["HashEncoder", "WOEEncoder", "TargetEncoder"])
def test_empty_selection_and_supervised_target_exclusion_preserve_schema(node):
    """A guaranteed no-op must not invent encoded targets or float columns."""
    schema = SkyulfSchema.from_columns(["cat", "label"], {"cat": "object", "label": "object"})
    calculator = NodeRegistry.get_calculator(node)()
    assert (
        calculator.infer_output_schema(schema, {"columns": [], "target_type": "binary"}) == schema
    )
    assert (
        calculator.infer_output_schema(
            schema, {"columns": ["label"], "target_column": "label", "target_type": "binary"}
        )
        == schema
    )


def test_hash_preview_abstains_for_buckets_outside_signed_integer_range():
    """Huge configured widths can produce value-dependent unsigned buckets and cannot promise int64."""
    schema = SkyulfSchema.from_columns(["cat"], {"cat": "object"})
    calculator = NodeRegistry.get_calculator("HashEncoder")()
    assert calculator.infer_output_schema(schema, {"columns": ["cat"], "n_features": 2**64}) is None
    edge = calculator.infer_output_schema(schema, {"columns": ["cat"], "n_features": 2**63})
    assert edge is not None and edge.dtypes["cat"] == "int64"


def test_label_default_encodes_only_the_named_embedded_target():
    """LabelEncoder's target-only default must not become categorical feature auto-detection."""
    schema = SkyulfSchema.from_columns(["cat", "label"], {"cat": "object", "label": "object"})
    preview = NodeRegistry.get_calculator("LabelEncoder")().infer_output_schema(
        schema, {"target_column": "label"}
    )
    assert preview is not None and preview.dtypes == {"cat": "object", "label": "int64"}


def test_explicit_unknown_dtype_and_multiclass_remain_honest():
    """Explicit selection can determine output dtype without input metadata; multiclass still needs data."""
    schema = SkyulfSchema.from_columns(["cat", "keep"])
    ordinal = NodeRegistry.get_calculator("OrdinalEncoder")().infer_output_schema(
        schema, {"columns": ["cat"]}
    )
    assert ordinal is not None and ordinal.dtypes == {"cat": "float32"}
    assert (
        NodeRegistry.get_calculator("TargetEncoder")().infer_output_schema(
            schema, {"columns": ["cat"], "target_type": "multiclass"}
        )
        is None
    )
