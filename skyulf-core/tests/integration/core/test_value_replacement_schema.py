"""Replacement previews retain only dtype guarantees available before seeing rows."""

from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["int8", "int64", "Int64", "UInt64", "int64[pyarrow]"])
@pytest.mark.parametrize("value", [None, 2.0])
def test_integer_preview_matches_full_singleton_and_empty_runtime(engine, dtype, value):
    """Preview must expose nullable promotion without falsely widening exact integers."""
    frame = pd.DataFrame({"x": pd.Series([1, 3], dtype=dtype), "other": [8, 9]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    schema = SkyulfSchema.from_dataframe(frame)
    config = {"columns": ["x"], "mapping": {"1": value}}
    calculator = ValueReplacementCalculator()
    inferred = calculator.infer_output_schema(schema, config)
    state = calculator.fit(frame, config)
    for chunk in (frame, frame.tail(1), frame.head(0)):
        result = ValueReplacementApplier().apply(chunk, state)
        assert inferred == SkyulfSchema.from_dataframe(result)
    assert schema == SkyulfSchema.from_dataframe(frame)


@pytest.mark.parametrize(
    "rules",
    [
        {"mapping": {"1": None}},
        {"mapping": {"x": {"1": None}, "unselected": {"1": "text"}}},
        {"to_replace": "1", "value": None},
        {"to_replace": ["1"], "value": [None]},
        {"to_replace": {"1": None}},
        {"mapping": {"1": "text"}, "replacements": [{"old": "1", "new": None}]},
        {"mapping": {"1": None}, "to_replace": [1], "value": ["text"]},
    ],
)
def test_schema_uses_effective_runtime_rules_without_running_fit_or_apply(rules, monkeypatch):
    """Preview must respect all public rule forms and precedence without executing the pipeline."""
    schema = SkyulfSchema.from_columns(["x", "unselected"], {"x": "int64", "unselected": "int64"})
    config = {"columns": ["x"], **rules}
    before = deepcopy(config)

    def forbidden(*args, **kwargs):
        """Schema preview must not execute a fitted transform or fit request data."""
        raise AssertionError("Unexpected execution during schema preview")

    monkeypatch.setattr(ValueReplacementCalculator, "fit", forbidden)
    monkeypatch.setattr(ValueReplacementApplier, "apply", forbidden)
    inferred = ValueReplacementCalculator().infer_output_schema(schema, config)
    assert inferred == SkyulfSchema.from_columns(
        ["x", "unselected"], {"x": "Int64", "unselected": "int64"}
    )
    assert config == before


@pytest.mark.parametrize(
    "dtype,mapping",
    [
        ("float32", {1: 1.123456789123}),
        ("Float32", {1: 1.123456789123}),
        ("float64", {1: "text"}),
        ("int64", {1: "text"}),
        ("string", {"a": 1}),
        ("category", {"a": "new"}),
        ("unknown", {1: 2}),
        ("int64", {1: 0.5}),
    ],
)
def test_uncertain_preview_keeps_columns_without_claiming_an_output_dtype(dtype, mapping):
    """Data-dependent or unsupported replacements must not advertise the input dtype as output."""
    schema = SkyulfSchema.from_columns(["x", "other"], {"x": dtype, "other": "int64"})
    inferred = ValueReplacementCalculator().infer_output_schema(
        schema, {"columns": ["x"], "mapping": mapping}
    )
    assert inferred == SkyulfSchema.from_columns(["x", "other"], {"other": "int64"})
    assert schema.dtypes["x"] == dtype


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"columns": [], "mapping": {1: None}},
        {"columns": ["missing"], "mapping": {1: None}},
        {"columns": ["x"], "mapping": {}},
        {"columns": ["x"], "mapping": {"other": {1: None}}},
        {"columns": ["x"], "mapping": {"bad": 0.5}},
        {"columns": ["x"], "mapping": {True: 0.5}},
        {"columns": ["x"], "mapping": {2**80: 0.5}},
    ],
)
def test_noop_rules_preserve_known_schema(config):
    """An excluded or mathematically impossible key must not erase a known dtype."""
    schema = SkyulfSchema.from_columns(["x"], {"x": "int64"})
    assert ValueReplacementCalculator().infer_output_schema(schema, config) == schema


def test_object_replacement_preview_retains_existing_no_downcast_contract():
    """Object columns remain object even when configured replacements happen to be numbers."""
    schema = SkyulfSchema.from_columns(["x"], {"x": "object"})
    assert (
        ValueReplacementCalculator().infer_output_schema(
            schema, {"columns": ["x"], "mapping": {1: 0.5}}
        )
        == schema
    )


@pytest.mark.parametrize(
    "rules",
    [
        {"mapping": {"1": None, 1: 2}},
        {"mapping": {"x": {"1": None, 1: 2}}},
        {"to_replace": {"1": None, 1: 2}},
        {"replacements": [{"old": "1", "new": None}, {"old": 1, "new": 2}]},
    ],
)
def test_mapping_preview_collapses_coerced_keys_like_runtime(rules):
    """A later typed dictionary key can override a null rule without promoting the output."""
    frame = pd.DataFrame({"x": [1, 3]})
    schema = SkyulfSchema.from_dataframe(frame)
    config = {"columns": ["x"], **rules}
    calculator = ValueReplacementCalculator()
    state = calculator.fit(frame, config)
    inferred = calculator.infer_output_schema(schema, config)
    assert inferred == SkyulfSchema.from_dataframe(ValueReplacementApplier().apply(frame, state))
    assert inferred == schema
