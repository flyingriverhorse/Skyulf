"""Unit tests for the ValueReplacement cleaning node.

Covers: Calculator.fit branches (mapping/replacements/to_replace), Applier
apply for pandas + polars (mapping, nested mapping, to_replace/value),
NaN replacement, unseen values, edge cases (empty df, no valid columns),
and fit -> apply round trips.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl

from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)

# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


def test_fit_uses_mapping_from_config() -> None:
    """fit must pass an explicit `mapping` dict straight through."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    params = ValueReplacementCalculator().fit(df, {"columns": ["a"], "mapping": {1: 100}})
    assert params["mapping"] == {1: 100}
    assert params["columns"] == ["a"]


def test_fit_builds_mapping_from_replacements_list() -> None:
    """The `replacements` list of {old, new} dicts must be converted to a mapping."""
    df = pd.DataFrame({"a": [1, 2]})
    config = {"columns": ["a"], "replacements": [{"old": 1, "new": 10}, {"old": 2, "new": 20}]}
    params = ValueReplacementCalculator().fit(df, config)
    assert params["mapping"] == {1: 10, 2: 20}


def test_fit_replacements_overrides_mapping() -> None:
    """`replacements`, when present, must take priority over `mapping`."""
    df = pd.DataFrame({"a": [1]})
    config = {
        "columns": ["a"],
        "mapping": {1: 999},
        "replacements": [{"old": 1, "new": 5}],
    }
    params = ValueReplacementCalculator().fit(df, config)
    assert params["mapping"] == {1: 5}


def test_fit_stores_to_replace_and_value() -> None:
    """to_replace / value keys must be preserved in the artifact."""
    df = pd.DataFrame({"a": [1]})
    params = ValueReplacementCalculator().fit(df, {"columns": ["a"], "to_replace": -1, "value": 0})
    assert params["to_replace"] == -1
    assert params["value"] == 0
    assert params["mapping"] is None


def test_fit_infer_output_schema_passes_through() -> None:
    """infer_output_schema must return the input schema unchanged (column-preserving)."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], {"a": "int64"})
    result = ValueReplacementCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# Applier.apply — pandas, flat mapping
# ---------------------------------------------------------------------------


def test_apply_pandas_flat_mapping_replaces_values() -> None:
    """A flat mapping must replace matched values across all listed columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a", "b"], "mapping": {1: 100, 2: 200}}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [100, 200, 3]
    assert result["b"].tolist() == [100, 200, 3]


def test_apply_pandas_nested_mapping_is_per_column() -> None:
    """A nested mapping (col -> {old: new}) must only affect its own column."""
    df = pd.DataFrame({"a": [1, 2], "b": [1, 2]})
    params: Dict[str, Any] = {
        "columns": ["a", "b"],
        "mapping": {"a": {1: "one"}, "b": {2: "two"}},
    }
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == ["one", 2]
    assert result["b"].tolist() == [1, "two"]


def test_apply_pandas_mapping_replaces_nan() -> None:
    """NaN must be replaceable via the mapping (using np.nan as a key)."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    params: Dict[str, Any] = {"columns": ["a"], "mapping": {np.nan: -1.0}}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [1.0, -1.0, 3.0]


def test_apply_pandas_unseen_values_are_untouched() -> None:
    """Values not present in the mapping keys must remain unchanged."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"], "mapping": {99: -1}}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [1, 2, 3]


def test_apply_pandas_to_replace_scalar_with_value() -> None:
    """Scalar `to_replace` + `value` must replace matching entries with value."""
    df = pd.DataFrame({"a": [1, 2, 1]})
    params: Dict[str, Any] = {"columns": ["a"], "to_replace": 1, "value": -9}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [-9, 2, -9]


def test_apply_pandas_to_replace_mapping_like() -> None:
    """A dict-shaped `to_replace` must be applied like pandas .replace(dict)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"], "to_replace": {1: 10, 3: 30}}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [10, 2, 30]


def test_apply_pandas_no_mapping_no_to_replace_is_noop() -> None:
    """With neither mapping nor to_replace set, the frame must be unchanged."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"]}
    result = ValueReplacementApplier().apply(df, params)
    pd.testing.assert_frame_equal(result, df)


def test_apply_pandas_no_valid_columns_is_noop() -> None:
    """If none of the requested columns exist, the frame must pass through unchanged."""
    df = pd.DataFrame({"a": [1, 2]})
    params: Dict[str, Any] = {"columns": ["missing"], "mapping": {1: 100}}
    result = ValueReplacementApplier().apply(df, params)
    pd.testing.assert_frame_equal(result, df)


def test_apply_pandas_empty_dataframe() -> None:
    """Applying to an empty DataFrame must not raise and must stay empty."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    params: Dict[str, Any] = {"columns": ["a"], "mapping": {1: 100}}
    result = ValueReplacementApplier().apply(df, params)
    assert result.shape == (0, 1)


# ---------------------------------------------------------------------------
# Applier.apply — polars
# ---------------------------------------------------------------------------


def test_apply_polars_flat_mapping_replaces_values() -> None:
    """Polars path: flat mapping replaces values across listed columns."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"], "mapping": {1: 100, 2: 200}}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [100, 200, 3]


def test_apply_polars_nested_mapping_is_per_column() -> None:
    """Polars path: nested mapping only replaces values in its own column."""
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    params: Dict[str, Any] = {
        "columns": ["a", "b"],
        "mapping": {"a": {1: 100}, "b": {2: 200}},
    }
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [100, 2]
    assert result["b"].tolist() == [1, 200]


def test_apply_polars_to_replace_scalar_with_value() -> None:
    """Polars path: scalar to_replace/value pair must replace matching entries."""
    df = pl.DataFrame({"a": [1, 2, 1]})
    params: Dict[str, Any] = {"columns": ["a"], "to_replace": 1, "value": -9}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [-9, 2, -9]


def test_apply_polars_to_replace_mapping_like() -> None:
    """Polars path: a dict-shaped to_replace must be applied like pandas .replace(dict)."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"], "to_replace": {1: 10, 3: 30}}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [10, 2, 30]


def test_apply_polars_no_valid_columns_is_noop() -> None:
    """Polars path: no valid columns must leave the frame unchanged."""
    df = pl.DataFrame({"a": [1, 2]})
    params: Dict[str, Any] = {"columns": ["missing"], "mapping": {1: 100}}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [1, 2]


def test_apply_polars_no_mapping_no_to_replace_is_noop() -> None:
    """Polars path: with neither mapping nor to_replace, frame is unchanged."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    params: Dict[str, Any] = {"columns": ["a"]}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# fit -> apply round trip
# ---------------------------------------------------------------------------


def test_fit_then_apply_round_trip_replacements_list() -> None:
    """fit() built from a `replacements` list must apply correctly end-to-end."""
    df = pd.DataFrame({"grade": ["A", "B", "C"]})
    config = {
        "columns": ["grade"],
        "replacements": [{"old": "A", "new": 4}, {"old": "B", "new": 3}, {"old": "C", "new": 2}],
    }
    calc = ValueReplacementCalculator()
    applier = ValueReplacementApplier()
    params = calc.fit(df, config)
    result = applier.apply(df, params)
    assert result["grade"].tolist() == [4, 3, 2]


def test_fit_then_apply_round_trip_to_replace_value() -> None:
    """fit() with to_replace/value must round trip through apply correctly."""
    df = pd.DataFrame({"a": [-1, 0, -1, 5]})
    calc = ValueReplacementCalculator()
    applier = ValueReplacementApplier()
    params = calc.fit(df, {"columns": ["a"], "to_replace": -1, "value": 0})
    result = applier.apply(df, params)
    assert result["a"].tolist() == [0, 0, 0, 5]
