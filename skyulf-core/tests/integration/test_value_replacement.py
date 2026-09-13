"""Unit tests for the ValueReplacement cleaning node.

Covers: Calculator.fit branches (mapping/replacements/to_replace), Applier
apply for pandas + polars (mapping, nested mapping, to_replace/value),
NaN replacement, unseen values, edge cases (empty df, no valid columns),
and fit -> apply round trips.
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)

_fit_mapping_cases = TestCaseLoader("preprocessing/value_replacement", group="fit_mapping").load()
_roundtrip_cases = TestCaseLoader(
    "preprocessing/value_replacement", group="fit_apply_roundtrip"
).load()
_apply_scenario_cases = TestCaseLoader(
    "preprocessing/value_replacement", group="apply_scenarios"
).load()


def _coerce_numeric_keys(mapping: Any) -> Any:
    """Convert JSON string dict keys that look like integers back to ``int``.

    JSON object keys are always strings, so a fixture-encoded mapping like
    ``{"1": 100}`` must be restored to ``{1: 100}`` before being handed to
    pandas/polars ``.replace()``, which matches on the original column dtype.
    """
    if not isinstance(mapping, dict):
        return mapping
    result: dict[Any, Any] = {}
    for key, value in mapping.items():
        coerced_value = _coerce_numeric_keys(value) if isinstance(value, dict) else value
        try:
            coerced_key: Any = int(key)
        except (TypeError, ValueError):
            coerced_key = key
        result[coerced_key] = coerced_value
    return result


# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_fit_mapping_cases)
def test_fit_mapping_scenarios(
    config: dict[str, Any], expected_mapping: dict[str, Any], expected_columns: Any
) -> None:
    """Fit must resolve `mapping`/`replacements` config into the artifact's mapping."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    cfg = dict(config)
    if cfg.get("mapping") is not None:
        cfg["mapping"] = _coerce_numeric_keys(cfg["mapping"])
    params = ValueReplacementCalculator().fit(df, cfg)
    assert params["mapping"] == _coerce_numeric_keys(expected_mapping)
    if expected_columns is not None:
        assert params["columns"] == expected_columns


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
# Applier.apply — pandas + polars, config-driven scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_apply_scenario_cases)
def test_apply_scenarios(
    engine: str, data: dict[str, Any], config: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Flat/nested mapping, to_replace, and no-op scenarios across pandas + polars."""
    cfg = dict(config)
    for key in ("mapping", "to_replace"):
        if cfg.get(key) is not None:
            cfg[key] = _coerce_numeric_keys(cfg[key])
    df = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    result = ValueReplacementApplier().apply(df, cfg)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    for col, values in expected.items():
        assert result[col].tolist() == values


def test_apply_pandas_mapping_replaces_nan() -> None:
    """NaN must be replaceable via the mapping (using np.nan as a key)."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    params: dict[str, Any] = {"columns": ["a"], "mapping": {np.nan: -1.0}}
    result = ValueReplacementApplier().apply(df, params)
    assert result["a"].tolist() == [1.0, -1.0, 3.0]


def test_apply_pandas_empty_dataframe() -> None:
    """Applying to an empty DataFrame must not raise and must stay empty."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    params: dict[str, Any] = {"columns": ["a"], "mapping": {1: 100}}
    result = ValueReplacementApplier().apply(df, params)
    assert result.shape == (0, 1)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_apply_empty_columns_is_noop(engine: str) -> None:
    """Empty `columns` must leave the data unchanged (the UI's documented contract)."""
    data = {"a": [-999, 1], "b": [-999, 2]}
    df = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    params: dict[str, Any] = {"columns": [], "mapping": {-999: 0}}
    result = ValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["a"].tolist() == [-999, 1]
    assert result["b"].tolist() == [-999, 2]


def test_apply_unrecognized_mapping_key_does_not_mutate_boolean_column() -> None:
    """An unrecognized string mapping key for a boolean column should not mutate values."""
    params: dict[str, Any] = {"columns": ["flag"], "mapping": {"banana": True}}
    for engine in ("pandas", "polars"):
        data = {"flag": [True, False]}
        df = pd.DataFrame(data, dtype="boolean") if engine == "pandas" else pl.DataFrame(data)
        result = ValueReplacementApplier().apply(df, params)
        if hasattr(result, "to_pandas"):
            result = result.to_pandas()
        assert result["flag"].tolist() == [True, False]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,values,mapping,expected",
    [
        ("Int64", [1, 2, None], {"1": 10}, [10, 2, None]),
        ("Float64", [1.5, 2.5, None], {"1.5": 10.5}, [10.5, 2.5, None]),
        ("boolean", [True, False, None], {" TRUE ": False, "0": True}, [False, True, None]),
        ("string", ["red", "green", None], {"red": "warm"}, ["warm", "green", None]),
        ("Int64", [1, 2, None], {"banana": 99}, [1, 2, None]),
        ("Int64", [1, 2, None], {"1": 10, "banana": 99, "1.5": 99}, [10, 2, None]),
        ("Float64", [1.5, 2.5, None], {"1.5": 10.5, "banana": 99.0}, [10.5, 2.5, None]),
        ("boolean", [True, False, None], {"true": False, "banana": True}, [False, False, None]),
    ],
)
def test_json_mapping_keys_match_nullable_column_types(
    engine: str, dtype: str, values: list, mapping: dict, expected: list
) -> None:
    """Saved JSON string keys must replace numeric and boolean values without losing nulls."""
    config = json.loads(json.dumps({"columns": ["value"], "mapping": mapping}))
    source = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    original = source.copy(deep=True)
    frame = source if engine == "pandas" else pl.from_pandas(source)
    artifact = ValueReplacementCalculator().fit(frame, config)
    result = ValueReplacementApplier().apply(frame, artifact)
    expected_frame = pd.DataFrame({"value": pd.Series(expected, dtype=dtype)})

    if isinstance(frame, pl.DataFrame):
        assert result.equals(pl.from_pandas(expected_frame))
        assert frame.equals(pl.from_pandas(original))
    else:
        pd.testing.assert_frame_equal(result, expected_frame)
        pd.testing.assert_frame_equal(frame, original)
    assert config["mapping"] == mapping


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_nested_mapping_respects_selected_columns_and_preserves_target(engine: str) -> None:
    """Partial per-column mappings must leave unmapped features, excluded columns, and y intact."""
    source = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "excluded": [1, 2, 3]})
    target = pd.Series([0, 1, 0], name="target")
    config = {
        "columns": ["a", "b"],
        "mapping": {"a": {"1": 10}, "excluded": {"2": 20}, "missing": {"3": 30}},
    }
    frame = source if engine == "pandas" else pl.from_pandas(source)
    artifact = ValueReplacementCalculator().fit((frame, target), config)
    result, result_target = ValueReplacementApplier().apply((frame, target), artifact)
    expected = source.copy()
    expected["a"] = [10, 2, 3]

    if isinstance(frame, pl.DataFrame):
        assert result.equals(pl.from_pandas(expected))
        assert frame.equals(pl.from_pandas(source))
    else:
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(frame, source)
    pd.testing.assert_series_equal(result_target, target)


# ---------------------------------------------------------------------------
# fit -> apply round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_roundtrip_cases)
def test_fit_then_apply_round_trip(
    data: dict[str, Any], config: dict[str, Any], column: str, expected: list
) -> None:
    """fit() output must apply correctly end-to-end for replacements/to_replace configs."""
    df = pd.DataFrame(data)
    calc = ValueReplacementCalculator()
    applier = ValueReplacementApplier()
    params = calc.fit(df, config)
    result = applier.apply(df, params)
    assert result[column].tolist() == expected


# ---------------------------------------------------------------------------
# Real-shaped dataset integration check
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.

    Verifies that ValueReplacementCalculator / Applier correctly maps
    categorical ``plan_type`` values to numeric codes across the full 15-row
    dataset with mixed dtypes, and that no rows are dropped or duplicated.
    """

    def test_plan_type_mapping_to_numeric_codes(self) -> None:
        df = load_sample_dataset("customers")
        calc = ValueReplacementCalculator()
        applier = ValueReplacementApplier()
        config = {
            "columns": ["plan_type"],
            "mapping": {"basic": 1, "premium": 2, "enterprise": 3},
        }
        params = calc.fit(df, config)
        result = applier.apply(df, params)
        assert len(result) == len(df)
        assert set(result["plan_type"].unique()).issubset({1, 2, 3})
