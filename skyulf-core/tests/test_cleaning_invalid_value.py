"""Unit tests for the InvalidValueReplacement cleaning node.

Covers: mask helpers, inf-replacement helpers, resolver, Calculator.fit
branches, Applier.apply (pandas + polars), edge cases, and engine-parity.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skyulf.preprocessing.cleaning.invalid_value import (
    InvalidValueReplacementApplier,
    InvalidValueReplacementCalculator,
    _invalid_rule_pandas_mask,
    _invalid_rule_polars,
    _resolve_invalid_replacement,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _basic_df() -> pd.DataFrame:
    """Small numeric DataFrame with positive, negative, zero, and NaN values."""
    return pd.DataFrame(
        {
            "x": [-3.0, -1.0, 0.0, 1.0, 5.0, np.nan, 10.0],
            "y": [2.0, -4.0, 0.0, 6.0, -2.0, 8.0, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# _resolve_invalid_replacement
# ---------------------------------------------------------------------------


def test_resolve_replacement_uses_value_when_set() -> None:
    """Explicit `value` key must take priority over `replacement`."""
    params: Dict[str, Any] = {"replacement": np.nan, "value": 0}
    assert _resolve_invalid_replacement(params) == 0


def test_resolve_replacement_falls_back_to_replacement() -> None:
    """`replacement` is used when `value` is absent."""
    params: Dict[str, Any] = {"replacement": -999}
    assert _resolve_invalid_replacement(params) == -999


def test_resolve_replacement_default_is_nan() -> None:
    """Empty params yield NaN as the default fill value."""
    result = _resolve_invalid_replacement({})
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# _invalid_rule_pandas_mask
# ---------------------------------------------------------------------------


def test_pandas_mask_negative_marks_negatives() -> None:
    """Negative rule must flag only values < 0."""
    s = pd.Series([-1.0, 0.0, 1.0])
    mask = _invalid_rule_pandas_mask(s, "negative", None, None)
    assert list(mask) == [True, False, False]


def test_pandas_mask_negative_to_nan_alias() -> None:
    """Legacy 'negative_to_nan' must be treated as 'negative'."""
    s = pd.Series([-2.0, 3.0])
    mask = _invalid_rule_pandas_mask(s, "negative_to_nan", None, None)
    assert list(mask) == [True, False]


def test_pandas_mask_zero_marks_zeros() -> None:
    """Zero rule must flag exactly the zero entries."""
    s = pd.Series([0.0, 1.0, -1.0, 0.0])
    mask = _invalid_rule_pandas_mask(s, "zero", None, None)
    assert list(mask) == [True, False, False, True]


def test_pandas_mask_custom_range_both_bounds() -> None:
    """Values outside [min, max] should be flagged."""
    s = pd.Series([0.0, 5.0, 11.0, -1.0])
    mask = _invalid_rule_pandas_mask(s, "custom_range", 0.0, 10.0)
    assert list(mask) == [False, False, True, True]


def test_pandas_mask_custom_range_min_only() -> None:
    """Only min set: values below min are flagged."""
    s = pd.Series([1.0, 5.0, -3.0])
    mask = _invalid_rule_pandas_mask(s, "custom_range", 2.0, None)
    assert list(mask) == [True, False, True]


def test_pandas_mask_custom_range_max_only() -> None:
    """Only max set: values above max are flagged."""
    s = pd.Series([1.0, 5.0, 11.0])
    mask = _invalid_rule_pandas_mask(s, "custom_range", None, 8.0)
    assert list(mask) == [False, False, True]


def test_pandas_mask_custom_range_no_bounds_returns_none() -> None:
    """custom_range with neither bound yields None (no-op)."""
    s = pd.Series([1.0, 2.0])
    assert _invalid_rule_pandas_mask(s, "custom_range", None, None) is None


def test_pandas_mask_unknown_rule_returns_none() -> None:
    """An unrecognised rule should return None rather than raise."""
    s = pd.Series([1.0, 2.0])
    assert _invalid_rule_pandas_mask(s, "does_not_exist", None, None) is None


# ---------------------------------------------------------------------------
# _invalid_rule_polars (unit tests on the expression helpers)
# ---------------------------------------------------------------------------


def _eval_polars_expr(df: pl.DataFrame, col: str, expr: Any) -> list:
    """Materialise a Polars expression for inspection in tests."""
    return df.with_columns(expr.alias("out"))["out"].to_list()


def test_polars_rule_negative_replaces_negatives() -> None:
    """Polars negative rule replaces values < 0 with the fill value."""
    df = pl.DataFrame({"v": [-2.0, 0.0, 3.0]})
    expr = _invalid_rule_polars(pl.col("v"), "negative", 0.0, None, None)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [0.0, 0.0, 3.0]


def test_polars_rule_zero_replaces_zeros() -> None:
    """Polars zero rule must replace only the zero entry."""
    df = pl.DataFrame({"v": [0.0, 1.0, -1.0]})
    expr = _invalid_rule_polars(pl.col("v"), "zero", -99.0, None, None)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [-99.0, 1.0, -1.0]


def test_polars_rule_custom_range_both_bounds() -> None:
    """Polars custom_range with both bounds replaces out-of-range values."""
    df = pl.DataFrame({"v": [-1.0, 5.0, 11.0]})
    expr = _invalid_rule_polars(pl.col("v"), "custom_range", 0.0, 0.0, 10.0)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [0.0, 5.0, 0.0]


def test_polars_rule_custom_range_min_only_replaces_below_min() -> None:
    """Polars custom_range with only min_value must flag values below min."""
    df = pl.DataFrame({"v": [1.0, 5.0, -3.0]})
    expr = _invalid_rule_polars(pl.col("v"), "custom_range", 0.0, 2.0, None)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [0.0, 5.0, 0.0]


def test_polars_rule_custom_range_max_only_replaces_above_max() -> None:
    """Polars custom_range with only max_value must flag values above max."""
    df = pl.DataFrame({"v": [1.0, 5.0, 11.0]})
    expr = _invalid_rule_polars(pl.col("v"), "custom_range", 0.0, None, 8.0)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [1.0, 5.0, 0.0]


def test_polars_rule_custom_range_no_bounds_is_noop() -> None:
    """custom_range with no bounds must leave the column unchanged."""
    df = pl.DataFrame({"v": [1.0, 2.0]})
    expr = _invalid_rule_polars(pl.col("v"), "custom_range", 0.0, None, None)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [1.0, 2.0]


def test_polars_rule_unknown_is_noop() -> None:
    """An unrecognised rule must leave the column unchanged."""
    df = pl.DataFrame({"v": [1.0, 2.0]})
    expr = _invalid_rule_polars(pl.col("v"), "unknown_rule", 0.0, None, None)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Calculator.fit
# ---------------------------------------------------------------------------


def test_calculator_fit_negative_rule() -> None:
    """fit stores the rule and column list correctly for negative replacement."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(df, {"columns": ["x"], "rule": "negative"})
    assert params["rule"] == "negative"
    assert "x" in params["columns"]


def test_calculator_fit_zero_rule() -> None:
    """fit stores zero rule correctly."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(df, {"columns": ["x"], "rule": "zero"})
    assert params["rule"] == "zero"


def test_calculator_fit_custom_range() -> None:
    """fit passes min_value and max_value through to the artifact."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(
        df, {"columns": ["x"], "rule": "custom_range", "min_value": 0, "max_value": 9}
    )
    assert params["min_value"] == 0
    assert params["max_value"] == 9


def test_calculator_fit_replace_inf_flags() -> None:
    """fit preserves the replace_inf / replace_neg_inf flags."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(
        df, {"columns": ["x"], "replace_inf": True, "replace_neg_inf": True}
    )
    assert params["replace_inf"] is True
    assert params["replace_neg_inf"] is True


def test_calculator_fit_empty_columns_list_short_circuits() -> None:
    """Explicit empty columns list must short-circuit to {} (no-op)."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(df, {"columns": []})
    assert params == {}


def test_calculator_fit_mode_alias() -> None:
    """Legacy 'mode' key must be accepted as a synonym for 'rule'."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(df, {"columns": ["x"], "mode": "negative"})
    assert params["rule"] == "negative"


def test_calculator_fit_value_key_stored() -> None:
    """Explicit 'value' replacement sentinel is preserved in the artifact."""
    df = _basic_df()
    params = InvalidValueReplacementCalculator().fit(
        df, {"columns": ["x"], "rule": "negative", "value": 0}
    )
    assert params["value"] == 0


# ---------------------------------------------------------------------------
# Applier.apply — pandas path
# ---------------------------------------------------------------------------


def test_applier_negative_rule_replaces_with_nan() -> None:
    """Negative values must become NaN under the negative rule."""
    df = _basic_df()
    calc = InvalidValueReplacementCalculator()
    applier = InvalidValueReplacementApplier()
    params = calc.fit(df, {"columns": ["x"], "rule": "negative"})
    result = applier.apply(df, params)
    # -3.0 and -1.0 were negative; they should be NaN now
    assert result["x"].iloc[0] != result["x"].iloc[0]  # NaN check
    assert result["x"].iloc[3] == 1.0  # positive value unchanged


def test_applier_zero_rule_replaces_zeros() -> None:
    """Zero rule must replace zeros with NaN (default replacement)."""
    df = pd.DataFrame({"v": [0.0, 1.0, 0.0, 5.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "rule": "zero",
        "replacement": np.nan,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert np.isnan(result["v"].iloc[0])
    assert result["v"].iloc[1] == 1.0


def test_applier_custom_range_replaces_out_of_range() -> None:
    """Values outside [1, 9] must be replaced with the sentinel."""
    df = pd.DataFrame({"v": [-5.0, 5.0, 15.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "rule": "custom_range",
        "min_value": 1.0,
        "max_value": 9.0,
        "replacement": -1.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result["v"].iloc[0] == -1.0  # < min
    assert result["v"].iloc[1] == 5.0  # in range
    assert result["v"].iloc[2] == -1.0  # > max


def test_applier_replace_inf_only() -> None:
    """Only +inf should be replaced when replace_inf=True, replace_neg_inf=False."""
    df = pd.DataFrame({"v": [np.inf, -np.inf, 1.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "replace_inf": True,
        "replace_neg_inf": False,
        "replacement": 0.0,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result["v"].iloc[0] == 0.0
    assert result["v"].iloc[1] == -np.inf  # untouched
    assert result["v"].iloc[2] == 1.0


def test_applier_replace_neg_inf_only() -> None:
    """Only -inf should be replaced when replace_neg_inf=True, replace_inf=False."""
    df = pd.DataFrame({"v": [np.inf, -np.inf, 2.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "replace_inf": False,
        "replace_neg_inf": True,
        "replacement": 0.0,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result["v"].iloc[0] == np.inf  # untouched
    assert result["v"].iloc[1] == 0.0
    assert result["v"].iloc[2] == 2.0


def test_applier_empty_params_is_noop() -> None:
    """Empty params dict must return the DataFrame unchanged."""
    df = _basic_df()
    result = InvalidValueReplacementApplier().apply(df, {})
    pd.testing.assert_frame_equal(result, df)


def test_applier_empty_dataframe() -> None:
    """Applying to an empty DataFrame must not raise and must return empty."""
    df = pd.DataFrame({"v": pd.Series([], dtype=float)})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "rule": "negative",
        "replacement": np.nan,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result.shape == (0, 1)


def test_applier_all_nan_column() -> None:
    """An all-NaN column should survive the transformation without error."""
    df = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "rule": "negative",
        "replacement": 0.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result["v"].isna().all()


def test_applier_custom_value_sentinel() -> None:
    """When 'value' is set it overrides the 'replacement' fill value."""
    df = pd.DataFrame({"v": [-1.0, 2.0, -3.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "rule": "negative",
        "replacement": np.nan,
        "value": -999.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    assert result["v"].iloc[0] == -999.0
    assert result["v"].iloc[2] == -999.0


# ---------------------------------------------------------------------------
# fit → apply equivalence
# ---------------------------------------------------------------------------


def test_fit_then_apply_matches_direct_params() -> None:
    """fit+apply must produce the same result as manually constructed params."""
    df = pd.DataFrame({"v": [-2.0, 0.0, 5.0, np.inf]})
    calc = InvalidValueReplacementCalculator()
    applier = InvalidValueReplacementApplier()
    params = calc.fit(df, {"columns": ["v"], "rule": "negative", "replace_inf": True})
    result = applier.apply(df, params)
    # Only positive finite values should survive
    assert result["v"].iloc[2] == 5.0
    assert np.isnan(result["v"].iloc[0])  # was -2
    assert np.isnan(result["v"].iloc[3])  # was inf


# ---------------------------------------------------------------------------
# Polars applier path — uncovered branches
# ---------------------------------------------------------------------------


def test_polars_applier_replace_inf_via_polars_engine() -> None:
    """Polars applier path must replace +inf when replace_inf=True."""
    df = pl.DataFrame({"v": [float("inf"), -float("inf"), 1.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "replace_inf": True,
        "replace_neg_inf": False,
        "replacement": 0.0,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["v"].iloc[0] == 0.0
    assert result["v"].iloc[1] == -float("inf")  # untouched


def test_polars_applier_replace_neg_inf_via_polars_engine() -> None:
    """Polars applier path must replace -inf when replace_neg_inf=True."""
    df = pl.DataFrame({"v": [float("inf"), -float("inf"), 2.0]})
    params: Dict[str, Any] = {
        "columns": ["v"],
        "replace_inf": False,
        "replace_neg_inf": True,
        "replacement": 0.0,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert result["v"].iloc[0] == float("inf")  # untouched
    assert result["v"].iloc[1] == 0.0


def test_polars_applier_no_valid_columns_is_noop() -> None:
    """Polars applier must short-circuit when no valid column is found."""
    df = pl.DataFrame({"v": [1.0, 2.0]})
    params: Dict[str, Any] = {
        "columns": ["nonexistent"],
        "rule": "negative",
        "replacement": 0.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    result = InvalidValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert list(result["v"]) == [1.0, 2.0]


def test_infer_output_schema_returns_input_schema_unchanged() -> None:
    """infer_output_schema must pass through the input schema untouched."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a", "b"], {"a": "float64", "b": "float64"})
    result = InvalidValueReplacementCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# Engine parity (pandas vs polars Applier paths)
# ---------------------------------------------------------------------------

# The `engine_parity` profile is registered in test_engine_parity.py; use
# the settings directly here to avoid a duplicate-registration error.
_FINITE_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def _numeric_frame_for_iv(draw: st.DrawFn, min_rows: int = 5, max_rows: int = 30) -> pd.DataFrame:
    """Generate a DataFrame with two numeric columns for parity tests."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    a = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    b = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
    return pd.DataFrame({"a": a, "b": b})


@settings(max_examples=25, deadline=None)
@given(df=_numeric_frame_for_iv())
def test_invalid_value_apply_engine_parity_negative(df: pd.DataFrame) -> None:
    """pandas and polars paths must produce identical results for the negative rule."""
    params: Dict[str, Any] = {
        "columns": ["a", "b"],
        "rule": "negative",
        "replacement": 0.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    applier = InvalidValueReplacementApplier()
    pd_result = applier.apply(df, params)
    pl_result = applier.apply(pl.from_pandas(df), params)
    # Convert polars result to pandas for comparison
    if hasattr(pl_result, "to_pandas"):
        pl_result = pl_result.to_pandas()
    pd.testing.assert_frame_equal(
        pd_result.reset_index(drop=True),
        pl_result.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )


@settings(max_examples=25, deadline=None)
@given(df=_numeric_frame_for_iv())
def test_invalid_value_apply_engine_parity_custom_range(df: pd.DataFrame) -> None:
    """Pandas and polars custom_range paths must agree on which values are replaced."""
    params: Dict[str, Any] = {
        "columns": ["a", "b"],
        "rule": "custom_range",
        "min_value": -100.0,
        "max_value": 100.0,
        "replacement": 0.0,
        "replace_inf": False,
        "replace_neg_inf": False,
    }
    applier = InvalidValueReplacementApplier()
    pd_result = applier.apply(df, params)
    pl_result = applier.apply(pl.from_pandas(df), params)
    if hasattr(pl_result, "to_pandas"):
        pl_result = pl_result.to_pandas()
    pd.testing.assert_frame_equal(
        pd_result.reset_index(drop=True),
        pl_result.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )
