"""Unit tests for the InvalidValueReplacement cleaning node.

Covers: mask helpers, inf-replacement helpers, resolver, Calculator.fit
branches, Applier.apply (pandas + polars), edge cases, and engine-parity.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.cleaning.invalid_value import (
    InvalidValueReplacementApplier,
    InvalidValueReplacementCalculator,
    _invalid_rule_pandas_mask,
    _invalid_rule_polars,
    _resolve_invalid_replacement,
)

_pandas_mask_cases = TestCaseLoader("preprocessing/invalid_value_pandas_mask").load()
_polars_rule_cases = TestCaseLoader("preprocessing/invalid_value_polars_rule").load()
_applier_pandas_cases = TestCaseLoader("preprocessing/invalid_value_applier_pandas").load()
_polars_applier_cases = TestCaseLoader("preprocessing/invalid_value_polars_applier").load()

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


@pytest.mark.parametrize(*_pandas_mask_cases)
def test_pandas_mask(
    values: list,
    rule: str,
    min_value: Optional[float],
    max_value: Optional[float],
    expected: Optional[list],
) -> None:
    """_invalid_rule_pandas_mask must flag values per rule, or return None for no-ops."""
    s = pd.Series(values)
    mask = _invalid_rule_pandas_mask(s, rule, min_value, max_value)
    if expected is None:
        assert mask is None
    else:
        assert list(mask) == expected


# ---------------------------------------------------------------------------
# _invalid_rule_polars (unit tests on the expression helpers)
# ---------------------------------------------------------------------------


def _eval_polars_expr(df: pl.DataFrame, col: str, expr: Any) -> list:
    """Materialise a Polars expression for inspection in tests."""
    return df.with_columns(expr.alias("out"))["out"].to_list()


@pytest.mark.parametrize(*_polars_rule_cases)
def test_polars_rule(
    values: list,
    rule: str,
    replacement: float,
    min_value: Optional[float],
    max_value: Optional[float],
    expected: list,
) -> None:
    """_invalid_rule_polars must replace flagged values with the given replacement."""
    df = pl.DataFrame({"v": values})
    expr = _invalid_rule_polars(pl.col("v"), rule, replacement, min_value, max_value)
    result = df.with_columns(expr.alias("out"))["out"].to_list()
    assert result == expected


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


@pytest.mark.parametrize(*_applier_pandas_cases)
def test_applier_pandas(values: list, params: Dict[str, Any], expected: list) -> None:
    """InvalidValueReplacementApplier must replace flagged values per rule config."""
    df = pd.DataFrame({"v": values})
    result = InvalidValueReplacementApplier().apply(df, params)
    for got, exp in zip(result["v"], expected):
        if exp is None:
            assert np.isnan(got)
        else:
            assert got == exp


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


@pytest.mark.parametrize(*_polars_applier_cases)
def test_polars_applier(values: list, params: Dict[str, Any], expected: list) -> None:
    """Polars applier path must replace +inf/-inf per the given flags."""
    df = pl.DataFrame({"v": values})
    result = InvalidValueReplacementApplier().apply(df, params)
    if hasattr(result, "to_pandas"):
        result = result.to_pandas()
    assert list(result["v"]) == expected


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


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``income`` values — closer to production data than the
    small synthetic frame used elsewhere in this file.
    """

    def test_custom_range_replaces_income_outliers_and_preserves_missing(self) -> None:
        df = load_sample_dataset("customers")
        params: Dict[str, Any] = {
            "columns": ["income"],
            "rule": "custom_range",
            "min_value": 30000.0,
            "max_value": 90000.0,
            "replacement": -1.0,
            "replace_inf": False,
            "replace_neg_inf": False,
        }
        out = InvalidValueReplacementApplier().apply(df, params)

        out_of_range = (df["income"] < 30000.0) | (df["income"] > 90000.0)
        assert (out.loc[out_of_range.fillna(False), "income"] == -1.0).all()
        # Missing income values must remain missing, not be replaced.
        assert out.loc[df["income"].isna(), "income"].isna().all()
        in_range = ~out_of_range.fillna(True) & df["income"].notna()
        pd.testing.assert_series_equal(
            out.loc[in_range, "income"], df.loc[in_range, "income"], check_names=False
        )
