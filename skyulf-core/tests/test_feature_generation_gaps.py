"""Coverage-gap tests for feature_generation `_common`, `_pandas_ops`, and `polynomial`.

Complements ``test_feature_generation_full.py`` (arithmetic/ratio/similarity/
datetime operations) by exercising the group_agg pandas-op edge cases, the
similarity-fallback path (no rapidfuzz), the polynomial features node, and
several `_common.py` helpers directly.
"""

from typing import Any, Dict

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_generation._common import (
    _coerce_float,
    _resolve_group_agg_cols,
    _resolve_output_col,
    _resolve_similarity_pair,
    _safe_divide,
)
from skyulf.preprocessing.feature_generation._pandas_ops import _pandas_group_agg

_APPLIER = FeatureGenerationApplier()
_CALC = FeatureGenerationCalculator()


# ---------------------------------------------------------------------------
# _common.py helpers
# ---------------------------------------------------------------------------


def test_coerce_float_valid_numeric_string() -> None:
    """A numeric string coerces to its float value."""
    assert _coerce_float("3.5") == 3.5


def test_coerce_float_none_returns_none() -> None:
    """``None`` input returns ``None`` without raising."""
    assert _coerce_float(None) is None


def test_coerce_float_nan_returns_none() -> None:
    """A NaN float value is normalised to ``None``."""
    assert _coerce_float(float("nan")) is None


def test_coerce_float_non_numeric_string_returns_none() -> None:
    """A non-numeric string is treated as missing."""
    assert _coerce_float("abc") is None


def test_safe_divide_handles_zero_denominator() -> None:
    """Zero denominators are replaced by epsilon instead of raising/inf."""
    num = pd.Series([1.0, 2.0])
    den = pd.Series([0.0, 2.0])
    result = _safe_divide(num, den, epsilon=1e-9)
    assert result.notna().all()
    assert result.iloc[1] == 1.0


def test_safe_divide_handles_nan_denominator() -> None:
    """NaN denominators are filled with epsilon before dividing."""
    num = pd.Series([1.0])
    den = pd.Series([float("nan")])
    result = _safe_divide(num, den, epsilon=1e-9)
    assert result.notna().all()


def test_resolve_output_col_no_collision() -> None:
    """When the requested name is free, it is returned unchanged."""
    op = {"output_column": "my_col"}
    assert _resolve_output_col(op, 0, ["a", "b"], allow_overwrite=False) == "my_col"


def test_resolve_output_col_default_name_from_operation_type() -> None:
    """Without an explicit name, the default is ``<operation_type>_<i>``."""
    op = {"operation_type": "arithmetic"}
    assert _resolve_output_col(op, 3, [], allow_overwrite=False) == "arithmetic_3"


def test_resolve_output_col_with_prefix() -> None:
    """An ``output_prefix`` is prepended to the default generated name."""
    op = {"operation_type": "ratio", "output_prefix": "custom"}
    assert _resolve_output_col(op, 0, [], allow_overwrite=False) == "custom_ratio_0"


def test_resolve_output_col_collision_appends_suffix() -> None:
    """A colliding name gets a numeric suffix instead of overwriting."""
    op = {"output_column": "dup"}
    result = _resolve_output_col(op, 0, ["dup", "dup_1"], allow_overwrite=False)
    assert result == "dup_2"


def test_resolve_output_col_overwrite_allowed_keeps_name() -> None:
    """When overwrite is allowed, a colliding name is reused as-is."""
    op = {"output_column": "dup"}
    result = _resolve_output_col(op, 0, ["dup"], allow_overwrite=True)
    assert result == "dup"


def test_resolve_similarity_pair_missing_columns_returns_none() -> None:
    """Unresolvable similarity pairs (missing columns) return ``None``."""
    op = {"input_columns": ["x"], "secondary_columns": ["missing"]}
    assert _resolve_similarity_pair(op, ["x"]) is None


def test_resolve_similarity_pair_uses_second_input_column_as_fallback() -> None:
    """Without ``secondary_columns``, the second input column is used as pair-b."""
    op = {"input_columns": ["x", "y"]}
    assert _resolve_similarity_pair(op, ["x", "y"]) == ("x", "y")


def test_resolve_group_agg_cols_missing_secondary_returns_none() -> None:
    """A group_agg op without a resolvable target column returns ``None``."""
    op = {"input_columns": ["g"], "secondary_columns": []}
    assert _resolve_group_agg_cols(op, ["g"]) is None


def test_resolve_group_agg_cols_defaults_method_to_mean() -> None:
    """When ``method`` is absent, group_agg defaults to "mean"."""
    op = {"input_columns": ["g"], "secondary_columns": ["v"]}
    assert _resolve_group_agg_cols(op, ["g", "v"]) == ("g", "v", "mean")


# ---------------------------------------------------------------------------
# _pandas_ops.py: group_agg edge cases
# ---------------------------------------------------------------------------


def test_pandas_group_agg_unknown_method_returns_none() -> None:
    """An unsupported aggregation method is rejected (no column produced)."""
    df = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    op = {"input_columns": ["g"], "secondary_columns": ["v"], "method": "bogus"}
    assert _pandas_group_agg(op, df, 1e-9) is None


def test_pandas_group_agg_std_computation() -> None:
    """The "std" aggregation broadcasts each group's std back to every row."""
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 10.0]})
    op = {"input_columns": ["g"], "secondary_columns": ["v"], "method": "std"}
    result = _pandas_group_agg(op, df, 1e-9)
    assert result is not None
    assert result.iloc[2] == 0.0  # group "b" has zero variance


# ---------------------------------------------------------------------------
# Similarity fallback (SequenceMatcher, when rapidfuzz absent)
# ---------------------------------------------------------------------------


def test_similarity_token_sort_ratio_method() -> None:
    """The ``token_sort_ratio`` method still produces a bounded score."""
    df = pd.DataFrame({"x": ["quick brown fox"], "y": ["brown quick fox"]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "similarity",
                    "method": "token_sort_ratio",
                    "input_columns": ["x", "y"],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert 0 <= out["similarity_0"].iloc[0] <= 100


# ---------------------------------------------------------------------------
# Empty / edge-case frames for the full apply pipeline
# ---------------------------------------------------------------------------


def test_feature_generation_empty_dataframe_returns_empty() -> None:
    """Running ops against a zero-row frame must not raise."""
    df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {"operation_type": "arithmetic", "method": "add", "input_columns": ["a", "b"]}
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert len(out) == 0
    assert "arithmetic_0" in out.columns


def test_feature_generation_single_row() -> None:
    """A single-row frame is processed correctly for arithmetic ops."""
    df = pd.DataFrame({"a": [5.0], "b": [2.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {"operation_type": "arithmetic", "method": "multiply", "input_columns": ["a", "b"]}
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert out["arithmetic_0"].iloc[0] == 10.0


def test_feature_generation_all_nan_column_arithmetic() -> None:
    """An all-NaN column is coerced to the fillna default before arithmetic."""
    df = pd.DataFrame({"a": [None, None], "b": [1.0, 2.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {"operation_type": "arithmetic", "method": "add", "input_columns": ["a", "b"]}
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    # "a" is all-NaN -> filled with 0 -> add reduces to just "b"'s values.
    assert out["arithmetic_0"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------


def test_polynomial_features_fit_produces_feature_names() -> None:
    """Fitting records the expanded polynomial feature names."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"columns": ["a", "b"], "degree": 2})
    assert art["type"] == "polynomial_features"
    assert "feature_names" in art
    assert len(art["feature_names"]) > 0


def test_polynomial_features_apply_adds_interaction_column_pandas() -> None:
    """Applying degree-2 polynomial features adds an "a b" interaction column."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"columns": ["a", "b"], "degree": 2})
    out = PolynomialFeaturesApplier().apply(df, art)
    assert "poly_a_b" in out.columns
    # a*b for row 0 = 1*4 = 4
    assert out["poly_a_b"].iloc[0] == 4.0


def test_polynomial_features_apply_polars_roundtrip() -> None:
    """The polars apply path concatenates polynomial columns onto the frame."""
    import polars as pl

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"columns": ["a", "b"], "degree": 2})
    df_pl = pl.from_pandas(df)
    out = PolynomialFeaturesApplier().apply(df_pl, art)
    assert isinstance(out, pl.DataFrame)
    assert "poly_a_b" in out.columns


def test_polynomial_features_include_input_features() -> None:
    """``include_input_features=True`` keeps the original degree-1 columns too."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    art = PolynomialFeaturesCalculator().fit(
        df, {"columns": ["a", "b"], "degree": 2, "include_input_features": True}
    )
    out = PolynomialFeaturesApplier().apply(df, art)
    assert "poly_a" in out.columns
    assert "poly_b" in out.columns


# ---------------------------------------------------------------------------
# _pandas_ops.py: arithmetic with constant operands, ratio/similarity misses,
# unknown op types, and the malformed-op exception-swallow path.
# ---------------------------------------------------------------------------


def test_arithmetic_subtract_with_constant_value() -> None:
    """A constant operand is subtracted from the running column total."""
    df = pd.DataFrame({"a": [10.0, 20.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "subtract",
                    "input_columns": ["a"],
                    "constants": [3.0],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert out["arithmetic_0"].tolist() == [7.0, 17.0]


def test_arithmetic_multiply_with_constant_value() -> None:
    """A constant operand scales the running column product."""
    df = pd.DataFrame({"a": [2.0, 3.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "multiply",
                    "input_columns": ["a"],
                    "constants": [5.0],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert out["arithmetic_0"].tolist() == [10.0, 15.0]


def test_arithmetic_divide_by_only_constants() -> None:
    """Dividing two constants (no input columns) still yields a broadcast series."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "divide",
                    "input_columns": [],
                    "constants": [10.0, 2.0],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert out["arithmetic_0"].tolist() == [5.0, 5.0]


def test_arithmetic_divide_no_columns_no_constants_skips() -> None:
    """With neither input columns nor constants, the op produces no column."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    params = _CALC.fit(
        df,
        {"operations": [{"operation_type": "arithmetic", "method": "divide", "input_columns": []}]},
    )
    out = _APPLIER.apply(df, params)
    assert "arithmetic_0" not in out.columns


def test_ratio_missing_denominator_columns_skips() -> None:
    """A ratio op with no resolvable denominator columns produces no output."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "ratio",
                    "input_columns": ["a"],
                    "secondary_columns": ["missing"],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert "ratio_0" not in out.columns


def test_similarity_unresolvable_pair_skips() -> None:
    """A similarity op that cannot resolve a column pair produces no output."""
    df = pd.DataFrame({"a": ["x", "y"]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "similarity",
                    "input_columns": ["a"],
                    "secondary_columns": ["missing"],
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    assert "similarity_0" not in out.columns


def test_unknown_operation_type_is_skipped() -> None:
    """An unrecognized ``operation_type`` is silently ignored, not an error."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    params = _CALC.fit(df, {"operations": [{"operation_type": "not_a_real_op"}]})
    out = _APPLIER.apply(df, params)
    assert list(out.columns) == ["a"]


def test_malformed_operation_is_swallowed_without_raising() -> None:
    """A malformed op (e.g. bad round_digits) is caught and skipped, not raised."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    params = _CALC.fit(
        df,
        {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "add",
                    "input_columns": ["a", "b"],
                    "round_digits": "not-a-number",
                }
            ]
        },
    )
    out = _APPLIER.apply(df, params)
    # The malformed op is skipped entirely; no output column and no crash.
    assert "arithmetic_0" not in out.columns


def test_polynomial_features_empty_columns_returns_empty_artifact() -> None:
    """No configured/detected columns yields an empty artifact (no-op)."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"columns": []})
    assert art == {}


def test_polynomial_features_auto_detect_columns() -> None:
    """``auto_detect=True`` picks up numeric columns when none are configured."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"auto_detect": True, "degree": 2})
    assert set(art["columns"]) == {"a", "b"}


def test_polynomial_features_apply_missing_columns_is_noop() -> None:
    """Applying against a frame missing the fitted columns leaves it unchanged."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    art = PolynomialFeaturesCalculator().fit(df, {"columns": ["a", "b"], "degree": 2})
    other_df = pd.DataFrame({"c": [1.0, 2.0]})
    out = PolynomialFeaturesApplier().apply(other_df, art)
    assert list(out.columns) == ["c"]


@settings(max_examples=25, deadline=None)
@given(
    a=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=3, max_size=15),
    b=st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=3, max_size=15),
)
def test_safe_divide_never_raises_or_produces_inf(a: list, b: list) -> None:
    """Property: ``_safe_divide`` never produces inf/NaN for arbitrary finite inputs."""
    n = min(len(a), len(b))
    num = pd.Series(a[:n])
    den = pd.Series(b[:n])
    result = _safe_divide(num, den, epsilon=1e-9)
    assert result.notna().all()
    assert not result.isin([float("inf"), float("-inf")]).any()
