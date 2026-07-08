"""Unit tests for skyulf.preprocessing.casting.

Covers CastingCalculator.fit, CastingApplier.apply (pandas path), the
TYPE_ALIASES table, helper functions (_coerce_boolean_value, _cast_float,
_cast_int, _cast_bool, _cast_datetime), and engine-parity via hypothesis.

All tests use real DataFrames — no mocking of pandas.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.casting import (
    TYPE_ALIASES,
    CastingApplier,
    CastingCalculator,
    _cast_bool,
    _cast_datetime,
    _cast_float,
    _cast_int,
    _coerce_boolean_value,
    _drop_fractional_or_raise,
)

_coerce_boolean_value_cases = TestCaseLoader("preprocessing/casting").load()

# ---------------------------------------------------------------------------
# TYPE_ALIASES sanity check
# ---------------------------------------------------------------------------


def test_type_aliases_contains_common_names() -> None:
    """TYPE_ALIASES must map every expected alias to a canonical dtype string."""
    for alias in ("float", "int", "string", "bool", "datetime", "category"):
        assert alias in TYPE_ALIASES, f"Missing alias: {alias}"


def test_type_aliases_float_resolves_to_float64() -> None:
    """'float' alias must resolve to 'float64'."""
    assert TYPE_ALIASES["float"] == "float64"


def test_type_aliases_bool_resolves_to_boolean() -> None:
    """'bool' alias must resolve to pandas nullable 'boolean'."""
    assert TYPE_ALIASES["bool"] == "boolean"


# ---------------------------------------------------------------------------
# _coerce_boolean_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_coerce_boolean_value_cases)
def test_coerce_boolean_value_parametrized(value: Any, expected: Any) -> None:
    """_coerce_boolean_value must convert common truthy/falsy strings and ints."""
    assert _coerce_boolean_value(value) == expected


def test_coerce_boolean_value_nan_returns_none() -> None:
    """NaN input must return None (maps to pd.NA in the boolean series)."""
    assert _coerce_boolean_value(float("nan")) is None


def test_coerce_boolean_value_numeric_not_0_or_1_returns_none() -> None:
    """Numeric values other than 0 or 1 must not map to bool."""
    assert _coerce_boolean_value(2) is None
    assert _coerce_boolean_value(-1) is None


# ---------------------------------------------------------------------------
# _cast_float
# ---------------------------------------------------------------------------


def test_cast_float_converts_string_series() -> None:
    """String numerics must be converted to float64."""
    s = pd.Series(["1.5", "2.0", "3.7"])
    result = _cast_float(s, "float64", coerce_on_error=True)
    assert result.dtype == np.float64
    assert list(result) == [1.5, 2.0, 3.7]


def test_cast_float_coerce_bad_value_to_nan() -> None:
    """Non-numeric strings with coerce_on_error=True must become NaN."""
    s = pd.Series(["1.0", "bad", "3.0"])
    result = _cast_float(s, "float64", coerce_on_error=True)
    assert np.isnan(result.iloc[1])


def test_cast_float_raise_on_bad_value() -> None:
    """coerce_on_error=False must raise for non-numeric strings."""
    s = pd.Series(["1.0", "bad"])
    with pytest.raises(ValueError):
        _cast_float(s, "float64", coerce_on_error=False)


# ---------------------------------------------------------------------------
# _cast_int
# ---------------------------------------------------------------------------


def test_cast_int_integer_strings() -> None:
    """String integers must be cast to int64."""
    s = pd.Series(["1", "2", "3"])
    result = _cast_int(s, "col", "int64", coerce_on_error=True)
    assert result.dtype in (np.int64, pd.Int64Dtype())


def test_cast_int_fractional_coerced_to_nan() -> None:
    """Fractional floats with coerce_on_error=True must be NaN-padded to Int64."""
    s = pd.Series([1.0, 2.7, 3.0])
    result = _cast_int(s, "col", "int64", coerce_on_error=True)
    # 2.7 should become NaN — forces nullable Int64.
    assert result.isna().any()


def test_cast_int_fractional_raises_without_coerce() -> None:
    """Fractional values with coerce_on_error=False must raise ValueError."""
    s = pd.Series([1.0, 2.7])
    with pytest.raises(ValueError, match="fractional"):
        _cast_int(s, "col", "int64", coerce_on_error=False)


# ---------------------------------------------------------------------------
# _drop_fractional_or_raise
# ---------------------------------------------------------------------------


def test_drop_fractional_no_fractional_returns_unchanged() -> None:
    """A series with no fractional values must be returned as-is."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = _drop_fractional_or_raise(s, "col", coerce_on_error=False)
    pd.testing.assert_series_equal(result, s)


def test_drop_fractional_coerce_sets_nan() -> None:
    """Fractional cells must be set to NaN when coerce_on_error=True."""
    s = pd.Series([1.0, 2.5, 3.0])
    result = _drop_fractional_or_raise(s, "col", coerce_on_error=True)
    assert np.isnan(result.iloc[1])
    assert result.iloc[0] == 1.0


# ---------------------------------------------------------------------------
# _cast_bool
# ---------------------------------------------------------------------------


def test_cast_bool_from_int_series() -> None:
    """Integer 0/1 series must be cast to pandas boolean dtype."""
    s = pd.Series([0, 1, 0, 1])
    result = _cast_bool(s, coerce_on_error=True)
    assert result.dtype == pd.BooleanDtype()


def test_cast_bool_from_truthy_strings() -> None:
    """'true'/'false' strings must be mapped to boolean values."""
    s = pd.Series(["true", "false", "yes", "no"])
    result = _cast_bool(s, coerce_on_error=True)
    assert bool(result.iloc[0]) is True
    assert bool(result.iloc[1]) is False


def test_cast_bool_undecidable_string_becomes_na() -> None:
    """Strings that cannot be decoded must become pd.NA rather than raising."""
    s = pd.Series(["maybe", "true"])
    result = _cast_bool(s, coerce_on_error=True)
    assert pd.isna(result.iloc[0])
    assert bool(result.iloc[1]) is True


# ---------------------------------------------------------------------------
# _cast_datetime
# ---------------------------------------------------------------------------


def test_cast_datetime_iso_strings() -> None:
    """ISO 8601 strings must be parsed to datetime64 without errors."""
    s = pd.Series(["2024-01-01", "2024-06-15", "2024-12-31"])
    result = _cast_datetime(s, coerce_on_error=True)
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert result.iloc[0] == pd.Timestamp("2024-01-01")


def test_cast_datetime_bad_string_coerced_to_nat() -> None:
    """Non-parseable strings with coerce_on_error=True must become NaT."""
    s = pd.Series(["2024-01-01", "not-a-date"])
    result = _cast_datetime(s, coerce_on_error=True)
    assert pd.isna(result.iloc[1])


def test_cast_datetime_bad_string_raises_without_coerce() -> None:
    """Non-parseable strings with coerce_on_error=False must raise."""
    s = pd.Series(["not-a-date"])
    with pytest.raises(ValueError):
        _cast_datetime(s, coerce_on_error=False)


# ---------------------------------------------------------------------------
# CastingCalculator.fit
# ---------------------------------------------------------------------------


def test_casting_calculator_fit_column_types_config() -> None:
    """Fit with column_types config must build a type_map with resolved dtypes."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    config: dict[str, Any] = {"column_types": {"a": "float"}}
    artifact = CastingCalculator().fit(df, config)
    assert artifact["type_map"]["a"] == "float64"
    assert artifact["coerce_on_error"] is True


def test_casting_calculator_fit_target_type_with_columns() -> None:
    """Fit with target_type + columns must map all columns to the resolved type."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    config: dict[str, Any] = {"target_type": "float", "columns": ["x", "y"]}
    artifact = CastingCalculator().fit(df, config)
    assert artifact["type_map"]["x"] == "float64"
    assert artifact["type_map"]["y"] == "float64"


def test_casting_calculator_fit_skips_nonexistent_columns() -> None:
    """Fit must silently ignore columns that don't exist in the DataFrame."""
    df = pd.DataFrame({"a": [1]})
    config: dict[str, Any] = {"column_types": {"a": "int", "z": "float"}}
    artifact = CastingCalculator().fit(df, config)
    assert "z" not in artifact["type_map"]
    assert "a" in artifact["type_map"]


def test_casting_calculator_fit_empty_config_returns_empty_map() -> None:
    """Fit with no type instructions must return an empty type_map."""
    df = pd.DataFrame({"a": [1.0]})
    artifact = CastingCalculator().fit(df, {})
    assert artifact["type_map"] == {}


def test_casting_calculator_fit_coerce_on_error_false() -> None:
    """Fit must respect coerce_on_error=False from config."""
    df = pd.DataFrame({"a": [1.0]})
    config: dict[str, Any] = {"column_types": {"a": "int"}, "coerce_on_error": False}
    artifact = CastingCalculator().fit(df, config)
    assert artifact["coerce_on_error"] is False


# ---------------------------------------------------------------------------
# CastingApplier.apply (pandas path)
# ---------------------------------------------------------------------------


def test_casting_applier_casts_float_column() -> None:
    """Applier must convert a string column to float64 using the type_map."""
    df = pd.DataFrame({"val": ["1.5", "2.0", "3.7"]})
    params: dict[str, Any] = {"type_map": {"val": "float64"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert result["val"].dtype == np.float64


def test_casting_applier_casts_int_column() -> None:
    """Applier must convert float column to integer (Int64 if NaN present)."""
    df = pd.DataFrame({"n": [1.0, 2.0, 3.0]})
    params: dict[str, Any] = {"type_map": {"n": "int64"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert pd.api.types.is_integer_dtype(result["n"])


def test_casting_applier_casts_bool_column() -> None:
    """Applier must convert 0/1 integer column to boolean dtype."""
    df = pd.DataFrame({"flag": [0, 1, 0]})
    params: dict[str, Any] = {"type_map": {"flag": "boolean"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert result["flag"].dtype == pd.BooleanDtype()


def test_casting_applier_casts_datetime_column() -> None:
    """Applier must parse ISO date strings to datetime64 dtype."""
    df = pd.DataFrame({"dt": ["2024-01-01", "2024-06-15"]})
    params: dict[str, Any] = {"type_map": {"dt": "datetime64[ns]"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert pd.api.types.is_datetime64_any_dtype(result["dt"])


def test_casting_applier_casts_category_column() -> None:
    """Applier must convert a string column to the pandas category dtype."""
    df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
    params: dict[str, Any] = {"type_map": {"cat": "category"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert result["cat"].dtype.name == "category"


def test_casting_applier_empty_type_map_returns_unchanged() -> None:
    """An empty type_map must return the DataFrame unchanged."""
    df = pd.DataFrame({"a": [1, 2]})
    params: dict[str, Any] = {"type_map": {}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    pd.testing.assert_frame_equal(result, df)


def test_casting_applier_skips_nonexistent_columns() -> None:
    """Applier must silently skip columns not present in the DataFrame."""
    df = pd.DataFrame({"a": [1.0]})
    params: dict[str, Any] = {"type_map": {"z": "float64"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    # 'z' must not appear; 'a' must be untouched.
    assert "z" not in result.columns
    pd.testing.assert_frame_equal(result, df)


def test_casting_applier_all_nan_column_coerced_to_float() -> None:
    """All-NaN column cast to float must remain all-NaN without raising."""
    df = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
    params: dict[str, Any] = {"type_map": {"v": "float64"}, "coerce_on_error": True}
    result = CastingApplier().apply(df, params)
    assert result["v"].isna().all()


def test_casting_applier_coerce_false_raises_on_bad_cast() -> None:
    """coerce_on_error=False must propagate exceptions from bad casts."""
    df = pd.DataFrame({"a": ["not_a_number", "also_bad"]})
    params: dict[str, Any] = {"type_map": {"a": "float64"}, "coerce_on_error": False}
    with pytest.raises(ValueError):
        CastingApplier().apply(df, params)


def test_casting_applier_does_not_mutate_input() -> None:
    """Applier must return a new DataFrame and not modify the input in-place."""
    df = pd.DataFrame({"val": ["1.0", "2.0"]})
    original_dtype = df["val"].dtype
    params: dict[str, Any] = {"type_map": {"val": "float64"}, "coerce_on_error": True}
    CastingApplier().apply(df, params)
    assert df["val"].dtype == original_dtype


def test_cast_bool_coerce_false_raises_for_uncoerceable_series() -> None:
    """_cast_bool with coerce_on_error=False must raise for uncastable series."""
    # A series that pandas cannot cast directly to boolean and has no coerce path.
    # Using walrus-operator strings that _coerce_boolean_value would return None for.
    s = pd.Series(["maybe", "perhaps"])
    with pytest.raises((TypeError, ValueError)):
        _cast_bool(s, coerce_on_error=False)


# ---------------------------------------------------------------------------
# CastingCalculator.infer_output_schema
# ---------------------------------------------------------------------------


def test_infer_output_schema_rewrites_dtypes() -> None:
    """infer_output_schema must update dtype labels for configured columns."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a", "b"], dtypes={"a": "string", "b": "string"})
    config: dict[str, Any] = {"column_types": {"a": "float", "b": "int"}}
    result = CastingCalculator().infer_output_schema(schema, config)
    assert result.dtypes["a"] == "float64"
    assert result.dtypes["b"] == "int64"


def test_infer_output_schema_target_type_with_columns() -> None:
    """infer_output_schema must apply a single target_type across all named columns."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["x", "y"], dtypes={"x": "string", "y": "string"})
    config: dict[str, Any] = {"target_type": "float", "columns": ["x", "y"]}
    result = CastingCalculator().infer_output_schema(schema, config)
    assert result.dtypes["x"] == "float64"
    assert result.dtypes["y"] == "float64"


def test_infer_output_schema_empty_config_returns_same_schema() -> None:
    """infer_output_schema with no type instructions must return the input schema."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], dtypes={"a": "string"})
    result = CastingCalculator().infer_output_schema(schema, {})
    assert result == schema


# ---------------------------------------------------------------------------
# Full fit → apply roundtrip
# ---------------------------------------------------------------------------


def test_calculator_fit_then_applier_apply_roundtrip() -> None:
    """Fit then apply must produce correct output on the same data."""
    df = pd.DataFrame({"a": ["1.5", "2.0", "3.0"], "b": [True, False, True]})
    config: dict[str, Any] = {"column_types": {"a": "float", "b": "boolean"}}
    artifact = CastingCalculator().fit(df, config)
    result = CastingApplier().apply(df, artifact)
    assert result["a"].dtype == np.float64
    assert result["b"].dtype == pd.BooleanDtype()


def test_apply_on_new_data_uses_fitted_type_map() -> None:
    """Applier must cast new data using the artifact computed on training data."""
    train_df = pd.DataFrame({"score": ["10", "20", "30"]})
    config: dict[str, Any] = {"column_types": {"score": "int"}}
    artifact = CastingCalculator().fit(train_df, config)

    new_df = pd.DataFrame({"score": ["40", "50"]})
    result = CastingApplier().apply(new_df, artifact)
    assert pd.api.types.is_integer_dtype(result["score"])
    assert list(result["score"]) == [40, 50]


# ---------------------------------------------------------------------------
# Engine-parity test (hypothesis)
# ---------------------------------------------------------------------------

try:
    import polars as pl

    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

if _POLARS_AVAILABLE:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st

    _FINITE_FLOAT = st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
    )

    @st.composite
    def _float_frame(draw: st.DrawFn) -> pd.DataFrame:
        """Generate a small numeric DataFrame with columns 'x' and 'y'."""
        n = draw(st.integers(min_value=5, max_value=30))
        x = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
        y = draw(st.lists(_FINITE_FLOAT, min_size=n, max_size=n))
        return pd.DataFrame({"x": x, "y": y})

    @settings(max_examples=25, deadline=None)
    @given(df=_float_frame())
    def test_casting_fit_artifact_engine_parity(df: pd.DataFrame) -> None:
        """CastingCalculator.fit must produce identical type_maps on pandas and polars.

        The fit step is config-only (no per-engine math), so both engines must
        return exactly the same artifact dictionary.
        """
        assume(len(df) >= 5)
        config: dict[str, Any] = {"column_types": {"x": "float64", "y": "float64"}}
        pd_artifact = CastingCalculator().fit(df, dict(config))
        pl_artifact = CastingCalculator().fit(pl.from_pandas(df), dict(config))
        # type_map must be identical — no numeric tolerance needed, just equality.
        assert pd_artifact["type_map"] == pl_artifact["type_map"]
        assert pd_artifact["coerce_on_error"] == pl_artifact["coerce_on_error"]

    def test_casting_apply_polars_float_cast() -> None:
        """Polars apply path must cast numeric column to Float64."""
        df_pd = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df_pl = pl.from_pandas(df_pd)
        params: dict[str, Any] = {
            "type_map": {"x": "float64", "y": "int64"},
            "coerce_on_error": True,
        }
        result = CastingApplier().apply(df_pl, params)
        # Result must remain a polars frame with correct dtypes.
        assert result["x"].dtype == pl.Float64
        assert result["y"].dtype == pl.Int64

    def test_casting_apply_polars_empty_type_map_returns_unchanged() -> None:
        """Polars apply with empty type_map must return the frame unchanged."""
        df_pl = pl.DataFrame({"a": [1, 2]})
        params: dict[str, Any] = {"type_map": {}, "coerce_on_error": True}
        result = CastingApplier().apply(df_pl, params)
        assert result.equals(df_pl)

    def test_casting_apply_polars_skips_nonexistent_columns() -> None:
        """Polars apply must silently skip columns absent from the frame."""
        df_pl = pl.DataFrame({"a": [1, 2]})
        params: dict[str, Any] = {"type_map": {"z": "float64"}, "coerce_on_error": True}
        result = CastingApplier().apply(df_pl, params)
        assert result.equals(df_pl)

    def test_casting_apply_polars_bool_cast() -> None:
        """Polars apply path must cast column to Boolean."""
        df_pl = pl.DataFrame({"flag": [0, 1, 0]})
        params: dict[str, Any] = {"type_map": {"flag": "bool"}, "coerce_on_error": True}
        result = CastingApplier().apply(df_pl, params)
        assert result["flag"].dtype == pl.Boolean

    # -----------------------------------------------------------------------
    # _resolve_polars_dtype (datetime-prefix branch + unsupported-dtype None)
    # -----------------------------------------------------------------------

    from skyulf.preprocessing.casting import _resolve_polars_dtype

    def test_resolve_polars_dtype_datetime_prefix_variant() -> None:
        """A 'datetime...' string not in the alias table must map via the
        startswith('datetime') fallback to pl.Datetime."""
        assert _resolve_polars_dtype("datetime64[ns]") is pl.Datetime

    def test_resolve_polars_dtype_unsupported_returns_none() -> None:
        """An unrecognised dtype string must resolve to None (unsupported)."""
        assert _resolve_polars_dtype("totally_unsupported_dtype") is None

    def test_casting_apply_polars_skips_unsupported_dtype_column() -> None:
        """Polars apply must silently skip a column whose target dtype is
        unsupported (_resolve_polars_dtype returns None), leaving it untouched."""
        df_pl = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        params: dict[str, Any] = {
            "type_map": {"a": "totally_unsupported_dtype", "b": "string"},
            "coerce_on_error": True,
        }
        result = CastingApplier().apply(df_pl, params)
        # 'a' must remain untouched (Int64), 'b' must be cast to String.
        assert result["a"].dtype == df_pl["a"].dtype
        assert result["b"].dtype == pl.String


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which contains a ``signup_date`` string column and ``plan_type`` strings —
    closer to production data than the small synthetic frames used elsewhere.
    """

    def test_cast_signup_date_to_datetime(self) -> None:
        """Casting the ``signup_date`` string column to datetime must succeed and
        produce a datetime64 column with no NaT values (all dates are valid ISO strings).
        """
        df = load_sample_dataset("customers")
        artifact = CastingCalculator().fit(df, {"column_types": {"signup_date": "datetime"}})
        result = CastingApplier().apply(df, artifact)

        assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])
        assert result["signup_date"].notna().all()

    def test_cast_plan_type_to_category(self) -> None:
        """Casting ``plan_type`` to category dtype must produce a pandas Categorical column
        with the three distinct values from the dataset: basic, premium, enterprise.
        """
        df = load_sample_dataset("customers")
        artifact = CastingCalculator().fit(df, {"column_types": {"plan_type": "category"}})
        result = CastingApplier().apply(df, artifact)

        assert result["plan_type"].dtype.name == "category"
        assert set(result["plan_type"].cat.categories) == {"basic", "enterprise", "premium"}
