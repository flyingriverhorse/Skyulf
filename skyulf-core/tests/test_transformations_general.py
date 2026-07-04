"""Unit tests for GeneralTransformationCalculator and GeneralTransformationApplier.

Covers: all simple ops, power transforms, edge cases, polars path, and
hypothesis engine-parity between pandas and polars apply paths.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from skyulf.preprocessing.transformations.general import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
    _apply_power_to_pandas_col,
    _apply_power_to_polars_col,
    _fit_power_for_column,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calc() -> GeneralTransformationCalculator:
    """Fresh GeneralTransformationCalculator instance."""
    return GeneralTransformationCalculator()


@pytest.fixture
def appl() -> GeneralTransformationApplier:
    """Fresh GeneralTransformationApplier instance."""
    return GeneralTransformationApplier()


@pytest.fixture
def pos_df() -> pd.DataFrame:
    """Small positive-valued numeric DataFrame safe for all ops including box-cox."""
    return pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0, 16.0], "b": [0.5, 1.5, 3.0, 6.0, 12.0]})


# ---------------------------------------------------------------------------
# fit() — artifact structure
# ---------------------------------------------------------------------------


class TestFitArtifact:
    def test_fit_simple_op_records_method(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """fit() with a simple op stores method name and column without extra params."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "log"}]})
        assert art["type"] == "general_transformation"
        assert art["transformations"] == [{"column": "a", "method": "log"}]

    def test_fit_unknown_column_is_dropped(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """Columns absent from the DataFrame must be silently dropped from the artifact."""
        art = calc.fit(pos_df, {"transformations": [{"column": "not_here", "method": "log"}]})
        assert art["transformations"] == []

    def test_fit_empty_transformations(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """Empty transformation list produces an artifact with an empty list."""
        art = calc.fit(pos_df, {"transformations": []})
        assert art["transformations"] == []

    def test_fit_multiple_columns_all_stored(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """All valid column configs are stored in the artifact."""
        art = calc.fit(
            pos_df,
            {
                "transformations": [
                    {"column": "a", "method": "log"},
                    {"column": "b", "method": "sqrt"},
                ]
            },
        )
        assert len(art["transformations"]) == 2

    def test_fit_yeo_johnson_stores_lambdas(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """yeo-johnson fit must store fitted lambdas for later inverse / apply."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        item = art["transformations"][0]
        assert "lambdas" in item
        # One column → exactly one lambda value
        assert len(item["lambdas"]) == 1

    def test_fit_yeo_johnson_stores_scaler_params(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """yeo-johnson with standardize=True must store mean/scale for the scaler."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        item = art["transformations"][0]
        assert "scaler_params" in item
        assert item["scaler_params"]["mean"] is not None
        assert item["scaler_params"]["scale"] is not None

    def test_fit_box_cox_works_on_positive_data(self, calc: Any) -> None:
        """box-cox succeeds and stores lambdas when all values are strictly positive."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "box-cox"}]})
        assert len(art["transformations"]) == 1
        assert "lambdas" in art["transformations"][0]

    def test_fit_box_cox_skipped_on_nonpositive(self, calc: Any) -> None:
        """box-cox must be skipped (not raise) when the column contains non-positive values."""
        df = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0, 3.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "box-cox"}]})
        assert art["transformations"] == []

    def test_fit_mixed_valid_invalid_columns(self, calc: Any, pos_df: pd.DataFrame) -> None:
        """Valid columns are stored; invalid/missing ones are dropped without error."""
        art = calc.fit(
            pos_df,
            {
                "transformations": [
                    {"column": "a", "method": "square"},
                    {"column": "missing", "method": "log"},
                ]
            },
        )
        columns = [t["column"] for t in art["transformations"]]
        assert "a" in columns
        assert "missing" not in columns


# ---------------------------------------------------------------------------
# apply() — simple ops (pandas engine)
# ---------------------------------------------------------------------------


class TestApplySimpleOpsPandas:
    @pytest.mark.parametrize(
        "method,expected_fn",
        [
            ("log", lambda v: np.log1p(v)),
            ("sqrt", lambda v: np.sqrt(v)),
            ("square", lambda v: np.square(v)),
            ("cube_root", lambda v: np.cbrt(v)),
        ],
    )
    def test_op_transforms_values_correctly(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame, method: str, expected_fn: Any
    ) -> None:
        """Each op must produce values matching its mathematical definition."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": method}]})
        result = appl.apply(pos_df, art)
        np.testing.assert_allclose(
            result["a"].values,
            expected_fn(pos_df["a"].values),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_exp_clips_large_values(self, calc: Any, appl: Any) -> None:
        """exp clips input at 700 (by default) to avoid overflow."""
        df = pd.DataFrame({"x": [1.0, 10.0, 800.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "exp"}]})
        result = appl.apply(df, art)
        # 800.0 is clipped to 700 before exp, so result == exp(700)
        assert np.isfinite(result["x"].values).all()

    def test_reciprocal_zero_becomes_nan(self, calc: Any, appl: Any) -> None:
        """1/0 must yield NaN rather than inf, matching the pandas ops contract."""
        df = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "reciprocal"}]})
        result = appl.apply(df, art)
        assert np.isnan(result["x"].iloc[0])
        np.testing.assert_allclose(result["x"].iloc[1:].values, [0.5, 0.25], rtol=1e-6)

    def test_log_negative_values_become_nan(self, calc: Any, appl: Any) -> None:
        """Negative inputs to log must produce NaN, not raise."""
        df = pd.DataFrame({"x": [-5.0, -1.0, 0.0, 1.0, 4.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "log"}]})
        result = appl.apply(df, art)
        assert np.isnan(result["x"].iloc[0])
        assert np.isnan(result["x"].iloc[1])

    def test_sqrt_negative_values_become_nan(self, calc: Any, appl: Any) -> None:
        """Negative inputs to sqrt must produce NaN, not raise."""
        df = pd.DataFrame({"x": [-4.0, 0.0, 4.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "sqrt"}]})
        result = appl.apply(df, art)
        assert np.isnan(result["x"].iloc[0])
        np.testing.assert_allclose(result["x"].iloc[1:].values, [0.0, 2.0], atol=1e-6)

    def test_square_root_alias_matches_sqrt(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame
    ) -> None:
        """'square_root' alias must produce identical results to 'sqrt'."""
        r_sqrt = appl.apply(
            pos_df, calc.fit(pos_df, {"transformations": [{"column": "a", "method": "sqrt"}]})
        )
        r_alias = appl.apply(
            pos_df,
            calc.fit(pos_df, {"transformations": [{"column": "a", "method": "square_root"}]}),
        )
        pd.testing.assert_series_equal(r_sqrt["a"], r_alias["a"])

    def test_exp_alias_matches_exp(self, calc: Any, appl: Any, pos_df: pd.DataFrame) -> None:
        """'exponential' alias must produce identical results to 'exp'."""
        r_exp = appl.apply(
            pos_df, calc.fit(pos_df, {"transformations": [{"column": "a", "method": "exp"}]})
        )
        r_alias = appl.apply(
            pos_df,
            calc.fit(pos_df, {"transformations": [{"column": "a", "method": "exponential"}]}),
        )
        pd.testing.assert_series_equal(r_exp["a"], r_alias["a"])

    def test_unknown_method_leaves_column_unchanged(self, appl: Any, pos_df: pd.DataFrame) -> None:
        """An unknown method in the params dict silently skips the column."""
        params = {"transformations": [{"column": "a", "method": "does_not_exist"}]}
        result = appl.apply(pos_df, params)
        pd.testing.assert_series_equal(result["a"], pos_df["a"])

    def test_missing_column_in_apply_is_skipped(self, appl: Any, pos_df: pd.DataFrame) -> None:
        """Columns referenced in params but absent from DataFrame are skipped."""
        params = {"transformations": [{"column": "z", "method": "log"}]}
        result = appl.apply(pos_df, params)
        pd.testing.assert_frame_equal(result, pos_df)

    def test_empty_transformations_returns_identity(self, appl: Any, pos_df: pd.DataFrame) -> None:
        """No transformations config → DataFrame passes through unchanged."""
        result = appl.apply(pos_df, {"transformations": []})
        pd.testing.assert_frame_equal(result, pos_df)

    def test_transform_only_target_column(self, calc: Any, appl: Any, pos_df: pd.DataFrame) -> None:
        """Applying to column 'a' must leave column 'b' completely untouched."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "square"}]})
        result = appl.apply(pos_df, art)
        pd.testing.assert_series_equal(result["b"], pos_df["b"])


# ---------------------------------------------------------------------------
# apply() — power transforms
# ---------------------------------------------------------------------------


class TestApplyPowerTransforms:
    def test_yeo_johnson_full_flow(self, calc: Any, appl: Any, pos_df: pd.DataFrame) -> None:
        """yeo-johnson fit → apply returns same shape with all-finite values."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        result = appl.apply(pos_df, art)
        assert result["a"].shape == pos_df["a"].shape
        assert result["a"].notna().all()

    def test_yeo_johnson_changes_distribution(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame
    ) -> None:
        """yeo-johnson is not a no-op — it must alter the column values."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        result = appl.apply(pos_df, art)
        assert not np.allclose(result["a"].to_numpy(), pos_df["a"].to_numpy())

    def test_box_cox_positive_data_full_flow(self, calc: Any, appl: Any) -> None:
        """box-cox on strictly positive data returns finite values."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "box-cox"}]})
        result = appl.apply(df, art)
        assert result["x"].notna().all()
        assert np.isfinite(result["x"].values).all()

    def test_power_transform_missing_lambdas_is_noop(self, appl: Any, pos_df: pd.DataFrame) -> None:
        """If lambdas are None in params, the power-transform helper returns the frame unchanged."""
        # Inject a malformed artifact that is missing lambdas
        params = {"transformations": [{"column": "a", "method": "yeo-johnson"}]}
        # No lambdas key → _apply_power_to_pandas_col returns df_out unchanged
        result = appl.apply(pos_df, params)
        pd.testing.assert_series_equal(result["a"], pos_df["a"])

    def test_yeo_johnson_polars_path(self, calc: Any, appl: Any, pos_df: pd.DataFrame) -> None:
        """yeo-johnson apply on polars input returns a polars DataFrame."""
        pl_df = pl.from_pandas(pos_df)
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        result = appl.apply(pl_df, art)
        assert isinstance(result, pl.DataFrame)
        assert "a" in result.columns

    def test_yeo_johnson_pandas_polars_values_close(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame
    ) -> None:
        """yeo-johnson pandas and polars paths must produce numerically close values."""
        art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        pd_result = appl.apply(pos_df, art)
        pl_result = appl.apply(pl.from_pandas(pos_df), art)
        np.testing.assert_allclose(
            pd_result["a"].values,
            pl_result["a"].to_numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_does_not_raise(self, calc: Any, appl: Any) -> None:
        """A one-row DataFrame must transform without raising."""
        df = pd.DataFrame({"x": [4.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "sqrt"}]})
        result = appl.apply(df, art)
        np.testing.assert_allclose(result["x"].values, [2.0])

    def test_all_nan_column_survives_transform(self, calc: Any, appl: Any) -> None:
        """An all-NaN column should remain all-NaN after a simple op, without raising."""
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "log"}]})
        result = appl.apply(df, art)
        assert result["x"].isna().all()

    def test_constant_column_transforms_correctly(self, calc: Any, appl: Any) -> None:
        """A constant column should return a constant transformed column."""
        df = pd.DataFrame({"x": [3.0, 3.0, 3.0, 3.0]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "square"}]})
        result = appl.apply(df, art)
        np.testing.assert_allclose(result["x"].values, 9.0)

    def test_large_dataframe_does_not_raise(self, calc: Any, appl: Any) -> None:
        """Large DataFrames must transform without memory or performance failures."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.uniform(0.1, 100.0, 10_000)})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "log"}]})
        result = appl.apply(df, art)
        assert result["x"].notna().all()

    def test_multiple_ops_applied_in_order(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame
    ) -> None:
        """Multiple transforms config applies each op to its designated column."""
        art = calc.fit(
            pos_df,
            {
                "transformations": [
                    {"column": "a", "method": "log"},
                    {"column": "b", "method": "square"},
                ]
            },
        )
        result = appl.apply(pos_df, art)
        np.testing.assert_allclose(result["a"].values, np.log1p(pos_df["a"].values), rtol=1e-6)
        np.testing.assert_allclose(result["b"].values, np.square(pos_df["b"].values), rtol=1e-6)


# ---------------------------------------------------------------------------
# Polars engine path
# ---------------------------------------------------------------------------


class TestPolarsPath:
    @pytest.mark.parametrize("method", ["log", "sqrt", "square", "exp", "cube_root", "reciprocal"])
    def test_simple_op_returns_polars_frame(
        self, calc: Any, appl: Any, pos_df: pd.DataFrame, method: str
    ) -> None:
        """Every simple op must return a polars DataFrame when given polars input."""
        pl_df = pl.from_pandas(pos_df)
        art = calc.fit(pl_df, {"transformations": [{"column": "a", "method": method}]})
        result = appl.apply(pl_df, art)
        assert isinstance(result, pl.DataFrame)

    def test_polars_empty_transformations_returns_polars(
        self, appl: Any, pos_df: pd.DataFrame
    ) -> None:
        """Empty transformations on polars input must return an unchanged polars frame."""
        pl_df = pl.from_pandas(pos_df)
        result = appl.apply(pl_df, {"transformations": []})
        assert isinstance(result, pl.DataFrame)
        pd.testing.assert_frame_equal(result.to_pandas(), pos_df)

    def test_polars_missing_column_skipped(self, appl: Any, pos_df: pd.DataFrame) -> None:
        """Missing columns in polars apply path are silently skipped."""
        pl_df = pl.from_pandas(pos_df)
        result = appl.apply(pl_df, {"transformations": [{"column": "z_missing", "method": "log"}]})
        assert isinstance(result, pl.DataFrame)
        pd.testing.assert_frame_equal(result.to_pandas(), pos_df)

    def test_polars_fit_yields_same_lambdas_as_pandas(
        self, calc: Any, pos_df: pd.DataFrame
    ) -> None:
        """Polars fit path for yeo-johnson must yield identical lambdas as pandas path.

        _fit_power_for_column converts polars → pandas internally so artifacts should match.
        """
        pd_art = calc.fit(pos_df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
        pl_art = calc.fit(
            pl.from_pandas(pos_df),
            {"transformations": [{"column": "a", "method": "yeo-johnson"}]},
        )
        np.testing.assert_allclose(
            pd_art["transformations"][0]["lambdas"],
            pl_art["transformations"][0]["lambdas"],
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# Internal helpers — direct unit tests
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_fit_power_for_column_returns_lambdas(self, pos_df: pd.DataFrame) -> None:
        """_fit_power_for_column must return a dict with 'lambdas' for yeo-johnson."""
        result = _fit_power_for_column(pos_df, "a", "yeo-johnson", is_polars=False)
        assert "lambdas" in result
        assert len(result["lambdas"]) == 1

    def test_fit_power_for_column_polars(self, pos_df: pd.DataFrame) -> None:
        """_fit_power_for_column polars path converts to pandas and produces lambdas."""
        pl_df = pl.from_pandas(pos_df)
        result = _fit_power_for_column(pl_df, "a", "yeo-johnson", is_polars=True)
        assert "lambdas" in result

    def test_apply_power_pandas_missing_lambdas_is_noop(self, pos_df: pd.DataFrame) -> None:
        """_apply_power_to_pandas_col with lambdas=None returns the frame unchanged."""
        item: dict = {"column": "a", "method": "yeo-johnson"}
        result = _apply_power_to_pandas_col(pos_df.copy(), item)
        pd.testing.assert_frame_equal(result, pos_df)

    def test_apply_power_polars_missing_lambdas_is_noop(self, pos_df: pd.DataFrame) -> None:
        """_apply_power_to_polars_col with lambdas=None returns the frame unchanged (line 32)."""
        pl_df = pl.from_pandas(pos_df)
        item: dict = {"column": "a", "method": "yeo-johnson"}
        result = _apply_power_to_polars_col(pl_df, item)
        assert result is pl_df

    def test_apply_power_polars_malformed_lambdas_swallows_exception(
        self, pos_df: pd.DataFrame
    ) -> None:
        """A malformed lambdas value must be caught and the frame returned unchanged (lines 52-54)."""
        pl_df = pl.from_pandas(pos_df)
        item: dict = {"column": "a", "method": "yeo-johnson", "lambdas": ["not_a_number"]}
        result = _apply_power_to_polars_col(pl_df, item)
        # Exception is swallowed; original frame returned untouched.
        assert result.equals(pl_df)

    def test_apply_power_pandas_malformed_lambdas_swallows_exception(
        self, pos_df: pd.DataFrame
    ) -> None:
        """A malformed lambdas value must be caught and the column left unchanged (lines 82-83)."""
        item: dict = {"column": "a", "method": "yeo-johnson", "lambdas": ["not_a_number"]}
        original = pos_df.copy()
        result = _apply_power_to_pandas_col(pos_df.copy(), item)
        pd.testing.assert_series_equal(result["a"], original["a"])

    def test_fit_power_for_column_swallows_exception(
        self, calc: GeneralTransformationCalculator
    ) -> None:
        """fit() must catch exceptions from _fit_power_for_column and skip the item (lines 196-198)."""
        # Non-numeric column values cause PowerTransformer.fit() to raise internally.
        df = pd.DataFrame({"x": ["a", "b", "c", "d", "e"]})
        art = calc.fit(df, {"transformations": [{"column": "x", "method": "yeo-johnson"}]})
        assert art["transformations"] == []

    def test_apply_polars_unknown_method_is_noop(
        self, appl: GeneralTransformationApplier, pos_df: pd.DataFrame
    ) -> None:
        """An unknown method on the polars apply path must be skipped (line 109)."""
        pl_df = pl.from_pandas(pos_df)
        params = {"transformations": [{"column": "a", "method": "does_not_exist"}]}
        result = appl.apply(pl_df, params)
        assert result["a"].to_list() == pos_df["a"].to_list()


# ---------------------------------------------------------------------------
# Hypothesis engine-parity (pandas apply == polars apply)
# ---------------------------------------------------------------------------

_FINITE_POSITIVE = st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False)


@st.composite
def _pos_frame(draw: st.DrawFn, min_rows: int = 5, max_rows: int = 30) -> pd.DataFrame:
    """Generate a small, strictly positive numeric DataFrame with columns 'a' and 'b'."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    a = draw(st.lists(_FINITE_POSITIVE, min_size=n, max_size=n))
    b = draw(st.lists(_FINITE_POSITIVE, min_size=n, max_size=n))
    return pd.DataFrame({"a": a, "b": b})


@pytest.mark.parametrize("method", ["log", "sqrt", "square", "exp", "cube_root"])
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=_pos_frame())
def test_general_transform_engine_parity(method: str, df: pd.DataFrame) -> None:
    """Pandas and polars apply paths must produce numerically identical values.

    Using strictly positive inputs avoids documented NaN divergences for log/sqrt.
    """
    assume(len(df) >= 2)
    calc = GeneralTransformationCalculator()
    appl = GeneralTransformationApplier()
    art = calc.fit(df, {"transformations": [{"column": "a", "method": method}]})
    pd_vals = appl.apply(df, art)["a"].to_numpy()
    pl_vals = appl.apply(pl.from_pandas(df), art)["a"].to_numpy()
    np.testing.assert_allclose(
        pd_vals, pl_vals, rtol=1e-5, atol=1e-5, err_msg=f"Engine divergence for method={method!r}"
    )


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=_pos_frame())
def test_yeo_johnson_engine_parity(df: pd.DataFrame) -> None:
    """yeo-johnson pandas vs polars apply paths must produce numerically close values."""
    assume(df["a"].nunique() > 1)
    calc = GeneralTransformationCalculator()
    appl = GeneralTransformationApplier()
    art = calc.fit(df, {"transformations": [{"column": "a", "method": "yeo-johnson"}]})
    pd_vals = appl.apply(df, art)["a"].to_numpy()
    pl_vals = appl.apply(pl.from_pandas(df), art)["a"].to_numpy()
    np.testing.assert_allclose(
        pd_vals, pl_vals, rtol=1e-4, atol=1e-4, err_msg="Engine divergence for yeo-johnson"
    )
