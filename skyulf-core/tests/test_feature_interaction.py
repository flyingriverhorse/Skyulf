"""Tests for the FeatureInteraction Calculator/Applier pair.

Covers 2-way and 3-way interaction generation, deterministic
regularization-friendly naming, ``interaction_only`` behavior, and
pandas/polars engine parity.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.feature_generation import (
    FeatureInteractionApplier,
    FeatureInteractionCalculator,
)


def _sample_pandas_df() -> pd.DataFrame:
    """Build a small numeric pandas DataFrame for interaction tests."""
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 30.0, 40.0],
            "x3": [2.0, 2.0, 2.0, 2.0],
        }
    )


class TestTwoWayInteractions:
    """2-way (degree=2) interaction generation."""

    def test_generates_correct_values_and_names(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2})
        result = applier.apply(df, params)

        assert "x1_x_x2" in result.columns
        np.testing.assert_allclose(result["x1_x_x2"], df["x1"] * df["x2"])

    def test_column_order_does_not_affect_name(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()

        params_a = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2})
        params_b = calc.fit(df, {"columns": ["x2", "x1"], "degree": 2})

        result_a = applier.apply(df, params_a)
        result_b = applier.apply(df, params_b)

        assert "x1_x_x2" in result_a.columns
        assert "x1_x_x2" in result_b.columns
        np.testing.assert_allclose(result_a["x1_x_x2"], result_b["x1_x_x2"])


class TestThreeWayInteractions:
    """3-way (degree=3) interaction generation."""

    def test_generates_triple_product(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2", "x3"], "degree": 3})
        result = applier.apply(df, params)

        assert "x1_x_x2_x_x3" in result.columns
        np.testing.assert_allclose(result["x1_x_x2_x_x3"], df["x1"] * df["x2"] * df["x3"])


class TestFourWayInteractions:
    """4-way (degree=4) interaction generation."""

    def test_generates_quadruple_product(self) -> None:
        df = _sample_pandas_df()
        df["x4"] = [5.0, 5.0, 5.0, 5.0]
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2", "x3", "x4"], "degree": 4})
        result = applier.apply(df, params)

        assert "x1_x_x2_x_x3_x_x4" in result.columns
        np.testing.assert_allclose(
            result["x1_x_x2_x_x3_x_x4"], df["x1"] * df["x2"] * df["x3"] * df["x4"]
        )


class TestInteractionOnly:
    """``interaction_only`` must skip self-products."""

    def test_interaction_only_skips_self_products(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2, "interaction_only": True})
        result = applier.apply(df, params)

        assert "x1_x_x1" not in result.columns
        assert "x2_x_x2" not in result.columns

    def test_interaction_only_false_includes_self_products(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2, "interaction_only": False})
        result = applier.apply(df, params)

        assert "x1_x_x1" in result.columns
        np.testing.assert_allclose(result["x1_x_x1"], df["x1"] * df["x1"])


class TestIncludeBias:
    """Bias column support."""

    def test_include_bias_adds_constant_column(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2, "include_bias": True})
        result = applier.apply(df, params)

        assert "interaction_bias" in result.columns
        assert (result["interaction_bias"] == 1.0).all()


class TestValidation:
    """Config validation errors."""

    def test_unsupported_degree_raises(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"columns": ["x1", "x2"], "degree": 5})

    def test_non_numeric_column_raises(self) -> None:
        df = _sample_pandas_df()
        df["cat"] = ["a", "b", "c", "d"]
        calc = FeatureInteractionCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"columns": ["x1", "cat"], "degree": 2})

    def test_missing_column_raises(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        with pytest.raises(ValueError):
            calc.fit(df, {"columns": ["x1", "does_not_exist"], "degree": 2})


class TestFitApplyRoundTrip:
    """Full fit -> apply round trip."""

    def test_round_trip_preserves_original_columns(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df, {"columns": ["x1", "x2", "x3"], "degree": 2})
        result = applier.apply(df, params)

        for col in df.columns:
            assert col in result.columns
        # 3 choose 2 = 3 new interaction columns.
        assert len(result.columns) == len(df.columns) + 3

    def test_deterministic_feature_names_regardless_of_order(self) -> None:
        df = _sample_pandas_df()
        calc = FeatureInteractionCalculator()
        params_1 = calc.fit(df, {"columns": ["x3", "x1", "x2"], "degree": 2})
        params_2 = calc.fit(df, {"columns": ["x1", "x2", "x3"], "degree": 2})

        assert sorted(params_1["feature_names"]) == sorted(params_2["feature_names"])


class TestEngineParity:
    """Pandas and polars engines must produce numerically identical output."""

    def test_two_way_parity(self) -> None:
        df_pd = _sample_pandas_df()
        df_pl = pl.from_pandas(df_pd)

        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df_pd, {"columns": ["x1", "x2"], "degree": 2})

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        np.testing.assert_allclose(
            result_pd["x1_x_x2"].to_numpy(),
            result_pl["x1_x_x2"].to_numpy(),
        )

    def test_three_way_parity(self) -> None:
        df_pd = _sample_pandas_df()
        df_pl = pl.from_pandas(df_pd)

        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df_pd, {"columns": ["x1", "x2", "x3"], "degree": 3})

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        np.testing.assert_allclose(
            result_pd["x1_x_x2_x_x3"].to_numpy(),
            result_pl["x1_x_x2_x_x3"].to_numpy(),
        )

    def test_four_way_parity(self) -> None:
        df_pd = _sample_pandas_df()
        df_pd["x4"] = [5.0, 5.0, 5.0, 5.0]
        df_pl = pl.from_pandas(df_pd)

        calc = FeatureInteractionCalculator()
        applier = FeatureInteractionApplier()
        params = calc.fit(df_pd, {"columns": ["x1", "x2", "x3", "x4"], "degree": 4})

        result_pd = applier.apply(df_pd, params)
        result_pl = applier.apply(df_pl, params)

        np.testing.assert_allclose(
            result_pd["x1_x_x2_x_x3_x_x4"].to_numpy(),
            result_pl["x1_x_x2_x_x3_x_x4"].to_numpy(),
        )
