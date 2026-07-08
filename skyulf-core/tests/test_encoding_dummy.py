"""Unit tests for the DummyEncoder Calculator/Applier (fit + apply, dual-engine)."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.encoding.dummy import (
    DummyEncoderApplier,
    DummyEncoderCalculator,
)


def _fit_apply(df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run DummyEncoderCalculator.fit then DummyEncoderApplier.apply on ``df``."""
    params = DummyEncoderCalculator().fit(df, config)
    result = DummyEncoderApplier().apply(df, dict(params))
    return dict(params), result


def test_fit_learns_sorted_string_categories() -> None:
    """fit() records sorted unique string categories per column."""
    df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    params = DummyEncoderCalculator().fit(df, {"columns": ["color"]})
    assert params["categories"]["color"] == ["blue", "green", "red"]
    assert params["drop_first"] is False


def test_fit_apply_round_trip_correct_indicators() -> None:
    """Dummy-encoded columns hold correct 0/1 indicators matching source values."""
    df = pd.DataFrame({"color": ["red", "blue", "green"]})
    params, out = _fit_apply(df, {"columns": ["color"]})

    assert "color" not in out.columns
    assert out.loc[0, "color_red"] == 1
    assert out.loc[0, "color_blue"] == 0
    assert out.loc[1, "color_blue"] == 1
    assert out.loc[2, "color_green"] == 1
    assert params["categories"]["color"] == ["blue", "green", "red"]


def test_drop_first_reduces_column_count() -> None:
    """drop_first=True omits the indicator for the first sorted category."""
    df = pd.DataFrame({"color": ["red", "blue", "green"]})
    _, out = _fit_apply(df, {"columns": ["color"], "drop_first": True})
    assert "color_blue" not in out.columns  # "blue" sorts first, dropped
    assert "color_green" in out.columns
    assert "color_red" in out.columns


def test_unseen_category_at_apply_time_yields_all_zero_row() -> None:
    """A category unseen during fit is coerced to NaN and encodes to an all-zero row."""
    train = pd.DataFrame({"color": ["red", "blue"]})
    test = pd.DataFrame({"color": ["red", "purple"]})  # "purple" never seen at fit

    params = DummyEncoderCalculator().fit(train, {"columns": ["color"]})
    out = DummyEncoderApplier().apply(test, dict(params))

    indicator_cols = [c for c in out.columns if c.startswith("color_")]
    assert set(indicator_cols) == {"color_blue", "color_red"}
    unseen_row = out.loc[1, indicator_cols]
    assert (unseen_row == 0).all()
    assert out.loc[0, "color_red"] == 1


def test_empty_dataframe_returns_empty_categories() -> None:
    """Fitting on a zero-row DataFrame yields an empty category list, no crash."""
    df = pd.DataFrame({"color": pd.Series([], dtype="object")})
    params = DummyEncoderCalculator().fit(df, {"columns": ["color"]})
    assert params["categories"]["color"] == []


def test_single_row_dataframe() -> None:
    """A single-row frame produces exactly one indicator column set to 1."""
    df = pd.DataFrame({"color": ["red"]})
    params, out = _fit_apply(df, {"columns": ["color"]})
    assert params["categories"]["color"] == ["red"]
    assert out.loc[0, "color_red"] == 1


def test_all_nan_column_yields_no_categories() -> None:
    """An entirely-NaN column has no learned categories and no dummy columns after apply."""
    df = pd.DataFrame({"color": [None, None, None]})
    params, out = _fit_apply(df, {"columns": ["color"]})
    assert params["categories"]["color"] == []
    assert not any(c.startswith("color_") for c in out.columns)


def test_no_columns_selected_returns_input_unchanged() -> None:
    """Explicitly picking zero columns is a no-op (user_picked_no_columns short-circuit)."""
    df = pd.DataFrame({"color": ["red", "blue"]})
    params = DummyEncoderCalculator().fit(df, {"columns": []})
    assert params == {}
    out = DummyEncoderApplier().apply(df, dict(params))
    pd.testing.assert_frame_equal(out, df)


def test_target_column_excluded_from_encoding() -> None:
    """The target column is excluded from encoding since DummyEncoder destroys columns."""
    X = pd.DataFrame({"color": ["red", "blue"]})
    y = pd.Series(["red", "blue"], name="target")
    params = DummyEncoderCalculator().fit((X, y), {"columns": ["color", "target"]})
    assert "target" not in params["columns"]


def test_polars_apply_path_matches_pandas_values() -> None:
    """Polars apply path yields identical indicator values to the pandas path."""
    df_pd = pd.DataFrame({"color": ["red", "blue", "red"]})
    df_pl = pl.from_pandas(df_pd)

    params = DummyEncoderCalculator().fit(df_pd, {"columns": ["color"]})
    out_pd = DummyEncoderApplier().apply(df_pd, dict(params))
    out_pl = DummyEncoderApplier().apply(df_pl, dict(params))

    out_pl_pd = out_pl.to_pandas()
    indicator_cols = [c for c in out_pd.columns if c.startswith("color_")]
    for col in indicator_cols:
        np.testing.assert_array_equal(
            out_pd[col].to_numpy().astype(int), out_pl_pd[col].to_numpy().astype(int)
        )


def test_drop_first_not_applied_when_single_category() -> None:
    """_drop_first_if_needed leaves a single-category list untouched even with drop_first=True."""
    from skyulf.preprocessing.encoding.dummy import _drop_first_if_needed

    assert _drop_first_if_needed(["only"], drop_first=True) == ["only"]


def test_polars_apply_drop_first_removes_first_category_column() -> None:
    """Polars apply path honors drop_first, matching the pandas-side behavior."""
    df_pd = pd.DataFrame({"color": ["red", "blue", "green"]})
    params = DummyEncoderCalculator().fit(df_pd, {"columns": ["color"], "drop_first": True})

    df_pl = pl.from_pandas(df_pd)
    out_pl = DummyEncoderApplier().apply(df_pl, dict(params))

    assert "color_blue" not in out_pl.columns  # "blue" sorts first, dropped
    assert "color_green" in out_pl.columns
    assert "color_red" in out_pl.columns


def test_polars_apply_no_valid_columns_is_noop() -> None:
    """Polars apply returns X, y unchanged when configured columns aren't present in X."""
    df_pl = pl.DataFrame({"other": [1, 2]})
    params = {"columns": ["color"], "categories": {"color": ["blue", "red"]}, "drop_first": False}
    out = DummyEncoderApplier().apply(df_pl, dict(params))
    assert out.equals(df_pl)


@given(
    cats=st.lists(st.sampled_from(["a", "b", "c"]), min_size=5, max_size=40),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fit_engine_parity_pandas_vs_polars(cats: list[str]) -> None:
    """Fitting on pandas vs polars must learn identical sorted category lists."""
    df_pd = pd.DataFrame({"color": cats})
    df_pl = pl.from_pandas(df_pd)
    config = {"columns": ["color"]}

    pd_params = DummyEncoderCalculator().fit(df_pd, dict(config))
    pl_params = DummyEncoderCalculator().fit(df_pl, dict(config))

    assert pd_params["categories"]["color"] == pl_params["categories"]["color"]
    assert pd_params["columns"] == pl_params["columns"]


# ---------------------------------------------------------------------------
# Real-shaped dataset integration
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has ``plan_type`` (3 categories: basic, premium, enterprise) with no
    missing values — verifying correct dummy encoding on real-world cardinality.
    """

    def test_dummy_encode_plan_type_produces_three_indicator_columns(self) -> None:
        """Dummy-encoding ``plan_type`` must produce exactly three indicator columns
        (one per category) and drop the original column from the output.
        """
        df = load_sample_dataset("customers")
        params = DummyEncoderCalculator().fit(df, {"columns": ["plan_type"]})
        result = DummyEncoderApplier().apply(df, dict(params))

        assert "plan_type" not in result.columns
        expected_cols = {"plan_type_basic", "plan_type_enterprise", "plan_type_premium"}
        assert expected_cols.issubset(set(result.columns))
        # Each row must have exactly one indicator set to 1.
        indicator_sum = result[list(expected_cols)].sum(axis=1)
        assert (indicator_sum == 1).all()
