"""Unit tests for the OneHotEncoder Calculator/Applier (fit + apply, dual-engine)."""

from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.preprocessing.encoding.one_hot import (
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)


def _fit_apply(df: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run OneHotEncoderCalculator.fit then OneHotEncoderApplier.apply on ``df``."""
    params = OneHotEncoderCalculator().fit(df, config)
    result = OneHotEncoderApplier().apply(df, dict(params))
    return dict(params), result


def test_fit_apply_round_trip_basic_values() -> None:
    """One-hot encoding produces correct 0/1 indicator columns for known categories."""
    df = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    params, out = _fit_apply(df, {"columns": ["color"]})

    assert params["columns"] == ["color"]
    assert "color" not in out.columns
    expected_cols = {"color_blue", "color_green", "color_red"}
    assert expected_cols.issubset(set(out.columns))
    assert out.loc[0, "color_red"] == 1
    assert out.loc[0, "color_blue"] == 0
    assert out.loc[1, "color_blue"] == 1
    assert out.loc[3, "color_green"] == 1


def test_drop_original_false_keeps_source_column() -> None:
    """When drop_original is False the raw categorical column is retained."""
    df = pd.DataFrame({"color": ["red", "blue"]})
    _, out = _fit_apply(df, {"columns": ["color"], "drop_original": False})
    assert "color" in out.columns
    assert "color_red" in out.columns


def test_drop_first_removes_one_category_column() -> None:
    """drop_first=True drops the first (alphabetically sorted) sklearn category."""
    df = pd.DataFrame({"color": ["blue", "red", "blue"]})
    params, out = _fit_apply(df, {"columns": ["color"], "drop_first": True})
    # sklearn OneHotEncoder sorts categories alphabetically and drops the first.
    assert "color_blue" not in out.columns
    assert "color_red" in out.columns
    assert len(params["feature_names"]) == 1


def test_unseen_category_at_apply_time_is_ignored() -> None:
    """Categories unseen during fit produce all-zero indicator rows (handle_unknown=ignore)."""
    train = pd.DataFrame({"color": ["red", "blue"]})
    test = pd.DataFrame({"color": ["red", "green"]})  # "green" never seen at fit time

    params = OneHotEncoderCalculator().fit(train, {"columns": ["color"]})
    out = OneHotEncoderApplier().apply(test, dict(params))

    feature_cols = [c for c in out.columns if c.startswith("color_")]
    assert "color_green" not in feature_cols
    unseen_row = out.loc[1, feature_cols]
    assert (unseen_row == 0).all()


def test_include_missing_creates_missing_token_column() -> None:
    """NaN values are mapped to the sentinel missing token when include_missing is set."""
    df = pd.DataFrame({"color": ["red", None, "blue"]})
    params, out = _fit_apply(df, {"columns": ["color"], "include_missing": True})
    missing_cols = [c for c in params["feature_names"] if "__mlops_missing__" in c]
    assert missing_cols
    assert out.loc[1, missing_cols[0]] == 1


def test_empty_dataframe_raises_value_error() -> None:
    """Fitting sklearn's OneHotEncoder on a zero-row DataFrame raises (no samples to fit)."""
    df = pd.DataFrame({"color": pd.Series([], dtype="object")})
    with pytest.raises(ValueError):
        OneHotEncoderCalculator().fit(df, {"columns": ["color"]})


def test_single_row_dataframe() -> None:
    """A single-row frame yields one indicator column set to 1 for that lone category."""
    df = pd.DataFrame({"color": ["red"]})
    params, out = _fit_apply(df, {"columns": ["color"]})
    assert params["feature_names"] == ["color_red"]
    assert out.loc[0, "color_red"] == 1


def test_all_nan_column_yields_single_none_category() -> None:
    """An entirely-NaN column is treated as a single ``None`` category by sklearn."""
    df = pd.DataFrame({"color": [None, None, None]})
    params, out = _fit_apply(df, {"columns": ["color"]})
    assert params["feature_names"] == ["color_None"]
    assert (out["color_None"] == 1).all()


def test_no_columns_selected_returns_input_unchanged() -> None:
    """Explicitly picking zero columns is a no-op (user_picked_no_columns short-circuit)."""
    df = pd.DataFrame({"color": ["red", "blue"]})
    params = OneHotEncoderCalculator().fit(df, {"columns": []})
    assert params == {}
    out = OneHotEncoderApplier().apply(df, dict(params))
    pd.testing.assert_frame_equal(out, df)


def test_drop_first_with_single_category_warns_and_yields_zero_features() -> None:
    """A constant column combined with drop_first=True yields 0 encoded features."""
    df = pd.DataFrame({"color": ["red", "red", "red"]})
    params, out = _fit_apply(df, {"columns": ["color"], "drop_first": True})
    assert params["feature_names"] == []
    assert not any(c.startswith("color_") for c in out.columns)


def test_max_categories_limits_encoded_columns() -> None:
    """max_categories caps the number of learned categories (sklearn infrequent bucket)."""
    df = pd.DataFrame({"color": ["a", "b", "c", "d", "e"]})
    params, _ = _fit_apply(df, {"columns": ["color"], "max_categories": 2})
    assert len(params["feature_names"]) <= 2


def test_target_column_excluded_from_encoding() -> None:
    """The target column is excluded from encoding since OneHot destroys columns."""
    X = pd.DataFrame({"color": ["red", "blue"], "target": ["red", "blue"]})
    y = X["target"]
    params = OneHotEncoderCalculator().fit((X.drop(columns=["target"]), y), {"columns": ["color"]})
    assert "target" not in params.get("columns", [])


def test_polars_apply_path_matches_pandas_values() -> None:
    """Polars apply path densifies and concatenates identically to the pandas path."""
    df_pd = pd.DataFrame({"color": ["red", "blue", "red"]})
    df_pl = pl.from_pandas(df_pd)

    params = OneHotEncoderCalculator().fit(df_pd, {"columns": ["color"]})
    out_pd = OneHotEncoderApplier().apply(df_pd, dict(params))
    out_pl = OneHotEncoderApplier().apply(df_pl, dict(params))

    out_pl_pd = out_pl.to_pandas()
    for col in params["feature_names"]:
        np.testing.assert_array_equal(out_pd[col].to_numpy(), out_pl_pd[col].to_numpy())


@given(
    cats=st.lists(st.sampled_from(["a", "b", "c"]), min_size=5, max_size=40),
)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_fit_engine_parity_pandas_vs_polars(cats: list[str]) -> None:
    """Fitting on pandas vs polars must learn the same categories and feature names."""
    df_pd = pd.DataFrame({"color": cats})
    df_pl = pl.from_pandas(df_pd)
    config = {"columns": ["color"]}

    pd_params = OneHotEncoderCalculator().fit(df_pd, dict(config))
    pl_params = OneHotEncoderCalculator().fit(df_pl, dict(config))

    assert pd_params["feature_names"] == pl_params["feature_names"]
    assert pd_params["columns"] == pl_params["columns"]


# ---------------------------------------------------------------------------
# Apply-time edge cases and error handling
# ---------------------------------------------------------------------------


def test_apply_missing_columns_key_is_noop() -> None:
    """A params dict with no 'columns' key (falsy) short-circuits apply to a no-op."""
    df = pd.DataFrame({"color": ["red", "blue"]})
    out = OneHotEncoderApplier().apply(df, {})
    pd.testing.assert_frame_equal(out, df)


def test_apply_columns_present_but_no_encoder_is_noop() -> None:
    """Columns exist but there's no fitted encoder_object: apply is a no-op."""
    df = pd.DataFrame({"color": ["red", "blue"]})
    params = {"columns": ["color"], "encoder_object": None}
    out = OneHotEncoderApplier().apply(df, dict(params))
    pd.testing.assert_frame_equal(out, df)


def test_polars_apply_no_valid_columns_is_noop() -> None:
    """Polars apply returns X, y unchanged when configured columns aren't in X."""
    df_pl = pl.DataFrame({"other": [1, 2]})
    params, _ = _fit_apply(pd.DataFrame({"color": ["red", "blue"]}), {"columns": ["color"]})
    out = OneHotEncoderApplier().apply(df_pl, dict(params))
    assert out.equals(df_pl)


def test_polars_apply_include_missing_fills_null_token() -> None:
    """Polars apply path fills nulls with the sentinel token when include_missing is set."""
    df_pd = pd.DataFrame({"color": ["red", None, "blue"]})
    params = OneHotEncoderCalculator().fit(df_pd, {"columns": ["color"], "include_missing": True})

    df_pl = pl.DataFrame({"color": ["red", None, "blue"]})
    out_pl = OneHotEncoderApplier().apply(df_pl, dict(params))
    missing_cols = [c for c in params["feature_names"] if "__mlops_missing__" in c]
    assert missing_cols
    assert out_pl[missing_cols[0]][1] == 1


def test_polars_apply_exception_is_caught_and_returns_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the polars apply path is caught, logged, and X returned as-is."""
    df_pl = pl.DataFrame({"color": ["red", "blue"]})

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {
        "columns": ["color"],
        "encoder_object": _BrokenEncoder(),
        "feature_names": ["color_x"],
    }
    with caplog.at_level("ERROR"):
        out = OneHotEncoderApplier().apply(df_pl, dict(params))

    assert out.equals(df_pl)
    assert any("OneHot Encoding failed" in rec.message for rec in caplog.records)


def test_pandas_apply_exception_is_caught_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A transform-time exception in the pandas apply path is caught and logged."""
    df_pd = pd.DataFrame({"color": ["red", "blue"]})

    class _BrokenEncoder:
        def transform(self, _x: Any) -> Any:
            raise ValueError("boom")

    params = {
        "columns": ["color"],
        "encoder_object": _BrokenEncoder(),
        "feature_names": ["color_x"],
    }
    with caplog.at_level("ERROR"):
        out = OneHotEncoderApplier().apply(df_pd, dict(params))

    pd.testing.assert_frame_equal(out, df_pd)
    assert any("OneHot Encoding failed" in rec.message for rec in caplog.records)


def test_to_dense_handles_pandas_wrapped_output() -> None:
    """_to_dense unwraps a pandas-like `.values` output when there's no `.toarray`."""
    from skyulf.preprocessing.encoding.one_hot import _to_dense

    wrapped = pd.DataFrame({"a": [1, 2]})
    result = _to_dense(wrapped)
    np.testing.assert_array_equal(result, wrapped.values)


def test_to_dense_handles_sparse_output_via_toarray() -> None:
    """_to_dense densifies sparse-matrix-like output via `.toarray()`."""
    from skyulf.preprocessing.encoding.one_hot import _to_dense

    class _FakeSparse:
        def toarray(self) -> np.ndarray:
            return np.array([[1, 0], [0, 1]])

    result = _to_dense(_FakeSparse())
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]]))


def test_polars_fit_include_missing_fills_null_before_fitting() -> None:
    """Polars fit path fills nulls with the sentinel token before fitting sklearn's encoder."""
    df_pl = pl.DataFrame({"color": ["red", None, "blue"]})
    params = OneHotEncoderCalculator().fit(df_pl, {"columns": ["color"], "include_missing": True})
    missing_names = [c for c in params["feature_names"] if "__mlops_missing__" in c]
    assert missing_names


def test_warn_degenerate_categories_returns_early_without_categories_attr() -> None:
    """_warn_degenerate_categories is a no-op for objects lacking `categories_`."""
    from skyulf.preprocessing.encoding.one_hot import _warn_degenerate_categories

    class _NoCategories:
        pass

    # Should not raise even though `_NoCategories` has no `categories_` attribute.
    _warn_degenerate_categories(cast(Any, _NoCategories()), ["color"], None)


def test_zero_category_column_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """_warn_degenerate_categories logs a "will be dropped" warning for 0-category columns."""
    from skyulf.preprocessing.encoding.one_hot import _warn_degenerate_categories

    class _FakeEncoder:
        categories_ = [np.array([], dtype=object)]

    with caplog.at_level("WARNING"):
        _warn_degenerate_categories(cast(Any, _FakeEncoder()), ["color"], None)

    assert any("0 categories" in rec.message for rec in caplog.records)


def test_polars_fit_no_resolvable_columns_returns_empty() -> None:
    """Polars fit falls back to auto-detection; a purely-numeric frame yields no columns."""
    df_pl = pl.DataFrame({"amount": [1, 2, 3]})
    params = OneHotEncoderCalculator().fit(df_pl, {})
    assert params == {}


def test_pandas_fit_no_resolvable_columns_returns_empty() -> None:
    """Pandas fit falls back to auto-detection; a purely-numeric frame yields no columns."""
    df_pd = pd.DataFrame({"amount": [1, 2, 3]})
    params = OneHotEncoderCalculator().fit(df_pd, {})
    assert params == {}


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has no missing values in ``plan_type`` — closer to production data
    than the small synthetic frames used elsewhere in this file.
    """

    def test_plan_type_encoding_produces_one_column_per_category(self) -> None:
        df = load_sample_dataset("customers")
        params, out = _fit_apply(df, {"columns": ["plan_type"]})

        assert "plan_type" not in out.columns
        expected_cols = {f"plan_type_{v}" for v in df["plan_type"].unique()}
        assert expected_cols.issubset(set(out.columns))
        for value in df["plan_type"].unique():
            rows = df["plan_type"] == value
            assert (out.loc[rows, f"plan_type_{value}"] == 1).all()
