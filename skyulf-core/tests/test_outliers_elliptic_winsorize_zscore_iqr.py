"""Behavioral tests for the IQR, Z-Score, Winsorize, and EllipticEnvelope outlier nodes.

Complements ``test_engine_parity.py`` (which only checks pandas/polars artifact
parity) and ``test_outliers_manual_bounds.py`` (a different node). These tests
assert exact apply-time behavior: IQR/Z-Score/EllipticEnvelope *remove* outlier
rows, Winsorize *clips* values in place, and cover edge cases (empty frame,
single row, constant column, non-numeric column, too-few-samples).
"""

import typing
from typing import Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.core.schema import SkyulfSchema
from skyulf.preprocessing.outliers.elliptic import (
    EllipticEnvelopeApplier,
    EllipticEnvelopeCalculator,
)
from skyulf.preprocessing.outliers.iqr import IQRApplier, IQRCalculator
from skyulf.preprocessing.outliers.winsorize import WinsorizeApplier, WinsorizeCalculator
from skyulf.preprocessing.outliers.zscore import ZScoreApplier, ZScoreCalculator

# ---------------------------------------------------------------------------
# IQR
# ---------------------------------------------------------------------------


class TestIQRCalculator:
    def test_bounds_match_q1_q3_formula(self) -> None:
        """Bounds must equal Q1 - k*IQR and Q3 + k*IQR for the configured multiplier."""
        values = list(range(1, 11))  # 1..10
        df = pd.DataFrame({"val": values})
        params = IQRCalculator().fit(df, {"columns": ["val"], "multiplier": 1.5})

        series = pd.Series(values, dtype=float)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        expected_lower = q1 - 1.5 * iqr
        expected_upper = q3 + 1.5 * iqr

        bounds = params["bounds"]["val"]
        assert bounds["lower"] == pytest.approx(expected_lower)
        assert bounds["upper"] == pytest.approx(expected_upper)
        assert params["multiplier"] == 1.5
        assert params["warnings"] == []

    def test_user_picked_no_columns_returns_empty(self) -> None:
        """Explicit empty columns list means the node is a no-op."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        assert IQRCalculator().fit(df, {"columns": []}) == {}

    def test_constant_column_zero_iqr_bounds_equal_value(self) -> None:
        """A constant column has IQR=0, so both bounds collapse to the constant value."""
        df = pd.DataFrame({"val": [5.0] * 10})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        bounds = params["bounds"]["val"]
        assert bounds["lower"] == pytest.approx(5.0)
        assert bounds["upper"] == pytest.approx(5.0)

    def test_non_numeric_column_produces_warning_and_no_bounds(self) -> None:
        """A column that is entirely non-numeric must warn and be excluded from bounds."""
        df = pd.DataFrame({"val": ["a", "b", "c"]})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        assert params["bounds"] == {}
        assert any("val" in w for w in params["warnings"])

    def test_empty_dataframe_returns_empty_bounds(self) -> None:
        """Fitting on a zero-row frame must not raise and yields no bounds."""
        df = pd.DataFrame({"val": pd.Series([], dtype=float)})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        assert params["bounds"] == {}
        assert any("val" in w for w in params["warnings"])

    def test_infer_output_schema_passes_through(self) -> None:
        """IQR removes rows, not columns, so the output schema equals the input schema."""
        schema = object()
        assert IQRCalculator().infer_output_schema(typing.cast(SkyulfSchema, schema), {}) is schema


class TestIQRApplier:
    def test_outlier_rows_are_removed_not_clipped(self) -> None:
        """Rows outside [lower, upper] must be dropped entirely; values are unmodified."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        out = IQRApplier().apply(df, params)
        assert 1000.0 not in out["val"].values
        # Surviving values must be untouched (no clipping happened).
        assert set(out["val"].tolist()) == {1.0, 2.0, 3.0, 4.0, 5.0}

    def test_nan_rows_retained(self) -> None:
        """NaN values are treated as inliers and must survive filtering."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, float("nan"), 1000.0]})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        out = IQRApplier().apply(df, params)
        assert out["val"].isna().sum() == 1
        assert 1000.0 not in out["val"].dropna().values

    def test_no_bounds_is_passthrough(self) -> None:
        """Empty params (no columns fitted) must return the frame unchanged."""
        df = pd.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = IQRApplier().apply(df, {})
        assert len(out) == len(df)

    def test_unknown_column_in_bounds_ignored(self) -> None:
        """Bounds referencing a column absent from the frame must not raise."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = {"bounds": {"nonexistent": {"lower": 0.0, "upper": 1.0}}}
        out = IQRApplier().apply(df, params)
        assert len(out) == len(df)

    def test_polars_engine_matches_pandas_engine(self) -> None:
        """Polars apply path must drop the same rows as the pandas path."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        params = IQRCalculator().fit(df, {"columns": ["val"]})
        pd_out = IQRApplier().apply(df, params)
        pl_out = IQRApplier().apply(pl.from_pandas(df), params)
        assert sorted(pd_out["val"].tolist()) == sorted(pl_out["val"].to_list())

    def test_tuple_xy_input_filters_y_in_sync(self) -> None:
        """apply on an (X, y) tuple must filter y rows to match the surviving X rows."""
        X = pd.DataFrame({"val": [1.0, 2.0, 3.0, 1000.0]})
        y = pd.Series([10, 20, 30, 40])
        params = IQRCalculator().fit(X, {"columns": ["val"]})
        X_out, y_out = IQRApplier().apply((X, y), params)
        assert len(X_out) == len(y_out)
        assert 40 not in y_out.values

    def test_polars_no_bounds_passthrough(self) -> None:
        """Empty bounds dict must short-circuit the polars path and return input unchanged."""
        df = pl.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = IQRApplier().apply(df, {})
        assert out["val"].to_list() == [1.0, 2.0, 1000.0]

    def test_polars_unknown_column_in_bounds_skipped(self) -> None:
        """Polars path must skip bound entries for columns absent from the frame."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = {"bounds": {"nonexistent": {"lower": 0.0, "upper": 1.0}}}
        out = IQRApplier().apply(df, params)
        assert out["val"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------


class TestZScoreCalculator:
    def test_stats_match_mean_and_population_std(self) -> None:
        """fit must store the population mean/std (ddof=0) used for z computation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"val": values})
        params = ZScoreCalculator().fit(df, {"columns": ["val"], "threshold": 3.0})
        series = pd.Series(values)
        assert params["stats"]["val"]["mean"] == pytest.approx(series.mean())
        assert params["stats"]["val"]["std"] == pytest.approx(series.std(ddof=0))
        assert params["threshold"] == 3.0

    def test_zero_variance_column_warns_and_excluded(self) -> None:
        """A constant column has std=0 and must be skipped (division-by-zero guard)."""
        df = pd.DataFrame({"val": [7.0] * 8})
        params = ZScoreCalculator().fit(df, {"columns": ["val"]})
        assert params["stats"] == {}
        assert any("Zero variance" in w for w in params["warnings"])

    def test_user_picked_no_columns_returns_empty(self) -> None:
        """Explicit empty columns list means the node is a no-op."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        assert ZScoreCalculator().fit(df, {"columns": []}) == {}

    def test_non_numeric_column_warns(self) -> None:
        """Entirely non-numeric column must warn and be excluded from stats."""
        df = pd.DataFrame({"val": ["x", "y"]})
        params = ZScoreCalculator().fit(df, {"columns": ["val"]})
        assert params["stats"] == {}
        assert any("val" in w for w in params["warnings"])

    def test_infer_output_schema_passes_through(self) -> None:
        """Z-score removes rows, not columns; output schema equals input schema."""
        schema = object()
        assert (
            ZScoreCalculator().infer_output_schema(typing.cast(SkyulfSchema, schema), {}) is schema
        )


class TestZScoreApplier:
    def test_rows_beyond_threshold_removed(self) -> None:
        """Rows with |z| > threshold must be dropped; the rest kept verbatim.

        Stats are supplied directly (rather than via fit on this tiny sample)
        because a single 500.0 outlier would otherwise inflate std enough to
        pull its own z-score back under the threshold.
        """
        params: Dict[str, Any] = {
            "stats": {"val": {"mean": 10.0, "std": 0.2}},
            "threshold": 3.0,
        }
        df = pd.DataFrame({"val": [10.0, 10.1, 9.9, 10.2, 9.8, 500.0]})
        out = ZScoreApplier().apply(df, params)
        assert 500.0 not in out["val"].values
        assert len(out) == 5

    def test_boundary_within_threshold_kept(self) -> None:
        """A value exactly at the threshold boundary must be kept (<=, not <)."""
        stat_mean, stat_std, threshold = 0.0, 1.0, 3.0
        params: Dict[str, Any] = {
            "stats": {"val": {"mean": stat_mean, "std": stat_std}},
            "threshold": threshold,
        }
        df = pd.DataFrame({"val": [0.0, 3.0, -3.0, 3.0001]})
        out = ZScoreApplier().apply(df, params)
        assert sorted(out["val"].tolist()) == [-3.0, 0.0, 3.0]

    def test_std_zero_column_skipped_in_apply(self) -> None:
        """If stats carry std=0 for a column, that column's mask must be a no-op."""
        params = {"stats": {"val": {"mean": 5.0, "std": 0.0}}, "threshold": 3.0}
        df = pd.DataFrame({"val": [5.0, 5.0, 999.0]})
        out = ZScoreApplier().apply(df, params)
        assert len(out) == len(df)

    def test_no_stats_is_passthrough(self) -> None:
        """Empty stats (no columns fitted) must return the frame unchanged."""
        df = pd.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = ZScoreApplier().apply(df, {})
        assert len(out) == len(df)

    def test_polars_engine_matches_pandas_engine(self) -> None:
        """Polars apply path must drop the same rows as the pandas path."""
        values = [10.0, 10.1, 9.9, 10.2, 9.8, 500.0]
        df = pd.DataFrame({"val": values})
        params = ZScoreCalculator().fit(df, {"columns": ["val"], "threshold": 3.0})
        pd_out = ZScoreApplier().apply(df, params)
        pl_out = ZScoreApplier().apply(pl.from_pandas(df), params)
        assert sorted(pd_out["val"].tolist()) == sorted(pl_out["val"].to_list())

    def test_polars_no_stats_passthrough(self) -> None:
        """Empty stats dict must short-circuit the polars path and return input unchanged."""
        df = pl.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = ZScoreApplier().apply(df, {})
        assert out["val"].to_list() == [1.0, 2.0, 1000.0]

    def test_polars_unknown_column_in_stats_skipped(self) -> None:
        """Polars path must skip stat entries for columns absent from the frame."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = {"stats": {"nonexistent": {"mean": 0.0, "std": 1.0}}, "threshold": 3.0}
        out = ZScoreApplier().apply(df, params)
        assert out["val"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------


class TestWinsorizeCalculator:
    def test_bounds_match_percentile_formula(self) -> None:
        """Bounds must equal the configured lower/upper percentiles of the data."""
        values = list(range(1, 101))  # 1..100
        df = pd.DataFrame({"val": values})
        params = WinsorizeCalculator().fit(
            df, {"columns": ["val"], "lower_percentile": 5.0, "upper_percentile": 95.0}
        )
        series = pd.Series(values, dtype=float)
        expected_lower = series.quantile(0.05)
        expected_upper = series.quantile(0.95)
        bounds = params["bounds"]["val"]
        assert bounds["lower"] == pytest.approx(expected_lower)
        assert bounds["upper"] == pytest.approx(expected_upper)

    def test_user_picked_no_columns_returns_empty(self) -> None:
        """Explicit empty columns list means the node is a no-op."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        assert WinsorizeCalculator().fit(df, {"columns": []}) == {}

    def test_non_numeric_column_warns(self) -> None:
        """Entirely non-numeric column must warn and be excluded from bounds."""
        df = pd.DataFrame({"val": ["a", "b"]})
        params = WinsorizeCalculator().fit(df, {"columns": ["val"]})
        assert params["bounds"] == {}
        assert any("val" in w for w in params["warnings"])

    def test_infer_output_schema_passes_through(self) -> None:
        """Winsorize clips values in place; output schema equals input schema."""
        schema = object()
        assert (
            WinsorizeCalculator().infer_output_schema(typing.cast(SkyulfSchema, schema), {})
            is schema
        )


class TestWinsorizeApplier:
    def test_values_are_clipped_not_removed(self) -> None:
        """Out-of-range values must be clipped to the bound, and row count preserved."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        params = WinsorizeCalculator().fit(
            df, {"columns": ["val"], "lower_percentile": 0.0, "upper_percentile": 80.0}
        )
        out = WinsorizeApplier().apply(df, params)
        # Row count is unchanged -- winsorize clips, it does not drop rows.
        assert len(out) == len(df)
        upper_bound = params["bounds"]["val"]["upper"]
        assert out["val"].max() == pytest.approx(upper_bound)
        assert out["val"].iloc[-1] == pytest.approx(upper_bound)

    def test_non_numeric_column_untouched(self) -> None:
        """A bound referencing a non-numeric dtype column must be skipped, not raise."""
        df = pd.DataFrame({"val": [1.0, 1000.0], "label": ["a", "b"]})
        params = {"bounds": {"label": {"lower": 0.0, "upper": 1.0}}}
        out = WinsorizeApplier().apply(df, params)
        assert out["label"].tolist() == ["a", "b"]

    def test_pandas_unknown_column_in_bounds_skipped(self) -> None:
        """Pandas path must skip bound entries for columns absent from the frame."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = {"bounds": {"nonexistent": {"lower": 0.0, "upper": 1.0}}}
        out = WinsorizeApplier().apply(df, params)
        assert out["val"].tolist() == [1.0, 2.0, 3.0]

    def test_no_bounds_is_passthrough(self) -> None:
        """Empty bounds (no columns fitted) must return the frame unchanged."""
        df = pd.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = WinsorizeApplier().apply(df, {})
        assert out["val"].tolist() == [1.0, 2.0, 1000.0]

    def test_polars_no_bounds_passthrough(self) -> None:
        """Empty bounds dict must short-circuit the polars path and return input unchanged."""
        df = pl.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = WinsorizeApplier().apply(df, {})
        assert out["val"].to_list() == [1.0, 2.0, 1000.0]

    def test_polars_unknown_column_in_bounds_skipped(self) -> None:
        """Polars path must skip bound entries for columns absent from the frame."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = {"bounds": {"nonexistent": {"lower": 0.0, "upper": 1.0}}}
        out = WinsorizeApplier().apply(df, params)
        assert out["val"].to_list() == [1.0, 2.0, 3.0]

    def test_polars_engine_matches_pandas_engine(self) -> None:
        """Polars clip path must produce the same clipped values as the pandas path."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        params = WinsorizeCalculator().fit(
            df, {"columns": ["val"], "lower_percentile": 0.0, "upper_percentile": 80.0}
        )
        pd_out = WinsorizeApplier().apply(df, params)
        pl_out = WinsorizeApplier().apply(pl.from_pandas(df), params)
        np.testing.assert_allclose(pd_out["val"].to_numpy(), pl_out["val"].to_numpy())

    def test_integer_column_cast_to_float_in_polars(self) -> None:
        """Polars path casts to float64 before clipping to avoid integer truncation."""
        df_pd = pd.DataFrame({"val": [1, 2, 3, 4, 5, 1000]})
        params = WinsorizeCalculator().fit(
            df_pd, {"columns": ["val"], "lower_percentile": 0.0, "upper_percentile": 80.0}
        )
        pl_out = WinsorizeApplier().apply(pl.from_pandas(df_pd), params)
        assert pl_out["val"].dtype == pl.Float64


# ---------------------------------------------------------------------------
# Elliptic Envelope
# ---------------------------------------------------------------------------


class TestEllipticEnvelopeCalculator:
    def test_fit_produces_model_per_column(self) -> None:
        """fit must produce a fitted EllipticEnvelope model per requested column."""
        rng = np.random.RandomState(0)
        values = rng.normal(0, 1, 50).tolist()
        df = pd.DataFrame({"val": values})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"], "contamination": 0.1})
        assert "val" in params["models"]
        assert params["contamination"] == 0.1
        assert params["warnings"] == []

    def test_too_few_samples_warns_and_skips_model(self) -> None:
        """Columns with fewer than 5 valid samples must warn and produce no model."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"]})
        assert params["models"] == {}
        assert any("Too few samples" in w for w in params["warnings"])

    def test_no_auto_detected_columns_returns_empty(self) -> None:
        """When auto-detection finds no numeric column, fit must short-circuit to {}."""
        df = pd.DataFrame({"label": ["a", "b", "c", "d", "e"]})
        assert EllipticEnvelopeCalculator().fit(df, {}) == {}

    def test_fit_failure_is_caught_and_warned(self) -> None:
        """A degenerate (zero-variance) column can make EllipticEnvelope.fit raise.

        The calculator must catch it, record a warning, and skip that column's
        model rather than propagating the exception.
        """
        df = pd.DataFrame({"val": [5.0] * 20})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"]})
        assert "val" not in params["models"]
        assert params["warnings"]

    def test_user_picked_no_columns_returns_empty(self) -> None:
        """Explicit empty columns list means the node is a no-op."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert EllipticEnvelopeCalculator().fit(df, {"columns": []}) == {}

    def test_infer_output_schema_passes_through(self) -> None:
        """EllipticEnvelope filters rows, not columns; schema is passed through."""
        schema = object()
        assert (
            EllipticEnvelopeCalculator().infer_output_schema(typing.cast(SkyulfSchema, schema), {})
            is schema
        )


class TestEllipticEnvelopeApplier:
    def test_extreme_outlier_row_removed(self) -> None:
        """A single far-outlying value must be classified as an outlier and dropped."""
        rng = np.random.RandomState(1)
        values = rng.normal(0, 1, 60).tolist()
        values.append(500.0)  # blatant outlier
        df = pd.DataFrame({"val": values})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"], "contamination": 0.05})
        out = EllipticEnvelopeApplier().apply(df, params)
        assert 500.0 not in out["val"].values
        assert len(out) < len(df)

    def test_no_models_is_passthrough(self) -> None:
        """Empty params (no columns fitted) must return the frame unchanged."""
        df = pd.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = EllipticEnvelopeApplier().apply(df, {})
        assert len(out) == len(df)

    def test_polars_no_models_passthrough(self) -> None:
        """Empty models dict must short-circuit the polars path and return input unchanged."""
        df = pl.DataFrame({"val": [1.0, 2.0, 1000.0]})
        out = EllipticEnvelopeApplier().apply(df, {})
        assert out["val"].to_list() == [1.0, 2.0, 1000.0]

    def test_all_nan_column_skipped_by_filter_helper(self) -> None:
        """A column that is entirely NaN yields an empty valid_idx and is skipped."""
        from skyulf.preprocessing.outliers.elliptic import _elliptic_filter_pandas

        rng = np.random.RandomState(6)
        fit_values = rng.normal(0, 1, 20).tolist()
        params = EllipticEnvelopeCalculator().fit(
            pd.DataFrame({"val": fit_values}), {"columns": ["val"]}
        )
        all_nan_df = pd.DataFrame({"val": [float("nan")] * 5})
        mask = _elliptic_filter_pandas(all_nan_df, params["models"])
        # No valid samples to predict on -> every row stays an inlier (True).
        assert mask.all()

    def test_predict_failure_is_caught_and_column_treated_as_inlier(self) -> None:
        """If model.predict raises, the column's mask must default to all-True (fail open)."""
        from skyulf.preprocessing.outliers.elliptic import _elliptic_filter_pandas

        class _RaisingModel:
            def predict(self, arr: Any) -> Any:
                raise RuntimeError("boom")

        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        mask = _elliptic_filter_pandas(df, {"val": _RaisingModel()})
        assert mask.all()

    def test_nan_rows_kept_regardless_of_model_prediction(self) -> None:
        """NaN values must never be dropped -- they are excluded from model.predict."""
        rng = np.random.RandomState(2)
        values = rng.normal(0, 1, 40).tolist()
        df = pd.DataFrame({"val": values})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"]})

        df_with_nan = df.copy()
        df_with_nan.loc[0, "val"] = float("nan")
        out = EllipticEnvelopeApplier().apply(df_with_nan, params)
        assert out["val"].isna().any()

    def test_unknown_column_in_models_ignored(self) -> None:
        """A model keyed by a column absent from the frame must not raise."""
        rng = np.random.RandomState(3)
        values = rng.normal(0, 1, 30).tolist()
        df = pd.DataFrame({"other": values})
        fit_df = pd.DataFrame({"val": values})
        params = EllipticEnvelopeCalculator().fit(fit_df, {"columns": ["val"]})
        out = EllipticEnvelopeApplier().apply(df, params)
        assert len(out) == len(df)

    def test_polars_engine_matches_pandas_engine(self) -> None:
        """Polars apply path (convert->filter->convert back) must match pandas filtering."""
        rng = np.random.RandomState(4)
        values = rng.normal(0, 1, 60).tolist()
        values.append(500.0)
        df = pd.DataFrame({"val": values})
        params = EllipticEnvelopeCalculator().fit(df, {"columns": ["val"], "contamination": 0.05})
        pd_out = EllipticEnvelopeApplier().apply(df, params)
        pl_out = EllipticEnvelopeApplier().apply(pl.from_pandas(df), params)
        assert sorted(pd_out["val"].tolist()) == sorted(pl_out["val"].to_list())

    def test_polars_tuple_xy_filters_y_in_sync(self) -> None:
        """Polars engine with an (X, y) tuple must filter y rows to match survivors."""
        rng = np.random.RandomState(7)
        values = rng.normal(0, 1, 60).tolist()
        values.append(500.0)
        X = pl.DataFrame({"val": values})
        y = pl.Series("target", list(range(len(values))))
        params = EllipticEnvelopeCalculator().fit(
            pd.DataFrame({"val": values}), {"columns": ["val"], "contamination": 0.05}
        )
        X_out, y_out = EllipticEnvelopeApplier().apply((X, y), params)
        assert X_out.shape[0] == y_out.shape[0]
        assert (len(values) - 1) not in y_out.to_list()

    def test_tuple_xy_input_filters_y_in_sync(self) -> None:
        """apply on an (X, y) tuple must filter y rows to match the surviving X rows."""
        rng = np.random.RandomState(5)
        values = rng.normal(0, 1, 60).tolist()
        values.append(500.0)
        X = pd.DataFrame({"val": values})
        y = pd.Series(range(len(values)))
        params = EllipticEnvelopeCalculator().fit(X, {"columns": ["val"], "contamination": 0.05})
        X_out, y_out = EllipticEnvelopeApplier().apply((X, y), params)
        assert len(X_out) == len(y_out)
        assert len(values) - 1 not in y_out.values


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has a missing ``income`` value — closer to production data than the
    small synthetic frames used elsewhere in this file.
    """

    def test_iqr_removes_income_outliers_and_keeps_missing_rows(self) -> None:
        df = load_sample_dataset("customers")
        params = IQRCalculator().fit(df, {"columns": ["income"]})
        out = IQRApplier().apply(df, params)

        bounds = params["bounds"]["income"]
        kept_income = out["income"]
        in_bounds = (kept_income >= bounds["lower"]) & (kept_income <= bounds["upper"])
        assert (in_bounds | kept_income.isna()).all()
        # Rows with a missing income must be retained, not treated as outliers.
        assert df["income"].isna().sum() == out["income"].isna().sum()
