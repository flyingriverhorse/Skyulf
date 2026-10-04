"""Finite profiling views and exact integer drift retain the reported population."""

import math
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from skyulf.profiling import analyzer as analyzer_module
from skyulf.profiling import drift as drift_module
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.correlations import calculate_correlations
from skyulf.profiling.distributions import calculate_histogram
from skyulf.profiling.drift import DriftCalculator


@pytest.fixture(autouse=True)
def verify_requested_source_root():
    """Local snapshot checks must not accidentally exercise the editable main checkout."""
    expected = os.environ.get("SKYULF_EXPECT_SOURCE_ROOT")
    if expected:
        for module in (analyzer_module, drift_module):
            assert Path(module.__file__).resolve().is_relative_to(Path(expected).resolve())


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_public_profile_scores_finite_observations_without_rewriting_input(invalid):
    """One invalid cell must not erase valid statistics or the outlier row identity."""
    rng = np.random.default_rng(203)
    x = rng.normal(size=64)
    y = 0.4 * x + rng.normal(size=64)
    x[-1], y[-1] = 40.0, -30.0
    x[3] = invalid
    frame = pl.DataFrame({"x": x, "y": y})
    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()
    valid = np.isfinite(x)
    stats = profile.columns["x"].numeric_stats
    assert stats is not None and stats.mean == pytest.approx(x[valid].mean())
    assert stats.std == pytest.approx(x[valid].std(ddof=1))
    assert stats.min == x[valid].min() and stats.max == x[valid].max()
    histogram = profile.columns["x"].histogram
    assert histogram is not None and sum(bucket.count for bucket in histogram) == 63
    assert profile.columns["x"].normality_test is not None
    expected_corr = np.corrcoef(x[valid], y[valid])[0, 1]
    assert profile.correlations is not None
    assert profile.correlations.values[0][1] == pytest.approx(expected_corr)
    assert profile.vif is not None
    assert profile.vif["x"] == pytest.approx(1 / (1 - expected_corr**2))
    assert profile.outliers is not None
    assert profile.outliers.analyzed_rows == 63 and profile.outliers.total_rows == 64
    assert all(point.index != 3 for point in profile.outliers.top_outliers)
    assert any(point.index == 63 for point in profile.outliers.top_outliers)
    assert all(
        point.values == frame.row(point.index, named=True)
        for point in profile.outliers.top_outliers
    )
    assert profile.missing_cells_percentage == 0
    assert profile.columns["x"].missing_count == 0
    assert profile.row_count == 64 and profile.sample_data == frame.to_dicts()
    assert analyzer.df.equals(frame)
    exclusions = [alert for alert in profile.alerts if alert.type == "Non-finite Values"]
    assert len(exclusions) == 1 and exclusions[0].column == "x"
    assert "1" in exclusions[0].message and "finite" in exclusions[0].message


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan")])
def test_public_leaf_calculations_keep_finite_pairs_and_histogram_counts(invalid):
    """Direct helpers must exclude an invalid pair without shifting the other column."""
    frame = pl.DataFrame({"x": [1.0, invalid, 3.0, 5.0, None], "y": [7.0, 500.0, 2.0, 6.0, 9.0]})
    matrix = calculate_correlations(frame.lazy(), ["x", "y"])
    histogram = calculate_histogram(frame.lazy(), "x", bins=3)
    assert matrix is not None
    assert matrix.values[0][1] == pytest.approx(np.corrcoef([1.0, 3.0, 5.0], [7.0, 2.0, 6.0])[0, 1])
    assert histogram is not None and sum(bucket.count for bucket in histogram) == 3
    assert frame.height == 5 and frame["x"].null_count() == 1


@pytest.mark.parametrize("values", [[float("inf")] * 12, [None] * 12])
def test_public_profile_keeps_unavailable_calculations_when_no_valid_values_remain(values):
    """Excluding unusable values must not invent descriptive or joint statistics."""
    frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64), "y": np.arange(12.0)})
    profile = EDAAnalyzer(frame).analyze()
    stats = profile.columns["x"].numeric_stats
    assert stats is not None and stats.mean is None and stats.std is None
    assert profile.columns["x"].histogram is None
    assert profile.correlations is None and profile.vif is None
    assert profile.row_count == 12 and profile.sample_data == frame.to_dicts()


def test_public_profile_does_not_invent_vif_after_invalid_rows_leave_one_observation():
    """The original complete-case minimum still applies after infinity exclusion."""
    frame = pl.DataFrame({"x": [float("inf")] * 9 + [1.0], "y": np.arange(10.0)})
    profile = EDAAnalyzer(frame).analyze()
    assert profile.vif is None
    assert any(alert.type == "VIF Unavailable" for alert in profile.alerts)
    assert profile.row_count == 10


def test_public_profile_retains_missing_row_imputation_for_outliers():
    """Infinity exclusion must preserve the existing imputation policy for null cells."""
    values: list[float | None] = [float(i) for i in range(40)]
    values[2] = None
    frame = pl.DataFrame({"x": values, "y": np.arange(40.0) ** 2})
    profile = EDAAnalyzer(frame).analyze()
    assert profile.outliers is not None and profile.outliers.analyzed_rows == 40
    assert profile.columns["x"].missing_count == 1


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int64, [2**53, 2**53]),
        (pl.Int64, [-(2**63), 2**63 - 2]),
        (pl.UInt64, [2**64 - 4, 2**64 - 2]),
        (pl.Int128, [2**100, 2**100 + 2]),
        (pl.Int128, [-(2**126), 2**126]),
    ],
    ids=["int64-constant", "int64-wide", "uint64", "int128-close", "int128-wide"],
)
def test_public_integer_drift_keeps_unit_transport_distance(dtype, values):
    """Integer differences must be taken before float conversion even across a wide span."""
    reference = pl.DataFrame({"value": pl.Series(values * 8 + [None], dtype=dtype)})
    current = pl.DataFrame(
        {"value": pl.Series([value + 1 for value in values] * 8 + [None], dtype=dtype)}
    )
    report = DriftCalculator(reference, current).calculate_drift()
    column = report.column_drifts["value"]
    metrics = {metric.metric: metric for metric in column.metrics}
    distance = metrics["wasserstein_distance"]
    assert distance.raw_value == pytest.approx(1.0)
    expected_std = abs(values[1] - values[0]) / 2
    assert distance.value == pytest.approx(1 / expected_std if expected_std else 1.0)
    assert metrics["ks_statistic"].value == pytest.approx(1.0 if values[0] == values[1] else 0.5)
    assert report.reference_rows == 17 and report.current_rows == 17
    assert all(math.isfinite(metric.value) for metric in column.metrics)
    assert reference["value"].to_list() == values * 8 + [None]
    assert current["value"].dtype == dtype


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64, pl.Int128])
def test_public_integer_drift_handles_unequal_populations_exactly(dtype):
    """Transport weights must retain their empirical counts when population sizes differ."""
    origin = 2**53
    reference = pl.DataFrame({"value": pl.Series([origin, origin + 2], dtype=dtype)})
    current = pl.DataFrame({"value": pl.Series([origin, origin + 1, origin + 2], dtype=dtype)})
    column = DriftCalculator(reference, current).calculate_drift().column_drifts["value"]
    metrics = {metric.metric: metric for metric in column.metrics}
    assert metrics["wasserstein_distance"].raw_value == pytest.approx(1 / 3)
    assert metrics["wasserstein_distance"].value == pytest.approx(1 / 3)
    assert metrics["ks_statistic"].value == pytest.approx(1 / 6)


@pytest.mark.parametrize("values", [[2**100, 2**100 + 3], [None, None]])
def test_public_int128_drift_keeps_stable_and_missing_controls(values):
    """Int128 support must not create drift or force a native conversion for no-data columns."""
    frame = pl.DataFrame({"value": pl.Series(values, dtype=pl.Int128)})
    report = DriftCalculator(frame, frame.clone()).calculate_drift()
    assert report.drifted_columns_count == 0
    assert all(
        metric.value == 0 or metric.metric == "ks_test_p_value"
        for column in report.column_drifts.values()
        for metric in column.metrics
    )


@pytest.mark.parametrize("dtype", [pl.Float64, pl.Decimal(12, 2)])
def test_public_noninteger_drift_preserves_fractional_distance(dtype):
    """An integer precision repair must retain the established floating and Decimal paths."""
    reference = pl.DataFrame({"value": pl.Series([0, 1], dtype=dtype)})
    current = pl.DataFrame({"value": pl.Series([Decimal("0.25"), Decimal("1.25")]).cast(dtype)})
    column = DriftCalculator(reference, current).calculate_drift().column_drifts["value"]
    distance = next(metric for metric in column.metrics if metric.metric == "wasserstein_distance")
    assert distance.raw_value == pytest.approx(0.25)
    assert distance.value == pytest.approx(0.5)


def test_decimal_eda_preserves_precision_when_statistics_are_unavailable():
    """Decimal support must not silently collapse distinct values in approximate metrics."""
    frame = pl.DataFrame(
        {"value": [Decimal("9007199254740993.01"), None, Decimal("9007199254740993.02")]}
    )
    profile = EDAAnalyzer(frame).analyze()
    assert profile.columns["value"].dtype == "Numeric"
    assert profile.columns["value"].numeric_stats is None
    assert profile.sample_data == frame.to_dicts()
    assert any(
        alert.type == "Numeric Precision" and alert.column == "value" for alert in profile.alerts
    )


@pytest.mark.parametrize(
    ("dtype", "origin"),
    [(pl.Int64, 2**53), (pl.Int64, -(2**62)), (pl.UInt64, 2**63), (pl.Int128, 2**100)],
)
def test_large_integer_drift_matches_translated_small_coordinate_oracle(dtype, origin):
    """Translation must preserve transport and KS for unequal, repeated empirical populations."""
    from scipy.stats import ks_2samp, wasserstein_distance

    reference_offsets = [-7, -1, 4, 4]
    current_offsets = [-6, 2, 3, 7, 7, 7]
    reference = pl.DataFrame(
        {"value": pl.Series([origin + value for value in reference_offsets], dtype=dtype)}
    )
    current = pl.DataFrame(
        {"value": pl.Series([origin + value for value in current_offsets], dtype=dtype)}
    )
    column = DriftCalculator(reference, current).calculate_drift().column_drifts["value"]
    metrics = {metric.metric: metric for metric in column.metrics}
    expected = wasserstein_distance(reference_offsets, current_offsets)
    assert metrics["wasserstein_distance"].raw_value == pytest.approx(expected)
    assert metrics["wasserstein_distance"].value == pytest.approx(
        expected / np.std(reference_offsets)
    )
    assert metrics["ks_statistic"].value == pytest.approx(
        ks_2samp(reference_offsets, current_offsets).statistic
    )
    assert column.distribution is not None
    assert all(bucket.bin_end > bucket.bin_start for bucket in column.distribution.bins)


@pytest.mark.parametrize(
    "reference_dtype,current_dtype", [(pl.Int64, pl.UInt64), (pl.UInt64, pl.Int64)]
)
@pytest.mark.parametrize("shift", [0, 1])
def test_signed_unsigned_integer_drift_preserves_ks_identity(reference_dtype, current_dtype, shift):
    """Combining native integer widths must not round away distinct empirical CDFs."""
    origin = 2**53
    reference = pl.DataFrame({"value": pl.Series([origin] * 6, dtype=reference_dtype)})
    current = pl.DataFrame({"value": pl.Series([origin + shift] * 6, dtype=current_dtype)})
    column = DriftCalculator(reference, current).calculate_drift().column_drifts["value"]
    metrics = {metric.metric: metric for metric in column.metrics}
    assert metrics["ks_statistic"].value == shift
    assert metrics["ks_test_p_value"].value == pytest.approx(1 / 462 if shift else 1.0)
    assert metrics["wasserstein_distance"].raw_value == shift
    assert column.drift_detected is bool(shift)


@pytest.mark.parametrize(
    "reference_dtype,current_dtype",
    [(pl.Int64, pl.UInt64), (pl.UInt64, pl.Int64), (pl.Int64, pl.UInt32)],
)
def test_signed_unsigned_ks_matches_exact_coordinate_oracle(reference_dtype, current_dtype):
    """Joint ordering must preserve ties and the KS p-value across safe or widening casts."""
    from scipy.stats import ks_2samp

    origin = 2**53 if current_dtype != pl.UInt32 else 0
    ref_offsets, curr_offsets = [0, 2, 2, 5], [1, 1, 2, 4, 7]
    reference = pl.DataFrame(
        {"value": pl.Series([origin + value for value in ref_offsets], dtype=reference_dtype)}
    )
    current = pl.DataFrame(
        {"value": pl.Series([origin + value for value in curr_offsets], dtype=current_dtype)}
    )
    column = DriftCalculator(reference, current).calculate_drift().column_drifts["value"]
    metrics = {metric.metric: metric for metric in column.metrics}
    expected = ks_2samp(ref_offsets, curr_offsets)
    assert metrics["ks_statistic"].value == pytest.approx(expected.statistic)
    assert metrics["ks_test_p_value"].value == pytest.approx(expected.pvalue)
