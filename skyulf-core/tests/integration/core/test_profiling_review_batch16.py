"""Temporal drift, bounded VIF solves and target overlap must retain their public contracts."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.drift import DriftCalculator


def _dates(unit="us", zone=None, start=2026):
    """Create a nonconstant time population without relying on platform epoch inference."""
    first = datetime(start, 1, 1, tzinfo=UTC if zone else None)
    return pl.Series(
        "event", [first + timedelta(days=i) for i in range(128)], dtype=pl.Datetime(unit, zone)
    )


@pytest.mark.parametrize("reference_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("current_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("aware", [False, True])
def test_equivalent_datetime_representations_are_measured_without_drift(
    reference_unit, current_unit, aware
):
    """Time unit and aware-zone differences describe the same instants, not schema drift."""
    reference = _dates(reference_unit, "UTC" if aware else None)
    current = reference.dt.cast_time_unit(current_unit)
    if aware:
        current = current.dt.convert_time_zone("Europe/Vilnius")
    report = DriftCalculator(reference.to_frame(), current.to_frame()).calculate_drift()
    assert report.drifted_columns_count == 0
    result = report.column_drifts["event"]
    assert result.drift_detected is False
    assert all(not metric.has_drift for metric in result.metrics)
    assert any(metric.metric == "ks_statistic" and metric.value == 0 for metric in result.metrics)
    assert reference.dtype == pl.Datetime(reference_unit, "UTC" if aware else None)


@pytest.mark.parametrize("unit,year", [("ms", 2500), ("us", 2500), ("ns", 2026)])
@pytest.mark.parametrize("zone", [None, "UTC"])
def test_real_datetime_shift_is_scored_in_seconds(unit, year, zone):
    """A real date shift must be detected even beyond the Int64 nanosecond epoch range."""
    reference = _dates(unit, zone, year).to_frame()
    current = reference.select(pl.col("event") + pl.duration(days=365))
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.drifted_columns_count == 1
    metrics = {metric.metric: metric for metric in report.column_drifts["event"].metrics}
    assert metrics["ks_statistic"].value == 1.0
    assert metrics["wasserstein_distance"].raw_value == pytest.approx(365 * 86400)
    assert reference["event"][0].year == year


def test_nanosecond_datetime_shift_preserves_small_physical_differences():
    """Temporal comparison must retain nanoseconds before computing floating distances."""
    first = 1767225600000000000
    reference = pl.Series("event", [first + i for i in range(128)], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )
    current = pl.Series("event", [first + 512 + i for i in range(128)], dtype=pl.Int64).cast(
        reference.dtype
    )
    report = DriftCalculator(reference.to_frame(), current.to_frame()).calculate_drift()
    metrics = {metric.metric: metric for metric in report.column_drifts["event"].metrics}
    assert metrics["ks_statistic"].value == 1.0
    assert metrics["wasserstein_distance"].raw_value == pytest.approx(512e-9)
    assert report.drifted_columns_count == 1


@pytest.mark.parametrize("first_aware", [False, True])
def test_naive_and_aware_datetimes_require_explicit_timezone_policy(first_aware):
    """Missing timezone information cannot be silently interpreted as UTC."""
    reference = _dates(zone="UTC" if first_aware else None)
    current = _dates(zone=None if first_aware else "UTC")
    report = DriftCalculator(reference.to_frame(), current.to_frame()).calculate_drift()
    assert report.column_drifts["event"].metrics[0].metric == "type_drift"
    assert report.drifted_columns_count == 1


def test_date_columns_are_scored_without_coercing_datetime_schema():
    """Date shifts can be measured while Date versus Datetime remains a schema distinction."""
    reference = _dates().cast(pl.Date).to_frame()
    current = reference.select(pl.col("event") + pl.duration(days=365))
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.drifted_columns_count == 1
    assert report.column_drifts["event"].metrics[0].metric != "type_drift"
    incompatible = DriftCalculator(reference, _dates().to_frame()).calculate_drift()
    assert incompatible.column_drifts["event"].metrics[0].metric == "type_drift"


@pytest.mark.parametrize("noise", [0.0, 1e-6])
def test_ill_conditioned_vif_uses_only_small_residual_designs(monkeypatch, noise):
    """One row pass must replace repeated full-population least-squares solves."""
    base = np.tile([-1.0, -1.0, 1.0, 1.0], 128)
    independent = np.tile([-1.0, 1.0, -1.0, 1.0], 128)
    frame = pl.DataFrame(
        {"a": base, "b": 2 * base + noise * independent, "other": base * independent}
    )
    shapes = []
    original = np.linalg.lstsq

    def record_design(design, target, *args, **kwargs):
        """Observe real solver dimensions without changing its numerical result."""
        shapes.append(design.shape)
        return original(design, target, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", record_design)
    profile = EDAAnalyzer(frame).analyze()
    assert profile.vif is not None
    expected = 999.0 if noise == 0 else 1 + 4 / noise**2
    assert profile.vif["a"] == pytest.approx(expected, rel=1e-5)
    assert profile.vif["b"] == pytest.approx(expected, rel=1e-5)
    assert profile.vif["other"] == pytest.approx(1.0)
    assert shapes and max(rows for rows, _ in shapes) <= frame.width


@pytest.mark.parametrize("overlap", [0, 1, 2, 3, 20])
def test_target_correlation_requires_three_complete_pairs(overlap):
    """Two points always imply perfect correlation and cannot justify a leakage alert."""
    values = [float(i + 1) for i in range(overlap)] + [None] * (128 - overlap)
    frame = pl.DataFrame(
        {"x": pl.Series(values, dtype=pl.Float64), "target": np.arange(128, dtype=float)}
    )
    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="regression")
    correlations = profile.target_correlations or {}
    leakage = [alert for alert in profile.alerts if alert.type == "Leakage" and alert.column == "x"]
    if overlap < 3:
        assert "x" not in correlations
        assert leakage == []
    else:
        assert correlations["x"] == pytest.approx(1.0)
        assert len(leakage) == 1
    assert frame["x"].null_count() == 128 - overlap


@pytest.mark.parametrize("finite_pairs", [2, 3])
def test_target_overlap_and_coefficient_use_the_same_finite_rows(finite_pairs):
    """Infinity must neither count as evidence nor erase valid target-feature pairs."""
    values = (
        [float(i + 1) for i in range(finite_pairs)] + [float("inf")] + [None] * (127 - finite_pairs)
    )
    frame = pl.DataFrame(
        {"x": pl.Series(values, dtype=pl.Float64), "target": np.arange(128, dtype=float)}
    )
    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="regression")
    correlations = profile.target_correlations or {}
    if finite_pairs < 3:
        assert "x" not in correlations
    else:
        assert correlations["x"] == pytest.approx(1.0)
    assert frame["x"].is_infinite().sum() == 1
