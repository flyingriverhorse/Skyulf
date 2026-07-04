"""Tests for skyulf.profiling.drift.DriftCalculator."""

import numpy as np
import polars as pl

from skyulf.profiling.drift import DriftCalculator, DriftReport


def test_calculate_drift_detects_shifted_distribution() -> None:
    """A current dataset shifted well away from the reference should be flagged as drifted."""
    rng = np.random.default_rng(0)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 500)})
    current = pl.DataFrame({"feature": rng.normal(5, 1, 500)})

    report = DriftCalculator(reference, current).calculate_drift()

    assert isinstance(report, DriftReport)
    assert report.reference_rows == 500
    assert report.current_rows == 500
    assert "feature" in report.column_drifts
    assert report.column_drifts["feature"].drift_detected is True
    assert report.drifted_columns_count == 1
    metric_names = {m.metric for m in report.column_drifts["feature"].metrics}
    assert {"wasserstein_distance", "ks_test_p_value", "psi", "kl_divergence"} == metric_names


def test_calculate_drift_no_drift_for_identical_distribution() -> None:
    """Sampling from the same distribution twice (large N) should not trigger drift."""
    rng = np.random.default_rng(1)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 5000)})
    current = pl.DataFrame({"feature": rng.normal(0, 1, 5000)})

    report = DriftCalculator(reference, current).calculate_drift()

    metrics = {m.metric: m.value for m in report.column_drifts["feature"].metrics}
    # PSI is the most robust indicator for "no real population shift" on large samples.
    assert metrics["psi"] < 0.2


def test_calculate_drift_reports_missing_and_new_columns() -> None:
    """Schema drift: columns present only in reference/current should be reported."""
    reference = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    current = pl.DataFrame({"a": [1.0, 2.0, 3.0], "c": [1.0, 2.0, 3.0]})

    report = DriftCalculator(reference, current).calculate_drift()

    assert report.missing_columns == ["b"]
    assert report.new_columns == ["c"]
    assert "a" in report.column_drifts
    assert "b" not in report.column_drifts


def test_calculate_drift_skips_non_numeric_columns() -> None:
    """Non-numeric common columns should be excluded from drift metrics entirely."""
    reference = pl.DataFrame({"cat": ["a", "b", "c"]})
    current = pl.DataFrame({"cat": ["a", "b", "c"]})

    report = DriftCalculator(reference, current).calculate_drift()

    assert report.column_drifts == {}
    assert report.drifted_columns_count == 0


def test_calculate_drift_handles_empty_data_after_nulls() -> None:
    """A column that is entirely null in either frame should be safely skipped."""
    reference = pl.DataFrame({"feature": pl.Series([None, None], dtype=pl.Float64)})
    current = pl.DataFrame({"feature": [1.0, 2.0]})

    report = DriftCalculator(reference, current).calculate_drift()

    assert report.column_drifts == {}


def test_calculate_psi_is_zero_for_constant_reference() -> None:
    """PSI should short-circuit to 0.0 when the reference array has no variance."""
    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
    psi = calc._calculate_psi(np.array([5.0, 5.0, 5.0]), np.array([5.0, 6.0, 7.0]))
    assert psi == 0.0


def test_calculate_kl_is_zero_for_constant_reference() -> None:
    """KL divergence should short-circuit to 0.0 when the reference array has no variance."""
    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
    kl = calc._calculate_kl(np.array([5.0, 5.0, 5.0]), np.array([5.0, 6.0, 7.0]))
    assert kl == 0.0


def test_calculate_distribution_handles_constant_arrays() -> None:
    """A constant reference/current pair should still produce a valid histogram."""
    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
    dist = calc._calculate_distribution(np.array([3.0, 3.0]), np.array([3.0, 3.0]))
    assert len(dist.bins) == 20
    assert sum(b.reference_count for b in dist.bins) == 2
    assert sum(b.current_count for b in dist.bins) == 2


def test_calculate_drift_custom_thresholds_override_defaults() -> None:
    """Custom thresholds should be merged with (and override) the defaults."""
    rng = np.random.default_rng(2)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 300)})
    current = pl.DataFrame({"feature": rng.normal(0.05, 1, 300)})

    lenient_report = DriftCalculator(reference, current).calculate_drift(
        thresholds={"ks": 0.0, "psi": 100.0, "wasserstein": 100.0, "kl_divergence": 100.0}
    )
    assert lenient_report.column_drifts["feature"].drift_detected is False
