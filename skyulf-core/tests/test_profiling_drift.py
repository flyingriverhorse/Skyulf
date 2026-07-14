"""Tests for skyulf.profiling.drift.DriftCalculator."""

import numpy as np
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

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


def test_calculate_drift_computes_categorical_psi_for_low_cardinality_string_column() -> None:
    """A low-cardinality string column should now get PSI-based categorical
    drift detection instead of being skipped entirely."""
    reference = pl.DataFrame({"cat": ["a", "b", "c"]})
    current = pl.DataFrame({"cat": ["a", "b", "c"]})

    report = DriftCalculator(reference, current).calculate_drift()

    assert "cat" in report.column_drifts
    col_drift = report.column_drifts["cat"]
    assert col_drift.drift_detected is False
    metric_names = {m.metric for m in col_drift.metrics}
    assert metric_names == {"psi_categorical"}


def test_calculate_drift_detects_categorical_distribution_shift() -> None:
    """A current dataset whose category proportions have shifted heavily
    away from the reference should be flagged as drifted."""
    rng = np.random.default_rng(3)
    reference = pl.DataFrame({"cat": rng.choice(["a", "b", "c"], size=500, p=[0.8, 0.1, 0.1])})
    current = pl.DataFrame({"cat": rng.choice(["a", "b", "c"], size=500, p=[0.1, 0.1, 0.8])})

    report = DriftCalculator(reference, current).calculate_drift()

    assert "cat" in report.column_drifts
    assert report.column_drifts["cat"].drift_detected is True
    assert report.drifted_columns_count == 1


def test_calculate_drift_skips_high_cardinality_categorical_column() -> None:
    """A near-unique-per-row string column (free text / IDs) must be skipped
    rather than blowing up the PSI computation on effectively-unique values."""
    n = 200
    reference = pl.DataFrame({"cat": [f"id_{i}" for i in range(n)]})
    current = pl.DataFrame({"cat": [f"id_{i}" for i in range(n)]})

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


def test_scipy_missing_disables_flag_and_blocks_drift_calculation() -> None:
    """If scipy cannot be imported, SCIPY_AVAILABLE is False and calculate_drift raises."""
    import importlib
    import sys

    import skyulf.profiling.drift as drift_module

    original_scipy_stats = sys.modules.get("scipy.stats")
    sys.modules["scipy.stats"] = None  # ty: ignore[invalid-assignment]
    try:
        reloaded = importlib.reload(drift_module)
        assert reloaded.SCIPY_AVAILABLE is False
        calc = reloaded.DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
        with pytest.raises(ImportError, match="scipy is required"):
            calc.calculate_drift()
    finally:
        # Restore real scipy.stats and reload the module so later tests see a working scipy.
        if original_scipy_stats is not None:
            sys.modules["scipy.stats"] = original_scipy_stats
        else:
            del sys.modules["scipy.stats"]
        importlib.reload(drift_module)


def test_calculate_drift_skips_column_when_cast_fails() -> None:
    """A current column that cannot be cast to the reference dtype should be skipped."""
    reference = pl.DataFrame({"a": [1, 2, 3]})
    current = pl.DataFrame({"a": [[1], [2], [3]]})  # List(Int64) can't cast to Int64.

    report = DriftCalculator(reference, current).calculate_drift()

    assert "a" not in report.column_drifts


def test_calculate_drift_moderate_psi_shift_suggestion() -> None:
    """A moderate (but not critical) PSI shift should add the 'monitor closely' suggestion."""
    rng = np.random.default_rng(42)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 1000)})
    current = pl.DataFrame({"feature": rng.normal(0.4, 1, 1000)})

    report = DriftCalculator(reference, current).calculate_drift(
        thresholds={"psi": 0.01, "ks": 0.05, "wasserstein": 100.0, "kl_divergence": 100.0}
    )
    drift = report.column_drifts["feature"]
    metrics = {m.metric: m.value for m in drift.metrics}
    assert 0.1 < metrics["psi"] <= 0.25
    assert any("Monitor model performance" in s for s in drift.suggestions)


def test_calculate_drift_ks_drift_without_psi_drift_suggestion() -> None:
    """When KS flags drift but PSI does not, the 'check for outliers' suggestion should appear."""
    rng = np.random.default_rng(7)
    reference = pl.DataFrame({"feature": rng.normal(0, 1, 2000)})
    current = pl.DataFrame({"feature": rng.normal(0.08, 1, 2000)})

    report = DriftCalculator(reference, current).calculate_drift(
        thresholds={"ks": 0.9, "psi": 100.0, "wasserstein": 100.0, "kl_divergence": 100.0}
    )
    drift = report.column_drifts["feature"]
    assert drift.drift_detected is True
    assert any("population stability" in s for s in drift.suggestions)


def test_calculate_psi_returns_zero_for_empty_arrays() -> None:
    """PSI should short-circuit to 0.0 when either input array is empty."""
    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
    assert calc._calculate_psi(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert calc._calculate_psi(np.array([1.0, 2.0]), np.array([])) == 0.0


def test_calculate_kl_returns_zero_for_empty_arrays() -> None:
    """KL divergence should short-circuit to 0.0 when either input array is empty."""
    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))
    assert calc._calculate_kl(np.array([]), np.array([1.0, 2.0])) == 0.0
    assert calc._calculate_kl(np.array([1.0, 2.0]), np.array([])) == 0.0


def test_calculate_distribution_handles_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_calculate_distribution should fall back to empty bins if histogram computation errors."""
    import skyulf.profiling.drift as drift_module

    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(drift_module.np, "histogram", boom)
    dist = calc._calculate_distribution(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert dist.bins == []


def test_calculate_psi_handles_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """_calculate_psi should fall back to 0.0 if the percentile computation errors."""
    import skyulf.profiling.drift as drift_module

    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(drift_module.np, "percentile", boom)
    psi = calc._calculate_psi(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert psi == 0.0


def test_calculate_kl_handles_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """_calculate_kl should fall back to 0.0 if the percentile computation errors."""
    import skyulf.profiling.drift as drift_module

    calc = DriftCalculator(pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.0]}))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(drift_module.np, "percentile", boom)
    kl = calc._calculate_kl(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert kl == 0.0


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``age``/``income`` values — closer to production data
    than the small synthetic ``pl.DataFrame`` fixtures used elsewhere in this
    file.
    """

    def test_calculate_drift_between_plan_type_segments(self) -> None:
        df = pl.from_pandas(load_sample_dataset("customers"))
        # Compare the "basic" plan segment (reference) against everyone else
        # (current) on the numeric age/income columns, both of which contain
        # real missing values that drop_nulls() must tolerate.
        reference = df.filter(pl.col("plan_type") == "basic").select(["age", "income"])
        current = df.filter(pl.col("plan_type") != "basic").select(["age", "income"])

        report = DriftCalculator(reference, current).calculate_drift()

        assert report.missing_columns == []
        assert report.new_columns == []
        assert "age" in report.column_drifts
        assert "income" in report.column_drifts
        # Both columns retain some non-null rows on each side, so metrics must be computed.
        for column in ("age", "income"):
            metric_names = {m.metric for m in report.column_drifts[column].metrics}
            assert {"wasserstein_distance", "ks_test_p_value", "psi", "kl_divergence"} == (
                metric_names
            )
