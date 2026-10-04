"""Distribution decisions need practical shifts and calibrated statistical evidence."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.stats import fisher_exact

from skyulf.profiling.drift import DriftCalculator


def report(reference, current, thresholds=None):
    """Exercise the public calculator without depending on private statistical helpers."""
    return DriftCalculator(pl.DataFrame(reference), pl.DataFrame(current)).calculate_drift(
        thresholds
    )


@pytest.mark.parametrize("size", [1, 2, 3])
def test_tiny_numeric_populations_are_explicitly_insufficient(size):
    """A large observed difference cannot create evidence that these sample sizes cannot supply."""
    result = report({"x": list(range(size))}, {"x": list(range(100, 100 + size))})
    column = result.column_drifts["x"]
    assert column.drift_detected is False
    assert column.evidence.status == "insufficient_data"
    assert column.evidence.reference_count == column.evidence.current_count == size
    assert column.evidence.reason
    assert any(metric.has_drift for metric in column.metrics if metric.metric != "ks_test_p_value")
    assert not any("retrain" in suggestion.lower() for suggestion in column.suggestions)


@pytest.mark.parametrize("kind", ["numeric", "categorical_sparse", "categorical_dense"])
def test_null_population_calibration_avoids_fixed_threshold_false_alarms(kind):
    """A fixed seeded null simulation should respect feature-family error control, unlike raw OR flags."""
    false_reports = 0
    for seed in range(40):
        rng = np.random.default_rng(seed + 93041)
        if kind == "numeric":
            reference = {f"x{i}": rng.normal(size=30) for i in range(5)}
            current = {f"x{i}": rng.normal(size=30) for i in range(5)}
        else:
            probabilities = [0.48, 0.50, 0.02] if kind.endswith("sparse") else [0.34, 0.33, 0.33]
            reference = {
                f"x{i}": rng.choice(["a", "b", "rare"], size=30, p=probabilities) for i in range(5)
            }
            current = {
                f"x{i}": rng.choice(["a", "b", "rare"], size=30, p=probabilities) for i in range(5)
            }
        false_reports += int(report(reference, current).drifted_columns_count > 0)
    assert false_reports <= 5, f"{false_reports}/40 independent null reports signaled drift"


def test_multiple_feature_correction_includes_only_distribution_tests():
    """Duplicating a marginal p=.034 comparison must lose support under two-test Bonferroni."""
    original = np.arange(20)
    current = original + 9
    single = report({"x": original}, {"x": current}).column_drifts["x"]
    assert single.drift_detected
    assert single.evidence.status == "supported"
    assert single.evidence.p_value == pytest.approx(0.0335416594061465)
    combined = report(
        {"x": original, "other": original, "gone": original}, {"x": current, "other": current}
    )
    assert combined.drifted_columns_count == 1  # The missing column remains structural drift.
    for column in combined.column_drifts.values():
        assert not column.drift_detected
        assert column.evidence.status == "not_detected"
        assert column.evidence.adjusted_p_value == pytest.approx(2 * single.evidence.p_value)


def test_statistical_support_remains_independent_of_effect_thresholds():
    """Large samples cannot force drift when the user's practical effect thresholds are not exceeded."""
    values = np.arange(20000) / 20000
    column = report(
        {"x": values},
        {"x": values + 0.03},
        {"psi": 100, "ks_statistic": 100, "wasserstein": 100, "kl_divergence": 100},
    ).column_drifts["x"]
    assert column.evidence.status == "supported"
    assert column.evidence.adjusted_p_value < 0.05
    assert column.drift_detected is False
    assert column.suggestions == []


@pytest.mark.parametrize("kind", ["numeric", "categorical"])
def test_constant_populations_are_valid_tests_when_replicated(kind):
    """Constants need valid unchanged/changed evidence without inventing variance or declaring unavailable."""
    first, second = (1, 9) if kind == "numeric" else ("a", "b")
    stable = report({"x": [first] * 20}, {"x": [first] * 20}).column_drifts["x"]
    shifted = report({"x": [first] * 20}, {"x": [second] * 20}).column_drifts["x"]
    assert stable.evidence.status == "not_detected" and stable.evidence.p_value == 1
    assert not stable.drift_detected
    assert shifted.evidence.status == "supported" and shifted.drift_detected


@pytest.mark.parametrize("values", [["a", "b"], ["a", "b", "c"]])
def test_sparse_categorical_samples_do_not_use_unreliable_asymptotic_evidence(values):
    """Tiny category counts must retain descriptive PSI while refusing unsupported distribution alarms."""
    result = report({"x": values}, {"x": ["z"] * len(values)}).column_drifts["x"]
    assert result.evidence.status == "insufficient_data"
    assert result.drift_detected is False
    assert result.evidence.test.startswith("fisher")


def test_sparse_multicategory_evidence_matches_exact_corrected_category_tests():
    """Rare cells need exact tests with a within-feature correction rather than asymptotic chi-square."""
    reference = ["a"] * 30 + ["b"] * 29 + ["rare"]
    current = ["a"] * 4 + ["b"] * 55 + ["rare"]
    result = report({"x": reference}, {"x": current}).column_drifts["x"]
    p_values = [
        fisher_exact(
            [
                [reference.count(label), len(reference) - reference.count(label)],
                [current.count(label), len(current) - current.count(label)],
            ]
        ).pvalue
        for label in sorted(set(reference + current))
    ]
    assert result.evidence.test == "fisher_category_bonferroni"
    assert result.evidence.p_value == pytest.approx(min(1.0, 3 * min(p_values)))
    assert result.evidence.status == "supported" and result.drift_detected


def test_dense_categorical_shift_is_supported_without_a_universal_row_floor():
    """Balanced expected counts permit a small valid three-category distribution comparison."""
    reference = ["a"] * 20 + ["b"] * 5 + ["c"] * 5
    current = ["a"] * 5 + ["b"] * 5 + ["c"] * 20
    result = report({"x": reference}, {"x": current}).column_drifts["x"]
    assert result.evidence.test == "chi_square"
    assert result.evidence.status == "supported" and result.drift_detected


def test_nonfinite_missing_values_do_not_inflate_statistical_sample_counts():
    """Statistical sufficiency is per-feature and excludes NaN and null observations."""
    result = report(
        {"x": [0.0, 1.0, None, float("nan")]}, {"x": [10.0, 11.0, None, float("nan")]}
    ).column_drifts["x"]
    assert result.evidence.reference_count == result.evidence.current_count == 2
    assert result.evidence.status == "insufficient_data"


def test_exact_integer_and_temporal_shift_support_preserves_native_precision():
    """Sample-aware evidence must use the existing exact ordering before statistical tests."""
    integers = pl.Series("x", [2**100 + i for i in range(20)], dtype=pl.Int128)
    shifted = pl.Series("x", [2**100 + 100 + i for i in range(20)], dtype=pl.Int128)
    integer_result = (
        DriftCalculator(integers.to_frame(), shifted.to_frame())
        .calculate_drift()
        .column_drifts["x"]
    )
    dates = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(20)]
    time_result = report(
        {"x": dates}, {"x": [value + timedelta(days=100) for value in dates]}
    ).column_drifts["x"]
    assert integer_result.evidence is not None and time_result.evidence is not None
    assert integer_result.evidence.status == time_result.evidence.status == "supported"
    assert integer_result.metrics[0].raw_value == 100
    assert time_result.metrics[0].raw_value == 100 * 86400


def test_schema_type_change_bypasses_statistical_sample_requirements():
    """An incompatible schema is structural evidence even when the frame has one row."""
    result = report({"x": [1]}, {"x": ["bad"]})
    assert result.drifted_columns_count == 1
    assert result.column_drifts["x"].metrics[0].metric == "type_drift"
    assert result.column_drifts["x"].evidence is None


def test_multiple_comparisons_can_make_small_sample_resolution_insufficient():
    """Four-versus-four separation is resolvable alone but not under a two-feature correction."""
    reference = {"x": list(range(4))}
    current = {"x": list(range(10, 14))}
    single = report(reference, current).column_drifts["x"]
    assert single.evidence.status == "supported" and single.drift_detected
    combined = report({**reference, "other": reference["x"]}, {**current, "other": current["x"]})
    assert combined.drifted_columns_count == 0
    assert all(
        column.evidence.status == "insufficient_data" for column in combined.column_drifts.values()
    )


@pytest.mark.parametrize("repeats", [10, 20000])
def test_dense_category_expected_counts_do_not_overflow_native_integer_width(repeats):
    """Windows Int32 defaults cannot select a different statistical test for large valid counts."""
    values = [f"category_{i}" for i in range(10) for _ in range(repeats)]
    result = report({"x": values}, {"x": values}).column_drifts["x"]
    assert result.evidence.test == "chi_square"
    assert result.evidence.p_value == 1.0
    assert result.evidence.status == "not_detected" and not result.drift_detected
