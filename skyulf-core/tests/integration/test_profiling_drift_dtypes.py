"""Dtype changes must neither erase distributions nor hide incompatible columns."""

import json
import math
from datetime import date
from decimal import Decimal

import polars as pl
import pytest

from skyulf.profiling.drift import DriftCalculator


@pytest.mark.parametrize(
    ("reference_dtype", "current_dtype", "current_values", "raw_distance"),
    [
        (pl.Int64, pl.Float64, [0.9, 1.9], 0.9),
        (pl.Int8, pl.Int64, [256, 257], 256.0),
        (pl.UInt8, pl.Int64, [-2, -1], 2.0),
        (pl.Float32, pl.Float64, [1e-8, 1.00000001], 1e-8),
        (pl.Int64, pl.String, ["0.9", "1.9"], 0.9),
        (pl.Int64, pl.Categorical, ["0.9", "1.9"], 0.9),
        (pl.Float64, pl.Decimal(10, 1), [Decimal("0.9"), Decimal("1.9")], 0.9),
        (pl.Decimal(10, 0), pl.Float64, [0.9, 1.9], 0.9),
    ],
)
def test_numeric_dtype_changes_preserve_distribution(
    reference_dtype, current_dtype, current_values, raw_distance
):
    """Fractions, wider integers and numeric text must retain their actual transport distance."""
    reference = pl.DataFrame({"feature": pl.Series([0] * 50 + [1] * 50, dtype=reference_dtype)})
    current = pl.DataFrame(
        {
            "feature": pl.Series(
                [current_values[0]] * 50 + [current_values[1]] * 50, dtype=current_dtype
            )
        }
    )
    report = DriftCalculator(reference, current).calculate_drift()
    column = report.column_drifts["feature"]
    metrics = {metric.metric: metric for metric in column.metrics}
    # Reference std is exactly 0.5; each half of the population moves by the same amount.
    assert metrics["wasserstein_distance"].raw_value == pytest.approx(raw_distance, abs=1e-12)
    assert metrics["wasserstein_distance"].value == pytest.approx(2 * raw_distance, abs=1e-12)
    assert column.distribution is not None
    assert sum(bucket.current_count for bucket in column.distribution.bins) == 100
    assert current.schema["feature"] == current_dtype


def test_fractional_shift_reports_the_expected_metrics_and_histogram():
    """The public report must agree with hand-derived CDF and binned-frequency calculations."""
    reference = pl.DataFrame({"feature": [0] * 50 + [1] * 50})
    current = pl.DataFrame({"feature": [0.9] * 50 + [1.9] * 50})
    report = DriftCalculator(reference, current).calculate_drift()
    column = report.column_drifts["feature"]
    metrics = {metric.metric: metric for metric in column.metrics}
    # Reference histogram proportions are (0.5, 0.5); shifted proportions are (0, 1).
    # PSI floors the empty bin at 0.0001; KL normalizes the smoothed proportions.
    expected_psi = (0.0001 - 0.5) * math.log(0.0001 / 0.5) + 0.5 * math.log(2)
    q0, q1 = 0.0001 / 1.0001, 1 / 1.0001
    expected_kl = q0 * math.log(q0 / 0.5) + q1 * math.log(q1 / 0.5)
    assert metrics["ks_statistic"].value == pytest.approx(0.5)
    assert metrics["psi"].value == pytest.approx(expected_psi)
    assert metrics["kl_divergence"].value == pytest.approx(expected_kl)
    assert column.distribution is not None
    assert column.distribution.bins[-1].bin_end == pytest.approx(1.9)
    assert column.drift_detected is True
    assert report.drifted_columns_count == 1


@pytest.mark.parametrize(
    ("reference_dtype", "current_dtype"),
    [(pl.Int8, pl.Int64), (pl.Int64, pl.Float64), (pl.Float32, pl.Float64)],
)
def test_compatible_numeric_representation_change_is_stable(reference_dtype, current_dtype):
    """A wider numeric representation alone must not create a schema-drift alert."""
    reference = pl.DataFrame({"feature": pl.Series([0, 1] * 50, dtype=reference_dtype)})
    current = pl.DataFrame({"feature": pl.Series([0, 1] * 50, dtype=current_dtype)})
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.column_drifts["feature"].drift_detected is False
    assert report.drifted_columns_count == 0


@pytest.mark.parametrize("values", [["bad", "bad", "bad"], ["0", "1", "bad"]])
def test_incompatible_numeric_text_is_reported_without_dropping_rows(values):
    """A failed or partial parse must produce type drift rather than a stable surviving subset."""
    reference = pl.DataFrame({"feature": [0, 1, 2]})
    current = pl.DataFrame({"feature": values})
    report = DriftCalculator(reference, current).calculate_drift()
    column = report.column_drifts["feature"]
    assert report.current_rows == 3
    assert column.drift_detected is True
    assert [
        (metric.metric, metric.value, metric.threshold, metric.has_drift)
        for metric in column.metrics
    ] == [("type_drift", 1.0, 0.0, True)]
    assert column.distribution is None
    assert "Int64" in column.suggestions[0] and "String" in column.suggestions[0]


@pytest.mark.parametrize(
    ("reference_values", "current_values"),
    [
        ([0, 1], [False, True]),
        ([False, True], [0, 1]),
        (["0", "1"], [0, 1]),
        (["a", "b"], [[1], [2]]),
        ([date(2026, 1, 1), date(2026, 1, 2)], ["2026-01-01", "2026-01-02"]),
    ],
)
def test_incompatible_type_families_are_not_coerced(reference_values, current_values):
    """Lossy or semantically different types must remain visible even without comparable metrics."""
    report = DriftCalculator(
        pl.DataFrame({"feature": reference_values}), pl.DataFrame({"feature": current_values})
    ).calculate_drift(thresholds={"psi": 100, "ks_statistic": 100, "wasserstein": 100})
    assert report.column_drifts["feature"].metrics[0].metric == "type_drift"
    assert report.column_drifts["feature"].drift_detected is True
    assert report.drifted_columns_count == 1


@pytest.mark.parametrize("current_dtype", [pl.Categorical, pl.Enum(["a", "b"])])
def test_compatible_categorical_representations_still_measure_psi(current_dtype):
    """Category storage encodings must not replace a real frequency comparison with type drift."""
    reference = pl.DataFrame({"category": ["a", "b"] * 50})
    current = pl.DataFrame({"category": pl.Series(["a", "b"] * 50, dtype=current_dtype)})
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.column_drifts["category"].metrics[0].metric == "psi_categorical"
    assert report.drifted_columns_count == 0


def test_type_drift_counts_once_alongside_new_and_missing_columns():
    """Type drift must count a shared column once and preserve the existing JSON schema."""
    reference = pl.DataFrame({"typed": [0, 1], "stable": [0, 1], "missing": [0, 1]})
    current = pl.DataFrame({"typed": ["bad", "bad"], "stable": [0.0, 1.0], "new": [0, 1]})
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.missing_columns == ["missing"]
    assert report.new_columns == ["new"]
    assert report.drifted_columns_count == 3
    payload = json.loads(json.dumps(report.model_dump(), allow_nan=False))
    assert set(payload) == {
        "reference_rows",
        "current_rows",
        "drifted_columns_count",
        "column_drifts",
        "missing_columns",
        "new_columns",
    }
    assert payload["column_drifts"]["typed"]["drift_detected"] is True


@pytest.mark.parametrize("current_dtype", [pl.Null, pl.Float64])
def test_empty_current_column_keeps_the_existing_no_measurement_policy(current_dtype):
    """A no-data column must not gain a type alert merely from Polars inferring Null."""
    reference = pl.DataFrame({"feature": [0, 1]})
    current = pl.DataFrame({"feature": pl.Series([None, None], dtype=current_dtype)})
    report = DriftCalculator(reference, current).calculate_drift()
    assert report.column_drifts == {}
    assert report.drifted_columns_count == 0


def test_numeric_text_conversion_preserves_non_null_population():
    """Strict conversion may discard existing nulls but must retain every valid numeric observation."""
    reference = pl.DataFrame({"feature": [None, 0, 1]})
    current = pl.DataFrame({"feature": [None, "0.9", "1.9"]})
    report = DriftCalculator(reference, current).calculate_drift()
    column = report.column_drifts["feature"]
    assert column.distribution is not None
    assert sum(bucket.current_count for bucket in column.distribution.bins) == 2
    assert column.metrics[0].raw_value == pytest.approx(0.9)


@pytest.mark.parametrize(
    "values", [[0.0, float("inf")], [0.0, float("-inf")], ["0", "1e400"], ["0", "-inf"]]
)
def test_infinite_current_values_fail_before_reporting_undefined_distances(values):
    """Preserving current precision must not persist infinite scores or silently lose bad rows."""
    calculator = DriftCalculator(
        pl.DataFrame({"feature": [0, 1]}), pl.DataFrame({"feature": values})
    )
    with pytest.raises(ValueError, match="feature.*infinite.*finite"):
        calculator.calculate_drift()


def test_infinite_reference_values_fail_with_the_affected_column():
    """An invalid reference must receive the same actionable diagnosis as current data."""
    calculator = DriftCalculator(
        pl.DataFrame({"feature": [0.0, float("inf")]}),
        pl.DataFrame({"feature": [0, 1]}),
    )
    with pytest.raises(ValueError, match="feature.*infinite.*finite"):
        calculator.calculate_drift()
