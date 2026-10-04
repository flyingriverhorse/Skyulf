"""Temporal profiles exclude unusable dates and report only meaningful correlations."""

import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize("row_count", [999, 1000])
@pytest.mark.parametrize("date_kind", ["polars_datetime", "polars_date"])
@pytest.mark.parametrize("missing_dates", ["none", "one", "all"])
def test_public_temporal_profile_excludes_missing_dates_across_resampling_boundary(
    row_count, date_kind, missing_dates
):
    """One unusable timestamp must not remove otherwise valid temporal analysis."""
    base = datetime(2024, 1, 1)
    step = timedelta(days=1) if date_kind == "polars_date" else timedelta(hours=1)
    dates: list[Any] = [base + i * step for i in range(row_count)]
    if date_kind == "polars_date":
        dates = [value.date() for value in dates]
    values = np.arange(row_count, dtype=float)
    if missing_dates == "one":
        dates[-1] = None
        values[-1] = 999999.0
    elif missing_dates == "all":
        dates = [None] * row_count
    date_dtype = pl.Date if date_kind == "polars_date" else pl.Datetime
    native = pl.DataFrame({"date": pl.Series("date", dates, dtype=date_dtype), "value": values})
    profile = EDAAnalyzer(native).analyze(date_col="date")

    assert profile.row_count == row_count
    temporal = profile.timeseries
    assert temporal is not None
    assert temporal.date_col == "date"
    kept = row_count if missing_dates == "none" else row_count - 1
    if missing_dates == "all":
        assert temporal.trend == []
        assert temporal.autocorrelation == []
        assert temporal.stationarity_test is None
        assert temporal.seasonality.day_of_week == []
        assert temporal.seasonality.month_of_year == []
    else:
        if date_kind == "polars_date" and row_count >= 1000:
            expected_values = [
                sum(range(start, min(start + 7, kept))) / min(7, kept - start)
                for start in range(0, kept, 7)
            ]
        else:
            expected_values = list(range(kept))
        np.testing.assert_allclose(
            [point.values["value"] for point in temporal.trend], expected_values
        )
        assert all(row["day"] is not None for row in temporal.seasonality.day_of_week)
        assert all(row["month"] is not None for row in temporal.seasonality.month_of_year)
    assert native["date"].null_count() == {"none": 0, "one": 1, "all": row_count}[missing_dates]


@pytest.mark.parametrize("row_count", [999, 1000])
@pytest.mark.parametrize("pattern", ["null", "nan", "constant", "single", "sparse"])
def test_public_temporal_profile_omits_acf_without_meaningful_observations(row_count, pattern):
    """Empty, constant, or insufficiently observed metrics must not fabricate ACF lags."""
    base = datetime(2024, 1, 1)
    values: list[Any] = [None] * row_count
    if pattern == "nan":
        values = [np.nan] * row_count
    elif pattern == "constant":
        values = [3.0] * row_count
    elif pattern == "single":
        values[row_count // 2] = 5.0
    elif pattern == "sparse":
        for index, value in zip(range(0, 960, 120), range(8), strict=True):
            values[index] = float(value)
    data = pl.DataFrame(
        {
            "date": [base + timedelta(hours=i) for i in range(row_count)],
            "value": pl.Series("value", values, dtype=pl.Float64),
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        profile = EDAAnalyzer(data).analyze(date_col="date")

    temporal = profile.timeseries
    assert temporal is not None
    assert temporal.autocorrelation == []
    assert temporal.stationarity_test is None
    assert not any(
        "temporal.py" in warning.filename and "Mean of empty slice" in str(warning.message)
        for warning in captured
    )


@pytest.mark.parametrize("row_count", [999, 1000])
@pytest.mark.parametrize("sparse", [False, True])
def test_public_temporal_profile_keeps_finite_acf_for_observed_variable_metrics(row_count, sparse):
    """Sufficient finite observations retain meaningful correlation and stationarity output."""
    base = datetime(2024, 1, 1)
    values: list[Any] = np.random.default_rng(71).normal(size=row_count).tolist()
    if sparse:
        values = [value if index % 2 else None for index, value in enumerate(values)]
    data = pl.DataFrame(
        {
            "date": [base + timedelta(hours=i) for i in range(row_count)],
            "value": pl.Series("value", values, dtype=pl.Float64),
        }
    )

    temporal = EDAAnalyzer(data).analyze(date_col="date").timeseries

    assert temporal is not None
    assert temporal.autocorrelation is not None
    assert [entry["lag"] for entry in temporal.autocorrelation] == list(range(1, 31))
    assert all(np.isfinite(entry["corr"]) for entry in temporal.autocorrelation)
    assert any(entry["corr"] != 0 for entry in temporal.autocorrelation)
    if temporal.stationarity_test is not None:
        assert np.isfinite(temporal.stationarity_test["test_statistic"])
        assert np.isfinite(temporal.stationarity_test["p_value"])


@pytest.mark.parametrize("scale", [1e200, 1e307])
def test_temporal_diagnostics_omit_finite_values_when_statistics_overflow(scale):
    """Finite input must not turn overflowed variance into fabricated correlations or ADF output."""
    base = datetime(2024, 1, 1)
    data = pl.DataFrame(
        {
            "date": [base + timedelta(hours=i) for i in range(1000)],
            "value": [scale, scale / 2] * 500,
        }
    )

    temporal = EDAAnalyzer(data).analyze(date_col="date").timeseries

    assert temporal is not None
    assert temporal.autocorrelation == []
    assert temporal.stationarity_test is None


def test_public_temporal_profile_omits_nonfinite_adf_result_for_finite_variance():
    """A numerically unstable ADF result must not publish an infinite test statistic."""
    base = datetime(2024, 1, 1)
    data = pl.DataFrame(
        {
            "date": [base + timedelta(hours=i) for i in range(1000)],
            "value": np.arange(1000, dtype=float) * 1e-140,
        }
    )

    temporal = EDAAnalyzer(data).analyze(date_col="date").timeseries

    assert temporal is not None
    assert temporal.autocorrelation
    assert all(np.isfinite(entry["corr"]) for entry in temporal.autocorrelation)
    assert temporal.stationarity_test is None
