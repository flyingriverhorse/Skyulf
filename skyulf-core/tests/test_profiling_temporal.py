"""Tests for skyulf.profiling._analyzer.temporal.TemporalMixin."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer


def _small_ts_df(n: int = 50) -> pl.DataFrame:
    """Small (<1000-row) time series dataset with a numeric metric to track."""
    rng = np.random.default_rng(41)
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    value = rng.normal(10, 2, n)
    return pl.DataFrame({"date": dates, "value": value})


def _large_ts_df(n: int = 1200, span_days: int = 400) -> pl.DataFrame:
    """Large (>=1000-row) time series dataset spanning `span_days` days (→ '1d'/'1w' interval)."""
    rng = np.random.default_rng(42)
    base = datetime(2020, 1, 1)
    # Spread rows evenly across the span so group_by_dynamic has real buckets.
    offsets = np.linspace(0, span_days * 86400, n)
    dates = [base + timedelta(seconds=float(s)) for s in offsets]
    value = rng.normal(10, 2, n)
    return pl.DataFrame({"date": dates, "value": value})


def test_analyze_timeseries_small_dataset_happy_path() -> None:
    """Small dataset: raw trend points, seasonality, ACF, and (if available) ADF results."""
    analyzer = EDAAnalyzer(_small_ts_df())
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert result.date_col == "date"
    assert len(result.trend) > 0
    assert result.seasonality.day_of_week
    assert result.seasonality.month_of_year
    assert result.autocorrelation is not None
    assert len(result.autocorrelation) > 0


def test_analyze_timeseries_auto_detects_date_column() -> None:
    """No date_col given: highest-cardinality DateTime column should be auto-selected."""
    df = _small_ts_df()
    # Add a low-cardinality datetime column that should lose to 'date'.
    df = df.with_columns(pl.lit(datetime(2022, 1, 1)).alias("ingested_at"))
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"])
    assert result is not None
    assert result.date_col == "date"


def test_analyze_timeseries_no_date_columns_returns_none() -> None:
    """A dataset with no DateTime columns should return None."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    analyzer = EDAAnalyzer(df)
    assert analyzer._analyze_timeseries(["a"]) is None


def test_analyze_timeseries_explicit_date_col_not_in_columns_returns_none() -> None:
    """An explicit date_col that doesn't exist in the frame should return None (line 43)."""
    analyzer = EDAAnalyzer(_small_ts_df())
    assert analyzer._analyze_timeseries(["value"], date_col="does_not_exist") is None


def test_analyze_timeseries_small_dataset_skips_null_rows() -> None:
    """Rows with a null date or with all-null tracked values should be skipped (lines 64, 67)."""
    n = 30
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    # Introduce a handful of nulls in the tracked value; drop_nulls() removes those rows
    # upstream, so all remaining rows have non-null values — this still exercises the loop.
    value = [None if i % 6 == 0 else float(i) for i in range(n)]
    df = pl.DataFrame({"date": dates, "value": value})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert len(result.trend) == sum(1 for v in value if v is not None)


def test_analyze_timeseries_appends_target_col_when_not_in_first_three() -> None:
    """target_col present in numeric_cols but outside the first 3 should be appended (line 53)."""
    n = 60
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(47)
    df = pl.DataFrame(
        {
            "date": dates,
            "n1": rng.normal(0, 1, n),
            "n2": rng.normal(0, 1, n),
            "n3": rng.normal(0, 1, n),
            "n4": rng.normal(0, 1, n),
        }
    )
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(
        ["n1", "n2", "n3", "n4"], target_col="n4", date_col="date"
    )
    assert result is not None
    assert any("n4" in p.values for p in result.trend)


def test_analyze_timeseries_small_dataset_skips_null_rows_injected(monkeypatch) -> None:
    """A null date row and an all-null-values row should be skipped (lines 64, 67)."""
    import skyulf.profiling._analyzer.temporal as temporal_mod

    orig_collect = temporal_mod._collect
    state = {"n": 0}

    def _patched(lf):
        df = orig_collect(lf)
        state["n"] += 1
        if state["n"] == 1:
            # Inject a null-date row and an all-null-values row directly into trend_df.
            extra = pl.DataFrame(
                {"date": [None, datetime(2022, 6, 1)], "value": [1.0, None]},
                schema=df.schema,
            )
            df = pl.concat([df, extra])
        return df

    monkeypatch.setattr(temporal_mod, "_collect", _patched)
    analyzer = EDAAnalyzer(_small_ts_df(n=30))
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    # Neither injected row (null date, all-null values) should have produced a trend point.
    assert all(p.date != datetime(2022, 6, 1).isoformat() for p in result.trend)


def test_analyze_timeseries_large_dataset_skips_null_rows_injected(monkeypatch) -> None:
    """A null date row and an all-null-values row should be skipped (lines 101, 104).

    We only want to inject the bogus rows into the *trend* aggregation
    (the ``group_by_dynamic`` result over ``date``/``value``), not into any of
    the other ``_collect`` calls that ``EDAAnalyzer`` performs (e.g. during
    construction). Call order isn't guaranteed to be stable across test runs
    (it can shift depending on coverage instrumentation or fixture ordering),
    so we match on the resulting schema instead of a call counter.
    """
    import skyulf.profiling._analyzer.temporal as temporal_mod

    orig_collect = temporal_mod._collect
    state = {"injected": False}

    def _patched(lf):
        df = orig_collect(lf)
        if not state["injected"] and list(df.columns) == ["date", "value"]:
            state["injected"] = True
            extra = pl.DataFrame(
                {"date": [None, datetime(2020, 6, 1, 3, 33, 33)], "value": [1.0, None]},
                schema=df.schema,
            )
            df = pl.concat([df, extra])
        return df

    monkeypatch.setattr(temporal_mod, "_collect", _patched)
    analyzer = EDAAnalyzer(_large_ts_df(n=1200, span_days=400))
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert state["injected"]
    assert all(p.date != datetime(2020, 6, 1, 3, 33, 33).isoformat() for p in result.trend)


def test_analyze_timeseries_large_dataset_daily_interval() -> None:
    """A >=1000-row dataset spanning a moderate range should pick the '1d' interval."""
    analyzer = EDAAnalyzer(_large_ts_df(n=1200, span_days=400))
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert len(result.trend) > 0


def test_analyze_timeseries_large_dataset_weekly_interval() -> None:
    """A very long date range (>=604800s ideal) should pick the '1w' interval (line 82-84)."""
    analyzer = EDAAnalyzer(_large_ts_df(n=1200, span_days=365 * 20))
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert len(result.trend) > 0


def test_analyze_timeseries_large_dataset_second_interval() -> None:
    """A very short date range should trigger the '1s' adaptive interval (line 76)."""
    analyzer = EDAAnalyzer(_large_ts_df(n=1200, span_days=0))
    # Force a sub-minute total span so ideal_seconds < 60.
    base = datetime(2020, 1, 1)
    rng = np.random.default_rng(43)
    dates = [base + timedelta(seconds=float(s)) for s in np.linspace(0, 50, 1200)]
    df = pl.DataFrame({"date": dates, "value": rng.normal(10, 2, 1200)})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None


def test_analyze_timeseries_large_dataset_minute_interval() -> None:
    """A span of ~1.16 days (ideal_seconds in [60, 3600)) should trigger '1m' (line 78)."""
    base = datetime(2020, 1, 1)
    rng = np.random.default_rng(44)
    dates = [base + timedelta(seconds=float(s)) for s in np.linspace(0, 100000, 1200)]
    df = pl.DataFrame({"date": dates, "value": rng.normal(10, 2, 1200)})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None


def test_analyze_timeseries_large_dataset_hour_interval() -> None:
    """A span of ~11.6 days (ideal_seconds in [3600, 86400)) should trigger '1h' (line 80)."""
    base = datetime(2020, 1, 1)
    rng = np.random.default_rng(45)
    dates = [base + timedelta(seconds=float(s)) for s in np.linspace(0, 1_000_000, 1200)]
    df = pl.DataFrame({"date": dates, "value": rng.normal(10, 2, 1200)})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None


def test_analyze_timeseries_large_dataset_no_tracked_columns() -> None:
    """Empty numeric_cols on a large dataset should use pl.count() aggregation (line 87)."""
    analyzer = EDAAnalyzer(_large_ts_df(n=1200, span_days=400))
    result = analyzer._analyze_timeseries([], date_col="date")
    assert result is not None
    assert len(result.trend) > 0
    # No tracked metric means each trend point only carries a "count" value.
    assert all("count" in p.values for p in result.trend)


def test_analyze_timeseries_acf_fills_nan_values() -> None:
    """NaN values in the tracked series should be filled via np.nanmean before ACF (line 152)."""
    n = 60
    base = datetime(2022, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(46)
    value = rng.normal(10, 2, n)
    value[5] = np.nan
    value[10] = np.nan
    df = pl.DataFrame({"date": dates, "value": value.tolist()})
    analyzer = EDAAnalyzer(df)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert result.autocorrelation is not None
    assert len(result.autocorrelation) > 0


def test_analyze_timeseries_adfuller_exception_is_caught(monkeypatch) -> None:
    """adfuller raising should be caught, leaving stationarity_test as None (lines 183-184)."""
    import statsmodels.tsa.stattools as stattools

    def _boom(*args, **kwargs):
        raise RuntimeError("adf boom")

    monkeypatch.setattr(stattools, "adfuller", _boom)
    analyzer = EDAAnalyzer(_small_ts_df(n=30))
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is not None
    assert result.stationarity_test is None


def test_analyze_timeseries_outer_exception_returns_none(monkeypatch) -> None:
    """An unexpected internal error should be caught by the outer except (lines 193-195)."""
    analyzer = EDAAnalyzer(_small_ts_df())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer.lazy_df, "sort", _boom)
    result = analyzer._analyze_timeseries(["value"], date_col="date")
    assert result is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has a real ``signup_date`` string column plus missing ``age``/
    ``income`` rows — closer to production data than the small synthetic
    frames used elsewhere in this file.
    """

    def test_analyze_timeseries_on_signup_date_handles_missing_values(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        # signup_date is auto-cast from a string column by EDAAnalyzer.__init__.
        result = analyzer._analyze_timeseries(["age", "income"], date_col="signup_date")

        assert result is not None
        assert result.date_col == "signup_date"
        assert len(result.trend) > 0
