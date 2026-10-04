"""Seasonality metadata must describe the values stored under the legacy count key."""

from datetime import datetime

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import SeasonalityStats


@pytest.mark.parametrize("numeric_cols", [["sales"], ["visits", "sales"], []])
def test_seasonality_records_the_actual_measure(numeric_cols: list[str]) -> None:
    """A selected target must not cause the UI to mislabel another tracked metric."""
    analyzer = EDAAnalyzer(
        pl.DataFrame(
            {
                "date": [datetime(2026, 1, 5), datetime(2026, 1, 12)],
                "sales": [10.0, 30.0],
                "visits": [2.0, 6.0],
            }
        )
    )
    result = analyzer._analyze_timeseries(numeric_cols, target_col="sales", date_col="date")
    assert result is not None
    saved = result.model_dump(mode="json")["seasonality"]
    metric = numeric_cols[0] if numeric_cols else None
    expected = {"sales": 20.0, "visits": 4.0, None: 2}[metric]
    assert saved["metric"] == metric
    assert saved["aggregation"] == ("mean" if metric else "count")
    assert saved["day_of_week"] == [{"day": "Mon", "count": expected}]
    assert saved["month_of_year"] == [{"month": "Jan", "count": expected}]


def test_legacy_seasonality_does_not_invent_measure_metadata() -> None:
    """Old count fields can contain means, so absence must remain unknown."""
    result = SeasonalityStats(day_of_week=[{"day": "Mon", "count": 20}], month_of_year=[])
    assert result.aggregation is None
    assert result.metric is None
    assert result.day_of_week[0]["count"] == 20
