"""Native non-text dtypes must retain a usable profile without string aggregation."""

from datetime import time
from decimal import Decimal

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("value", [Decimal("12.34"), None, Decimal("56.78")]),
        pl.Series("value", [time(9, 30), None, time(16, 45)]),
        pl.Series("value", [[1, 2], None, [3]]),
    ],
    ids=["decimal", "time", "list"],
)
def test_unsupported_native_dtype_retains_data_and_reports_unavailable_stats(series):
    """One unsupported statistics dtype must not abort or mislabel the entire report."""
    frame = pl.DataFrame({"value": series, "signal": [1.0, 2.0, 6.0]})
    analyzer = EDAAnalyzer(frame)

    profile = analyzer.analyze()

    column = profile.columns["value"]
    assert column.dtype == "Unknown"
    assert column.missing_count == 1
    assert column.missing_percentage == pytest.approx(100 / 3)
    assert column.text_stats is None and column.numeric_stats is None
    assert column.histogram is None
    numeric = profile.columns["signal"].numeric_stats
    assert numeric is not None and numeric.mean == 3.0
    assert profile.sample_data == frame.to_dicts()
    assert analyzer.df.equals(frame)
    unavailable = [alert for alert in profile.alerts if alert.type == "Unsupported Type"]
    assert len(unavailable) == 1 and unavailable[0].column == "value"
    assert str(series.dtype) in unavailable[0].message
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert restored.columns["value"].missing_count == 1
    assert restored.alerts == profile.alerts
