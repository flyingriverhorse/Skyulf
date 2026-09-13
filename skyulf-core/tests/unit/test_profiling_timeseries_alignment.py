"""Time-series plotting preserves each metric's timestamp and missingness."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import TimeSeriesPoint
from skyulf.profiling.visualizer import EDAVisualizer


def test_public_plot_aligns_partially_observed_native_profile(monkeypatch):
    """Independent metric missingness must not crash public plotting or shift observations."""
    import matplotlib.pyplot as plt

    start = datetime(2026, 1, 1)
    frame = pl.DataFrame(
        {
            "date": [start + timedelta(seconds=index) for index in range(1000)],
            "complete": np.arange(1000, dtype=float),
            "sparse": [float(index) if index % 2 == 0 else None for index in range(1000)],
            "late": [float(index) if index % 2 else None for index in range(1000)],
        }
    )
    profile = EDAAnalyzer(frame).analyze(date_col="date")
    assert profile.timeseries is not None and len(profile.timeseries.trend) == 1000
    captured = {}

    def capture_plots():
        """Inspect the real plot before the public visualizer closes its figures."""
        for number in plt.get_fignums():
            for axis in plt.figure(number).axes:
                if axis.get_title().startswith("Time Series Trend"):
                    captured.update(
                        {
                            line.get_label(): (line.get_xdata(), line.get_ydata())
                            for line in axis.lines
                        }
                    )

    monkeypatch.setattr(plt, "show", capture_plots)
    EDAVisualizer(profile).plot()

    assert set(captured) == {"complete", "sparse", "late"}
    for dates, values in captured.values():
        assert len(dates) == len(values) == 1000
        assert dates[0] == start and dates[-1] == start + timedelta(seconds=999)
    np.testing.assert_array_equal(captured["sparse"][1][::2], np.arange(0, 1000, 2))
    assert np.isnan(captured["sparse"][1][1::2]).all()
    assert np.isnan(captured["late"][1][::2]).all()
    np.testing.assert_array_equal(captured["late"][1][1::2], np.arange(1, 1000, 2))


def test_bad_timestamps_do_not_shift_metric_values():
    """Discarding an invalid date must discard its values from every plotted metric."""
    dates, values = EDAVisualizer._timeseries_series(
        [
            TimeSeriesPoint(date="bad", values={"a": 99}),
            TimeSeriesPoint(date="2026-01-01", values={"a": 1}),
            TimeSeriesPoint(date="2026-01-02", values={"b": 2}),
        ]
    )
    assert dates == [datetime(2026, 1, 1), datetime(2026, 1, 2)]
    assert values["a"][0] == 1 and np.isnan(values["a"][1])
    assert np.isnan(values["b"][0]) and values["b"][1] == 2
