"""Profiling keeps usable Decimal, finite, temporal, and small-scale observations."""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.correlations import calculate_correlations


@pytest.mark.parametrize("row_count", [200, 1200])
def test_decimal_timeseries_keeps_trend_acf_and_stationarity(row_count):
    """Decimal metrics must reach the same temporal diagnostics as their float values."""
    values = np.round(np.random.default_rng(92).normal(size=row_count), 3)
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(row_count)],
            "amount": pl.Series([Decimal(str(value)) for value in values], dtype=pl.Decimal(20, 3)),
        }
    )
    analyzer = EDAAnalyzer(frame)
    temporal = analyzer.analyze().timeseries

    assert temporal is not None
    assert len(temporal.trend) == row_count
    np.testing.assert_allclose([point.values["amount"] for point in temporal.trend], values)
    assert temporal.autocorrelation is not None
    assert len(temporal.autocorrelation) == 30
    assert temporal.stationarity_test is not None
    assert np.isfinite(temporal.stationarity_test["test_statistic"])
    assert analyzer.df.equals(frame)


def test_decimal_features_remain_available_to_causal_discovery():
    """Nominal targets must not leave Decimal features as object arrays for Fisher-Z."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(92)
    values = np.round(rng.normal(size=200), 3)
    frame = pl.DataFrame(
        {
            "amount": pl.Series([Decimal(str(value)) for value in values], dtype=pl.Decimal(20, 3)),
            "signal": pl.Series(
                [
                    Decimal(str(value))
                    for value in np.round(values + rng.normal(scale=0.3, size=200), 3)
                ],
                dtype=pl.Decimal(20, 3),
            ),
            "target": ["a", "b"] * 100,
        }
    )
    analyzer = EDAAnalyzer(frame)
    graph = analyzer.analyze(target_col="target").causal_graph

    assert graph is not None
    assert {node.id for node in graph.nodes} == {"amount", "signal"}
    assert graph.edges
    assert analyzer.df.equals(frame)


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan"), None])
@pytest.mark.parametrize("numeric_target", [False, True])
def test_target_interactions_use_only_finite_observed_pairs(invalid, numeric_target):
    """Invalid observations cannot poison eta, box plots, or ANOVA in either target mode."""
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b", None, "a"],
            "value": [0.0, 2.0, 2.0, 4.0, invalid, 1000.0, None],
        }
    )
    analyzer = EDAAnalyzer(frame)
    target, task = ("value", "Regression") if numeric_target else ("group", "Classification")
    profile = analyzer.analyze(target_col=target, task_type=task)

    if not numeric_target:
        assert profile.target_correlations is not None
        assert profile.target_correlations["value"] == pytest.approx(2**-0.5)
    assert profile.target_interactions
    interaction = profile.target_interactions[0]
    assert {box.name: (box.stats.min, box.stats.max) for box in interaction.data} == {
        "a": (0.0, 2.0),
        "b": (2.0, 4.0),
    }
    assert interaction.p_value == pytest.approx(0.2928932188134525)


@pytest.mark.parametrize("row_count", [200, 1200])
def test_temporal_statistics_ignore_infinity_without_shifting_other_metrics(row_count):
    """Both raw and resampled trends retain aligned metrics while seasonal means stay finite."""
    values = np.random.default_rng(92).normal(size=row_count)
    values[5] = np.inf
    values[6] = -np.inf
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(row_count)],
            "value": values,
            "other": np.arange(row_count, dtype=float) + 10,
        }
    )
    analyzer = EDAAnalyzer(frame)
    temporal = analyzer.analyze().timeseries

    assert temporal is not None
    assert len(temporal.trend) == row_count
    assert temporal.trend[5].values == {"other": 15.0}
    assert temporal.trend[6].values == {"other": 16.0}
    for group in temporal.seasonality.month_of_year:
        expected = [
            value
            for date, value in zip(frame["date"], values, strict=True)
            if date.strftime("%b") == group["month"] and np.isfinite(value)
        ]
        assert group["count"] == pytest.approx(np.mean(expected))
    assert analyzer.df.equals(frame)


@pytest.mark.parametrize("task_type", ["Classification", "Regression"])
@pytest.mark.parametrize("invalid_column", ["feature", "target"])
def test_rule_tree_handles_infinities_like_existing_missing_values(task_type, invalid_column):
    """Non-finite numeric cells must not suppress an otherwise trainable surrogate tree."""
    frame = pl.DataFrame({"feature": np.arange(80.0), "target": [0.0] * 40 + [1.0] * 40})
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 10)
        .then(float("inf"))
        .otherwise(pl.col(invalid_column))
        .alias(invalid_column)
    )
    analyzer = EDAAnalyzer(frame)
    tree = analyzer.analyze(target_col="target", task_type=task_type).rule_tree

    assert tree is not None
    assert tree.nodes[0].samples == 80
    assert tree.accuracy is not None
    assert np.isfinite(tree.accuracy)
    assert tree.accuracy > 0.95
    assert analyzer.df.equals(frame)


def test_duration_does_not_compete_with_dates_or_break_decomposition():
    """Elapsed times remain unsupported measurements instead of overriding a valid date axis."""
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1) + timedelta(days=i // 2) for i in range(40)],
            "elapsed": [timedelta(seconds=i) for i in range(40)],
            "value": np.random.default_rng(92).normal(size=40),
        }
    )
    analyzer = EDAAnalyzer(frame)
    profile = analyzer.analyze()

    assert profile.columns["elapsed"].dtype == "Unknown"
    assert profile.timeseries is not None and profile.timeseries.date_col == "date"
    assert len(analyzer.get_decomposition_split(None, "count", "date", [])) == 20
    assert analyzer.analyze(date_col="elapsed").timeseries is None
    assert analyzer.df.equals(frame)


def test_duration_decomposition_returns_no_unsupported_split():
    """Requesting an elapsed-time split must not crash by casting Duration to text."""
    frame = pl.DataFrame({"elapsed": [timedelta(seconds=1), timedelta(seconds=2)]})

    assert EDAAnalyzer(frame).get_decomposition_split(None, "count", "elapsed", []) == []


@pytest.mark.parametrize(
    "scale,offset",
    [
        (1.0, 0.0),
        (1e-10, 0.0),
        (1e-100, 0.0),
        (1e100, 0.0),
        (1e-200, 0.0),
        (1e200, 0.0),
        (1.0, 1e15),
    ],
)
def test_correlation_retains_nonconstant_columns_independent_of_units(scale, offset):
    """Scaling valid measurements cannot erase their Pearson correlation matrix."""
    x = np.array([1.0, 3.0, 2.0, 5.0]) * scale + offset
    y = np.array([2.0, 5.0, 3.0, 9.0]) * scale + offset
    frame = pl.DataFrame({"x": x, "y": y, "constant": [scale] * 4, "empty": [None] * 4})
    result = calculate_correlations(frame.lazy(), frame.columns)

    assert result is not None
    assert result.columns == ["x", "y"]
    assert result.values[0][1] == pytest.approx(15.75 / (8.75 * 28.75) ** 0.5)


@pytest.mark.parametrize("scale", [1.0, 1e-100])
def test_correlation_normalization_uses_each_pairs_observed_overlap(scale):
    """An outlier missing from a pair cannot erase variation among its shared observations."""
    frame = pl.DataFrame(
        {
            "x": [scale, 2 * scale, 3 * scale, 4 * scale, 1e200],
            "y": [2 * scale, 4 * scale, 6 * scale, 8 * scale, None],
        }
    )
    matrix = calculate_correlations(frame.lazy(), frame.columns)

    assert matrix is not None
    assert matrix.values == [[1.0, 1.0], [1.0, 1.0]]
