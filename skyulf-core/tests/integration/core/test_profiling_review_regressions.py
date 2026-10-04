"""Public profiling regressions for constants, plots, partial trends and rule labels."""

from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile, RuleNode, RuleTree
from skyulf.profiling.visualizer import EDAVisualizer


@pytest.mark.parametrize(
    "values,semantic",
    [
        pytest.param([7] * 100, "Categorical", id="integer"),
        pytest.param(["fixed"] * 100, "Categorical", id="string"),
        pytest.param([True] * 100, "Boolean", id="boolean"),
        pytest.param([7.0] * 100, "Numeric", id="float-control"),
    ],
)
def test_d7_7_constant_flag_does_not_depend_on_numeric_dispatch(values, semantic):
    """Semantically categorical constants still carry no information for modeling."""
    profile = EDAAnalyzer(pl.DataFrame({"constant": values})).analyze()

    assert profile.columns["constant"].dtype == semantic
    assert profile.columns["constant"].is_constant is True
    assert any(alert.type == "Constant" and alert.column == "constant" for alert in profile.alerts)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([7, 8] * 50, id="integer"),
        pytest.param(["left", "right"] * 50, id="string"),
        pytest.param([True, False] * 50, id="boolean"),
        pytest.param([7.0, 8.0] * 50, id="float"),
    ],
)
def test_d7_7_varying_column_control_is_not_marked_constant(values):
    """A shared constant check must not turn ordinary categorical features into constants."""
    profile = EDAAnalyzer(pl.DataFrame({"variable": values})).analyze()

    assert profile.columns["variable"].is_constant is False
    assert not any(alert.type == "Constant" for alert in profile.alerts)


@pytest.mark.parametrize(
    "dtype,value",
    [
        pytest.param(pl.Int64, 7, id="integer"),
        pytest.param(pl.Float64, 7.0, id="float"),
        pytest.param(pl.String, "fixed", id="string"),
        pytest.param(pl.Boolean, True, id="boolean"),
        pytest.param(pl.Categorical, "fixed", id="categorical"),
        pytest.param(pl.Enum(["fixed"]), "fixed", id="enum"),
    ],
)
@pytest.mark.parametrize("observed_count", [0, 1, 2])
def test_d7_7_constant_flag_requires_repeated_observations(dtype, value, observed_count):
    """Missing values are excluded without treating an absent or lone observation as constant."""
    values = [value] * observed_count + [None] * (100 - observed_count)
    frame = pl.DataFrame({"measurement": pl.Series(values, dtype=dtype)})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.columns["measurement"].missing_count == 100 - observed_count
    assert profile.columns["measurement"].is_constant is (observed_count > 1)
    assert any(alert.type == "Constant" for alert in profile.alerts) is (observed_count > 1)


def _empty_profile(row_count=8, **updates):
    """Build the smallest real profile needed to exercise visualizer behavior."""
    return DatasetProfile(
        row_count=row_count,
        column_count=2,
        duplicate_rows=0,
        missing_cells_percentage=0.0,
        memory_usage_mb=0.0,
        columns={},
        **updates,
    )


@pytest.fixture
def caller_figures(monkeypatch):
    """Protect preexisting figures and remove only figures introduced by this test."""
    plt = pytest.importorskip("matplotlib.pyplot")
    existing = set(plt.get_fignums())
    sentinel = plt.figure()
    sentinel.add_subplot().plot([0, 1], [0, 1])
    retained = set(plt.get_fignums())
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        yield plt, retained
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)


@pytest.mark.parametrize(
    "first,second",
    [
        pytest.param([1.0] * 8, np.arange(8.0) ** 2, id="constant-kde"),
        pytest.param(np.arange(8.0), np.arange(8.0) ** 2, id="variable-control"),
        pytest.param([1.0], [2.0], id="single-observation"),
        pytest.param([0.0, 1.0, 1.0, 1.0], [None, 1.0, 4.0, 9.0], id="constant-after-dropna"),
    ],
)
def test_d7_8_scatter_plot_accepts_constant_measurements_without_leaking_figures(
    caller_figures, monkeypatch, first, second
):
    """A valid constant measurement must not make plotting crash or orphan owned figures."""
    plt, retained = caller_figures
    frame = pl.DataFrame({"first": first, "second": second})
    shown_axes = []

    def record_shown_figures():
        """Inspect rendered axes before the visualizer releases its figures."""
        shown_axes.extend(
            len(plt.figure(number).axes) for number in set(plt.get_fignums()) - retained
        )

    monkeypatch.setattr(plt, "show", record_shown_figures)

    EDAVisualizer(_empty_profile(row_count=len(frame)), frame).plot()

    assert any(count >= 4 for count in shown_axes)
    assert set(plt.get_fignums()) == retained


def test_d7_8_renderer_failure_closes_only_call_owned_figures(caller_figures, monkeypatch):
    """An unexpected renderer exception must leave the caller's figure registry unchanged."""
    plt, retained = caller_figures
    visualizer = EDAVisualizer(_empty_profile())

    def failed_renderer():
        """Create a real owned figure before simulating a rendering failure."""
        plt.figure()
        raise RuntimeError("simulated renderer failure")

    monkeypatch.setattr(visualizer, "_plot_scatter_matrix", failed_renderer)
    with pytest.raises(RuntimeError, match="simulated renderer failure"):
        visualizer.plot()

    assert set(plt.get_fignums()) == retained


@pytest.mark.parametrize("partial", [True, False], ids=["alternating-nulls", "complete-control"])
def test_d7_10_raw_trend_keeps_each_dates_observed_measurements(partial):
    """One metric's missing value must not erase another metric observed on the same date."""
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    first = [float(i) if not partial or i % 2 == 0 else None for i in range(6)]
    second = [float(10 + i) if not partial or i % 2 else None for i in range(6)]
    frame = pl.DataFrame(
        {
            "date": dates + [None, dates[-1] + timedelta(days=1)],
            "first": first + [999.0, None],
            "second": second + [999.0, None],
        }
    )

    temporal = EDAAnalyzer(frame).analyze(date_col="date").timeseries

    assert temporal is not None
    assert [point.date for point in temporal.trend] == [date.isoformat() for date in dates]
    expected = [
        {key: value for key, value in {"first": a, "second": b}.items() if value is not None}
        for a, b in zip(first, second, strict=True)
    ]
    assert [point.values for point in temporal.trend] == expected


@pytest.mark.parametrize(
    "task_type,class_name,expected_metric",
    [
        pytest.param("Classification", "1", "Accuracy: 75.0%", id="numeric-class"),
        pytest.param("Classification", "approved", "Accuracy: 75.0%", id="text-class-control"),
        pytest.param("Regression", "1.25", "R²: 0.75", id="regression-control"),
        pytest.param(None, "1.25", "R²: 0.75", id="legacy-regression-control"),
        pytest.param(None, "approved", "Accuracy: 75.0%", id="legacy-classification-control"),
    ],
)
def test_d7_12_explicit_task_controls_rule_tree_display(
    task_type, class_name, expected_metric, monkeypatch
):
    """Numeric class labels must not turn an explicitly classified tree into a regression report."""
    rich_console = pytest.importorskip("rich.console")
    stream = StringIO()
    console = rich_console.Console(file=stream, force_jupyter=False, color_system=None, width=120)
    monkeypatch.setattr(rich_console, "Console", lambda: console)
    tree = RuleTree(
        nodes=[
            RuleNode(
                id=0, impurity=0.0, samples=4, value=[1.0, 3.0], class_name=class_name, is_leaf=True
            )
        ],
        accuracy=0.75,
    )
    profile = _empty_profile(task_type=task_type, target_col="label", rule_tree=tree)

    EDAVisualizer(profile).summary()
    output = stream.getvalue()

    assert f"Decision Tree Rules ({expected_metric})" in output
    if expected_metric.startswith("Accuracy"):
        assert "Value =" not in output
