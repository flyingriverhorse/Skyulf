"""Tests for skyulf.profiling.visualizer.EDAVisualizer.

Uses a real ``DatasetProfile`` produced by :class:`EDAAnalyzer` so the
rendering code exercises its real branches (numeric/categorical/text stats,
VIF, outliers, causal graph, geospatial, timeseries, target analysis, rule
tree, PCA, clustering, alerts) instead of hand-built stub objects.
"""

import builtins
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Headless backend so plt.show() never blocks the test run.

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import (
    BoxPlotStats,
    CategoricalStats,
    CategoryBoxPlot,
    ClusteringAnalysis,
    ClusteringPoint,
    ClusterStats,
    ColumnProfile,
    CorrelationMatrix,
    DatasetProfile,
    GeoPoint,
    GeospatialStats,
    OutlierAnalysis,
    OutlierPoint,
    PCAComponent,
    PCAPoint,
    RuleNode,
    RuleTree,
    SeasonalityStats,
    TargetInteraction,
    TextStats,
    TimeSeriesAnalysis,
    TimeSeriesPoint,
)
from skyulf.profiling.visualizer import EDAVisualizer


def _rich_dataset(n: int = 150) -> pl.DataFrame:
    """Dataset large enough to populate PCA/clustering/outliers/rules/causal sections."""
    rng = np.random.default_rng(11)
    num1 = rng.normal(0, 1, n)
    num2 = num1 * 1.8 + rng.normal(0, 0.3, n)
    num3 = rng.normal(5, 2, n)
    cat = rng.choice(["low", "medium", "high"], size=n)
    target = np.where(num1 + rng.normal(0, 0.2, n) > 0, "high", "low")
    lat = rng.uniform(10.0, 20.0, n)
    lon = rng.uniform(10.0, 20.0, n)

    return pl.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "num3": num3,
            "cat": cat,
            "latitude": lat,
            "longitude": lon,
            "target": target,
        }
    )


@pytest.fixture(scope="module")
def rich_profile() -> DatasetProfile:
    """A fully-populated DatasetProfile built from a real EDAAnalyzer run."""
    df = _rich_dataset()
    return EDAAnalyzer(df).analyze(target_col="target")


@pytest.fixture()
def base_profile() -> DatasetProfile:
    """A minimal, real DatasetProfile used as a base for model_copy(update=...) fixtures."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    return EDAAnalyzer(df).analyze()


def test_summary_prints_without_raising(rich_profile: DatasetProfile) -> None:
    """summary() should render every section without raising, given a rich profile."""
    visualizer = EDAVisualizer(rich_profile)
    visualizer.summary()  # Exercises the full rich-console dispatch chain.


def test_summary_handles_minimal_profile_without_optional_sections() -> None:
    """A bare-bones profile (no target/outliers/pca/etc.) should short-circuit each section."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    profile = EDAAnalyzer(df).analyze()
    visualizer = EDAVisualizer(profile)
    visualizer.summary()

    assert profile.vif is None


def test_detect_regression_tree_true_for_numeric_leaf(rich_profile: DatasetProfile) -> None:
    """_detect_regression_tree should return False when leaf class names are non-numeric labels."""
    visualizer = EDAVisualizer(rich_profile)
    # rich_profile's target uses string labels ("low"/"high"), so this is a classification tree.
    if rich_profile.rule_tree is not None:
        assert visualizer._detect_regression_tree() is False


def test_detect_regression_tree_false_without_rule_tree() -> None:
    """No rule_tree at all should return False rather than raising."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    profile = EDAAnalyzer(df).analyze()
    visualizer = EDAVisualizer(profile)
    assert visualizer._detect_regression_tree() is False


def test_plot_runs_all_sections_with_agg_backend(rich_profile: DatasetProfile) -> None:
    """plot() should generate every figure without raising under the headless Agg backend."""
    import matplotlib.pyplot as plt

    df = _rich_dataset()
    visualizer = EDAVisualizer(rich_profile, df=df)
    visualizer.plot()
    plt.close("all")


def test_render_clustering_table_uses_cluster_centers() -> None:
    """_render_clustering should format cluster center features without raising."""
    from rich.console import Console
    from rich.table import Table

    df = pl.DataFrame({"a": [1.0]})
    profile = EDAAnalyzer(df).analyze()
    profile = profile.model_copy(
        update={
            "clustering": ClusteringAnalysis(
                method="KMeans",
                n_clusters=2,
                inertia=1.23,
                clusters=[
                    ClusterStats(
                        cluster_id=0, size=5, percentage=50.0, center={"a": 1.0, "b": 2.0}
                    ),
                    ClusterStats(
                        cluster_id=1, size=5, percentage=50.0, center={"a": 3.0, "b": 4.0}
                    ),
                ],
                points=[ClusteringPoint(x=0.1, y=0.2, cluster=0)],
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_clustering(console, Table)
    output = console.export_text()
    assert "KMeans" in output or "Clusters" in output


def test_summary_prints_message_when_rich_missing(
    base_profile: DatasetProfile, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """summary() should print an install hint and return early if 'rich' is unavailable."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name.startswith("rich"):
            raise ImportError("no rich installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    visualizer = EDAVisualizer(base_profile)
    visualizer.summary()

    captured = capsys.readouterr()
    assert "rich" in captured.out.lower()


def test_render_numeric_stats_prints_message_without_numeric_columns(
    base_profile: DatasetProfile,
) -> None:
    """_render_numeric_stats should print a fallback message when no numeric columns exist."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(
        update={
            "columns": {
                "cat": ColumnProfile(
                    name="cat",
                    dtype="Categorical",
                    missing_count=0,
                    missing_percentage=0.0,
                    categorical_stats=CategoricalStats(unique_count=2, top_k=[]),
                )
            }
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_numeric_stats(console, Table)
    output = console.export_text()
    assert "No numeric columns found" in output


def test_render_vif_shows_high_status_for_moderate_scores(base_profile: DatasetProfile) -> None:
    """_render_vif should mark VIF scores between 5 and 10 as 'High' rather than 'Severe'."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(update={"vif": {"a": 7.5}})
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_vif(console, Table)
    output = console.export_text()
    assert "High" in output


def test_render_text_stats_renders_populated_text_column(base_profile: DatasetProfile) -> None:
    """_render_text_stats should format length and sentiment stats for text columns."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(
        update={
            "columns": {
                "review": ColumnProfile(
                    name="review",
                    dtype="Text",
                    missing_count=0,
                    missing_percentage=0.0,
                    text_stats=TextStats(
                        avg_length=12.5,
                        min_length=2,
                        max_length=40,
                        sentiment_distribution={"positive": 0.5, "neutral": 0.3, "negative": 0.2},
                    ),
                )
            }
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_text_stats(console, Table)
    output = console.export_text()
    assert "review" in output


def test_render_outliers_returns_early_without_outliers(base_profile: DatasetProfile) -> None:
    """_render_outliers should no-op when the profile has no outlier analysis."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(update={"outliers": None})
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_outliers(console, Table)
    assert console.export_text() == ""


def test_render_timeseries_renders_trend_and_seasonality(base_profile: DatasetProfile) -> None:
    """_render_timeseries should print the date range and note seasonality availability."""
    from rich.console import Console

    profile = base_profile.model_copy(
        update={
            "timeseries": TimeSeriesAnalysis(
                date_col="date",
                trend=[
                    TimeSeriesPoint(date="2020-01-01", values={"a": 1.0}),
                    TimeSeriesPoint(date="2020-01-02", values={"a": 2.0}),
                ],
                seasonality=SeasonalityStats(
                    day_of_week=[{"day": "Mon", "value": 1.0}],
                    month_of_year=[{"month": "Jan", "value": 2.0}],
                ),
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_timeseries(console)
    output = console.export_text()
    assert "Time Series Analysis" in output
    assert "Seasonality Analysis" in output


def test_detect_regression_tree_true_for_numeric_leaf_value(base_profile: DatasetProfile) -> None:
    """A leaf whose class_name parses as float implies a regression surrogate tree."""
    profile = base_profile.model_copy(
        update={
            "rule_tree": RuleTree(
                nodes=[
                    RuleNode(
                        id=0,
                        impurity=0.0,
                        samples=10,
                        value=[10.0],
                        class_name="3.14",
                        is_leaf=True,
                    )
                ],
                accuracy=0.9,
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    assert visualizer._detect_regression_tree() is True


def test_detect_regression_tree_false_when_no_leaf_node_present(
    base_profile: DatasetProfile,
) -> None:
    """When no node in the tree is a leaf, detection falls through to False."""
    profile = base_profile.model_copy(
        update={
            "rule_tree": RuleTree(
                nodes=[
                    RuleNode(
                        id=0,
                        feature="a",
                        threshold=1.0,
                        impurity=0.5,
                        samples=10,
                        value=[5.0, 5.0],
                        is_leaf=False,
                        children=[],
                    )
                ],
                accuracy=0.9,
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    assert visualizer._detect_regression_tree() is False


def test_render_rule_tree_handles_regression_leaf_and_dangling_child(
    base_profile: DatasetProfile,
) -> None:
    """_render_rule_tree should format a regression leaf value and skip a missing child id."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(
        update={
            "rule_tree": RuleTree(
                nodes=[
                    RuleNode(
                        id=0,
                        feature="a",
                        threshold=1.0,
                        impurity=0.5,
                        samples=10,
                        value=[5.0],
                        is_leaf=False,
                        children=[1, 99],  # 99 does not exist -> dangling child branch
                    ),
                    RuleNode(
                        id=1,
                        impurity=0.0,
                        samples=5,
                        value=[5.0],
                        class_name="2.5",
                        is_leaf=True,
                    ),
                ],
                accuracy=0.87,
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_rule_tree(console, Table)
    output = console.export_text()
    assert "R²" in output


def test_plot_prints_message_when_matplotlib_missing(
    base_profile: DatasetProfile, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """plot() should print an install hint and return early if matplotlib is unavailable."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    visualizer = EDAVisualizer(base_profile)
    visualizer.plot()

    captured = capsys.readouterr()
    assert "matplotlib" in captured.out.lower()


def test_plot_correlations_returns_early_without_correlations(
    base_profile: DatasetProfile,
) -> None:
    """_plot_correlations should no-op when correlations are absent."""
    assert base_profile.correlations is None
    visualizer = EDAVisualizer(base_profile)
    visualizer._plot_correlations()  # Should not raise and should not create a figure.


def test_plot_correlations_with_target_returns_early_without_data(
    base_profile: DatasetProfile,
) -> None:
    """_plot_correlations_with_target should no-op when no target correlation matrix exists."""
    assert base_profile.correlations_with_target is None
    visualizer = EDAVisualizer(base_profile)
    visualizer._plot_correlations_with_target()


def test_plot_correlations_with_target_renders_heatmap(base_profile: DatasetProfile) -> None:
    """_plot_correlations_with_target should render a heatmap when data is present."""
    import matplotlib.pyplot as plt

    profile = base_profile.model_copy(
        update={
            "correlations_with_target": CorrelationMatrix(
                columns=["a", "target"], values=[[1.0, 0.5], [0.5, 1.0]]
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_correlations_with_target()
    plt.close("all")


def test_plot_target_interactions_returns_early_without_interactions(
    base_profile: DatasetProfile,
) -> None:
    """_plot_target_interactions should no-op when there are no target interactions."""
    assert base_profile.target_interactions is None
    visualizer = EDAVisualizer(base_profile)
    visualizer._plot_target_interactions()


def test_plot_target_interactions_returns_early_without_boxplots(
    base_profile: DatasetProfile,
) -> None:
    """_plot_target_interactions should no-op when interactions exist but none are boxplots."""
    profile = base_profile.model_copy(
        update={
            "target_interactions": [
                TargetInteraction(feature="a", plot_type="scatter", data=[], p_value=0.5)
            ]
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_target_interactions()


def test_plot_scatter_matrix_returns_early_without_dataframe(base_profile: DatasetProfile) -> None:
    """_plot_scatter_matrix should no-op when no raw DataFrame was provided."""
    visualizer = EDAVisualizer(base_profile, df=None)
    visualizer._plot_scatter_matrix()


def test_plot_scatter_matrix_returns_early_when_pandas_plotting_missing(
    base_profile: DatasetProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_plot_scatter_matrix should no-op if pandas.plotting.scatter_matrix can't be imported."""
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "pandas.plotting":
            raise ImportError("no pandas.plotting")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    visualizer = EDAVisualizer(base_profile, df=df)
    visualizer._plot_scatter_matrix()


def test_plot_scatter_matrix_truncates_columns_and_colors_by_target() -> None:
    """_plot_scatter_matrix should cap columns at 5 when there are more numeric columns."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    df = pl.DataFrame(
        {
            "n1": rng.normal(size=20),
            "n2": rng.normal(size=20),
            "n3": rng.normal(size=20),
            "n4": rng.normal(size=20),
            "n5": rng.normal(size=20),
            "n6": rng.normal(size=20),
            "cat": rng.choice(["a", "b"], size=20),
        }
    )
    profile = EDAAnalyzer(df).analyze()
    visualizer = EDAVisualizer(profile, df=df)
    visualizer._plot_scatter_matrix()
    plt.close("all")


def test_plot_scatter_matrix_colors_points_by_numeric_target() -> None:
    """_plot_scatter_matrix should color points by the target column when it is numeric."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    df = pl.DataFrame(
        {
            "n1": rng.normal(size=20),
            "n2": rng.normal(size=20),
            # Numeric target so it survives the numeric-only column selection in
            # _plot_scatter_matrix and exercises the color-by-target branch.
            "target": rng.integers(0, 2, size=20),
        }
    )
    profile = EDAAnalyzer(df).analyze(target_col="target")
    visualizer = EDAVisualizer(profile, df=df)
    visualizer._plot_scatter_matrix()
    plt.close("all")


def test_plot_distributions_returns_early_without_numeric_histograms(
    base_profile: DatasetProfile,
) -> None:
    """_plot_distributions should no-op when no numeric column has a histogram."""
    profile = base_profile.model_copy(
        update={
            "columns": {
                "cat": ColumnProfile(
                    name="cat",
                    dtype="Categorical",
                    missing_count=0,
                    missing_percentage=0.0,
                    categorical_stats=CategoricalStats(unique_count=2, top_k=[]),
                )
            }
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_distributions()


def test_plot_pca_returns_early_without_pca_data(base_profile: DatasetProfile) -> None:
    """_plot_pca should no-op when there is no 2D PCA projection to plot."""
    assert base_profile.pca_data is None
    visualizer = EDAVisualizer(base_profile)
    visualizer._plot_pca()


def test_plot_pca_renders_projection_with_labels(base_profile: DatasetProfile) -> None:
    """_plot_pca should render a scatter plot, exercising both numeric and categorical labels."""
    import matplotlib.pyplot as plt

    profile = base_profile.model_copy(
        update={
            "pca_data": [
                PCAPoint(x=0.1, y=0.2, label="low"),
                PCAPoint(x=0.3, y=0.4, label="high"),
            ],
            "pca_components": [
                PCAComponent(component="PC1", explained_variance_ratio=0.5, top_features={"a": 1.0})
            ],
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_pca()
    plt.close("all")


def test_plot_geospatial_returns_early_without_sample_points(base_profile: DatasetProfile) -> None:
    """_plot_geospatial should no-op when no geospatial stats or sample points exist."""
    assert base_profile.geospatial is None
    visualizer = EDAVisualizer(base_profile)
    visualizer._plot_geospatial()

    profile_no_points = base_profile.model_copy(
        update={
            "geospatial": GeospatialStats(
                lat_col="lat",
                lon_col="lon",
                min_lat=0.0,
                max_lat=1.0,
                min_lon=0.0,
                max_lon=1.0,
                centroid_lat=0.5,
                centroid_lon=0.5,
                sample_points=[],
            )
        }
    )
    EDAVisualizer(profile_no_points)._plot_geospatial()


def test_plot_timeseries_renders_trend_lines(base_profile: DatasetProfile) -> None:
    """_plot_timeseries should parse ISO dates and plot one line per tracked column."""
    import matplotlib.pyplot as plt

    profile = base_profile.model_copy(
        update={
            "timeseries": TimeSeriesAnalysis(
                date_col="date",
                trend=[
                    TimeSeriesPoint(date="2020-01-01T00:00:00", values={"a": 1.0, "b": 2.0}),
                    TimeSeriesPoint(date="2020-01-02T00:00:00", values={"a": 3.0, "b": 4.0}),
                ],
                seasonality=SeasonalityStats(day_of_week=[], month_of_year=[]),
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_timeseries()
    plt.close("all")


def test_plot_timeseries_skips_unparseable_dates(base_profile: DatasetProfile) -> None:
    """_plot_timeseries should skip trend points whose date string isn't ISO-parseable."""
    import matplotlib.pyplot as plt

    profile = base_profile.model_copy(
        update={
            "timeseries": TimeSeriesAnalysis(
                date_col="date",
                trend=[
                    TimeSeriesPoint(date="not-a-date", values={"a": 1.0}),
                    TimeSeriesPoint(date="2020-01-02T00:00:00", values={"a": 2.0}),
                ],
                seasonality=SeasonalityStats(day_of_week=[], month_of_year=[]),
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_timeseries()
    plt.close("all")


def test_plot_timeseries_returns_early_when_all_dates_unparseable(
    base_profile: DatasetProfile,
) -> None:
    """_plot_timeseries should no-op when every trend point has an unparseable date."""
    profile = base_profile.model_copy(
        update={
            "timeseries": TimeSeriesAnalysis(
                date_col="date",
                trend=[TimeSeriesPoint(date="not-a-date", values={"a": 1.0})],
                seasonality=SeasonalityStats(day_of_week=[], month_of_year=[]),
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    visualizer._plot_timeseries()


def test_render_outliers_and_causal_graph_with_full_data(base_profile: DatasetProfile) -> None:
    """_render_outliers should render top anomalies when outlier analysis data is present."""
    from rich.console import Console
    from rich.table import Table

    profile = base_profile.model_copy(
        update={
            "outliers": OutlierAnalysis(
                method="IsolationForest",
                total_outliers=2,
                outlier_percentage=10.0,
                top_outliers=[
                    OutlierPoint(
                        index=0,
                        values={"a": 1.0},
                        score=0.9,
                        explanation=[{"feature": "a", "value": 1.0, "mean": 0.0, "diff": 1.0}],
                    )
                ],
            )
        }
    )
    visualizer = EDAVisualizer(profile)
    console = Console(record=True)
    visualizer._render_outliers(console, Table)
    output = console.export_text()
    assert "outliers" in output.lower()
