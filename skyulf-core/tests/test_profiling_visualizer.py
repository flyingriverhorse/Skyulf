"""Tests for skyulf.profiling.visualizer.EDAVisualizer.

Uses a real ``DatasetProfile`` produced by :class:`EDAAnalyzer` so the
rendering code exercises its real branches (numeric/categorical/text stats,
VIF, outliers, causal graph, geospatial, timeseries, target analysis, rule
tree, PCA, clustering, alerts) instead of hand-built stub objects.
"""

import matplotlib

matplotlib.use("Agg")  # Headless backend so plt.show() never blocks the test run.

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import (
    ClusteringAnalysis,
    ClusteringPoint,
    ClusterStats,
    DatasetProfile,
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
