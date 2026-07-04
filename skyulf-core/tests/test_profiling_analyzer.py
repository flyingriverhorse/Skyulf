"""Tests for skyulf.profiling.analyzer.EDAAnalyzer.

Exercises the ``analyze()`` orchestrator across a variety of column types
(numeric / categorical / boolean / datetime / text / geospatial) and target
scenarios (regression, classification, no target, filters, exclusions,
empty data) since ``analyze()`` fans out into nearly every ``_analyzer``
mixin in one pass.
"""

from typing import List

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile


def _mixed_dataset(n: int = 120) -> pl.DataFrame:
    """Build a polars DataFrame covering numeric/categorical/date/text/geo columns."""
    rng = np.random.default_rng(42)

    num1 = rng.normal(0, 1, n)
    # num2 is a noisy linear function of num1 so correlations/VIF have signal.
    num2 = num1 * 2.0 + rng.normal(0, 0.1, n)

    cats = rng.choice(["A", "B", "C"], size=n, p=[0.85, 0.1, 0.05])
    flags = rng.choice([True, False], size=n)

    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 28, size=n)
    dates = [f"2021-{m:02d}-{d:02d}" for m, d in zip(months, days)]

    words = ["good", "bad", "excellent", "terrible", "average", "great"]
    text_col = [" ".join(rng.choice(words, size=4)) + f" note {i}" for i in range(n)]

    emails = [f"user{i % 5}@example.com" for i in range(n)]

    lat = rng.uniform(30.0, 45.0, n)
    lon = rng.uniform(-10.0, 10.0, n)

    target_reg = num1 * 3 + rng.normal(0, 0.5, n)
    target_clf = (num1 > 0).astype(int)

    df = pl.DataFrame(
        {
            "row_id": [f"id-{i}" for i in range(n)],
            "num1": num1,
            "num2": num2,
            "cat": cats,
            "flag": flags,
            "created_at": dates,
            "review_text": text_col,
            "contact_email": emails,
            "latitude": lat,
            "longitude": lon,
            "constant_col": [7.0] * n,
            "target_reg": target_reg,
            "target_clf": target_clf,
        }
    )
    # 60% missing so the "High Null" alert + "Drop" recommendation both fire.
    mostly_missing = [None if i % 5 != 0 else 1.0 for i in range(n)]
    df = df.with_columns(pl.Series("mostly_missing", mostly_missing))
    # A handful of missing values (~10%) so the "Impute" recommendation also fires.
    few_missing = [None if i % 10 == 0 else float(i) for i in range(n)]
    df = df.with_columns(pl.Series("few_missing", few_missing))
    return df


@pytest.fixture(scope="module")
def mixed_df() -> pl.DataFrame:
    """Shared mixed-type dataset used across analyzer tests in this module."""
    return _mixed_dataset()


def test_analyze_regression_target_builds_full_profile(mixed_df: pl.DataFrame) -> None:
    """analyze() with a numeric target should populate correlations, VIF, PCA and rules."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze(target_col="target_reg", task_type="Regression")

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == mixed_df.height
    assert profile.task_type == "Regression"
    assert "num1" in profile.columns
    assert profile.columns["num1"].dtype == "Numeric"
    assert profile.columns["constant_col"].is_constant is True
    # row_id is a high-cardinality string so it's classified as free-form Text.
    assert profile.columns["row_id"].dtype == "Text"

    # High correlation between num1 and target_reg should trigger a leakage alert.
    assert profile.target_correlations
    assert profile.target_correlations.get("num1", 0) > 0.9
    assert any(a.type == "Leakage" for a in profile.alerts)
    assert any(a.type == "Constant" for a in profile.alerts)
    assert any(a.type == "High Null" for a in profile.alerts)
    assert any(a.type == "PII" for a in profile.alerts)

    # VIF may be None when a constant column causes a singular correlation matrix;
    # the collinearity computation itself is covered by a dedicated unit test below.
    if profile.vif is not None:
        assert "num1" in profile.vif

    # Recommendations should include a Drop for the constant/ID/high-missing columns.
    actions = {r.action for r in profile.recommendations}
    assert "Drop" in actions
    assert "Impute" in actions

    # PCA / rule tree / correlations should be populated for >=2 numeric features.
    assert profile.pca_components is not None
    assert profile.correlations is not None
    assert profile.rule_tree is not None
    assert profile.rule_tree.accuracy is not None


def test_analyze_classification_target_uses_categorical_path(mixed_df: pl.DataFrame) -> None:
    """A categorical target should use eta-association + ANOVA box-plot interactions."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze(target_col="cat")

    assert profile.task_type is None or profile.task_type == "Classification"
    assert profile.target_correlations is not None
    # eta-association values should be within a valid correlation-like range.
    for val in profile.target_correlations.values():
        assert -1.0 <= val <= 1.0
    if profile.target_interactions:
        for interaction in profile.target_interactions:
            assert interaction.plot_type == "boxplot"


def test_analyze_detects_geospatial_and_dates(mixed_df: pl.DataFrame) -> None:
    """Lat/lon columns and date-like strings should be auto-detected."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze()

    assert profile.geospatial is not None
    assert profile.geospatial.lat_col == "latitude"
    assert profile.geospatial.lon_col == "longitude"
    assert profile.geospatial.min_lat <= profile.geospatial.max_lat

    # "created_at" should have been auto-cast to a Date/Datetime dtype.
    assert profile.columns["created_at"].dtype == "DateTime"
    assert profile.columns["created_at"].date_stats is not None


def test_analyze_text_and_categorical_stats(mixed_df: pl.DataFrame) -> None:
    """Text columns get common-word stats; skewed categoricals get top_k + rare labels."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze()

    text_profile = profile.columns["review_text"]
    assert text_profile.dtype == "Text"
    assert text_profile.text_stats is not None
    assert text_profile.text_stats.avg_length and text_profile.text_stats.avg_length > 0

    cat_profile = profile.columns["cat"]
    assert cat_profile.dtype == "Categorical"
    assert cat_profile.categorical_stats is not None
    assert cat_profile.categorical_stats.unique_count == 3
    assert len(cat_profile.categorical_stats.top_k) > 0


def test_analyze_applies_filters_and_exclusions(mixed_df: pl.DataFrame) -> None:
    """Filters narrow rows; exclude_cols removes columns from the resulting profile."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze(
        filters=[{"column": "cat", "operator": "==", "value": "A"}],
        exclude_cols=["review_text"],
    )

    assert profile.row_count <= mixed_df.height
    assert "review_text" not in profile.columns
    assert profile.excluded_columns == ["review_text"]
    assert profile.active_filters is not None
    assert profile.active_filters[0].column == "cat"


def test_analyze_filters_all_operators(mixed_df: pl.DataFrame) -> None:
    """Each supported filter operator should be applied without raising."""
    ops: List[dict] = [
        {"column": "num1", "operator": ">", "value": 0.0},
        {"column": "num1", "operator": "<", "value": 100.0},
        {"column": "num1", "operator": ">=", "value": -100.0},
        {"column": "num1", "operator": "<=", "value": 100.0},
        {"column": "cat", "operator": "!=", "value": "Z"},
        {"column": "cat", "operator": "in", "value": ["A", "B"]},
    ]
    for op in ops:
        analyzer = EDAAnalyzer(mixed_df)
        profile = analyzer.analyze(filters=[op])
        assert profile.row_count >= 0


def test_analyze_empty_filter_result_returns_empty_profile(mixed_df: pl.DataFrame) -> None:
    """A filter matching zero rows should short-circuit with an 'Empty Data' alert."""
    analyzer = EDAAnalyzer(mixed_df)
    profile = analyzer.analyze(
        filters=[{"column": "cat", "operator": "==", "value": "does-not-exist"}]
    )

    assert profile.row_count == 0
    assert profile.columns == {}
    assert profile.alerts[0].type == "Empty Data"
    assert profile.correlations is None


def test_analyze_handles_single_row_dataframe() -> None:
    """A one-row frame should not crash despite most stats being undefined."""
    df = pl.DataFrame({"a": [1.0], "b": ["x"]})
    analyzer = EDAAnalyzer(df)
    profile = analyzer.analyze()

    assert profile.row_count == 1
    assert profile.columns["a"].dtype == "Numeric"


def test_analyze_handles_all_null_column() -> None:
    """An all-null numeric column should produce None stats instead of raising."""
    df = pl.DataFrame(
        {
            "all_null": pl.Series([None, None, None, None], dtype=pl.Float64),
            "other": [1.0, 2.0, 3.0, 4.0],
        }
    )
    analyzer = EDAAnalyzer(df)
    profile = analyzer.analyze()

    assert profile.columns["all_null"].missing_percentage == 100.0
    assert profile.columns["all_null"].numeric_stats is not None
    assert profile.columns["all_null"].numeric_stats.mean is None


def test_get_semantic_type_classifies_common_dtypes() -> None:
    """_get_semantic_type should map dtypes/cardinality to the expected bucket."""
    # Low unique-ratio string column (<5%) should be classified as Categorical.
    cat_values = ["x"] * 98 + ["y", "z"]
    df = pl.DataFrame(
        {
            "f": [float(i) for i in range(100)],
            "flag": [True, False] * 50,
            "text": [f"a very long unique sentence number {i}" for i in range(100)],
            "cat_str": cat_values,
        }
    )
    analyzer = EDAAnalyzer(df)
    assert analyzer._get_semantic_type(df["f"]) == "Numeric"
    assert analyzer._get_semantic_type(df["flag"]) == "Boolean"
    assert analyzer._get_semantic_type(df["text"]) == "Text"
    assert analyzer._get_semantic_type(df["cat_str"]) == "Categorical"


def test_timeseries_analysis_large_dataset_uses_resampling() -> None:
    """>=1000 rows should trigger the group_by_dynamic trend + seasonality path."""
    rng = np.random.default_rng(1)
    n = 1500
    dates = pl.datetime_range(
        start=pl.datetime(2020, 1, 1), end=pl.datetime(2024, 1, 1), interval="1d", eager=True
    ).head(n)
    n = dates.len()
    values = rng.normal(10, 2, n)
    df = pl.DataFrame({"event_date": dates, "metric": values})

    analyzer = EDAAnalyzer(df)
    profile = analyzer.analyze(date_col="event_date")

    assert profile.timeseries is not None
    assert profile.timeseries.date_col == "event_date"
    assert len(profile.timeseries.trend) > 0
    assert profile.timeseries.seasonality.day_of_week
    assert profile.timeseries.seasonality.month_of_year


def test_discover_causal_graph_returns_none_for_small_sample() -> None:
    """The PC algorithm requires >=50 rows; smaller samples should return None."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._discover_causal_graph(["a", "b"])
    assert result is None


def test_discover_causal_graph_runs_on_correlated_data() -> None:
    """A larger correlated dataset should produce a causal graph with matching nodes."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 200)
    b = a * 1.5 + rng.normal(0, 0.2, 200)
    c = rng.normal(0, 1, 200)
    df = pl.DataFrame({"a": a, "b": b, "c": c})
    analyzer = EDAAnalyzer(df)
    graph = analyzer._discover_causal_graph(["a", "b", "c"])

    assert graph is not None
    assert {n.id for n in graph.nodes} == {"a", "b", "c"}


def test_discover_causal_graph_selects_top_correlated_with_target_hint() -> None:
    """With >15 numeric cols and a 'target'-like name, keep target + top-14 |corr| cols."""
    rng = np.random.default_rng(11)
    n = 60
    data = {f"feature_{i}": rng.normal(0, 1, n) for i in range(15)}
    # target correlates strongly with feature_0 so it should survive the top-14 cut.
    data["target"] = data["feature_0"] * 2.0 + rng.normal(0, 0.1, n)
    df = pl.DataFrame(data)
    analyzer = EDAAnalyzer(df)

    graph = analyzer._discover_causal_graph(list(df.columns))

    assert graph is not None
    node_ids = {n.id for n in graph.nodes}
    assert "target" in node_ids
    assert len(node_ids) == 15


def test_discover_causal_graph_selects_highest_variance_without_target_hint() -> None:
    """With >15 numeric cols and no target-like name, keep the 15 highest-variance cols."""
    rng = np.random.default_rng(12)
    n = 60
    data = {}
    for i in range(16):
        # Scale increases with i so higher-index columns have higher variance.
        data[f"col_{i}"] = rng.normal(0, 1 + i, n)
    df = pl.DataFrame(data)
    analyzer = EDAAnalyzer(df)

    graph = analyzer._discover_causal_graph(list(df.columns))

    assert graph is not None
    node_ids = {n.id for n in graph.nodes}
    assert len(node_ids) == 15
    # The lowest-variance column ("col_0") should have been dropped.
    assert "col_0" not in node_ids


def test_discover_causal_graph_returns_none_when_causallearn_missing(monkeypatch) -> None:
    """If causal-learn isn't importable, the PC algorithm step should degrade to None."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("causallearn"):
            raise ImportError("mocked missing causallearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rng = np.random.default_rng(13)
    df = pl.DataFrame({"a": rng.normal(0, 1, 60), "b": rng.normal(0, 1, 60)})
    analyzer = EDAAnalyzer(df)

    assert analyzer._discover_causal_graph(["a", "b"]) is None


def test_discover_causal_graph_maps_all_edge_endpoint_combinations(monkeypatch) -> None:
    """Directed/reversed/undirected/bidirected/no-edge endpoint codes should map correctly."""
    from types import SimpleNamespace

    import skyulf.profiling._analyzer.causal as causal_module

    # causal-learn endpoint encoding: -1 = tail, 1 = arrowhead. Build a 4x4 adjacency
    # matrix exercising every branch of the endpoint-to-edge-type mapping.
    adj = np.zeros((4, 4))
    # (a, b): directed a -> b
    adj[1, 0] = -1
    adj[0, 1] = 1
    # (a, c): reversed -> edge stored as c -> a
    adj[2, 0] = 1
    adj[0, 2] = -1
    # (a, d): no edge at all (both endpoints 0) -> skipped
    # (b, c): undirected
    adj[2, 1] = -1
    adj[1, 2] = -1
    # (b, d): bidirected
    adj[3, 1] = 1
    adj[1, 3] = 1
    # (c, d): left as no edge.

    fake_cg = SimpleNamespace(G=SimpleNamespace(graph=adj))

    def fake_pc(data, alpha, indep_test, show_progress):
        return fake_cg

    monkeypatch.setattr(
        "causallearn.search.ConstraintBased.PC.pc",
        fake_pc,
    )

    rng = np.random.default_rng(14)
    n = 60
    df = pl.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
            "d": rng.normal(0, 1, n),
        }
    )
    analyzer = causal_module.CausalMixin()
    analyzer.df = df  # type: ignore[attr-defined]

    graph = analyzer._discover_causal_graph(["a", "b", "c", "d"])

    assert graph is not None
    edges_by_type = {}
    for edge in graph.edges:
        edges_by_type.setdefault(edge.type, []).append((edge.source, edge.target))

    assert ("a", "b") in edges_by_type["directed"]
    assert ("c", "a") in edges_by_type["directed"]
    assert ("b", "c") in edges_by_type["undirected"]
    assert ("b", "d") in edges_by_type["bidirected"]
    # No-edge pairs (a,d) and (c,d) should not have produced any edge entries.
    all_pairs = {(e.source, e.target) for e in graph.edges} | {
        (e.target, e.source) for e in graph.edges
    }
    assert ("a", "d") not in all_pairs
    assert ("c", "d") not in all_pairs


def test_discover_causal_graph_returns_none_on_internal_exception() -> None:
    """Any unexpected internal error should be caught and degrade to None (not raise)."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    analyzer = EDAAnalyzer(df)
    # Passing a column name that isn't in the DataFrame triggers a polars error
    # inside the try block, which should be caught by the outer except.
    result = analyzer._discover_causal_graph(["a", "does_not_exist"])
    assert result is None


def test_calculate_vif_flags_collinear_columns() -> None:
    """VIF should be high for two nearly-collinear numeric columns."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 100)
    b = a + rng.normal(0, 0.01, 100)
    df = pl.DataFrame({"a": a, "b": b})
    analyzer = EDAAnalyzer(df)
    vif = analyzer._calculate_vif(["a", "b"])

    assert vif is not None
    assert vif["a"] > 5
    assert vif["b"] > 5


def test_calculate_vif_returns_none_for_single_column() -> None:
    """VIF is undefined for a single feature; the function should return None."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    analyzer = EDAAnalyzer(df)
    assert analyzer._calculate_vif(["a"]) is None


def test_analyze_flags_moderate_vif_with_info_severity() -> None:
    """A moderately (not severely) collinear feature set should raise an 'info' VIF alert."""
    rng = np.random.default_rng(5)
    n = 300
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    # c is a moderate (not near-perfect) linear combination of a and b to land VIF in (5, 10].
    c = 0.75 * a + 0.75 * b + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"a": a, "b": b, "c": c})

    profile = EDAAnalyzer(df).analyze()

    assert profile.vif is not None
    moderate_vif_cols = [col for col, val in profile.vif.items() if 5.0 < val <= 10.0]
    assert moderate_vif_cols, f"expected a moderate VIF column, got {profile.vif}"
    info_alerts = [
        a for a in profile.alerts if a.type == "Multicollinearity" and a.severity == "info"
    ]
    assert info_alerts


def test_analyze_flags_high_null_percentage() -> None:
    """A dataset that is mostly null should raise a 'High Null' frame-level alert."""
    df = pl.DataFrame(
        {
            "a": [None, None, None, None, 1.0],
            "b": [None, None, None, None, 2.0],
        }
    )
    profile = EDAAnalyzer(df).analyze()

    assert profile.missing_cells_percentage > 50
    dataset_level_alerts = [a for a in profile.alerts if "Dataset is" in a.message]
    assert dataset_level_alerts
    assert "empty" in dataset_level_alerts[0].message


def test_analyze_infers_native_categorical_semantic_type() -> None:
    """A native pl.Categorical column should map to the 'Categorical' semantic type."""
    df = pl.DataFrame({"native_cat": pl.Series(["x", "y", "x", "z"], dtype=pl.Categorical)})
    profile = EDAAnalyzer(df).analyze()

    assert profile.columns["native_cat"].dtype == "Categorical"
