"""Tests for skyulf.profiling._analyzer.multivariate.MultivariateMixin."""


import numpy as np
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling._analyzer import multivariate as multivariate_mod
from skyulf.profiling.analyzer import EDAAnalyzer


def _multivariate_df(n: int = 60) -> pl.DataFrame:
    """Small multi-feature numeric dataset with a target column for PCA/clustering."""
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1, n)
    b = a * 1.5 + rng.normal(0, 0.2, n)
    c = rng.normal(5, 2, n)
    target = rng.choice(["x", "y"], size=n)
    return pl.DataFrame({"a": a, "b": b, "c": c, "target": target})


def test_prepare_matrix_sample_returns_scaled_matrix() -> None:
    """Happy path: returns a scaled matrix, sample_df, and a scaler instance."""
    analyzer = EDAAnalyzer(_multivariate_df())
    X_scaled, sample_df, scaler = analyzer._prepare_matrix_sample(["a", "b", "c"])
    assert X_scaled is not None
    assert sample_df is not None
    assert scaler is not None
    assert X_scaled.shape[1] == 3


def test_prepare_matrix_sample_sklearn_unavailable(monkeypatch) -> None:
    """SKLEARN_AVAILABLE=False should short-circuit to (None, None, None) (line 35)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    monkeypatch.setattr(multivariate_mod, "SKLEARN_AVAILABLE", False)
    result = analyzer._prepare_matrix_sample(["a", "b", "c"])
    assert result == (None, None, None)


def test_prepare_matrix_sample_appends_target_col() -> None:
    """target_col not already in numeric_cols should be appended to cols_to_fetch (line 49)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    X_scaled, sample_df, _ = analyzer._prepare_matrix_sample(["a", "b"], target_col="c")
    assert X_scaled is not None
    assert sample_df is not None
    assert "c" in sample_df.columns


def test_prepare_matrix_sample_too_few_rows_returns_none() -> None:
    """Fewer than 5 sampled rows should short-circuit to (None, None, None)."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    analyzer = EDAAnalyzer(df)
    result = analyzer._prepare_matrix_sample(["a", "b"])
    assert result == (None, None, None)


def test_prepare_matrix_sample_fill_null_fallback(monkeypatch) -> None:
    """fill_null raising should fall back to pandas + SimpleImputer path (lines 66-71)."""
    analyzer = EDAAnalyzer(_multivariate_df())

    def _boom(self, *args, **kwargs):
        raise RuntimeError("fill_null exploded")

    monkeypatch.setattr(pl.DataFrame, "fill_null", _boom)
    X_scaled, sample_df, scaler = analyzer._prepare_matrix_sample(["a", "b", "c"])
    assert X_scaled is not None
    assert sample_df is not None
    assert scaler is not None


def test_prepare_matrix_sample_outer_exception(monkeypatch) -> None:
    """StandardScaler.fit_transform raising should be caught by the outer except (lines 78-80)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.preprocessing import StandardScaler

    def _boom(self, *args, **kwargs):
        raise RuntimeError("scaler exploded")

    monkeypatch.setattr(StandardScaler, "fit_transform", _boom)
    result = analyzer._prepare_matrix_sample(["a", "b", "c"])
    assert result == (None, None, None)


def test_prepare_matrix_sample_large_dataset_is_sampled() -> None:
    """row_count > limit should trigger the .sample() path instead of full select (line 49)."""
    rng = np.random.default_rng(5)
    n = 20
    df = pl.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    analyzer = EDAAnalyzer(df)
    X_scaled, sample_df, _ = analyzer._prepare_matrix_sample(["a", "b"], limit=5)
    assert X_scaled is not None
    assert sample_df is not None
    assert sample_df.height == 5


def test_prepare_matrix_sample_handles_non_finite_values() -> None:
    """Infinite values surviving fill_null should be replaced via nan_to_num (line 66)."""
    n = 10
    df = pl.DataFrame({"a": [1.0] * n, "b": [float("inf")] * (n - 1) + [1.0]})
    analyzer = EDAAnalyzer(df)
    X_scaled, sample_df, scaler = analyzer._prepare_matrix_sample(["a", "b"])
    assert X_scaled is not None
    assert np.isfinite(X_scaled).all()


def test_calculate_pca_happy_path() -> None:
    """3+ numeric columns should produce 3D PCA points and component loadings."""
    analyzer = EDAAnalyzer(_multivariate_df())
    points, components = analyzer._calculate_pca(["a", "b", "c"], target_col="target")
    assert points is not None
    assert components is not None
    assert len(components) == 3
    assert all(p.z is not None for p in points)
    assert all(p.label in ("x", "y") for p in points)


def test_calculate_pca_pads_when_fewer_than_three_components(monkeypatch) -> None:
    """Force PCA.fit_transform to return only 2 dims to hit the padding branch (lines 120-121)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.decomposition import PCA

    def _two_dim_fit_transform(self, X, *args, **kwargs):
        return np.asarray(X)[:, :2]

    monkeypatch.setattr(PCA, "fit_transform", _two_dim_fit_transform)
    points, components = analyzer._calculate_pca(["a", "b", "c"])
    assert points is not None
    # z should still be populated (padded with zeros) since shape[1] > 2 after padding.
    assert all(p.z is not None for p in points)


def test_calculate_pca_single_column_reshapes_1d(monkeypatch) -> None:
    """A single numeric column can make PCA.fit_transform return a 1D array (line 117)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.decomposition import PCA

    def _flat_fit_transform(self, X, *args, **kwargs):
        return np.asarray(X)[:, 0]

    monkeypatch.setattr(PCA, "fit_transform", _flat_fit_transform)
    points, _ = analyzer._calculate_pca(["a"])
    assert points is not None
    assert len(points) > 0


def test_calculate_pca_no_matrix_returns_none() -> None:
    """When _prepare_matrix_sample fails (too few rows), PCA should return (None, None)."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    analyzer = EDAAnalyzer(df)
    points, components = analyzer._calculate_pca(["a", "b"])
    assert points is None
    assert components is None


def test_calculate_pca_outer_exception(monkeypatch) -> None:
    """PCA.fit_transform raising should be caught by the outer except (lines 141-143)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.decomposition import PCA

    def _boom(self, *args, **kwargs):
        raise RuntimeError("pca exploded")

    monkeypatch.setattr(PCA, "fit_transform", _boom)
    points, components = analyzer._calculate_pca(["a", "b", "c"])
    assert points is None
    assert components is None


def test_perform_clustering_happy_path() -> None:
    """3+ numeric cols should produce a full ClusteringAnalysis with 3 clusters."""
    analyzer = EDAAnalyzer(_multivariate_df())
    result = analyzer._perform_clustering(["a", "b", "c"], target_col="target")
    assert result is not None
    assert result.n_clusters == 3
    assert len(result.clusters) == 3
    assert all(p.label in ("x", "y") for p in result.points)


def test_perform_clustering_pads_when_fewer_than_two_components(monkeypatch) -> None:
    """Force PCA.fit_transform to return a 1-dim projection to hit the padding branch (192-193)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.decomposition import PCA

    def _one_dim_fit_transform(self, X, *args, **kwargs):
        return np.asarray(X)[:, :1]

    monkeypatch.setattr(PCA, "fit_transform", _one_dim_fit_transform)
    result = analyzer._perform_clustering(["a", "b", "c"])
    assert result is not None
    assert len(result.points) > 0


def test_perform_clustering_single_column_reshapes_1d(monkeypatch) -> None:
    """Force PCA.fit_transform to return a 1D array inside clustering (line 190)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.decomposition import PCA

    def _flat_fit_transform(self, X, *args, **kwargs):
        return np.asarray(X)[:, 0]

    monkeypatch.setattr(PCA, "fit_transform", _flat_fit_transform)
    result = analyzer._perform_clustering(["a"])
    assert result is not None


def test_perform_clustering_no_matrix_returns_none() -> None:
    """Too few sampled rows should make clustering return None."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    analyzer = EDAAnalyzer(df)
    assert analyzer._perform_clustering(["a", "b"]) is None


def test_perform_clustering_outer_exception(monkeypatch) -> None:
    """KMeans.fit_predict raising should be caught by the outer except (lines 219-221)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.cluster import KMeans

    def _boom(self, *args, **kwargs):
        raise RuntimeError("kmeans exploded")

    monkeypatch.setattr(KMeans, "fit_predict", _boom)
    result = analyzer._perform_clustering(["a", "b", "c"])
    assert result is None


def test_detect_outliers_happy_path() -> None:
    """IsolationForest should flag some points as outliers on a dataset with extremes."""
    rng = np.random.default_rng(3)
    n = 100
    a = list(rng.normal(0, 1, n - 5)) + [50, -50, 60, -60, 55]
    b = list(rng.normal(0, 1, n - 5)) + [50, -50, 60, -60, 55]
    df = pl.DataFrame({"a": a, "b": b})
    analyzer = EDAAnalyzer(df)
    result = analyzer._detect_outliers(["a", "b"])
    assert result is not None
    assert result.total_outliers > 0
    assert len(result.top_outliers) > 0


def test_detect_outliers_no_outliers_returns_none(monkeypatch) -> None:
    """When IsolationForest finds zero outliers, the method should return None (line 245)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.ensemble import IsolationForest

    def _no_outliers(self, X, *args, **kwargs):
        return np.ones(len(X), dtype=int)

    monkeypatch.setattr(IsolationForest, "predict", _no_outliers)
    result = analyzer._detect_outliers(["a", "b", "c"])
    assert result is None


def test_detect_outliers_missing_columns_returns_none() -> None:
    """Non-existent numeric_cols should raise internally and be caught (lines 294-296)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    result = analyzer._detect_outliers(["does_not_exist"])
    assert result is None


def test_detect_outliers_outer_exception(monkeypatch) -> None:
    """IsolationForest.fit raising should be caught by the outer except (lines 294-296)."""
    analyzer = EDAAnalyzer(_multivariate_df())
    from sklearn.ensemble import IsolationForest

    def _boom(self, *args, **kwargs):
        raise RuntimeError("iforest exploded")

    monkeypatch.setattr(IsolationForest, "fit", _boom)
    result = analyzer._detect_outliers(["a", "b", "c"])
    assert result is None


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing ``age``/``income``/``lat``/``lon`` values — closer to
    production data than the small synthetic ``_multivariate_df()`` frame used
    elsewhere in this file.
    """

    def test_calculate_pca_handles_missing_numeric_values(self) -> None:
        df = load_sample_dataset("customers", engine="polars")
        analyzer = EDAAnalyzer(df)
        points, components = analyzer._calculate_pca(["age", "income", "lat", "lon"])

        # 15 rows is enough to clear the >= 5 row floor even after mean-imputation.
        assert points is not None
        assert components is not None
        assert len(points) == df.height
        assert all(p.z is not None for p in points)
