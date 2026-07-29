"""Tests for skyulf.modeling._evaluation.clustering (evaluate_clustering_model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from skyulf.modeling._evaluation.clustering import evaluate_clustering_model
from skyulf.modeling._evaluation.metrics import calculate_clustering_metrics
from skyulf.modeling._evaluation.schemas import ModelEvaluationReport


@pytest.fixture
def clustering_fitted():
    """Deterministic KMeans model fitted on 3 well-separated blobs."""
    rng = np.random.RandomState(0)
    blob0 = rng.normal(loc=(-5, -5), scale=0.3, size=(20, 2))
    blob1 = rng.normal(loc=(5, 5), scale=0.3, size=(20, 2))
    blob2 = rng.normal(loc=(5, -5), scale=0.3, size=(20, 2))
    X = pd.DataFrame(np.vstack([blob0, blob1, blob2]), columns=["x", "y"])
    model = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
    labels = model.predict(X)
    return model, X, labels


def test_evaluate_clustering_returns_model_evaluation_report(clustering_fitted):
    """The report should be a ModelEvaluationReport with clustering populated."""
    model, X, labels = clustering_fitted
    report = evaluate_clustering_model(model, X, labels, dataset_name="train")
    assert isinstance(report, ModelEvaluationReport)
    assert report.dataset_name == "train"
    assert report.clustering is not None
    assert report.classification is None
    assert report.regression is None


def test_evaluate_clustering_finds_three_well_separated_clusters(clustering_fitted):
    """Three well-separated blobs should be recovered with a high silhouette score."""
    model, X, labels = clustering_fitted
    report = evaluate_clustering_model(model, X, labels)
    assert report.clustering is not None
    assert report.clustering.n_clusters == 3
    assert report.metrics["silhouette_score"] > 0.8
    assert report.metrics["silhouette_sample_size"] == len(labels)


def test_evaluate_clustering_cluster_sizes_sum_to_total(clustering_fitted):
    """Per-cluster sizes should add up to the total number of rows."""
    model, X, labels = clustering_fitted
    report = evaluate_clustering_model(model, X, labels)
    assert report.clustering is not None
    assert sum(report.clustering.cluster_sizes.values()) == len(labels)


def test_evaluate_clustering_centroids_have_real_feature_names(clustering_fitted):
    """Centroid dicts should be keyed by the original DataFrame column names."""
    model, X, labels = clustering_fitted
    report = evaluate_clustering_model(model, X, labels)
    assert report.clustering is not None
    for centroid in report.clustering.centroids:
        assert set(centroid.center.keys()) == {"x", "y"}


def test_calculate_clustering_metrics_single_cluster_omits_quality_scores():
    """A degenerate single-cluster labeling should skip silhouette/CH/DB (undefined)."""
    X = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    labels = np.zeros(10, dtype=int)
    metrics = calculate_clustering_metrics(X, labels)
    assert metrics["n_clusters"] == 1
    assert "silhouette_score" not in metrics


def test_calculate_clustering_metrics_caps_silhouette_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large clustering inputs must use the configured deterministic silhouette cap."""
    captured: dict[str, object] = {}

    def fake_silhouette(
        X: np.ndarray,
        labels: np.ndarray,
        *,
        sample_size: int | None = None,
        random_state: int | None = None,
    ) -> float:
        captured["shape"] = X.shape
        captured["sample_size"] = sample_size
        captured["random_state"] = random_state
        return 0.5

    monkeypatch.setattr("sklearn.metrics.silhouette_score", fake_silhouette)
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) * 2})
    labels = np.array([0, 1] * 20)

    metrics = calculate_clustering_metrics(X, labels, silhouette_sample_size=10, random_state=7)

    assert captured["shape"] == (40, 2)
    assert captured["sample_size"] == 10
    assert captured["random_state"] == 7
    assert metrics["silhouette_score"] == 0.5
    assert metrics["silhouette_sample_size"] == 10.0


def test_calculate_clustering_metrics_scores_all_small_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs at or below the cap must not request sklearn subsampling."""
    captured: dict[str, object] = {}

    def fake_silhouette(X: np.ndarray, labels: np.ndarray, **kwargs: object) -> float:
        captured.update(kwargs)
        return 0.75

    monkeypatch.setattr("sklearn.metrics.silhouette_score", fake_silhouette)
    X = pd.DataFrame({"a": [0.0, 1.0, 10.0, 11.0], "b": [0.0, 1.0, 10.0, 11.0]})
    labels = np.array([0, 0, 1, 1])

    metrics = calculate_clustering_metrics(X, labels, silhouette_sample_size=10)

    assert captured == {}
    assert metrics["silhouette_sample_size"] == 4.0


@pytest.mark.parametrize("sample_size", [-1, 0, 1])
def test_calculate_clustering_metrics_rejects_invalid_silhouette_sample_size(
    sample_size: int,
) -> None:
    """Silhouette sample size must be large enough for sklearn scoring."""
    X = pd.DataFrame({"a": [0.0, 1.0, 10.0, 11.0], "b": [0.0, 1.0, 10.0, 11.0]})
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="silhouette_sample_size must be at least 2"):
        calculate_clustering_metrics(X, labels, silhouette_sample_size=sample_size)


def test_calculate_clustering_metrics_rejects_invalid_cap_before_degenerate_guard() -> None:
    """Invalid caps should raise even when cluster-quality metrics are otherwise undefined."""
    X = pd.DataFrame({"a": np.arange(4.0), "b": np.arange(4.0)})
    labels = np.zeros(4, dtype=int)

    with pytest.raises(ValueError, match="silhouette_sample_size must be at least 2"):
        calculate_clustering_metrics(X, labels, silhouette_sample_size=1)
