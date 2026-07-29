"""Tests for skyulf.modeling._evaluation.clustering (evaluate_clustering_model)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from skyulf.modeling._evaluation.clustering import evaluate_clustering_model
from skyulf.modeling._evaluation.metrics import (
    DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    calculate_clustering_metrics,
)
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
    """Large clustering inputs must pass a bounded preselected sample to silhouette."""
    captured: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []

    def fake_silhouette(X: np.ndarray, labels: np.ndarray, **kwargs: object) -> float:
        captured.append((X.copy(), labels.copy(), dict(kwargs)))
        return 0.5

    monkeypatch.setattr("sklearn.metrics.silhouette_score", fake_silhouette)
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) * 2})
    labels = np.array([0, 1] * 20)

    metrics = calculate_clustering_metrics(X, labels, silhouette_sample_size=10, random_state=7)

    sampled_X, sampled_labels, sampled_kwargs = captured[0]
    assert sampled_X.shape == (10, 2)
    assert sampled_kwargs == {}
    assert set(sampled_labels) == {0, 1}
    assert metrics["silhouette_score"] == 0.5
    assert metrics["silhouette_sample_size"] == 10.0


def test_calculate_clustering_metrics_keeps_large_imbalanced_samples_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large imbalanced inputs must keep every cluster represented in a deterministic sample."""
    captured: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []

    def fake_silhouette(X: np.ndarray, labels: np.ndarray, **kwargs: object) -> float:
        captured.append((X.copy(), labels.copy(), dict(kwargs)))
        return 0.33

    monkeypatch.setattr("sklearn.metrics.silhouette_score", fake_silhouette)
    monkeypatch.setattr("sklearn.metrics.calinski_harabasz_score", lambda X, labels: 1.0)
    monkeypatch.setattr("sklearn.metrics.davies_bouldin_score", lambda X, labels: 2.0)
    X = pd.DataFrame(
        {
            "row_id": np.arange(DEFAULT_SILHOUETTE_SAMPLE_SIZE + 1, dtype=float),
            "feature": np.linspace(0.0, 1.0, DEFAULT_SILHOUETTE_SAMPLE_SIZE + 1),
        }
    )
    labels = np.array([0] + [1] * DEFAULT_SILHOUETTE_SAMPLE_SIZE)

    metrics1 = calculate_clustering_metrics(X, labels, random_state=474)
    metrics2 = calculate_clustering_metrics(X, labels, random_state=474)

    first_X, first_labels, first_kwargs = captured[0]
    second_X, second_labels, second_kwargs = captured[1]
    assert first_X.shape == (DEFAULT_SILHOUETTE_SAMPLE_SIZE, 2)
    assert first_kwargs == second_kwargs == {}
    assert set(first_labels) == {0, 1}
    assert int((first_labels == 0).sum()) == 1
    assert np.array_equal(first_X, second_X)
    assert np.array_equal(first_labels, second_labels)
    assert metrics1["silhouette_score"] == metrics2["silhouette_score"] == 0.33
    assert (
        metrics1["silhouette_sample_size"]
        == metrics2["silhouette_sample_size"]
        == float(DEFAULT_SILHOUETTE_SAMPLE_SIZE)
    )


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


def test_calculate_clustering_metrics_rejects_sampled_cap_without_cluster_headroom() -> None:
    """Sampled silhouette caps must leave enough rows to represent every cluster."""
    X = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)})
    labels = np.array([0, 1] * 20)

    with pytest.raises(
        ValueError,
        match="silhouette_sample_size=2 is too small for 2 clusters",
    ):
        calculate_clustering_metrics(X, labels, silhouette_sample_size=2)
