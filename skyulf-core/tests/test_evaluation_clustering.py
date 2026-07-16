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
