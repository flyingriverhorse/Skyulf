"""End-to-end tests for the K-Means clustering node through StatefulEstimator.

Covers the "no target column" path added to `base.py::_extract_xy`,
`fit_predict`/`evaluate`'s clustering branch, and node-registry wiring —
mirroring the conventions in `test_modeling_base.py`.
"""

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.clustering import KMeansApplier, KMeansCalculator
from skyulf.registry import NodeRegistry


@pytest.fixture
def blobs_split_dataset():
    """Train/test SplitDataset of 3 well-separated 2D blobs, no target column."""
    rng = np.random.RandomState(0)
    blob0 = rng.normal(loc=(-5, -5), scale=0.3, size=(20, 2))
    blob1 = rng.normal(loc=(5, 5), scale=0.3, size=(20, 2))
    blob2 = rng.normal(loc=(5, -5), scale=0.3, size=(20, 2))
    X = pd.DataFrame(np.vstack([blob0, blob1, blob2]), columns=["x", "y"])
    train = X.iloc[:45].reset_index(drop=True)
    test = X.iloc[45:].reset_index(drop=True)
    return SplitDataset(train=train, test=test, validation=None)


def test_kmeans_registered_in_node_registry():
    """KMeans should be registered under the 'kmeans' node id with clustering metadata."""
    calculator_cls = NodeRegistry.get_calculator("kmeans")
    applier_cls = NodeRegistry.get_applier("kmeans")
    assert calculator_cls is KMeansCalculator
    assert applier_cls is KMeansApplier

    metadata = NodeRegistry.get_all_metadata()["kmeans"]
    assert metadata["category"] == "Modeling"
    assert "clustering" in metadata["tags"]


def test_kmeans_calculator_problem_type_is_clustering():
    """KMeansCalculator.problem_type should be 'clustering', not classification/regression."""
    assert KMeansCalculator().problem_type == "clustering"


def test_fit_predict_with_empty_target_column_produces_labels_for_every_split(
    blobs_split_dataset,
):
    """fit_predict() with target_column='' (the "no target" sentinel) should fit on
    train and predict cluster labels for both train and test splits."""
    estimator = StatefulEstimator(KMeansCalculator(), KMeansApplier(), "node1")
    predictions = estimator.fit_predict(blobs_split_dataset, "", {"params": {"n_clusters": 3}})
    assert set(predictions.keys()) == {"train", "test"}
    assert len(predictions["train"]) == len(blobs_split_dataset.train)
    assert len(predictions["test"]) == len(blobs_split_dataset.test)


def test_evaluate_with_empty_target_column_returns_clustering_report(blobs_split_dataset):
    """evaluate() should branch to the clustering path and return per-split reports
    with silhouette/CH/DB metrics and no crash from the missing y_true."""
    estimator = StatefulEstimator(KMeansCalculator(), KMeansApplier(), "node1")
    estimator.fit_predict(blobs_split_dataset, "", {"params": {"n_clusters": 3}})
    report = estimator.evaluate(blobs_split_dataset, "", job_id="job1")

    assert report["problem_type"] == "clustering"
    assert set(report["splits"].keys()) == {"train", "test"}

    train_split = report["splits"]["train"]
    assert train_split.clustering is not None
    assert train_split.clustering.n_clusters == 3
    assert train_split.metrics["silhouette_score"] > 0.8

    # raw_data should carry labels (not y_true/y_pred) for each split
    raw_train = report["raw_data"]["splits"]["train"]
    assert "labels" in raw_train
    assert len(raw_train["labels"]) == len(blobs_split_dataset.train)

    # raw_data should also embed the cluster-size/centroid summary, so the
    # API can serve it to the frontend without a second evaluation pass.
    assert raw_train["clustering"]["n_clusters"] == 3
    assert sum(raw_train["clustering"]["cluster_sizes"].values()) == len(blobs_split_dataset.train)
    assert len(raw_train["clustering"]["centroids"]) == 3

    # ...and the quality metrics (silhouette/CH/DB), so the frontend can show
    # them per-split without relying on job-level flattened metric keys.
    assert raw_train["metrics"]["silhouette_score"] > 0.8


def test_refit_with_empty_target_column_does_not_crash_on_y_concat(blobs_split_dataset):
    """refit() concatenates train+validation y; for clustering y is None and must be
    skipped rather than crash trying to pd.concat([None, None])."""
    validation = blobs_split_dataset.test
    dataset = SplitDataset(
        train=blobs_split_dataset.train, test=pd.DataFrame(), validation=validation
    )
    estimator = StatefulEstimator(KMeansCalculator(), KMeansApplier(), "node1")
    estimator.fit_predict(dataset, "", {"params": {"n_clusters": 3}})
    estimator.refit(dataset, "", {"params": {"n_clusters": 3}})
    assert estimator.model is not None
