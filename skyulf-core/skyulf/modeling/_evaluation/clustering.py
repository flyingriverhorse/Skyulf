"""Clustering (unsupervised) evaluation logic."""

from typing import Any

import numpy as np
import pandas as pd

from ...engines import SkyulfDataFrame
from ...modeling.sklearn_wrapper import SklearnBridge
from .common import sanitize_metrics
from .metrics import calculate_clustering_metrics
from .schemas import ClusterCentroid, ClusteringEvaluation, ModelEvaluationReport


def _feature_frame(X: pd.DataFrame | SkyulfDataFrame | Any) -> pd.DataFrame:
    """Best-effort coercion of ``X`` to a pandas DataFrame with real column names."""
    if isinstance(X, pd.DataFrame):
        return X
    if hasattr(X, "to_pandas"):
        return X.to_pandas()
    X_np, _ = SklearnBridge.to_sklearn((X, None))
    return pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(X_np.shape[1])])


def _compute_centroids(X: pd.DataFrame, labels: np.ndarray) -> list[ClusterCentroid]:
    """Compute per-cluster size/percentage/mean-feature-value ("centroid") stats."""
    total = len(labels)
    centroids: list[ClusterCentroid] = []
    for cluster_id in sorted(int(c) for c in np.unique(labels)):
        mask = labels == cluster_id
        size = int(mask.sum())
        center = X.loc[mask].mean(numeric_only=True).to_dict()
        centroids.append(
            ClusterCentroid(
                cluster_id=cluster_id,
                size=size,
                percentage=round((size / total) * 100, 2) if total else 0.0,
                center={k: round(float(v), 6) for k, v in center.items()},
            )
        )
    return centroids


def evaluate_clustering_model(
    model: Any,
    X: pd.DataFrame | SkyulfDataFrame,
    labels: Any,
    dataset_name: str = "test",
) -> ModelEvaluationReport:
    """Evaluate a fitted clustering model's predicted ``labels`` on ``X``.

    Unlike classification/regression, there is no ground-truth target: the
    quality metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) only
    require the feature matrix and the predicted cluster assignment.
    """
    labels_np = np.asarray(labels)
    X_df = _feature_frame(X)
    X_df = X_df.reset_index(drop=True)

    # Distance-based metrics (silhouette/Calinski-Harabasz/Davies-Bouldin)
    # require a purely numeric matrix — mirror the same numeric-only
    # filtering `KMeansCalculator.fit`/`KMeansApplier.predict` apply, so a
    # stray text/id column left in `X` (e.g. no encoding node upstream)
    # doesn't crash evaluation with "could not convert string to float".
    X_numeric = X_df.select_dtypes(include=["number", "bool"])
    metrics = calculate_clustering_metrics(X_numeric, labels_np)

    cluster_sizes = {str(int(c)): int((labels_np == c).sum()) for c in sorted(np.unique(labels_np))}

    clustering_eval = ClusteringEvaluation(
        n_clusters=len(np.unique(labels_np)),
        cluster_sizes=cluster_sizes,
        centroids=_compute_centroids(X_df, labels_np),
    )

    return ModelEvaluationReport(
        dataset_name=dataset_name,
        metrics=sanitize_metrics(metrics),
        classification=None,
        regression=None,
        clustering=clustering_eval,
    )
