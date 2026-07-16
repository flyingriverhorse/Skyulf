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


def _auto_profile_label(
    center: dict[str, float],
    overall_mean: dict[str, float],
    overall_std: dict[str, float],
    top_n: int = 2,
) -> str:
    """Auto-generate a human-readable "characteristic profile" for a cluster.

    Compares the cluster's centroid to the dataset-wide mean for each feature
    (a z-score), then names the ``top_n`` most distinguishing features as
    "High"/"Low <feature>". This needs no ground truth — it just describes
    what numerically sets this cluster apart, e.g. "High petal_length, High
    petal_width". It is NOT a real-world label (e.g. a species name); the
    user still supplies that meaning themselves, optionally cross-checked via
    a ``reference_column`` (see ``evaluate_clustering_model``).
    """
    scored: list[tuple[float, float, str]] = []
    for feature, value in center.items():
        mean = overall_mean.get(feature)
        std = overall_std.get(feature)
        if mean is None or not std:
            continue
        z = (value - mean) / std
        scored.append((abs(z), z, feature))

    if not scored:
        return "Average profile"

    scored.sort(key=lambda t: t[0], reverse=True)
    parts = [f"{'High' if z > 0 else 'Low'} {feature}" for _, z, feature in scored[:top_n]]
    return ", ".join(parts)


def _compute_centroids(
    X: pd.DataFrame, labels: np.ndarray, X_numeric: pd.DataFrame
) -> list[ClusterCentroid]:
    """Compute per-cluster size/percentage/mean-feature-value ("centroid") stats,
    plus an auto-generated characteristic-profile label for each cluster.
    """
    total = len(labels)
    overall_mean = {str(k): float(v) for k, v in X_numeric.mean(numeric_only=True).items()}
    overall_std = {str(k): float(v) for k, v in X_numeric.std(numeric_only=True).items()}

    centroids: list[ClusterCentroid] = []
    for cluster_id in sorted(int(c) for c in np.unique(labels)):
        mask = labels == cluster_id
        size = int(mask.sum())
        center = {str(k): float(v) for k, v in X.loc[mask].mean(numeric_only=True).items()}
        center_rounded = {k: round(v, 6) for k, v in center.items()}
        centroids.append(
            ClusterCentroid(
                cluster_id=cluster_id,
                size=size,
                percentage=round((size / total) * 100, 2) if total else 0.0,
                center=center_rounded,
                profile=_auto_profile_label(center, overall_mean, overall_std),
            )
        )
    return centroids


def _compute_reference_crosstab(
    labels: np.ndarray, reference_values: pd.Series
) -> dict[str, dict[str, int]]:
    """Cross-tabulate predicted cluster labels against a reference column's values.

    E.g. for Iris with ``reference_column="species"``: ``{"0": {"setosa": 46,
    "versicolor": 2}, "1": {...}, ...}`` — lets the user see "Cluster 0 is
    mostly setosa" without the model ever using the species column to fit.
    """
    table = pd.crosstab(pd.Series(labels, name="cluster"), reference_values)
    return {
        str(int(cluster_id)): {str(col): int(count) for col, count in row.items() if count}
        for cluster_id, row in table.iterrows()
        if isinstance(cluster_id, int | np.integer)
    }


def evaluate_clustering_model(
    model: Any,
    X: pd.DataFrame | SkyulfDataFrame,
    labels: Any,
    dataset_name: str = "test",
    reference_column: str = "",
) -> ModelEvaluationReport:
    """Evaluate a fitted clustering model's predicted ``labels`` on ``X``.

    Unlike classification/regression, there is no ground-truth target: the
    quality metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) only
    require the feature matrix and the predicted cluster assignment.

    ``reference_column``, if present in ``X``, is a user-designated column
    (e.g. a known label like species name) that was excluded from the
    features the model was fit on — it's used here only to build a
    ``reference_crosstab`` for post-hoc interpretation.
    """
    labels_np = np.asarray(labels)
    X_df = _feature_frame(X)
    X_df = X_df.reset_index(drop=True)

    reference_values = None
    if reference_column and reference_column in X_df.columns:
        reference_values = X_df[reference_column].reset_index(drop=True)
        X_df = X_df.drop(columns=[reference_column])

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
        centroids=_compute_centroids(X_df, labels_np, X_numeric),
        reference_crosstab=(
            _compute_reference_crosstab(labels_np, reference_values)
            if reference_values is not None
            else None
        ),
        reference_column=reference_column or None,
    )

    return ModelEvaluationReport(
        dataset_name=dataset_name,
        metrics=sanitize_metrics(metrics),
        classification=None,
        regression=None,
        clustering=clustering_eval,
    )
