"""Clustering (unsupervised) evaluation logic."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...engines import SkyulfDataFrame
from ...engines.polars_engine import POLARS_NUMERIC_BOOL_DTYPES, SkyulfPolarsWrapper
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


def _as_polars_frame(X: Any) -> pl.DataFrame | None:
    """Return raw Polars data for native clustering evaluation when available."""
    if isinstance(X, pl.DataFrame):
        return X
    if isinstance(X, SkyulfPolarsWrapper):
        return X._df
    return None


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


def _compute_centroids_polars(X_numeric: pl.DataFrame, labels: np.ndarray) -> list[ClusterCentroid]:
    """Native Polars equivalent of ``_compute_centroids`` (no Pandas conversion).

    Casts numeric/bool columns to Float64 before aggregating so Boolean
    columns average like pandas' ``mean(numeric_only=True)`` (True/False as
    1/0), and uses ``std(ddof=1)`` to match pandas' default sample std.
    """
    total = len(labels)
    columns = X_numeric.columns

    def _column_stats(frame: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
        if not columns or frame.height == 0:
            return {}, {}
        casted = frame.select([pl.col(c).cast(pl.Float64).alias(c) for c in columns])
        means = casted.select([pl.col(c).mean().alias(c) for c in columns]).row(0)
        stds = casted.select([pl.col(c).std(ddof=1).alias(c) for c in columns]).row(0)
        mean_dict = {c: float(v) for c, v in zip(columns, means, strict=True)}
        std_dict = {
            c: (float(v) if v is not None else 0.0) for c, v in zip(columns, stds, strict=True)
        }
        return mean_dict, std_dict

    overall_mean, overall_std = _column_stats(X_numeric)
    labeled = X_numeric.with_columns(pl.Series("__skyulf_cluster__", labels))

    centroids: list[ClusterCentroid] = []
    for cluster_id in sorted(int(c) for c in np.unique(labels)):
        mask = labels == cluster_id
        size = int(mask.sum())
        subset = labeled.filter(pl.col("__skyulf_cluster__") == cluster_id).drop(
            "__skyulf_cluster__"
        )
        center, _ = _column_stats(subset)
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


def _compute_reference_crosstab_polars(
    labels: np.ndarray, reference_values: pl.Series
) -> dict[str, dict[str, int]]:
    """Native Polars equivalent of ``_compute_reference_crosstab``.

    Groups directly by (cluster, reference value) instead of building a dense
    ``pd.crosstab`` grid — occurring combinations always have a positive
    count, so the result matches the Pandas path's zero-filtered dict.

    Rows with a null reference value are excluded, matching ``pd.crosstab``'s
    default behavior of dropping NaN entries from the reference column
    (``pd.crosstab`` never emits a ``"nan"``/``"None"`` row) -- without this,
    Polars' ``group_by`` (which treats null as a normal, valid group key)
    would report a bogus ``"None"`` reference category that doesn't exist on
    the Pandas engine for identical input.
    """
    ref_name = reference_values.name or "reference"
    frame = (
        pl.DataFrame({"__skyulf_cluster__": labels})
        .with_columns(reference_values.alias(ref_name))
        .filter(pl.col(ref_name).is_not_null())
    )
    counts = frame.group_by(["__skyulf_cluster__", ref_name]).agg(pl.len().alias("count"))

    result: dict[str, dict[str, int]] = {}
    for cluster_id, ref_value, count in counts.iter_rows():
        if not isinstance(cluster_id, int | np.integer) or not count:
            continue
        result.setdefault(str(int(cluster_id)), {})[str(ref_value)] = int(count)
    return result


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
    X: pd.DataFrame | pl.DataFrame | SkyulfDataFrame,
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

    pl_frame = _as_polars_frame(X)
    if pl_frame is not None:
        reference_pl = None
        if reference_column and reference_column in pl_frame.columns:
            reference_pl = pl_frame.get_column(reference_column)
            pl_frame = pl_frame.drop(reference_column)

        # Distance-based metrics (silhouette/Calinski-Harabasz/Davies-Bouldin)
        # require a purely numeric matrix — mirror the same numeric-only
        # filtering `KMeansCalculator.fit`/`KMeansApplier.predict` apply, so a
        # stray text/id column left in `X` (e.g. no encoding node upstream)
        # doesn't crash evaluation with "could not convert string to float".
        numeric_cols = [
            c
            for c, t in zip(pl_frame.columns, pl_frame.dtypes, strict=True)
            if t in POLARS_NUMERIC_BOOL_DTYPES
        ]
        if numeric_cols:
            X_numeric_pl = pl_frame.select(numeric_cols)
        else:
            # `pl_frame.select([])` collapses to 0 rows, unlike pandas'
            # `select_dtypes(...)` which preserves row count with 0 columns —
            # keep the height so downstream row-count validation matches.
            X_numeric_pl = pl.DataFrame(np.empty((pl_frame.height, 0)))
        metrics = calculate_clustering_metrics(X_numeric_pl, labels_np)
        centroids = _compute_centroids_polars(X_numeric_pl, labels_np)
        reference_crosstab = (
            _compute_reference_crosstab_polars(labels_np, reference_pl)
            if reference_pl is not None
            else None
        )
    else:
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
        centroids = _compute_centroids(X_df, labels_np, X_numeric)
        reference_crosstab = (
            _compute_reference_crosstab(labels_np, reference_values)
            if reference_values is not None
            else None
        )

    cluster_sizes = {str(int(c)): int((labels_np == c).sum()) for c in sorted(np.unique(labels_np))}

    clustering_eval = ClusteringEvaluation(
        n_clusters=len(np.unique(labels_np)),
        cluster_sizes=cluster_sizes,
        centroids=centroids,
        reference_crosstab=reference_crosstab,
        reference_column=reference_column or None,
    )

    return ModelEvaluationReport(
        dataset_name=dataset_name,
        metrics=sanitize_metrics(metrics),
        classification=None,
        regression=None,
        clustering=clustering_eval,
    )
