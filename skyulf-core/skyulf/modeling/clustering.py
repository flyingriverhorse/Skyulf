"""Clustering (unsupervised segmentation) models."""

import logging
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans

from ..core.meta.decorators import node_meta
from ..engines import SkyulfDataFrame
from ..registry import NodeRegistry
from .sklearn_wrapper import SklearnApplier, SklearnCalculator

logger = logging.getLogger(__name__)


def _select_numeric_features(X: Any) -> tuple[Any, list[str]]:
    """Drop non-numeric (e.g. text/id) columns before feeding a DataFrame to sklearn.

    Clustering distance metrics (e.g. K-Means' Euclidean distance) are only
    meaningful over numeric features — a stray text/id column left in by an
    upstream node isn't automatically encoded/dropped, and would otherwise
    fail (or, worse, silently corrupt distances) once converted to a numpy
    array. Returns ``(numeric_only_X, dropped_column_names)``.
    """
    if not isinstance(X, pd.DataFrame):
        return X, []

    numeric = X.select_dtypes(include=["number", "bool"])
    dropped = [c for c in X.columns if c not in numeric.columns]
    return numeric, dropped


# --- K-Means ---
class KMeansApplier(SklearnApplier):
    """K-Means Applier.

    Reuses `SklearnApplier.predict()` as-is: `KMeans.predict()` genuinely
    supports out-of-sample cluster assignment (unlike DBSCAN/Agglomerative,
    which only implement `fit_predict()` on the training data) — this is
    exactly why K-Means is the first clustering algorithm shipped here.
    """

    def predict(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> Any:
        # Mirror the numeric-only column selection done at fit time
        # (`KMeansCalculator.fit`) so the feature count seen by
        # `model_artifact.predict()` matches what it was trained on.
        numeric_df, dropped = _select_numeric_features(df)
        if dropped:
            logger.info(f"Dropping non-numeric column(s) before predicting: {dropped}")
        return super().predict(numeric_df, model_artifact)


@NodeRegistry.register("kmeans", KMeansApplier)
@node_meta(
    id="kmeans",
    name="K-Means",
    category="Modeling",
    description="Partition rows into a fixed number of clusters (segments) by similarity.",
    params={"n_clusters": 3, "n_init": 10, "random_state": 42},
    tags=["clustering", "requires_scaling"],
)
class KMeansCalculator(SklearnCalculator):
    """K-Means Calculator."""

    def __init__(self):
        super().__init__(
            model_class=KMeans,
            default_params={
                "n_clusters": 3,
                "n_init": 10,
                "random_state": 42,
            },
            problem_type="clustering",
        )

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback=None,
        log_callback=None,
        validation_data=None,
    ) -> Any:
        """Fit K-Means, restricting to numeric columns (text/id columns aren't clusterable)."""
        numeric_X, dropped = _select_numeric_features(X)
        if dropped:
            msg = f"Dropping non-numeric column(s) before clustering: {dropped}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        return super().fit(
            numeric_X,
            y,
            config,
            progress_callback=progress_callback,
            log_callback=log_callback,
            validation_data=validation_data,
        )
