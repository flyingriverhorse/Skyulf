"""Clustering (unsupervised segmentation) models."""

import logging
from typing import Any

import pandas as pd
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

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


def _drop_reference_column(X: Any, reference_column: str) -> Any:
    """Drop a user-designated "reference column" (e.g. a known label like
    species name) by name, regardless of dtype — unlike ``_select_numeric_features``,
    this also protects against a *numeric* reference column (e.g. a species code)
    silently riding along into the distance calculation.
    """
    if reference_column and hasattr(X, "columns") and reference_column in X.columns:
        return X.drop(columns=[reference_column])
    return X


class _NumericOnlyClusteringApplier(SklearnApplier):
    """Applier mixin shared by every clustering model: drop non-numeric columns
    (and any reference column recorded at fit time) before predicting, mirroring
    the filtering the matching Calculator applies at fit time (see
    ``_NumericOnlyClusteringCalculatorMixin``).
    """

    def predict(self, df: pd.DataFrame | SkyulfDataFrame, model_artifact: Any) -> Any:
        reference_column = getattr(model_artifact, "reference_column_", "")
        working_df = _drop_reference_column(df, reference_column)
        numeric_df, dropped = _select_numeric_features(working_df)
        if dropped:
            logger.info(f"Dropping non-numeric column(s) before predicting: {dropped}")
        return super().predict(numeric_df, model_artifact)


class _NumericOnlyClusteringCalculatorMixin:
    """Calculator mixin shared by every clustering model: restrict fitting to
    numeric columns (text/id columns aren't clusterable via distance metrics),
    plus an optional named "reference column" the user wants excluded from
    training but kept around (elsewhere) purely for post-hoc interpretation —
    e.g. a species name in the Iris dataset used to check "which cluster is
    which flower" without ever letting the model see it.
    """

    def fit(
        self,
        X: Any,
        y: Any,
        config: dict[str, Any],
        progress_callback=None,
        log_callback=None,
        validation_data=None,
    ) -> Any:
        reference_column = (config or {}).get("reference_column") or ""
        working_X = _drop_reference_column(X, reference_column)
        numeric_X, dropped = _select_numeric_features(working_X)
        if dropped:
            msg = f"Dropping non-numeric column(s) before clustering: {dropped}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        model = super().fit(  # ty: ignore[unresolved-attribute]
            numeric_X,
            y,
            config,
            progress_callback=progress_callback,
            log_callback=log_callback,
            validation_data=validation_data,
        )
        if reference_column:
            # Stashed on the fitted estimator (plain attribute, pickles fine)
            # so the Applier — which only receives `model_artifact`, not the
            # original `config` — still knows which column to exclude later.
            model.reference_column_ = reference_column
        return model


# --- K-Means ---
class KMeansApplier(_NumericOnlyClusteringApplier):
    """K-Means Applier.

    `KMeans.predict()` genuinely supports out-of-sample cluster assignment
    (unlike DBSCAN/Agglomerative, which only implement `fit_predict()` on the
    training data) — this is exactly why K-Means is deployable for inference.
    """


@NodeRegistry.register("kmeans", KMeansApplier)
@node_meta(
    id="kmeans",
    name="K-Means",
    category="Modeling",
    description="Partition rows into a fixed number of clusters (segments) by similarity.",
    params={"n_clusters": 3, "n_init": 10, "random_state": 42},
    tags=["clustering", "requires_scaling"],
)
class KMeansCalculator(_NumericOnlyClusteringCalculatorMixin, SklearnCalculator):
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


# --- Mini-Batch K-Means ---
class MiniBatchKMeansApplier(_NumericOnlyClusteringApplier):
    """Mini-Batch K-Means Applier — supports the same out-of-sample `.predict()` as K-Means."""


@NodeRegistry.register("minibatch_kmeans", MiniBatchKMeansApplier)
@node_meta(
    id="minibatch_kmeans",
    name="Mini-Batch K-Means",
    category="Modeling",
    description=(
        "Faster, approximate variant of K-Means that fits on small random "
        "batches — a good choice for larger datasets."
    ),
    params={"n_clusters": 3, "batch_size": 1024, "n_init": 10, "random_state": 42},
    tags=["clustering", "requires_scaling"],
)
class MiniBatchKMeansCalculator(_NumericOnlyClusteringCalculatorMixin, SklearnCalculator):
    """Mini-Batch K-Means Calculator."""

    def __init__(self):
        super().__init__(
            model_class=MiniBatchKMeans,
            default_params={
                "n_clusters": 3,
                "batch_size": 1024,
                "n_init": 10,
                "random_state": 42,
            },
            problem_type="clustering",
        )


# --- Gaussian Mixture ---
class GaussianMixtureApplier(_NumericOnlyClusteringApplier):
    """Gaussian Mixture Applier.

    `GaussianMixture.predict()` assigns each row to the most likely mixture
    component, so — like K-Means — it supports genuine out-of-sample
    inference. Unlike K-Means, it models each cluster as an (optionally
    non-spherical) Gaussian, so it can capture elongated/overlapping clusters.
    """


@NodeRegistry.register("gaussian_mixture", GaussianMixtureApplier)
@node_meta(
    id="gaussian_mixture",
    name="Gaussian Mixture",
    category="Modeling",
    description=(
        "Probabilistic clustering that models each segment as a Gaussian "
        "distribution — handles elongated/overlapping clusters better than K-Means."
    ),
    params={"n_components": 3, "covariance_type": "full", "random_state": 42},
    tags=["clustering", "requires_scaling"],
)
class GaussianMixtureCalculator(_NumericOnlyClusteringCalculatorMixin, SklearnCalculator):
    """Gaussian Mixture Calculator."""

    def __init__(self):
        super().__init__(
            model_class=GaussianMixture,
            default_params={
                "n_components": 3,
                "covariance_type": "full",
                "random_state": 42,
            },
            problem_type="clustering",
        )


# --- Birch ---
class BirchApplier(_NumericOnlyClusteringApplier):
    """Birch Applier — supports genuine out-of-sample `.predict()`."""


@NodeRegistry.register("birch", BirchApplier)
@node_meta(
    id="birch",
    name="Birch",
    category="Modeling",
    description=(
        "Memory-efficient hierarchical clustering that incrementally builds a "
        "tree summary of the data — well-suited to large datasets."
    ),
    params={"n_clusters": 3, "threshold": 0.5, "branching_factor": 50},
    tags=["clustering", "requires_scaling"],
)
class BirchCalculator(_NumericOnlyClusteringCalculatorMixin, SklearnCalculator):
    """Birch Calculator."""

    def __init__(self):
        super().__init__(
            model_class=Birch,
            default_params={
                "n_clusters": 3,
                "threshold": 0.5,
                "branching_factor": 50,
            },
            problem_type="clustering",
        )
