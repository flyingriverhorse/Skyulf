"""Resampling nodes (`Oversampling`, `Undersampling`).

Both Appliers route through :func:`apply_dual_engine`. Imblearn is purely
pandas/numpy-bound, so the Polars path round-trips through pandas (convert in,
run sampler, convert back) while keeping all engine handling out of class
bodies. Sampler construction is split per-method and per-family so each helper
stays at low CCN.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ._artifacts import OversamplingArtifact, UndersamplingArtifact
from ._schema import SkyulfSchema
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


SamplerBuilder = Callable[[str, Dict[str, Any]], Optional[Any]]


def _extract_y_polars(X: Any, y: Any, target_col: Optional[str]) -> Tuple[Any, Any]:
    """When ``y`` is missing, lift it out of the Polars frame using ``target_col``."""
    if y is not None:
        return X, y
    if target_col and target_col in X.columns:
        return X.drop(target_col), X.select(target_col).to_series()
    return X, None


def _extract_y_pandas(X: Any, y: Any, target_col: Optional[str]) -> Tuple[Any, Any]:
    """When ``y`` is missing, lift it out of the Pandas frame using ``target_col``."""
    if y is not None:
        return X, y
    if target_col and target_col in X.columns:
        return X.drop(columns=[target_col]), X[target_col]
    return X, None


def _to_pandas_y(y: Any) -> Any:
    """Best-effort conversion of ``y`` to a pandas Series."""
    if y is None:
        return None
    if hasattr(y, "to_pandas"):
        return y.to_pandas()
    return y


def _validate_numeric(X_pd: pd.DataFrame) -> None:
    """Reject non-numeric feature columns — imblearn requires all-numeric input."""
    non_numeric = X_pd.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(
            f"Resampling requires all features to be numeric. Found non-numeric columns: "
            f"{list(non_numeric)}. Please use an Encoder node "
            "(e.g., OneHotEncoder, OrdinalEncoder) before Resampling."
        )


def _finalize_resampled(
    X_res: Any, y_res: Any, columns: Any, fallback_name: Optional[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Wrap raw imblearn output back into named DataFrame/Series."""
    if not isinstance(X_res, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=columns)
    if not isinstance(y_res, pd.Series):
        y_res = pd.Series(y_res, name=fallback_name)
    return X_res, y_res


def _run_sampler(
    X_pd: pd.DataFrame,
    y_pd: Any,
    params: Dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Run the sampler chosen by ``builder``; return ``None`` if no sampler matches."""
    _validate_numeric(X_pd)
    method = params.get("method", default_method)
    sampler = builder(method, params)
    if sampler is None:
        return None
    X_res, y_res = sampler.fit_resample(X_pd, y_pd)
    fallback_name = getattr(y_pd, "name", None) if y_pd is not None else params.get("target_column")
    return _finalize_resampled(X_res, y_res, X_pd.columns, fallback_name)


def _resample_polars(
    X: Any,
    y: Any,
    params: Dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> Tuple[Any, Any]:
    """Polars apply path: convert → resample → convert back."""
    target_col = params.get("target_column")
    X, y = _extract_y_polars(X, y, target_col)
    if y is None:
        return X, y
    X_pd = X.to_pandas()
    y_pd = _to_pandas_y(y)
    out = _run_sampler(X_pd, y_pd, params, builder, default_method)
    if out is None:
        return X, y
    X_res, y_res = out
    return pl.from_pandas(X_res), pl.from_pandas(y_res)


def _resample_pandas(
    X: Any,
    y: Any,
    params: Dict[str, Any],
    builder: SamplerBuilder,
    default_method: str,
) -> Tuple[Any, Any]:
    """Pandas apply path: resample in place."""
    target_col = params.get("target_column")
    X, y = _extract_y_pandas(X, y, target_col)
    if y is None:
        return X, y
    out = _run_sampler(X, y, params, builder, default_method)
    if out is None:
        return X, y
    return out


# -----------------------------------------------------------------------------
# Oversampling
# -----------------------------------------------------------------------------


def _import_over_samplers() -> Dict[str, Any]:
    """Lazy import of imblearn oversampling classes."""
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import (
            ADASYN,
            SMOTE,
            SVMSMOTE,
            BorderlineSMOTE,
            KMeansSMOTE,
        )
    except ImportError as exc:
        logger.error("imblearn is required for oversampling. `pip install imbalanced-learn`")
        raise ImportError(
            "imblearn is required for oversampling. `pip install imbalanced-learn`"
        ) from exc
    return {
        "smote": SMOTE,
        "adasyn": ADASYN,
        "borderline_smote": BorderlineSMOTE,
        "svm_smote": SVMSMOTE,
        "kmeans_smote": KMeansSMOTE,
        "smote_tomek": SMOTETomek,
    }


def _build_oversampler(method: str, params: Dict[str, Any]) -> Optional[Any]:
    """Construct an over-sampler by ``method`` name; return ``None`` if unknown."""
    classes = _import_over_samplers()
    cls = classes.get(method)
    if cls is None:
        return None

    strategy = params.get("sampling_strategy", "auto")
    random_state = params.get("random_state", 42)
    k_neighbors = params.get("k_neighbors", 5)

    if method == "smote":
        return cls(sampling_strategy=strategy, random_state=random_state, k_neighbors=k_neighbors)
    if method == "adasyn":
        return cls(sampling_strategy=strategy, random_state=random_state, n_neighbors=k_neighbors)
    if method == "borderline_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=params.get("m_neighbors", 10),
            kind=params.get("kind", "borderline-1"),
        )
    if method == "svm_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            m_neighbors=params.get("m_neighbors", 10),
            out_step=params.get("out_step", 0.5),
        )
    if method == "kmeans_smote":
        return cls(
            sampling_strategy=strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            cluster_balance_threshold=params.get("cluster_balance_threshold", 0.1),
            density_exponent=params.get("density_exponent", "auto"),
        )
    # smote_tomek
    return cls(sampling_strategy=strategy, random_state=random_state)


class OversamplingApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=lambda Xi, yi, p: _resample_polars(Xi, yi, p, _build_oversampler, "smote"),
            pandas_func=lambda Xi, yi, p: _resample_pandas(Xi, yi, p, _build_oversampler, "smote"),
        )


@NodeRegistry.register("Oversampling", OversamplingApplier)
@node_meta(
    id="Oversampling",
    name="Oversampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by oversampling minority class.",
    params={"method": "smote", "target_column": "target", "sampling_strategy": "auto"},
)
class OversamplingCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Resampling changes row counts only; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: Dict[str, Any]) -> OversamplingArtifact:
        return {
            "type": "oversampling",
            "method": config.get("method", "smote"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "k_neighbors": config.get("k_neighbors", 5),
            "m_neighbors": config.get("m_neighbors", 10),
            "kind": config.get("kind", "borderline-1"),
            "svm_estimator": config.get("svm_estimator", None),
            "out_step": config.get("out_step", 0.5),
            "kmeans_estimator": config.get("kmeans_estimator", None),
            "cluster_balance_threshold": config.get("cluster_balance_threshold", 0.1),
            "density_exponent": config.get("density_exponent", "auto"),
            "n_jobs": config.get("n_jobs", -1),
        }


# -----------------------------------------------------------------------------
# Undersampling
# -----------------------------------------------------------------------------


def _import_under_samplers() -> Dict[str, Any]:
    """Lazy import of imblearn undersampling classes."""
    try:
        from imblearn.under_sampling import (
            EditedNearestNeighbours,
            NearMiss,
            RandomUnderSampler,
            TomekLinks,
        )
    except ImportError as exc:
        logger.error("imblearn is required for undersampling. `pip install imbalanced-learn`")
        raise ImportError(
            "imblearn is required for undersampling. `pip install imbalanced-learn`"
        ) from exc
    return {
        "random_under_sampling": RandomUnderSampler,
        "nearmiss": NearMiss,
        "tomek_links": TomekLinks,
        "edited_nearest_neighbours": EditedNearestNeighbours,
    }


def _build_undersampler(method: str, params: Dict[str, Any]) -> Optional[Any]:
    """Construct an under-sampler by ``method`` name; return ``None`` if unknown."""
    classes = _import_under_samplers()
    cls = classes.get(method)
    if cls is None:
        return None

    strategy = params.get("sampling_strategy", "auto")

    if method == "random_under_sampling":
        return cls(
            sampling_strategy=strategy,
            random_state=params.get("random_state", 42),
            replacement=params.get("replacement", False),
        )
    if method == "nearmiss":
        return cls(sampling_strategy=strategy, version=params.get("version", 1))
    if method == "tomek_links":
        return cls(sampling_strategy=strategy)
    # edited_nearest_neighbours
    return cls(
        sampling_strategy=strategy,
        n_neighbors=params.get("n_neighbors", 3),
        kind_sel=params.get("kind_sel", "all"),
    )


class UndersamplingApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=lambda Xi, yi, p: _resample_polars(
                Xi, yi, p, _build_undersampler, "random_under_sampling"
            ),
            pandas_func=lambda Xi, yi, p: _resample_pandas(
                Xi, yi, p, _build_undersampler, "random_under_sampling"
            ),
        )


@NodeRegistry.register("Undersampling", UndersamplingApplier)
@node_meta(
    id="Undersampling",
    name="Undersampling",
    category="Preprocessing",
    description="Resample dataset to balance classes by undersampling majority class.",
    params={
        "method": "random_under_sampling",
        "target_column": "target",
        "sampling_strategy": "auto",
    },
)
class UndersamplingCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Resampling changes row counts only; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: Dict[str, Any]) -> UndersamplingArtifact:
        return {
            "type": "undersampling",
            "method": config.get("method", "random_under_sampling"),
            "target_column": config.get("target_column"),
            "sampling_strategy": config.get("sampling_strategy", "auto"),
            "random_state": config.get("random_state", 42),
            "replacement": config.get("replacement", False),
            "version": config.get("version", 1),  # For NearMiss
            "n_neighbors": config.get("n_neighbors", 3),  # For EditedNearestNeighbours
            "kind_sel": config.get("kind_sel", "all"),  # For EditedNearestNeighbours
            "n_jobs": config.get("n_jobs", -1),
        }
