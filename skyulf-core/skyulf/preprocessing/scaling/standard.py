"""Standard scaler node (zero mean, unit variance)."""

import logging
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.preprocessing import StandardScaler

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import StandardScalerArtifact
from .._helpers import resolve_valid_columns, safe_scale
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _select_subset_pandas, _select_subset_polars

logger = logging.getLogger(__name__)


def _needs_fitted_artifact(
    valid: List[str], with_mean: bool, mean: Any, with_std: bool, scale: Any
) -> bool:
    """True when there's nothing to scale, or a flag's required artifact is missing.

    sklearn's ``StandardScaler`` leaves ``mean_``/``scale_`` as ``None``
    whenever the corresponding ``with_mean``/``with_std`` flag was ``False``
    at fit time, so only the artifact a flag actually needs is required here
    — requiring both unconditionally would wrongly skip mean-centering for a
    valid ``with_mean=True, with_std=False`` configuration.
    """
    if not valid:
        return True
    if with_mean and mean is None:
        return True
    return with_std and scale is None


class StandardScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        mean = params.get("mean")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        if _needs_fitted_artifact(valid, with_mean, mean, with_std, scale):
            return X, _y

        mean_arr = np.array(mean) if mean is not None else np.zeros(len(cols))
        scale_arr = np.array(scale) if scale is not None else np.ones(len(cols))

        def _standardized_expr(col_name: str) -> Any:
            idx = cols.index(col_name)
            e = pl.col(col_name)
            if with_mean:
                e = e - mean_arr[idx]
            if with_std:
                s = scale_arr[idx]
                e = e / (s if s != 0 else 1.0)
            return e.alias(col_name)

        exprs = [_standardized_expr(col_name) for col_name in valid]
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        mean = params.get("mean")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        if _needs_fitted_artifact(valid, with_mean, mean, with_std, scale):
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        vals = X_out[valid].values
        if with_mean:
            vals = vals - np.array(mean)[col_indices]
        if with_std:
            vals = vals / safe_scale(np.array(scale)[col_indices].copy())

        X_out[valid] = vals
        return X_out, _y


@NodeRegistry.register("StandardScaler", StandardScalerApplier)
@node_meta(
    id="StandardScaler",
    name="Standard Scaler",
    category="Preprocessing",
    description="Standardize features by removing the mean and scaling to unit variance.",
    params={"columns": [], "with_mean": True, "with_std": True},
)
class StandardScalerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Scalers preserve column set; values change but names/order do not.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> StandardScalerArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return cast(StandardScalerArtifact, {})
        return cast(
            StandardScalerArtifact,
            fit_dual_engine(X, config, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_polars(X, config)
        if not cols:
            return {}
        return _fit_standard(X_subset, cols, config)

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_pandas(X, config)
        if not cols:
            return {}
        return _fit_standard(X_subset, cols, config)


def _fit_standard(X_subset: Any, cols: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    with_mean = config.get("with_mean", True)
    with_std = config.get("with_std", True)
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    scaler.fit(X_np)

    # sklearn stubs widen mean_/var_ to include int|float; cast to keep .tolist().
    mean_ = cast(Any, scaler.mean_)
    var_ = cast(Any, scaler.var_)
    return {
        "type": "standard_scaler",
        "mean": mean_.tolist() if hasattr(mean_, "tolist") else mean_,
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
        "var": var_.tolist() if hasattr(var_, "tolist") else var_,
        "with_mean": with_mean,
        "with_std": with_std,
        "columns": cols,
    }
