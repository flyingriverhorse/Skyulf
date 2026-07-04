"""Robust scaler node (median centering, IQR scaling)."""

from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.preprocessing import RobustScaler

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import RobustScalerArtifact
from .._helpers import resolve_valid_columns, safe_scale
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _select_subset_pandas, _select_subset_polars


class RobustScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        center = params.get("center")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid:
            return X, _y

        with_centering = params.get("with_centering", True)
        with_scaling = params.get("with_scaling", True)

        exprs = []
        for col_name in valid:
            i = cols.index(col_name)
            e = pl.col(col_name)
            if with_centering and center is not None:
                e = e - center[i]
            if with_scaling and scale is not None:
                s = scale[i]
                e = e / (s if s != 0 else 1.0)
            exprs.append(e.alias(col_name))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        center = params.get("center")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid:
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        vals = X_out[valid].values

        if params.get("with_centering", True) and center is not None:
            vals = vals - np.array(center)[col_indices]
        if params.get("with_scaling", True) and scale is not None:
            vals = vals / safe_scale(np.array(scale)[col_indices].copy())

        X_out[valid] = vals
        return X_out, _y


@NodeRegistry.register("RobustScaler", RobustScalerApplier)
@node_meta(
    id="RobustScaler",
    name="Robust Scaler",
    category="Preprocessing",
    description="Scale features using statistics that are robust to outliers.",
    params={
        "quantile_range": [25.0, 75.0],
        "with_centering": True,
        "with_scaling": True,
        "columns": [],
    },
)
class RobustScalerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> RobustScalerArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return cast(RobustScalerArtifact, {})
        return cast(
            RobustScalerArtifact,
            fit_dual_engine(X, config, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_polars(X, config)
        if not cols:
            return {}
        return _fit_robust(X_subset, cols, config)

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_pandas(X, config)
        if not cols:
            return {}
        return _fit_robust(X_subset, cols, config)


def _fit_robust(X_subset: Any, cols: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    # Same JSON list -> tuple coercion as MinMaxScaler.
    quantile_range = tuple(config.get("quantile_range", (25.0, 75.0)))
    with_centering = config.get("with_centering", True)
    with_scaling = config.get("with_scaling", True)
    scaler = RobustScaler(
        quantile_range=quantile_range,
        with_centering=with_centering,
        with_scaling=with_scaling,
    )
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    scaler.fit(X_np)
    return {
        "type": "robust_scaler",
        "center": scaler.center_.tolist() if scaler.center_ is not None else None,
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
        "quantile_range": quantile_range,
        "with_centering": with_centering,
        "with_scaling": with_scaling,
        "columns": cols,
    }
