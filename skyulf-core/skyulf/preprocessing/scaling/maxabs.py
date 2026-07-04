"""MaxAbs scaler node (scale by maximum absolute value)."""

from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.preprocessing import MaxAbsScaler

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import MaxAbsScalerArtifact
from .._helpers import resolve_valid_columns, safe_scale
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _select_subset_pandas, _select_subset_polars


class MaxAbsScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or scale is None:
            return X, _y

        exprs = []
        for col_name in valid:
            s = scale[cols.index(col_name)]
            exprs.append((pl.col(col_name) / (s if s != 0 else 1.0)).alias(col_name))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or scale is None:
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        vals = X_out[valid].values
        vals = vals / safe_scale(np.array(scale)[col_indices].copy())
        X_out[valid] = vals
        return X_out, _y


@NodeRegistry.register("MaxAbsScaler", MaxAbsScalerApplier)
@node_meta(
    id="MaxAbsScaler",
    name="MaxAbs Scaler",
    category="Preprocessing",
    description="Scale each feature by its maximum absolute value.",
    params={"columns": []},
)
class MaxAbsScalerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> MaxAbsScalerArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return cast(MaxAbsScalerArtifact, {})
        return cast(
            MaxAbsScalerArtifact,
            fit_dual_engine(X, config, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_polars(X, config)
        if not cols:
            return {}
        return _fit_maxabs(X_subset, cols)

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_pandas(X, config)
        if not cols:
            return {}
        return _fit_maxabs(X_subset, cols)


def _fit_maxabs(X_subset: Any, cols: List[str]) -> Dict[str, Any]:
    scaler = MaxAbsScaler()
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    scaler.fit(X_np)
    return {
        "type": "maxabs_scaler",
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
        "max_abs": (scaler.max_abs_.tolist() if scaler.max_abs_ is not None else None),
        "columns": cols,
    }
