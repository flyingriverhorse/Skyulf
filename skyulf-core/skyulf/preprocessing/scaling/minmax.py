"""Min-max scaler node (scale features into a given range)."""

from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import MinMaxScalerArtifact
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _select_subset_pandas, _select_subset_polars


class MinMaxScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        min_val = params.get("min")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or min_val is None or scale is None:
            return X, _y

        exprs = [
            (pl.col(c) * scale[cols.index(c)] + min_val[cols.index(c)]).alias(c) for c in valid
        ]
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        min_val = params.get("min")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or min_val is None or scale is None:
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        vals = X_out[valid].values
        vals = vals * np.array(scale)[col_indices] + np.array(min_val)[col_indices]
        X_out[valid] = vals
        return X_out, _y


@NodeRegistry.register("MinMaxScaler", MinMaxScalerApplier)
@node_meta(
    id="MinMaxScaler",
    name="Min-Max Scaler",
    category="Preprocessing",
    description="Transform features by scaling each feature to a given range.",
    params={"feature_range": [0, 1], "columns": []},
)
class MinMaxScalerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> MinMaxScalerArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return cast(MinMaxScalerArtifact, {})
        return cast(
            MinMaxScalerArtifact,
            fit_dual_engine(X, config, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_polars(X, config)
        if not cols:
            return {}
        return _fit_minmax(X_subset, cols, config)

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        cols, X_subset = _select_subset_pandas(X, config)
        if not cols:
            return {}
        return _fit_minmax(X_subset, cols, config)


def _fit_minmax(X_subset: Any, cols: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    # `feature_range` may arrive as a JSON-loaded list (e.g. ``[0, 1]``) from the
    # frontend or pipeline config; sklearn enforces ``tuple`` via param validation.
    feature_range = tuple(config.get("feature_range", (0, 1)))
    scaler = MinMaxScaler(feature_range=feature_range)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    scaler.fit(X_np)
    return {
        "type": "minmax_scaler",
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "feature_range": feature_range,
        "columns": cols,
    }
