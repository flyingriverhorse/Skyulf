"""Scaling nodes (Standard / MinMax / Robust / MaxAbs).

Each Applier/Calculator dispatches on engine via ``apply_dual_engine`` /
``fit_dual_engine`` (see ``dispatcher.py``) so per-engine logic lives in
small, individually testable ``_apply_polars`` / ``_apply_pandas`` /
``_fit_polars`` / ``_fit_pandas`` static helpers. This keeps the public
``apply`` / ``fit`` methods at CCN 1 (Codacy ``lizard_ccn-medium``).
"""

import logging
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ..utils import (
    detect_numeric_columns,
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine, fit_dual_engine
from ._artifacts import (
    MaxAbsScalerArtifact,
    MinMaxScalerArtifact,
    RobustScalerArtifact,
    StandardScalerArtifact,
)
from ._helpers import resolve_valid_columns, safe_scale
from ._schema import SkyulfSchema
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shared fit helpers (engine-specific subset selection)
# -----------------------------------------------------------------------------


def _select_subset_polars(X: Any, config: Dict[str, Any]) -> Tuple[List[str], Any]:
    """Resolve numeric columns and return (cols, X[cols]) for a Polars frame."""
    cols = resolve_columns(X, config, detect_numeric_columns)
    if not cols:
        return [], None
    return cols, X.select(cols)


def _select_subset_pandas(X: Any, config: Dict[str, Any]) -> Tuple[List[str], Any]:
    """Resolve numeric columns and return (cols, X[cols]) for a Pandas frame."""
    cols = resolve_columns(X, config, detect_numeric_columns)
    if not cols:
        return [], None
    return cols, X[cols]


# -----------------------------------------------------------------------------
# Standard Scaler
# -----------------------------------------------------------------------------


class StandardScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        mean = params.get("mean")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or mean is None or scale is None:
            return X, _y

        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        mean_arr = np.array(mean)
        scale_arr = np.array(scale)

        exprs = []
        for col_name in valid:
            idx = cols.index(col_name)
            e = pl.col(col_name)
            if with_mean:
                e = e - mean_arr[idx]
            if with_std:
                s = scale_arr[idx]
                e = e / (s if s != 0 else 1.0)
            exprs.append(e.alias(col_name))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        mean = params.get("mean")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or mean is None or scale is None:
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        vals = X_out[valid].values

        if params.get("with_mean", True):
            vals = vals - np.array(mean)[col_indices]
        if params.get("with_std", True):
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> StandardScalerArtifact:
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


# -----------------------------------------------------------------------------
# MinMax Scaler
# -----------------------------------------------------------------------------


class MinMaxScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> MinMaxScalerArtifact:
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
    feature_range = config.get("feature_range", (0, 1))
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


# -----------------------------------------------------------------------------
# Robust Scaler
# -----------------------------------------------------------------------------


class RobustScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> RobustScalerArtifact:
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
    quantile_range = config.get("quantile_range", (25.0, 75.0))
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


# -----------------------------------------------------------------------------
# MaxAbs Scaler
# -----------------------------------------------------------------------------


class MaxAbsScalerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> MaxAbsScalerArtifact:
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
