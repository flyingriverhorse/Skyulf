import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)

# --- Standard Scaler ---


class StandardScalerApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)

        cols = params.get("columns", [])
        mean = params.get("mean")
        scale = params.get("scale")

        # Check valid cols (works for both Pandas and Polars/Wrapper)
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or mean is None or scale is None:
            return pack_pipeline_output(X, y, is_tuple)

        # Check Engine
        engine = get_engine(X)
        
        if engine.__name__ == "PolarsEngine":
            import polars as pl
            # Polars Native Implementation
            mean_arr = np.array(mean)
            scale_arr = np.array(scale)
            col_indices = [cols.index(c) for c in valid_cols]
            
            exprs = []
            for idx, col_name in zip(col_indices, valid_cols):
                e = pl.col(col_name)
                if params.get("with_mean", True):
                    e = e - mean_arr[idx]
                if params.get("with_std", True):
                    s = scale_arr[idx]
                    s = s if s != 0 else 1.0
                    e = e / s
                exprs.append(e)
            
            # Apply transformations
            # X is Polars DataFrame or Wrapper
            if hasattr(X, "with_columns"):
                X_out = X.with_columns(exprs)
            else:
                # Should be wrapper or raw polars
                X_out = X.with_columns(exprs)
                
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas/Numpy Implementation (Legacy)
        X_out = X.copy()
        mean_arr = np.array(mean)
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]

        vals = X_out[valid_cols].values
        if params.get("with_mean", True):
            vals = vals - mean_arr[col_indices]
        if params.get("with_std", True):
            safe_scale = scale_arr[col_indices]
            safe_scale[safe_scale == 0] = 1.0
            vals = vals / safe_scale

        X_out[valid_cols] = vals
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("StandardScaler", StandardScalerApplier)
class StandardScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'with_mean': True, 'with_std': True, 'columns': [...]}
        with_mean = config.get("with_mean", True)
        with_std = config.get("with_std", True)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # Use Bridge for fitting
        X_subset = X.select(cols) if hasattr(X, "select") else X[cols]
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        
        scaler.fit(X_np)

        return {
            "type": "standard_scaler",
            "mean": scaler.mean_.tolist() if scaler.mean_ is not None else None,
            "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
            "var": scaler.var_.tolist() if scaler.var_ is not None else None,
            "with_mean": with_mean,
            "with_std": with_std,
            "columns": cols,
        }


# --- MinMax Scaler ---


class MinMaxScalerApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        min_val = params.get("min")
        scale = params.get("scale")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or min_val is None or scale is None:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl

            exprs = []
            for i, col_name in enumerate(cols):
                if col_name in valid_cols:
                    # X * scale + min
                    exprs.append(
                        (pl.col(col_name) * scale[i] + min_val[i]).alias(col_name)
                    )

            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()
        min_arr = np.array(min_val)
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]

        vals = X_out[valid_cols].values
        vals = vals * scale_arr[col_indices] + min_arr[col_indices]
        X_out[valid_cols] = vals
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("MinMaxScaler", MinMaxScalerApplier)
class MinMaxScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'feature_range': (0, 1), 'columns': [...]}
        feature_range = config.get("feature_range", (0, 1))

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        scaler = MinMaxScaler(feature_range=feature_range)
        
        # Use Bridge for fitting
        X_subset = X.select(cols) if hasattr(X, "select") else X[cols]
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


# --- Robust Scaler ---


class RobustScalerApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        center = params.get("center")
        scale = params.get("scale")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl

            exprs = []
            for i, col_name in enumerate(cols):
                if col_name in valid_cols:
                    expr = pl.col(col_name)

                    if params.get("with_centering", True) and center is not None:
                        expr = expr - center[i]

                    if params.get("with_scaling", True) and scale is not None:
                        s = scale[i]
                        if s == 0:
                            s = 1.0
                        expr = expr / s

                    exprs.append(expr.alias(col_name))

            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid_cols]
        vals = X_out[valid_cols].values

        if params.get("with_centering", True) and center is not None:
            center_arr = np.array(center)
            vals = vals - center_arr[col_indices]

        if params.get("with_scaling", True) and scale is not None:
            scale_arr = np.array(scale)
            # Avoid division by zero
            safe_scale = scale_arr[col_indices]
            safe_scale[safe_scale == 0] = 1.0
            vals = vals / safe_scale

        X_out[valid_cols] = vals
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("RobustScaler", RobustScalerApplier)
class RobustScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'quantile_range': (25.0, 75.0), 'with_centering': True, 'with_scaling': True, 'columns': [...]}
        quantile_range = config.get("quantile_range", (25.0, 75.0))
        with_centering = config.get("with_centering", True)
        with_scaling = config.get("with_scaling", True)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        scaler = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
        
        # Use Bridge for fitting
        X_subset = X.select(cols) if hasattr(X, "select") else X[cols]
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


# --- MaxAbs Scaler ---


class MaxAbsScalerApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        scale = params.get("scale")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or scale is None:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl

            exprs = []
            for i, col_name in enumerate(cols):
                if col_name in valid_cols:
                    s = scale[i]
                    if s == 0:
                        s = 1.0
                    exprs.append((pl.col(col_name) / s).alias(col_name))

            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]

        vals = X_out[valid_cols].values
        # Avoid division by zero
        safe_scale = scale_arr[col_indices]
        safe_scale[safe_scale == 0] = 1.0
        vals = vals / safe_scale

        X_out[valid_cols] = vals
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("MaxAbsScaler", MaxAbsScalerApplier)
class MaxAbsScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        scaler = MaxAbsScaler()
        
        # Use Bridge for fitting
        X_subset = X.select(cols) if hasattr(X, "select") else X[cols]
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        
        scaler.fit(X_np)

        return {
            "type": "maxabs_scaler",
            "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
            "max_abs": (
                scaler.max_abs_.tolist() if scaler.max_abs_ is not None else None
            ),
            "columns": cols,
        }


