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
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..engines import SkyulfDataFrame, get_engine
from ..engines.sklearn_bridge import SklearnBridge

logger = logging.getLogger(__name__)

# --- Standard Scaler ---


class StandardScalerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
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
            
            X_pl: Any = X

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
            X_out = X_pl.with_columns(exprs)
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas/Numpy Implementation (Legacy)
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        X_out = X_pd.copy()
        
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
@node_meta(
    id="StandardScaler",
    name="Standard Scaler",
    category="Preprocessing",
    description="Standardize features by removing the mean and scaling to unit variance.",
    params={"columns": [], "with_mean": True, "with_std": True}
)
class StandardScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Config: {'with_mean': True, 'with_std': True, 'columns': [...]}
        with_mean = config.get("with_mean", True)
        with_std = config.get("with_std", True)
        
        # Casting for strict type checking
        if engine.name == "polars":
            X_pl: Any = X
            cols = resolve_columns(X_pl, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pl.select(cols)
        else:
            X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
            cols = resolve_columns(X_pd, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pd[cols]

        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # Use Bridge for fitting
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
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
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
            
            X_pl: Any = X

            exprs = []
            for i, col_name in enumerate(cols):
                if col_name in valid_cols:
                    # X * scale + min
                    exprs.append(
                        (pl.col(col_name) * scale[i] + min_val[i]).alias(col_name)
                    )

            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        X_out = X_pd.copy()
        
        min_arr = np.array(min_val)
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]

        vals = X_out[valid_cols].values
        vals = vals * scale_arr[col_indices] + min_arr[col_indices]
        X_out[valid_cols] = vals
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("MinMaxScaler", MinMaxScalerApplier)
@node_meta(
    id="MinMaxScaler",
    name="Min-Max Scaler",
    category="Preprocessing",
    description="Transform features by scaling each feature to a given range.",
    params={"feature_range": [0, 1], "columns": []}
)
class MinMaxScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Config: {'feature_range': (0, 1), 'columns': [...]}
        feature_range = config.get("feature_range", (0, 1))

        if engine.name == "polars":
            X_pl: Any = X
            cols = resolve_columns(X_pl, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pl.select(cols)
        else:
            X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
            cols = resolve_columns(X_pd, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pd[cols]

        scaler = MinMaxScaler(feature_range=feature_range)
        
        # Use Bridge for fitting
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
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
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
            
            X_pl: Any = X

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

            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        X_out = X_pd.copy()
        
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
@node_meta(
    id="RobustScaler",
    name="Robust Scaler",
    category="Preprocessing",
    description="Scale features using statistics that are robust to outliers.",
    params={"quantile_range": [25.0, 75.0], "with_centering": True, "with_scaling": True, "columns": []}
)
class RobustScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Config: {'quantile_range': (25.0, 75.0), 'with_centering': True, 'with_scaling': True, 'columns': [...]}
        quantile_range = config.get("quantile_range", (25.0, 75.0))
        with_centering = config.get("with_centering", True)
        with_scaling = config.get("with_scaling", True)

        if engine.name == "polars":
            X_pl: Any = X
            cols = resolve_columns(X_pl, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pl.select(cols)
        else:
            X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
            cols = resolve_columns(X_pd, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pd[cols]

        scaler = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
        
        # Use Bridge for fitting
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
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
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
            
            X_pl: Any = X

            exprs = []
            for i, col_name in enumerate(cols):
                if col_name in valid_cols:
                    s = scale[i]
                    if s == 0:
                        s = 1.0
                    exprs.append((pl.col(col_name) / s).alias(col_name))

            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
        X_out = X_pd.copy()
        
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
@node_meta(
    id="MaxAbsScaler",
    name="MaxAbs Scaler",
    category="Preprocessing",
    description="Scale each feature by its maximum absolute value.",
    params={"columns": []}
)
class MaxAbsScalerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        if engine.name == "polars":
            X_pl: Any = X
            cols = resolve_columns(X_pl, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pl.select(cols)
        else:
            X_pd: Any = X.to_pandas() if hasattr(X, "to_pandas") else X
            cols = resolve_columns(X_pd, config, detect_numeric_columns)
            if not cols:
                return {}
            X_subset = X_pd[cols]

        scaler = MaxAbsScaler()
        
        # Use Bridge for fitting
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


