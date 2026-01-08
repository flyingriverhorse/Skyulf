import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..utils import (
    detect_numeric_columns,
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
)
from .base import BaseApplier, BaseCalculator
from ..engines import SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)

# --- Power Transformer (Box-Cox, Yeo-Johnson) ---


class PowerTransformerApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)
        was_polars = engine.name == "polars"

        cols = params.get("columns", [])
        lambdas = params.get("lambdas")
        method = params.get("method", "yeo-johnson")
        standardize = params.get("standardize", True)
        scaler_params = params.get("scaler_params", {})

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or lambdas is None:
            return pack_pipeline_output(X, y, is_tuple)

        if was_polars:
            import polars as pl
            X_pd = X.to_pandas()
        else:
            X_pd = X

        df_out = X_pd.copy()

        # Reconstruct PowerTransformer state for application
        # We manually restore the lambdas and internal scaler to apply the transform
        # without re-fitting.

        X_vals = df_out[valid_cols].values
        # Filter lambdas and scaler params to match valid_cols
        col_indices = [cols.index(c) for c in valid_cols]
        lambdas_arr = np.array(lambdas)[col_indices]

        # 1. Power Transform

        try:
            pt = PowerTransformer(method=method, standardize=standardize)
            pt.lambdas_ = lambdas_arr

            if standardize:
                scaler = StandardScaler()

                mean = np.array(scaler_params.get("mean"))
                scale = np.array(scaler_params.get("scale"))

                if len(mean) == len(cols):
                    mean = mean[col_indices]
                if len(scale) == len(cols):
                    scale = scale[col_indices]

                scaler.mean_ = mean
                scaler.scale_ = scale
                scaler.var_ = np.square(scaler.scale_)  # Approximate if not stored
                pt._scaler = scaler

            # We need to trick sklearn into thinking it's fitted
            # Usually setting attributes is enough, but let's see.
            # PowerTransformer checks hasattr(self, "lambdas_")

            X_trans = pt.transform(X_vals)
            # sklearn can be configured with transform_output="pandas", which returns a DataFrame.
            X_trans_arr = X_trans.to_numpy() if hasattr(X_trans, "to_numpy") else X_trans
            df_out.loc[:, valid_cols] = np.asarray(X_trans_arr)

        except Exception as e:
            logger.error(f"PowerTransformer application failed: {e}")
            # Fallback?
            pass

        if was_polars:
            df_out = pl.from_pandas(df_out)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("PowerTransformer", PowerTransformerApplier)
@node_meta(
    id="PowerTransformer",
    name="Power Transformer",
    category="Preprocessing",
    description="Apply a power transform featurewise to make data more Gaussian-like.",
    params={"method": "yeo-johnson", "standardize": True, "columns": []}
)
class PowerTransformerCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        if engine.name == "polars":
            X = X.to_pandas()

        # Config: {'method': 'yeo-johnson' | 'box-cox', 'standardize': True, 'columns': [...]}
        method = config.get("method", "yeo-johnson")
        standardize = config.get("standardize", True)

        cols = resolve_columns(X, config, detect_numeric_columns)

        if not cols:
            return {}

        valid_cols = []
        if method == "box-cox":
            for col in cols:
                # Box-Cox requires strictly positive data
                if (X[col] <= 0).any():
                    continue
                valid_cols.append(col)
        else:
            valid_cols = cols

        if not valid_cols:
            return {}

        transformer = PowerTransformer(method=method, standardize=standardize)
        transformer.fit(X[valid_cols])

        # Capture internal scaler parameters if standardization is enabled
        scaler_params = {}
        if standardize and hasattr(transformer, "_scaler"):
            scaler = transformer._scaler
            if scaler:
                scaler_params = {
                    "mean": scaler.mean_.tolist() if scaler.mean_ is not None else None,
                    "scale": (
                        scaler.scale_.tolist() if scaler.scale_ is not None else None
                    ),
                }

        return {
            "type": "power_transformer",
            "lambdas": transformer.lambdas_.tolist(),
            "method": method,
            "standardize": standardize,
            "columns": valid_cols,
            "scaler_params": scaler_params,
        }


# --- Simple Transformations (Log, Sqrt, etc.) ---


class SimpleTransformationApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        transformations = params.get("transformations", [])
        if not transformations:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X
            X_out = X_pl

            for item in transformations:
                col = item.get("column")
                method = item.get("method")
                if col not in X_out.columns:
                    continue
                
                expr = pl.col(col)
                
                if method == "log":
                    expr = pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).log1p()
                elif method == "square_root":
                    # sqrt of neg is nan
                    expr = pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).sqrt()
                elif method == "cube_root":
                    expr = pl.col(col).cbrt()
                elif method == "reciprocal":
                    # 1/0 is inf in polars, typically nan in pandas logic above (replace(0, nan))
                    expr = 1.0 / pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col))
                elif method == "square":
                    expr = pl.col(col).pow(2)
                elif method == "exponential":
                    threshold = item.get("clip_threshold", 700)
                    expr = pl.col(col).clip(upper_bound=threshold).exp()
                
                X_out = X_out.with_columns(expr.alias(col))
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

        for item in transformations:
            col = item.get("column")
            method = item.get("method")

            if col not in df_out.columns:
                continue

            series = pd.to_numeric(df_out[col], errors="coerce")

            if method == "log":
                # log1p is safer for zeros
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.log1p(series)

            elif method == "square_root":
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.sqrt(series)

            elif method == "cube_root":
                df_out[col] = np.cbrt(series)

            elif method == "reciprocal":
                df_out[col] = 1.0 / series.replace(0, np.nan)

            elif method == "square":
                df_out[col] = np.square(series)

            elif method == "exponential":
                # Clip to avoid overflow (exp(709) ~ max float64)
                # We use a slightly lower bound to be safe, or user provided threshold
                threshold = item.get("clip_threshold", 700)
                series_clipped = series.clip(upper=threshold)
                df_out[col] = np.exp(series_clipped)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("SimpleTransformation", SimpleTransformationApplier)
@node_meta(
    id="SimpleTransformation",
    name="Simple Transformation",
    category="Preprocessing",
    description="Apply simple mathematical transformations (log, sqrt, etc.).",
    params={"func": "log", "columns": []}
)
class SimpleTransformationCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'}, ...]}
        return {
            "type": "simple_transformation",
            "transformations": params.get("transformations", []),
        }


# --- General Transformation (Combined) ---


class GeneralTransformationApplier(BaseApplier):
    def apply(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        transformations = params.get("transformations", [])
        if not transformations:
            return pack_pipeline_output(X, y, is_tuple)
        
        # Polars Path
        if engine.name == "polars":
            import polars as pl
            X_pl: Any = X
            X_out = X_pl
            
            for item in transformations:
                col = item.get("column")
                method = item.get("method")
                
                if col not in X_out.columns:
                    continue

                if method in ["box-cox", "yeo-johnson"]:
                    lambdas = item.get("lambdas")
                    scaler_params = item.get("scaler_params")
                    if lambdas is None:
                        continue
                    
                    try:
                        pt = PowerTransformer(method=method, standardize=True)
                        pt.lambdas_ = np.array(lambdas)
                        
                        if scaler_params:
                            scaler = StandardScaler()
                            # Handle potential None or list
                            m = scaler_params.get("mean")
                            s = scaler_params.get("scale")
                            if m is not None:
                                scaler.mean_ = np.array(m)
                            if s is not None:
                                scaler.scale_ = np.array(s)
                                scaler.var_ = np.square(scaler.scale_)
                            pt._scaler = scaler
                        
                        # Get numpy array from polars col
                        vals = X_out[col].to_numpy().reshape(-1, 1)
                        trans_vals = pt.transform(vals)
                        # flatten
                        flat_vals = trans_vals.ravel()
                        
                        X_out = X_out.with_columns(pl.Series(flat_vals).alias(col))
                        
                    except Exception as e:
                        logger.warning(f"Failed to apply {method} for column {col}: {e}")
                
                # Simple Transformations (Polars native)
                else: 
                    expr = pl.col(col)
                    if method == "log":
                        expr = pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).log1p()
                    elif method in ["sqrt", "square_root"]:
                        expr = pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).sqrt()
                    elif method == "cube_root":
                        expr = pl.col(col).cbrt()
                    elif method == "reciprocal":
                        expr = 1.0 / pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col))
                    elif method == "square":
                        expr = pl.col(col).pow(2)
                    elif method in ["exp", "exponential"]:
                        threshold = item.get("clip_threshold", 700)
                        expr = pl.col(col).clip(upper_bound=threshold).exp()
                    
                    X_out = X_out.with_columns(expr.alias(col))
            
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

        for item in transformations:
            col = item.get("column")
            method = item.get("method")

            if col not in df_out.columns:
                continue

            series = pd.to_numeric(df_out[col], errors="coerce")

            if method in ["box-cox", "yeo-johnson"]:
                lambdas = item.get("lambdas")
                scaler_params = item.get("scaler_params")

                if lambdas is None:
                    continue

                try:
                    pt = PowerTransformer(method=method, standardize=True)
                    pt.lambdas_ = np.array(lambdas)

                    if scaler_params:
                        scaler = StandardScaler()
                        scaler.mean_ = np.array(scaler_params.get("mean"))
                        scaler.scale_ = np.array(scaler_params.get("scale"))
                        scaler.var_ = np.square(scaler.scale_)
                        pt._scaler = scaler

                    # Reshape for sklearn
                    vals = series.values.reshape(-1, 1)
                    trans_vals = pt.transform(vals)
                    # sklearn can be configured with transform_output="pandas", which returns a DataFrame.
                    trans_vals_arr = (
                        trans_vals.to_numpy() if hasattr(trans_vals, "to_numpy") else trans_vals
                    )
                    df_out[col] = np.asarray(trans_vals_arr).ravel()
                except Exception as e:
                    logger.warning(f"Failed to apply {method} for column {col}: {e}")

            elif method == "log":
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.log1p(series)
            elif method == "sqrt" or method == "square_root":
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.sqrt(series)
            elif method == "cube_root":
                df_out[col] = np.cbrt(series)
            elif method == "reciprocal":
                df_out[col] = 1.0 / series.replace(0, np.nan)
            elif method == "square":
                df_out[col] = np.square(series)
            elif method == "exp" or method == "exponential":
                threshold = item.get("clip_threshold", 700)
                series_clipped = series.clip(upper=threshold)
                df_out[col] = np.exp(series_clipped)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("GeneralTransformation", GeneralTransformationApplier)
@node_meta(
    id="GeneralTransformation",
    name="General Transformation",
    category="Preprocessing",
    description="Apply various function transformations (log, sqrt, square, exp) to columns.",
    params={"transformations": []}
)
class GeneralTransformationCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'},
        #                              {'column': 'col2', 'method': 'yeo-johnson'}]}
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        transformations_config = config.get("transformations", [])
        fitted_transformations = []

        for item in transformations_config:
            col = item.get("column")
            method = item.get("method")

            if col not in X.columns:
                continue

            fitted_item = {"column": col, "method": method}

            if method in ["box-cox", "yeo-johnson"]:
                # Fit PowerTransformer
                try:
                    # Prepare data (Pandas Series/DataFrame)
                    if engine.name == "polars":
                        col_series = X[col].to_pandas()
                        col_df = col_series.to_frame()
                    else:
                        col_series = X[col]
                        col_df = X[[col]]

                    # Box-Cox requires strictly positive
                    if method == "box-cox" and (col_series <= 0).any():
                        logger.warning(
                            f"Skipping Box-Cox for column {col} because it contains non-positive values."
                        )
                        continue

                    # Default to standardize=True for power transforms
                    pt = PowerTransformer(method=method, standardize=True)
                    pt.fit(col_df)

                    fitted_item["lambdas"] = pt.lambdas_.tolist()

                    if hasattr(pt, "_scaler") and pt._scaler:
                        fitted_item["scaler_params"] = {
                            "mean": (
                                pt._scaler.mean_.tolist()
                                if pt._scaler.mean_ is not None
                                else None
                            ),
                            "scale": (
                                pt._scaler.scale_.tolist()
                                if pt._scaler.scale_ is not None
                                else None
                            ),
                        }
                except Exception as e:
                    logger.warning(f"Failed to fit {method} for column {col}: {e}")
                    continue

            fitted_transformations.append(fitted_item)

        return {
            "type": "general_transformation",
            "transformations": fitted_transformations,
        }
