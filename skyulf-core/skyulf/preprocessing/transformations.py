"""Transformation nodes (PowerTransformer / SimpleTransformation / GeneralTransformation).

Per-engine apply paths are split into ``_apply_polars`` / ``_apply_pandas``
static helpers and dispatched via ``apply_dual_engine``. The per-method
expression builders for the simple and general transformations live in
module-level dispatch dicts (``_POLARS_OPS`` / ``_PANDAS_OPS``) so each
helper stays at low CCN.
"""

import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry
from ..utils import (
    detect_numeric_columns,
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import (
    GeneralTransformationArtifact,
    PowerTransformerArtifact,
    SimpleTransformationArtifact,
)
from ._helpers import to_pandas
from ._schema import SkyulfSchema
from ..engines import EngineName, SkyulfDataFrame, get_engine

logger = logging.getLogger(__name__)


# =============================================================================
# Power Transformer (Box-Cox / Yeo-Johnson)
# =============================================================================


def _build_pretrained_power_transformer(
    method: str,
    standardize: bool,
    lambdas_arr: np.ndarray,
    scaler_params: Dict[str, Any],
    col_indices: List[int],
    n_total_cols: int,
) -> PowerTransformer:
    """Reconstruct a fitted PowerTransformer from stored lambdas + scaler params."""
    pt = PowerTransformer(method=method, standardize=standardize)
    pt.lambdas_ = lambdas_arr
    if not standardize:
        return pt

    scaler = StandardScaler()
    mean = np.array(scaler_params.get("mean"))
    scale = np.array(scaler_params.get("scale"))
    if len(mean) == n_total_cols:
        mean = mean[col_indices]
    if len(scale) == n_total_cols:
        scale = scale[col_indices]
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = np.square(scale)
    pt._scaler = scaler
    return pt


def _power_transform_array(
    X_vals: np.ndarray, params: Dict[str, Any], cols: List[str], valid_cols: List[str]
) -> np.ndarray:
    """Run the rebuilt PowerTransformer over a numpy array; return transformed array."""
    col_indices = [cols.index(c) for c in valid_cols]
    lambdas_arr = np.array(params["lambdas"])[col_indices]
    pt = _build_pretrained_power_transformer(
        method=params.get("method", "yeo-johnson"),
        standardize=params.get("standardize", True),
        lambdas_arr=lambdas_arr,
        scaler_params=params.get("scaler_params", {}) or {},
        col_indices=col_indices,
        n_total_cols=len(cols),
    )
    X_trans = pt.transform(X_vals)
    # sklearn may be configured with transform_output="pandas".
    return X_trans.to_numpy() if hasattr(X_trans, "to_numpy") else X_trans


def _filter_power_columns(X_pd: pd.DataFrame, cols: List[str], method: str) -> List[str]:
    """Box-Cox requires strictly positive data — drop columns that violate it."""
    if not cols:
        return []
    if method == "box-cox":
        return [c for c in cols if not (X_pd[c] <= 0).any()]
    return cols


def _extract_scaler_params(transformer: PowerTransformer, standardize: bool) -> Dict[str, Any]:
    """Pull mean/scale arrays out of a fitted PowerTransformer's internal scaler."""
    if not standardize:
        return {}
    scaler = getattr(transformer, "_scaler", None)
    if scaler is None:
        return {}
    return {
        "mean": scaler.mean_.tolist() if scaler.mean_ is not None else None,
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else None,
    }


class PowerTransformerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        if params.get("lambdas") is None:
            return X, _y
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return X, _y

        try:
            X_vals = X.select(valid_cols).to_numpy()
            X_trans = _power_transform_array(X_vals, params, cols, valid_cols)
            series = [pl.Series(name, X_trans[:, i]) for i, name in enumerate(valid_cols)]
            return X.with_columns(series), _y
        except Exception as e:
            logger.error(f"PowerTransformer (Polars) application failed: {e}")
            return X, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        if params.get("lambdas") is None:
            return X, _y
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return X, _y

        df_out = X.copy()
        try:
            X_vals = df_out[valid_cols].values
            X_trans = _power_transform_array(X_vals, params, cols, valid_cols)
            df_out.loc[:, valid_cols] = np.asarray(X_trans)
        except Exception as e:
            logger.error(f"PowerTransformer (Pandas) application failed: {e}")
        return df_out, _y


@NodeRegistry.register("PowerTransformer", PowerTransformerApplier)
@node_meta(
    id="PowerTransformer",
    name="Power Transformer",
    category="Preprocessing",
    description="Apply a power transform featurewise to make data more Gaussian-like.",
    params={"method": "yeo-johnson", "standardize": True, "columns": []},
)
class PowerTransformerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Power transforms are applied in place on the same columns.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> PowerTransformerArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        method = config.get("method", "yeo-johnson")
        standardize = config.get("standardize", True)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        valid_cols = _filter_power_columns(X_pd, cols, method)
        if not valid_cols:
            return {}

        transformer = PowerTransformer(method=method, standardize=standardize)
        transformer.fit(X_pd[valid_cols])

        return {
            "type": "power_transformer",
            "lambdas": transformer.lambdas_.tolist(),
            "method": method,
            "standardize": standardize,
            "columns": valid_cols,
            "scaler_params": _extract_scaler_params(transformer, standardize),
        }


# =============================================================================
# Per-method dispatch tables (used by Simple + General transformations)
# =============================================================================


def _polars_log(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).log1p()


def _polars_sqrt(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return pl.when(pl.col(col) < 0).then(None).otherwise(pl.col(col)).sqrt()


def _polars_cbrt(item: Dict[str, Any]) -> Any:
    import polars as pl

    return pl.col(item["column"]).cbrt()


def _polars_reciprocal(item: Dict[str, Any]) -> Any:
    import polars as pl

    col = item["column"]
    return 1.0 / pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col))


def _polars_square(item: Dict[str, Any]) -> Any:
    import polars as pl

    return pl.col(item["column"]).pow(2)


def _polars_exp(item: Dict[str, Any]) -> Any:
    import polars as pl

    threshold = item.get("clip_threshold", 700)
    return pl.col(item["column"]).clip(upper_bound=threshold).exp()


_POLARS_OPS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "log": _polars_log,
    "sqrt": _polars_sqrt,
    "square_root": _polars_sqrt,
    "cube_root": _polars_cbrt,
    "reciprocal": _polars_reciprocal,
    "square": _polars_square,
    "exp": _polars_exp,
    "exponential": _polars_exp,
}


def _pandas_log(series: pd.Series, _item: Dict[str, Any]) -> Any:
    if (series < 0).any():
        series = series.where(series >= 0, np.nan)
    return np.log1p(series)


def _pandas_sqrt(series: pd.Series, _item: Dict[str, Any]) -> Any:
    if (series < 0).any():
        series = series.where(series >= 0, np.nan)
    return np.sqrt(series)


def _pandas_cbrt(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return np.cbrt(series)


def _pandas_reciprocal(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return 1.0 / series.replace(0, np.nan)


def _pandas_square(series: pd.Series, _item: Dict[str, Any]) -> Any:
    return np.square(series)


def _pandas_exp(series: pd.Series, item: Dict[str, Any]) -> Any:
    threshold = item.get("clip_threshold", 700)
    return np.exp(series.clip(upper=threshold))


_PANDAS_OPS: Dict[str, Callable[[pd.Series, Dict[str, Any]], Any]] = {
    "log": _pandas_log,
    "sqrt": _pandas_sqrt,
    "square_root": _pandas_sqrt,
    "cube_root": _pandas_cbrt,
    "reciprocal": _pandas_reciprocal,
    "square": _pandas_square,
    "exp": _pandas_exp,
    "exponential": _pandas_exp,
}


# =============================================================================
# Simple Transformations
# =============================================================================


class SimpleTransformationApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        X_out = X
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in X_out.columns:
                continue
            op = _POLARS_OPS.get(method)
            if op is None:
                continue
            X_out = X_out.with_columns(op(item).alias(col))
        return X_out, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        df_out = X.copy()
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in df_out.columns:
                continue
            op = _PANDAS_OPS.get(method)
            if op is None:
                continue
            df_out[col] = op(pd.to_numeric(df_out[col], errors="coerce"), item)
        return df_out, _y


@NodeRegistry.register("SimpleTransformation", SimpleTransformationApplier)
@node_meta(
    id="SimpleTransformation",
    name="Simple Transformation",
    category="Preprocessing",
    description="Apply simple mathematical transformations (log, sqrt, etc.).",
    params={"func": "log", "columns": []},
)
class SimpleTransformationCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Simple transformations replace values in place; column set is preserved.
        return input_schema

    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> SimpleTransformationArtifact:
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'}, ...]}
        return {
            "type": "simple_transformation",
            "transformations": config.get("transformations", []),
        }


# =============================================================================
# General Transformation (Simple ops + power transforms)
# =============================================================================


def _apply_power_to_polars_col(X_out: Any, item: Dict[str, Any]) -> Any:
    """Apply a fitted Box-Cox / Yeo-Johnson to one Polars column in place."""
    import polars as pl

    col = item["column"]
    method = item["method"]
    lambdas = item.get("lambdas")
    if lambdas is None:
        return X_out

    try:
        pt = PowerTransformer(method=method, standardize=True)
        pt.lambdas_ = np.array(lambdas)
        scaler_params = item.get("scaler_params")
        if scaler_params:
            scaler = StandardScaler()
            m = scaler_params.get("mean")
            s = scaler_params.get("scale")
            if m is not None:
                scaler.mean_ = np.array(m)
            if s is not None:
                scaler.scale_ = np.array(s)
                scaler.var_ = np.square(scaler.scale_)
            pt._scaler = scaler

        vals = X_out[col].to_numpy().reshape(-1, 1)
        flat = pt.transform(vals).ravel()
        return X_out.with_columns(pl.Series(flat).alias(col))
    except Exception as e:
        logger.warning(f"Failed to apply {method} for column {col}: {e}")
        return X_out


def _apply_power_to_pandas_col(df_out: Any, item: Dict[str, Any]) -> Any:
    """Apply a fitted Box-Cox / Yeo-Johnson to one Pandas column in place."""
    col = item["column"]
    method = item["method"]
    lambdas = item.get("lambdas")
    if lambdas is None:
        return df_out

    try:
        pt = PowerTransformer(method=method, standardize=True)
        pt.lambdas_ = np.array(lambdas)
        scaler_params = item.get("scaler_params")
        if scaler_params:
            scaler = StandardScaler()
            scaler.mean_ = np.array(scaler_params.get("mean"))
            scaler.scale_ = np.array(scaler_params.get("scale"))
            scaler.var_ = np.square(scaler.scale_)
            pt._scaler = scaler

        series = pd.to_numeric(df_out[col], errors="coerce")
        vals = series.values.reshape(-1, 1)
        trans_vals = pt.transform(vals)
        # sklearn may be configured with transform_output="pandas".
        trans_arr = trans_vals.to_numpy() if hasattr(trans_vals, "to_numpy") else trans_vals
        df_out[col] = np.asarray(trans_arr).ravel()
    except Exception as e:
        logger.warning(f"Failed to apply {method} for column {col}: {e}")
    return df_out


_POWER_METHODS = {"box-cox", "yeo-johnson"}


class GeneralTransformationApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        X_out = X
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in X_out.columns:
                continue
            if method in _POWER_METHODS:
                X_out = _apply_power_to_polars_col(X_out, item)
                continue
            op = _POLARS_OPS.get(method)
            if op is None:
                continue
            X_out = X_out.with_columns(op(item).alias(col))
        return X_out, _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        transformations = params.get("transformations", [])
        if not transformations:
            return X, _y

        df_out = X.copy()
        for item in transformations:
            col = item.get("column")
            method = item.get("method")
            if col not in df_out.columns:
                continue
            if method in _POWER_METHODS:
                df_out = _apply_power_to_pandas_col(df_out, item)
                continue
            op = _PANDAS_OPS.get(method)
            if op is None:
                continue
            df_out[col] = op(pd.to_numeric(df_out[col], errors="coerce"), item)
        return df_out, _y


# -----------------------------------------------------------------------------
# General Transformation Calculator
# -----------------------------------------------------------------------------


def _fit_power_for_column(X: Any, col: str, method: str, is_polars: bool) -> Dict[str, Any]:
    """Fit a PowerTransformer for one column; return the per-column artifact dict."""
    if is_polars:
        col_series = X[col].to_pandas()
        col_df = col_series.to_frame()
    else:
        col_series = X[col]
        col_df = X[[col]]

    if method == "box-cox" and (col_series <= 0).any():
        logger.warning(
            f"Skipping Box-Cox for column {col} because it contains non-positive values."
        )
        return {}

    pt = PowerTransformer(method=method, standardize=True)
    pt.fit(col_df)

    fitted: Dict[str, Any] = {"lambdas": pt.lambdas_.tolist()}
    if hasattr(pt, "_scaler") and pt._scaler:
        fitted["scaler_params"] = {
            "mean": pt._scaler.mean_.tolist() if pt._scaler.mean_ is not None else None,
            "scale": pt._scaler.scale_.tolist() if pt._scaler.scale_ is not None else None,
        }
    return fitted


@NodeRegistry.register("GeneralTransformation", GeneralTransformationApplier)
@node_meta(
    id="GeneralTransformation",
    name="General Transformation",
    category="Preprocessing",
    description="Apply various function transformations (log, sqrt, square, exp) to columns.",
    params={"transformations": []},
)
class GeneralTransformationCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Transformations are keyed by source column and replace it in place;
        # column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> GeneralTransformationArtifact:
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'},
        #                              {'column': 'col2', 'method': 'yeo-johnson'}]}
        is_polars = get_engine(X).name == EngineName.POLARS
        fitted_transformations: List[Dict[str, Any]] = []

        for item in config.get("transformations", []):
            col = item.get("column")
            method = item.get("method")
            if col not in X.columns:
                continue

            fitted_item: Dict[str, Any] = {"column": col, "method": method}

            if method in _POWER_METHODS:
                try:
                    extras = _fit_power_for_column(X, col, method, is_polars)
                except Exception as e:
                    logger.warning(f"Failed to fit {method} for column {col}: {e}")
                    continue
                if not extras:
                    continue  # box-cox skipped on non-positive data
                fitted_item.update(extras)

            fitted_transformations.append(fitted_item)

        return {
            "type": "general_transformation",
            "transformations": fitted_transformations,
        }
