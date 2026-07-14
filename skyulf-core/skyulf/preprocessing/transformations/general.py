"""General Transformation node (simple ops + fitted power transforms)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ...core.meta.decorators import node_meta
from ...engines import EngineName, get_engine
from ...registry import NodeRegistry
from .._artifacts import GeneralTransformationArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._ops import _PANDAS_OPS, _POLARS_OPS

logger = logging.getLogger(__name__)

_POWER_METHODS = {"box-cox", "yeo-johnson"}


def _apply_power_to_polars_col(X_out: Any, item: dict[str, Any]) -> Any:
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


def _apply_power_to_pandas_col(df_out: Any, item: dict[str, Any]) -> Any:
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


class GeneralTransformationApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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


def _fit_power_for_column(X: Any, col: str, method: str, is_polars: bool) -> dict[str, Any]:
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

    fitted: dict[str, Any] = {"lambdas": pt.lambdas_.tolist()}
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
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Transformations are keyed by source column and replace it in place;
        # column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> GeneralTransformationArtifact:  # pylint: disable=arguments-differ
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'},
        #                              {'column': 'col2', 'method': 'yeo-johnson'}]}
        is_polars = get_engine(X).name == EngineName.POLARS
        fitted_transformations: list[dict[str, Any]] = []

        for item in config.get("transformations", []):
            col = item.get("column")
            method = item.get("method")
            if col not in X.columns:
                continue

            fitted_item: dict[str, Any] = {"column": col, "method": method}

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
