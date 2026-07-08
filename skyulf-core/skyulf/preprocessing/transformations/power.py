"""Power Transformer node (Box-Cox / Yeo-Johnson)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import PowerTransformerArtifact
from .._helpers import to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

logger = logging.getLogger(__name__)


def _build_pretrained_power_transformer(
    method: str,
    standardize: bool,
    lambdas_arr: np.ndarray,
    scaler_params: dict[str, Any],
    col_indices: list[int],
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
    X_vals: np.ndarray, params: dict[str, Any], cols: list[str], valid_cols: list[str]
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


def _filter_power_columns(X_pd: pd.DataFrame, cols: list[str], method: str) -> list[str]:
    """Box-Cox requires strictly positive data — drop columns that violate it."""
    if not cols:
        return []
    if method == "box-cox":
        return [c for c in cols if not (X_pd[c] <= 0).any()]
    return cols


def _extract_scaler_params(transformer: PowerTransformer, standardize: bool) -> dict[str, Any]:
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
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Power transforms are applied in place on the same columns.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> PowerTransformerArtifact:  # pylint: disable=arguments-differ
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
