"""General Transformation node (simple ops + fitted power transforms)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import PowerTransformer

from ...core.meta.decorators import node_meta
from ...engines import EngineName, get_engine
from ...registry import NodeRegistry
from .._artifacts import GeneralTransformationArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._ops import _PANDAS_OPS, _POLARS_OPS
from ._power_common import build_pretrained_power_transformer

logger = logging.getLogger(__name__)

_POWER_METHODS = {"box-cox", "yeo-johnson"}
_FLOAT_METHODS = {
    "log",
    "sqrt",
    "square_root",
    "cube_root",
    "reciprocal",
    "exp",
    "exponential",
    *_POWER_METHODS,
}


def _apply_power_to_polars_col(X_out: Any, item: dict[str, Any]) -> Any:
    """Apply a fitted Box-Cox / Yeo-Johnson to one Polars column in place."""
    col = item["column"]
    method = item["method"]
    lambdas = item.get("lambdas")
    if lambdas is None:
        return X_out

    try:
        pt = build_pretrained_power_transformer(
            method=method,
            standardize=item.get("standardize", True),
            lambdas_arr=np.array(lambdas),
            scaler_params=item.get("scaler_params"),
        )
        vals = X_out[col].to_numpy().reshape(-1, 1)
        flat = pt.transform(vals).ravel()
        return X_out.with_columns(pl.Series(flat).alias(col))
    except Exception as e:  # noqa: BLE001 - per-column transform failure is logged; column left unchanged
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
        pt = build_pretrained_power_transformer(
            method=method,
            standardize=item.get("standardize", True),
            lambdas_arr=np.array(lambdas),
            scaler_params=item.get("scaler_params"),
        )
        series = pd.to_numeric(df_out[col], errors="coerce")
        vals = series.to_numpy().reshape(-1, 1)
        trans_vals = pt.transform(vals)
        # sklearn may be configured with transform_output="pandas".
        trans_arr = trans_vals.to_numpy() if hasattr(trans_vals, "to_numpy") else trans_vals
        df_out[col] = np.asarray(trans_arr).ravel()
    except Exception as e:  # noqa: BLE001 - per-column transform failure is logged; column left unchanged
        logger.warning(f"Failed to apply {method} for column {col}: {e}")
    return df_out


class GeneralTransformationApplier(BaseApplier):
    """Apply per-column simple ops or fitted Box-Cox/Yeo-Johnson transforms."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Run each configured transformation on the active engine; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

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


def _fit_power_for_column(
    X: Any, col: str, method: str, is_polars: bool, standardize: bool = True
) -> dict[str, Any]:
    """Fit a PowerTransformer for one column; return the per-column artifact dict."""
    if is_polars:
        col_values = X[col].to_numpy()
        col_df = col_values.reshape(-1, 1)
    else:
        col_values = X[col]
        col_df = X[[col]]

    if method == "box-cox" and (col_values <= 0).any():
        logger.warning(
            f"Skipping Box-Cox for column {col} because it contains non-positive values."
        )
        return {}

    pt = PowerTransformer(method=method, standardize=standardize)
    pt.fit(col_df)

    fitted: dict[str, Any] = {"lambdas": pt.lambdas_.tolist(), "standardize": standardize}
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
    learns_from_data=True,
)
class GeneralTransformationCalculator(BaseCalculator):
    """Resolve configured transformations, fitting lambdas for the power methods.

    Rules run in list order. Each power rule is fitted on the training values
    produced by earlier rules on the same column, matching artifact replay.
    Each power rule accepts ``standardize`` (default ``True``). Set it to
    ``False`` to apply only the fitted Box-Cox or Yeo-Johnson transform.
    The artifact retains the choice for subsequent transforms.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return updated dtypes for transformed numeric columns."""
        schema = input_schema
        touched: set[str] = set()
        for item in config.get("transformations", []):
            col = item.get("column")
            method = item.get("method")
            if col not in schema.columns or method is None:
                continue
            if method == "square":
                continue
            if method in _FLOAT_METHODS:
                touched.add(col)

        for col in touched:
            schema = schema.with_dtype(col, "float64")
        return schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> GeneralTransformationArtifact:  # pylint: disable=arguments-differ
        """Fit power lambdas in rule order; retain simple ops in the artifact.

        Box-Cox is skipped (with a warning) for non-positive values at that
        point in the rule sequence. A fit failure logs and skips that rule.
        """
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'},
        #                              {'column': 'col2', 'method': 'yeo-johnson'}]}
        is_polars = get_engine(X).name == EngineName.POLARS
        fitted_transformations: list[dict[str, Any]] = []
        transformations = config.get("transformations", [])
        last_power_rule = {
            item.get("column"): index
            for index, item in enumerate(transformations)
            if item.get("method") in _POWER_METHODS
        }
        working_X = X
        applier = GeneralTransformationApplier()

        for index, item in enumerate(transformations):
            col = item.get("column")
            method = item.get("method")
            if col not in X.columns:
                continue

            fitted_item: dict[str, Any] = {"column": col, "method": method}

            if method in _POWER_METHODS:
                try:
                    extras = _fit_power_for_column(
                        working_X, col, method, is_polars, standardize=item.get("standardize", True)
                    )
                except Exception as e:  # noqa: BLE001 - per-column fit failure is logged; column skipped
                    logger.warning(f"Failed to fit {method} for column {col}: {e}")
                    continue
                if not extras:
                    continue  # box-cox skipped on non-positive data
                fitted_item.update(extras)

            fitted_transformations.append(fitted_item)
            # Only materialize rules whose output is needed by a later fit.
            # Reuse replay semantics without mutating the caller's training data.
            if index < last_power_rule.get(col, -1):
                working_X = applier.apply(working_X, {"transformations": [fitted_item]})

        return {
            "type": "general_transformation",
            "transformations": fitted_transformations,
        }
