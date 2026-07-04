"""Weight-of-Evidence (WOE) / Information-Value (IV) encoder.

Credit-risk standard: replaces each category of a categorical column with the
log-odds of the binary target for that category (WOE). Also records the
Information Value (IV) per column as artifact metadata — a quick univariate
predictive-power score.

Supervised + binary-target only. The math is engine-agnostic; the fit converts
the relevant columns to pandas to compute the mapping, while ``apply`` stays in
the caller's engine (pandas in/out, polars in/out).
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Return ``(valid_cols, mappings)`` or ``([], {})`` if nothing to do."""
    cols = params.get("columns", [])
    mappings = params.get("mappings", {})
    valid_cols = [c for c in cols if c in X.columns and c in mappings]
    if not valid_cols:
        return [], {}
    return valid_cols, mappings


def _woe_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols, mappings = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    default = float(params.get("default", 0.0))
    exprs = [
        pl.col(col)
        .cast(pl.Utf8)
        .replace_strict(mappings[col], default=default, return_dtype=pl.Float64)
        .alias(col)
        for col in valid_cols
    ]
    return X.with_columns(exprs), y


def _woe_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols, mappings = _resolve_apply_inputs(X, params)
    if not valid_cols:
        return X, y

    default = float(params.get("default", 0.0))
    X_out = X.copy()
    for col in valid_cols:
        mapped = X_out[col].astype(str).map(mappings[col])
        X_out[col] = mapped.fillna(default).astype(float)
    return X_out, y


class WOEEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_woe_apply_polars,
            pandas_func=_woe_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _extract_y(X: Any, y: Any, target_col: Optional[str]) -> Any:
    """Pull ``y`` out of ``X`` if it was not provided separately."""
    if y is not None or not target_col:
        return y
    if target_col in X.columns:
        getter = getattr(X, "get_column", None)
        return getter(target_col) if getter else X[target_col]
    return y


def _binary_target(y: Any) -> Optional[np.ndarray]:
    """Coerce ``y`` to a 0/1 numpy array, or ``None`` if not binary."""
    arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    classes = np.unique(arr[~_is_null_mask(arr)])
    if len(classes) != 2:
        return None
    positive = classes[-1]
    return (arr == positive).astype(float)


def _is_null_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of NaN/None entries, dtype-safe for object arrays."""
    try:
        return np.isnan(arr.astype(float))
    except (TypeError, ValueError):
        return np.array([v is None for v in arr])


def _column_woe(
    values: np.ndarray, y_bin: np.ndarray, reg: float
) -> Tuple[Dict[str, float], float]:
    """Compute the WOE map and IV for a single categorical column."""
    total_pos = float(y_bin.sum())
    total_neg = float(len(y_bin) - total_pos)
    mapping: Dict[str, float] = {}
    iv = 0.0
    for cat in np.unique(values):
        mask = values == cat
        pos = float(y_bin[mask].sum())
        neg = float(mask.sum() - pos)
        dist_pos = (pos + reg) / (total_pos + reg)
        dist_neg = (neg + reg) / (total_neg + reg)
        woe = math.log(dist_neg / dist_pos)
        mapping[str(cat)] = woe
        iv += (dist_neg - dist_pos) * woe
    return mapping, iv


def _build_woe_artifact(
    frame: Any, y_bin: np.ndarray, cols: List[str], reg: float
) -> Mapping[str, Any]:
    """Build the WOE artifact for the given pandas frame + binary target."""
    mappings: Dict[str, Dict[str, float]] = {}
    iv_scores: Dict[str, float] = {}
    for col in cols:
        values = frame[col].astype(str).to_numpy()
        mappings[col], iv_scores[col] = _column_woe(values, y_bin, reg)
    return {
        "type": "woe_encoder",
        "columns": cols,
        "mappings": mappings,
        "information_value": iv_scores,
        "default": 0.0,
    }


def _woe_fit_common(
    frame: Any, y: Any, cols: List[str], config: Dict[str, Any]
) -> Mapping[str, Any]:
    """Shared fit: validate binary target, then build the artifact."""
    y_bin = _binary_target(y)
    if y_bin is None:
        logger.warning("WOEEncoder requires a binary target (exactly 2 classes). Skipping.")
        return {}
    reg = float(config.get("regularization", 0.5))
    return _build_woe_artifact(frame, y_bin, cols, reg)


def _woe_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _extract_y(X, y, config.get("target_column"))
    if y is None:
        logger.warning("WOEEncoder requires a target variable (y). Skipping.")
        return {}
    cols = _exclude_target_column(
        resolve_columns(X, config, detect_categorical_columns), config, "WOEEncoder", y
    )
    if not cols:
        return {}
    return _woe_fit_common(X.select(cols).to_pandas(), y, cols, config)


def _woe_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _extract_y(X, y, config.get("target_column"))
    if y is None:
        logger.warning("WOEEncoder requires a target variable (y). Skipping.")
        return {}
    cols = _exclude_target_column(
        resolve_columns(X, config, detect_categorical_columns), config, "WOEEncoder", y
    )
    if not cols:
        return {}
    return _woe_fit_common(X[cols], y, cols, config)


@NodeRegistry.register("WOEEncoder", WOEEncoderApplier)
@node_meta(
    id="WOEEncoder",
    name="WOE / IV Encoder",
    category="Preprocessing",
    description=(
        "Weight-of-Evidence encoder for binary classification. Replaces each "
        "category with its log-odds and records Information Value per column."
    ),
    params={"regularization": 0.5, "columns": []},
)
class WOEEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return {}
        return cast(
            Mapping[str, Any],
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_woe_fit_polars,
                pandas_func=_woe_fit_pandas,
            ),
        )

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # WOE replaces values in source columns in place (now float-valued);
        # column names are unchanged.
        return input_schema


__all__ = ["WOEEncoderApplier", "WOEEncoderCalculator"]
