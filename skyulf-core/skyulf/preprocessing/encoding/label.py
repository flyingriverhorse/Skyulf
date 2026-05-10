"""Label Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import LabelEncoderArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _le_mapping(le: LabelEncoder) -> Dict[Any, int]:
    """Return ``{class -> int}`` mapping for a fitted LabelEncoder."""
    return dict(zip(le.classes_, le.transform(le.classes_)))


def _le_mapping_str(le: LabelEncoder) -> Dict[str, int]:
    """String-keyed variant for Polars `replace`."""
    return {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}


def _build_polars_feature_exprs(
    X: Any, cols: List[str], encoders: Dict[str, Any], missing_code: Any
) -> List[Any]:
    import polars as pl

    exprs: List[Any] = []
    for col in cols:
        if col in X.columns and col in encoders:
            mapping = _le_mapping_str(encoders[col])
            exprs.append(
                pl.col(col)
                .cast(pl.Utf8)
                .replace(mapping, default=missing_code)
                .cast(pl.Int64)
                .alias(col)
            )
    return exprs


def _label_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    encoders: Dict[str, Any] = params.get("encoders", {})
    cols: Optional[List[str]] = params.get("columns")
    missing_code = params.get("missing_code", -1)

    X_out = X.clone()
    y_out = y.clone() if y is not None else None

    if cols:
        exprs = _build_polars_feature_exprs(X_out, cols, encoders, missing_code)
        if exprs:
            X_out = X_out.with_columns(exprs)

    if y_out is not None and "__target__" in encoders:
        mapping = _le_mapping_str(encoders["__target__"])
        y_out = y_out.cast(pl.Utf8).replace(mapping, default=missing_code).cast(pl.Int64)

    return X_out, y_out


def _label_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    encoders: Dict[str, Any] = params.get("encoders", {})
    cols: Optional[List[str]] = params.get("columns")
    missing_code = params.get("missing_code", -1)

    X_out = X.copy()
    y_out = y.copy() if y is not None else None

    if cols:
        for col in cols:
            if col in X_out.columns and col in encoders:
                mapping = _le_mapping(encoders[col])
                X_out[col] = X_out[col].astype(str).map(mapping).fillna(missing_code)

    if y_out is not None and "__target__" in encoders:
        mapping = _le_mapping(encoders["__target__"])
        y_out = y_out.astype(str).map(mapping).fillna(missing_code)

    return X_out, y_out


class LabelEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_label_apply_polars,
            pandas_func=_label_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _y_to_str_array(y: Any) -> Any:
    """Best-effort conversion of ``y`` to a 1-D string numpy array."""
    if hasattr(y, "to_numpy"):
        return y.to_numpy().astype(str)
    return np.array(y).astype(str)


def _fit_le_on_array(arr: Any) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(arr)
    return le


def _polars_col_to_str_array(X: Any, col: str) -> Any:
    import polars as pl

    return X.select(pl.col(col).cast(pl.Utf8)).to_series().to_numpy()


def _maybe_pull_y_polars(X: Any, y: Any, target_col: Optional[str]) -> Any:
    if y is not None or not target_col or target_col not in X.columns:
        return y
    return X.get_column(target_col)


def _maybe_pull_y_pandas(X: Any, y: Any, target_col: Optional[str]) -> Any:
    if y is not None or not target_col or target_col not in X.columns:
        return y
    return X[target_col]


def _maybe_fit_target(
    y: Any, cols: Optional[List[str]], encoders: Dict[str, Any], counts: Dict[str, int]
) -> None:
    """Fit a LabelEncoder on ``y`` when columns is empty OR y's name is in cols."""
    if y is None:
        return
    if cols:
        y_name = getattr(y, "name", None)
        if not (y_name and y_name in cols):
            return
    le = _fit_le_on_array(_y_to_str_array(y))
    encoders["__target__"] = le
    counts["__target__"] = len(le.classes_)


def _fit_feature_encoders(
    valid_cols: List[str],
    column_to_array: Any,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Fit one LabelEncoder per column. ``column_to_array(col) -> 1-D str array``."""
    encoders: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for col in valid_cols:
        le = _fit_le_on_array(column_to_array(col))
        encoders[col] = le
        counts[col] = len(le.classes_)
    return encoders, counts


def _build_label_artifact(
    encoders: Dict[str, Any],
    cols: Optional[List[str]],
    counts: Dict[str, int],
    config: Dict[str, Any],
) -> Mapping[str, Any]:
    return {
        "type": "label_encoder",
        "encoders": encoders,
        "columns": cols,
        "classes_count": counts,
        "missing_code": config.get("missing_code", -1),
    }


def _label_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _maybe_pull_y_polars(X, y, config.get("target_column"))
    cols: Optional[List[str]] = config.get("columns")
    encoders: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    if cols:
        valid = [c for c in cols if c in X.columns]
        encoders, counts = _fit_feature_encoders(valid, lambda c: _polars_col_to_str_array(X, c))

    _maybe_fit_target(y, cols, encoders, counts)
    return _build_label_artifact(encoders, cols, counts, config)


def _label_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    y = _maybe_pull_y_pandas(X, y, config.get("target_column"))
    cols: Optional[List[str]] = config.get("columns")
    encoders: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    if cols:
        valid = [c for c in cols if c in X.columns]
        encoders, counts = _fit_feature_encoders(valid, lambda c: X[c].astype(str))

    _maybe_fit_target(y, cols, encoders, counts)
    return _build_label_artifact(encoders, cols, counts, config)


@NodeRegistry.register("LabelEncoder", LabelEncoderApplier)
@node_meta(
    id="LabelEncoder",
    name="Label Encoder",
    category="Preprocessing",
    description="Encode target labels with value between 0 and n_classes-1.",
    params={"columns": [], "missing_code": -1},
)
class LabelEncoderCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Label encoding replaces categorical values with ints in place;
        # column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> LabelEncoderArtifact:
        return cast(
            LabelEncoderArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_label_fit_polars,
                pandas_func=_label_fit_pandas,
            ),
        )


__all__ = ["LabelEncoderApplier", "LabelEncoderCalculator"]
