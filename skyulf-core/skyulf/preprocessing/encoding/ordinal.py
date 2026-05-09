"""Ordinal Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Mapping, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import OrdinalArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _parse_categories_order, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_apply_inputs(X: Any, params: Dict[str, Any]) -> Tuple[List[str], Any, Dict[str, Any]]:
    """Return ``(valid_cols, feature_encoder, target_encoders)``."""
    cols = params.get("columns", [])
    encoder = params.get("encoder_object")
    target_encoders: Dict[str, Any] = params.get("encoders", {})
    valid_cols = [c for c in cols if c in X.columns]
    return valid_cols, encoder, target_encoders


def _apply_features_polars(X: Any, valid_cols: List[str], encoder: Any) -> Any:
    import polars as pl

    try:
        X_subset = X.select(valid_cols).select([pl.col(c).cast(pl.Utf8) for c in valid_cols])
        X_np, _ = SklearnBridge.to_sklearn(X_subset)
        encoded = encoder.transform(X_np)
        new_cols_pl = [pl.Series(col, encoded[:, i]) for i, col in enumerate(valid_cols)]
        return X.with_columns(new_cols_pl)
    except Exception as e:
        logger.error(f"Ordinal Encoding failed: {e}")
        return X


def _apply_features_pandas(X: Any, valid_cols: List[str], encoder: Any) -> Any:
    X_out = X.copy()
    try:
        X_subset = X_out[valid_cols].astype(str)
        X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
        X_out[valid_cols] = encoder.transform(X_input)
    except Exception as e:
        logger.error(f"Ordinal Encoding failed: {e}")
    return X_out


def _apply_target_polars(y: Any, enc: OrdinalEncoder) -> Any:
    import polars as pl

    try:
        y_arr = y.to_numpy().astype(str).reshape(-1, 1)
        encoded = enc.transform(y_arr).flatten()
        y_name = y.name if hasattr(y, "name") else "target"
        return pl.Series(y_name, encoded.astype(np.float32))
    except Exception as e:
        logger.error(f"Ordinal Encoding target failed: {e}")
        return y


def _apply_target_pandas(y: Any, enc: OrdinalEncoder) -> Any:
    try:
        y_arr = (
            y.to_numpy().astype(str).reshape(-1, 1)
            if hasattr(y, "to_numpy")
            else np.array(y).astype(str).reshape(-1, 1)
        )
        encoded = enc.transform(y_arr).flatten()
        return pd.Series(
            encoded,
            index=y.index if hasattr(y, "index") else None,
            name=y.name if hasattr(y, "name") else None,
        )
    except Exception as e:
        logger.error(f"Ordinal Encoding target failed: {e}")
        return y


def _ordinal_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols, encoder, target_encoders = _resolve_apply_inputs(X, params)
    if not valid_cols and "__target__" not in target_encoders:
        return X, y

    X_out: Any = X
    if valid_cols and encoder:
        X_out = _apply_features_polars(X, valid_cols, encoder)

    y_out = y
    if y is not None and "__target__" in target_encoders:
        y_out = _apply_target_polars(y, target_encoders["__target__"])
    return X_out, y_out


def _ordinal_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols, encoder, target_encoders = _resolve_apply_inputs(X, params)
    if not valid_cols and "__target__" not in target_encoders:
        return X, y

    X_out: Any = X
    if valid_cols and encoder:
        X_out = _apply_features_pandas(X, valid_cols, encoder)

    y_out = y
    if y is not None and "__target__" in target_encoders:
        y_out = _apply_target_pandas(y, target_encoders["__target__"])
    return X_out, y_out


class OrdinalEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_ordinal_apply_polars,
            pandas_func=_ordinal_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _resolve_handle_unknown(config: Dict[str, Any]) -> str:
    """OrdinalEncoder only accepts 'error' or 'use_encoded_value'."""
    raw = config.get("handle_unknown", "use_encoded_value")
    return raw if raw in ("error", "use_encoded_value") else "use_encoded_value"


def _make_ordinal_encoder(
    categories: Union[str, List[List[str]]], handle_unknown: str, unknown_value: Any
) -> OrdinalEncoder:
    return OrdinalEncoder(
        categories=categories,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value,
        dtype=np.float32,
    )


def _fit_target_encoder(
    y_series: Any,
    categories: Union[str, List[List[str]]],
    handle_unknown: str,
    unknown_value: Any,
) -> OrdinalEncoder:
    """Fit a one-column OrdinalEncoder on ``y``."""
    enc = _make_ordinal_encoder(categories, handle_unknown, unknown_value)
    y_arr = (
        y_series.to_numpy().astype(str).reshape(-1, 1)
        if hasattr(y_series, "to_numpy")
        else np.array(y_series).astype(str).reshape(-1, 1)
    )
    enc.fit(y_arr)
    return enc


def _resolve_target_categories(raw_order: Any, n_features: int) -> Union[str, List[List[str]]]:
    """Slice the per-target row out of the categories_order table."""
    parsed = _parse_categories_order(raw_order, n_features + 1)
    if isinstance(parsed, list) and len(parsed) == n_features + 1:
        return [parsed[-1]]
    return "auto"


def _build_subset_polars(X: Any, feature_cols: List[str]) -> Any:
    import polars as pl

    return X.select(feature_cols).select([pl.col(c).cast(pl.Utf8) for c in feature_cols])


def _fit_feature_encoder(
    X_subset: Any,
    feature_cols: List[str],
    config: Dict[str, Any],
) -> Tuple[OrdinalEncoder, List[int]]:
    cats = _parse_categories_order(config.get("categories_order"), len(feature_cols))
    enc = _make_ordinal_encoder(
        cats, _resolve_handle_unknown(config), config.get("unknown_value", -1)
    )
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    enc.fit(X_np)
    counts = [len(c) for c in enc.categories_]
    return enc, counts


def _ordinal_fit_no_columns(y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    """Fit-time fallback when the user picked no feature columns."""
    target_encoders: Dict[str, Any] = {}
    if y is not None:
        cats_y = _parse_categories_order(config.get("categories_order"), 1)
        target_encoders["__target__"] = _fit_target_encoder(
            y, cats_y, _resolve_handle_unknown(config), config.get("unknown_value", -1)
        )
    return {
        "type": "ordinal",
        "columns": [],
        "encoder_object": None,
        "encoders": target_encoders,
        "categories_count": [],
    }


def _should_encode_target(X: Any, y: Any, config: Dict[str, Any]) -> bool:
    """True iff ``y`` exists and the configured target name is in `columns`
    but not in ``X``."""
    if y is None:
        return False
    target_col = config.get("target_column") or getattr(y, "name", None)
    if not target_col:
        return False
    cols_raw: List[str] = config.get("columns") or []
    return target_col in cols_raw and target_col not in X.columns


def _maybe_fit_features(
    X: Any, feature_cols: List[str], config: Dict[str, Any], build_subset: Any
) -> Tuple["OrdinalEncoder | None", List[int]]:
    if not feature_cols:
        return None, []
    return _fit_feature_encoder(build_subset(X, feature_cols), feature_cols, config)


def _maybe_fit_target_block(
    y: Any, n_features: int, config: Dict[str, Any], encode_target: bool
) -> Dict[str, Any]:
    if not (encode_target and y is not None):
        return {}
    cats_y = _resolve_target_categories(config.get("categories_order"), n_features)
    enc = _fit_target_encoder(
        y, cats_y, _resolve_handle_unknown(config), config.get("unknown_value", -1)
    )
    return {"__target__": enc}


def _ordinal_fit_dispatch(
    X: Any,
    y: Any,
    config: Dict[str, Any],
    build_subset: Any,
) -> Mapping[str, Any]:
    """Engine-agnostic fit body. ``build_subset(X, feature_cols) -> X_subset``."""
    if user_picked_no_columns(config):
        return _ordinal_fit_no_columns(y, config)

    encode_target = _should_encode_target(X, y, config)
    cols = resolve_columns(X, config, detect_categorical_columns)
    feature_cols = [c for c in cols if c in X.columns]
    if not feature_cols and not encode_target:
        return {}

    feature_encoder, counts = _maybe_fit_features(X, feature_cols, config, build_subset)
    target_encoders = _maybe_fit_target_block(y, len(feature_cols), config, encode_target)

    return {
        "type": "ordinal",
        "columns": feature_cols,
        "encoder_object": feature_encoder,
        "encoders": target_encoders,
        "categories_count": counts,
    }


def _ordinal_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    return _ordinal_fit_dispatch(X, y, config, _build_subset_polars)


def _ordinal_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    return _ordinal_fit_dispatch(X, y, config, lambda Xi, fc: Xi[fc].astype(str))


@NodeRegistry.register("OrdinalEncoder", OrdinalEncoderApplier)
@node_meta(
    id="OrdinalEncoder",
    name="Ordinal Encoder",
    category="Preprocessing",
    description="Encodes categorical features as an integer array.",
    params={
        "columns": [],
        "handle_unknown": "use_encoded_value",
        "unknown_value": -1,
        "categories_order": "",
    },
)
class OrdinalEncoderCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Ordinal encoding replaces categorical values with ints in place;
        # column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> OrdinalArtifact:
        return cast(
            OrdinalArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_ordinal_fit_polars,
                pandas_func=_ordinal_fit_pandas,
            ),
        )


__all__ = ["OrdinalEncoderApplier", "OrdinalEncoderCalculator"]
