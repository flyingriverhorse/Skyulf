"""Ordinal Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ...utils import (
    pack_pipeline_output,
    resolve_columns,
    unpack_pipeline_input,
    user_picked_no_columns,
)
from ..base import BaseApplier, BaseCalculator
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...engines import EngineName, SkyulfDataFrame, get_engine
from ...engines.sklearn_bridge import SklearnBridge
from ._common import detect_categorical_columns, _exclude_target_column, _parse_categories_order

logger = logging.getLogger(__name__)


class OrdinalEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        encoder = params.get("encoder_object")
        target_encoders: Dict[str, Any] = params.get("encoders", {})

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols and "__target__" not in target_encoders:
            return pack_pipeline_output(X, y, is_tuple)

        X_out: Any = X

        # --- Encode feature columns ---
        if valid_cols and encoder:
            if engine.name == EngineName.POLARS:
                import polars as pl

                X_pl: Any = X
                try:
                    X_subset = X_pl.select(valid_cols)
                    X_subset = X_subset.select([pl.col(c).cast(pl.Utf8) for c in valid_cols])
                    X_np, _ = SklearnBridge.to_sklearn(X_subset)
                    encoded_array = encoder.transform(X_np)
                    new_cols_pl = [
                        pl.Series(col, encoded_array[:, i]) for i, col in enumerate(valid_cols)
                    ]
                    X_out = X_pl.with_columns(new_cols_pl)
                except Exception as e:
                    logger.error(f"Ordinal Encoding failed: {e}")
            else:
                X_out = X.copy()
                try:
                    X_subset = X_out[valid_cols].astype(str)
                    X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
                    encoded_array = encoder.transform(X_input)
                    X_out[valid_cols] = encoded_array
                except Exception as e:
                    logger.error(f"Ordinal Encoding failed: {e}")

        # --- Encode target y ---
        y_out = y
        if y is not None and "__target__" in target_encoders:
            enc = target_encoders["__target__"]
            try:
                if engine.name == EngineName.POLARS:
                    import polars as pl

                    y_arr = y.to_numpy().astype(str).reshape(-1, 1)
                    encoded = enc.transform(y_arr).flatten()
                    y_name = y.name if hasattr(y, "name") else "target"
                    y_out = pl.Series(y_name, encoded.astype(np.float32))
                else:
                    y_arr = (
                        y.to_numpy().astype(str).reshape(-1, 1)
                        if hasattr(y, "to_numpy")
                        else np.array(y).astype(str).reshape(-1, 1)
                    )
                    encoded = enc.transform(y_arr).flatten()
                    y_out = pd.Series(
                        encoded,
                        index=y.index if hasattr(y, "index") else None,
                        name=y.name if hasattr(y, "name") else None,
                    )
            except Exception as e:
                logger.error(f"Ordinal Encoding target failed: {e}")

        return pack_pipeline_output(X_out, y_out, is_tuple)


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
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # OrdinalEncoder only accepts 'error' or 'use_encoded_value'.
        _raw_hu = config.get("handle_unknown", "use_encoded_value")
        handle_unknown = (
            _raw_hu if _raw_hu in ("error", "use_encoded_value") else "use_encoded_value"
        )
        unknown_value = config.get("unknown_value", -1)

        target_col: str | None = config.get("target_column")
        if target_col is None and y is not None:
            target_col = getattr(y, "name", None)

        target_encoders: Dict[str, Any] = {}
        categories_order_raw = config.get("categories_order")

        def _fit_enc_on_y(
            y_series: Any,
            categories: Union[str, List[List[str]]] = "auto",
        ) -> OrdinalEncoder:
            enc = OrdinalEncoder(
                categories=categories,
                handle_unknown=handle_unknown,
                unknown_value=unknown_value,
                dtype=np.float32,
            )
            y_arr = (
                y_series.to_numpy().astype(str).reshape(-1, 1)
                if hasattr(y_series, "to_numpy")
                else np.array(y_series).astype(str).reshape(-1, 1)
            )
            enc.fit(y_arr)
            return enc

        if user_picked_no_columns(config):
            if y is not None:
                cats_y = _parse_categories_order(categories_order_raw, 1)
                target_encoders["__target__"] = _fit_enc_on_y(y, cats_y)
            return {
                "type": "ordinal",
                "columns": [],
                "encoder_object": None,
                "encoders": target_encoders,
                "categories_count": [],
            }

        cols_raw: List[str] = config.get("columns") or []
        encode_target = bool(
            target_col and target_col in cols_raw and y is not None and target_col not in X.columns
        )

        cols = resolve_columns(X, config, detect_categorical_columns)
        feature_cols = [c for c in cols if c in X.columns]

        if not feature_cols and not encode_target:
            return {}

        feature_encoder: OrdinalEncoder | None = None
        categories_count: List[int] = []

        if feature_cols:
            cats_feat = _parse_categories_order(categories_order_raw, len(feature_cols))
            feature_encoder = OrdinalEncoder(
                categories=cats_feat,
                handle_unknown=handle_unknown,
                unknown_value=unknown_value,
                dtype=np.float32,
            )
            if engine.name == EngineName.POLARS:
                import polars as pl

                X_pl: Any = X
                X_subset = X_pl.select(feature_cols)
                X_subset = X_subset.select([pl.col(c).cast(pl.Utf8) for c in feature_cols])
            else:
                X_subset = X[feature_cols].astype(str)
            X_np, _ = SklearnBridge.to_sklearn(X_subset)
            feature_encoder.fit(X_np)
            categories_count = [len(cats) for cats in feature_encoder.categories_]

        if encode_target and y is not None:
            n_feat = len(feature_cols)
            cats_tgt = _parse_categories_order(categories_order_raw, n_feat + 1)
            if isinstance(cats_tgt, list) and len(cats_tgt) == n_feat + 1:
                cats_for_y: Union[str, List[List[str]]] = [cats_tgt[-1]]
            else:
                cats_for_y = "auto"
            target_encoders["__target__"] = _fit_enc_on_y(y, cats_for_y)

        return {
            "type": "ordinal",
            "columns": feature_cols,
            "encoder_object": feature_encoder,
            "encoders": target_encoders,
            "categories_count": categories_count,
        }


__all__ = ["OrdinalEncoderApplier", "OrdinalEncoderCalculator"]
