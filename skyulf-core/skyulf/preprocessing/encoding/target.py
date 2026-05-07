"""Target Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, Tuple, Union

import pandas as pd
from sklearn.preprocessing import TargetEncoder

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
from ._common import detect_categorical_columns, _exclude_target_column

logger = logging.getLogger(__name__)


class TargetEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        encoder = params.get("encoder_object")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            try:
                X_subset = X_pl.select(valid_cols)
                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                encoded_array = encoder.transform(X_np)

                new_cols = [pl.Series(col, encoded_array[:, i]) for i, col in enumerate(valid_cols)]
                X_out = X_pl.with_columns(new_cols)
                return pack_pipeline_output(X_out, y, is_tuple)
            except Exception as e:
                logger.error(f"Target Encoding failed: {e}")
                return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
        X_out = X.copy()

        try:
            X_subset = X_out[valid_cols]
            X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
            encoded_array = encoder.transform(X_input)
            X_out[valid_cols] = encoded_array
        except Exception as e:
            logger.error(f"Target Encoding failed: {e}")

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("TargetEncoder", TargetEncoderApplier)
@node_meta(
    id="TargetEncoder",
    name="Target Encoder",
    category="Preprocessing",
    description="Encode categorical features using target statistics.",
    params={"smooth": "auto", "target_type": "auto", "columns": []},
)
class TargetEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        target_col = config.get("target_column")
        if y is None and target_col:
            if engine.name == EngineName.POLARS:
                X_pl_y: Any = X
                if target_col in X_pl_y.columns:
                    y = X_pl_y.get_column(target_col)
            else:
                if target_col in X.columns:
                    y = X[target_col]

        if y is None:
            logger.warning("TargetEncoder requires a target variable (y). Skipping.")
            return {}

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "TargetEncoder", y)
        if not cols:
            return {}

        smooth = config.get("smooth", "auto")
        target_type = config.get("target_type", "auto")

        encoder = TargetEncoder(smooth=smooth, target_type=target_type)

        if engine.name == EngineName.POLARS:
            X_pl: Any = X
            X_subset = X_pl.select(cols)
        else:
            X_subset = X[cols]

        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        y_np = y
        if hasattr(y, "to_numpy"):
            y_np = y.to_numpy()
        elif hasattr(y, "to_pandas"):
            y_np = y.to_pandas().to_numpy()

        encoder.fit(X_np, y_np)

        return {"type": "target_encoder", "columns": cols, "encoder_object": encoder}


__all__ = ["TargetEncoderApplier", "TargetEncoderCalculator"]
