"""Hash Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, Tuple, Union

import pandas as pd

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
from ._common import detect_categorical_columns, _exclude_target_column

logger = logging.getLogger(__name__)


class HashEncoderApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        params: Dict[str, Any],
    ) -> Any:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        n_features = params.get("n_features", 10)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path — uses Polars native hash for speed.
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            exprs = [
                (pl.col(col).cast(pl.Utf8).hash() % n_features).alias(col) for col in valid_cols
            ]
            X_out = X_pl.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()
        for col in valid_cols:
            X_out[col] = X_out[col].astype(str).apply(lambda x: hash(x) % n_features)

        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("HashEncoder", HashEncoderApplier)
@node_meta(
    id="HashEncoder",
    name="Hash Encoder",
    category="Preprocessing",
    description="Encode categorical features using hashing.",
    params={"n_features": 8, "columns": []},
)
class HashEncoderCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, y, _ = unpack_pipeline_input(df)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "HashEncoder", y)
        if not cols:
            return {}

        n_features = config.get("n_features", 10)
        return {"type": "hash_encoder", "columns": cols, "n_features": n_features}


__all__ = ["HashEncoderApplier", "HashEncoderCalculator"]
