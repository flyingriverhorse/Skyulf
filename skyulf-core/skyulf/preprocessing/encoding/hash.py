"""Hash Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, Optional

from ...utils import (
    resolve_columns,
    user_picked_no_columns,
)
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from .._artifacts import HashEncoderArtifact
from .._schema import SkyulfSchema
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...engines import EngineName, get_engine
from ._common import detect_categorical_columns, _exclude_target_column

logger = logging.getLogger(__name__)


class HashEncoderApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        _y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        cols = params.get("columns", [])
        n_features = params.get("n_features", 10)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return X

        # Polars Path — uses Polars native hash for speed.
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            exprs = [
                (pl.col(col).cast(pl.Utf8).hash() % n_features).alias(col) for col in valid_cols
            ]
            X_out = X_pl.with_columns(exprs)
            return X_out

        # Pandas Path
        X_out = X.copy()
        for col in valid_cols:
            X_out[col] = X_out[col].astype(str).apply(lambda x: hash(x) % n_features)

        return X_out


@NodeRegistry.register("HashEncoder", HashEncoderApplier)
@node_meta(
    id="HashEncoder",
    name="Hash Encoder",
    category="Preprocessing",
    description="Encode categorical features using hashing.",
    params={"n_features": 8, "columns": []},
)
class HashEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(
        self,
        X: Any,
        y: Any,
        config: Dict[str, Any],
    ) -> HashEncoderArtifact:

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "HashEncoder", y)
        if not cols:
            return {}

        n_features = config.get("n_features", 10)
        return {"type": "hash_encoder", "columns": cols, "n_features": n_features}

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # Hash encoder replaces values in source columns in place
        # (`pl.col(col)...alias(col)`). Schema is unchanged.
        return input_schema


__all__ = ["HashEncoderApplier", "HashEncoderCalculator"]
