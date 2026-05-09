"""Hash Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import HashEncoderArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)


def _resolve_valid_cols(X: Any, params: Dict[str, Any]) -> List[str]:
    """Filter requested columns down to those present in ``X``."""
    cols = params.get("columns", [])
    return [c for c in cols if c in X.columns]


def _hash_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Polars apply path — uses Polars native ``hash()`` for speed."""
    import polars as pl

    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    n_features = params.get("n_features", 10)
    exprs = [(pl.col(col).cast(pl.Utf8).hash() % n_features).alias(col) for col in valid_cols]
    return X.with_columns(exprs), y


def _hash_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Pandas apply path — Python's built-in ``hash`` per value."""
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    n_features = params.get("n_features", 10)
    X_out = X.copy()
    for col in valid_cols:
        X_out[col] = X_out[col].astype(str).apply(lambda x: hash(x) % n_features)
    return X_out, y


class HashEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_hash_apply_polars,
            pandas_func=_hash_apply_pandas,
        )


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
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> HashEncoderArtifact:
        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "HashEncoder", y)
        if not cols:
            return {}

        return {
            "type": "hash_encoder",
            "columns": cols,
            "n_features": config.get("n_features", 10),
        }

    def infer_output_schema(
        self,
        input_schema: SkyulfSchema,
        config: Dict[str, Any],
    ) -> Optional[SkyulfSchema]:
        # Hash encoder replaces values in source columns in place
        # (`pl.col(col)...alias(col)`). Schema is unchanged.
        return input_schema


__all__ = ["HashEncoderApplier", "HashEncoderCalculator"]
