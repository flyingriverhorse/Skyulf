"""Dummy Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict

import pandas as pd

from ...utils import (
    resolve_columns,
    user_picked_no_columns,
)
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from .._artifacts import DummyEncoderArtifact
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...engines import EngineName, get_engine
from ._common import detect_categorical_columns, _exclude_target_column

logger = logging.getLogger(__name__)


class DummyEncoderApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        _y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        cols = params.get("columns", [])
        categories = params.get("categories", {})
        drop_first = params.get("drop_first", False)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return X

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            X_out = X_pl

            for col in valid_cols:
                known_cats = categories.get(col, [])
                cats_to_encode = known_cats
                if drop_first and len(cats_to_encode) > 1:
                    cats_to_encode = cats_to_encode[1:]

                dummy_exprs = [
                    (pl.col(col).cast(pl.Utf8) == str(cat)).cast(pl.Int8).alias(f"{col}_{cat}")
                    for cat in cats_to_encode
                ]
                X_out = X_out.with_columns(dummy_exprs)

            X_out = X_out.drop(valid_cols)
            return X_out

        # Pandas Path
        X_out = X.copy()

        for col in valid_cols:
            known_cats = categories.get(col, [])
            X_out[col] = pd.Categorical(X_out[col].astype(str), categories=known_cats)

        dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype=int)
        X_out = X_out.drop(columns=valid_cols)
        X_out = pd.concat([X_out, dummies], axis=1)

        return X_out


@NodeRegistry.register("DummyEncoder", DummyEncoderApplier)
@node_meta(
    id="DummyEncoder",
    name="Dummy Encoder",
    category="Preprocessing",
    description="Convert categorical variables into dummy/indicator variables (pandas.get_dummies).",
    params={"columns": [], "drop_first": False},
)
class DummyEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(
        self,
        X: Any,
        y: Any,
        config: Dict[str, Any],
    ) -> DummyEncoderArtifact:
        engine = get_engine(X)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "DummyEncoder", y)

        categories: Dict[str, Any] = {}

        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X
            for col in cols:
                cats = X_pl.select(pl.col(col).cast(pl.Utf8).unique().sort()).to_series().to_list()
                categories[col] = [str(c) for c in cats if c is not None]
        else:
            for col in cols:
                categories[col] = sorted(X[col].dropna().unique().astype(str).tolist())

        return {
            "type": "dummy_encoder",
            "columns": cols,
            "categories": categories,
            "drop_first": config.get("drop_first", False),
        }


__all__ = ["DummyEncoderApplier", "DummyEncoderCalculator"]
