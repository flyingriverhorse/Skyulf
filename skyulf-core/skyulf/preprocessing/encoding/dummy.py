"""Dummy Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, List, Mapping, Tuple, cast

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import DummyEncoderArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _exclude_target_column, detect_categorical_columns

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Apply
# -----------------------------------------------------------------------------


def _resolve_valid_cols(X: Any, params: Dict[str, Any]) -> List[str]:
    cols = params.get("columns", [])
    return [c for c in cols if c in X.columns]


def _drop_first_if_needed(cats: List[Any], drop_first: bool) -> List[Any]:
    """Drop the first category when ``drop_first`` is enabled (and we have ≥ 2)."""
    if drop_first and len(cats) > 1:
        return cats[1:]
    return cats


def _dummy_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    categories = params.get("categories", {})
    drop_first = params.get("drop_first", False)
    X_out = X
    for col in valid_cols:
        cats = _drop_first_if_needed(categories.get(col, []), drop_first)
        exprs = [
            (pl.col(col).cast(pl.Utf8) == str(cat)).cast(pl.Int8).alias(f"{col}_{cat}")
            for cat in cats
        ]
        X_out = X_out.with_columns(exprs)
    return X_out.drop(valid_cols), y


def _dummy_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    categories = params.get("categories", {})
    drop_first = params.get("drop_first", False)
    X_out = X.copy()
    for col in valid_cols:
        known_cats = categories.get(col, [])
        X_out[col] = pd.Categorical(X_out[col].astype(str), categories=known_cats)

    dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype=int)
    X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, dummies], axis=1), y


class DummyEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            polars_func=_dummy_apply_polars,
            pandas_func=_dummy_apply_pandas,
        )


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------


def _build_dummy_artifact(
    cols: List[str],
    categories: Dict[str, List[str]],
    config: Dict[str, Any],
) -> Mapping[str, Any]:
    return {
        "type": "dummy_encoder",
        "columns": cols,
        "categories": categories,
        "drop_first": config.get("drop_first", False),
    }


def _dummy_fit_polars(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    import polars as pl

    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "DummyEncoder", y)

    categories: Dict[str, List[str]] = {}
    for col in cols:
        cats = X.select(pl.col(col).cast(pl.Utf8).unique().sort()).to_series().to_list()
        categories[col] = [str(c) for c in cats if c is not None]
    return _build_dummy_artifact(cols, categories, config)


def _dummy_fit_pandas(X: Any, y: Any, config: Dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "DummyEncoder", y)

    categories: Dict[str, List[str]] = {
        col: sorted(X[col].dropna().unique().astype(str).tolist()) for col in cols
    }
    return _build_dummy_artifact(cols, categories, config)


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
    def fit(self, X: Any, y: Any, config: Dict[str, Any]) -> DummyEncoderArtifact:
        if user_picked_no_columns(config):
            return {}
        return cast(
            DummyEncoderArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                polars_func=_dummy_fit_polars,
                pandas_func=_dummy_fit_pandas,
            ),
        )


__all__ = ["DummyEncoderApplier", "DummyEncoderCalculator"]
