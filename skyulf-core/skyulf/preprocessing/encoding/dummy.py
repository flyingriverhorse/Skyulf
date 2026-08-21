"""Dummy Encoder node (Calculator + Applier)."""

import logging
from collections.abc import Mapping
from typing import Any, cast

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


def _resolve_valid_cols(X: Any, params: dict[str, Any]) -> list[str]:
    cols = params.get("columns", [])
    return [c for c in cols if c in X.columns]


def _drop_first_if_needed(cats: list[Any], drop_first: bool) -> list[Any]:
    """Drop the first category when ``drop_first`` is enabled (and we have ≥ 2)."""
    if drop_first and len(cats) > 1:
        return cats[1:]
    return cats


def _pandas_col_to_str(series: Any) -> Any:
    """Render a pandas Series as strings, matching the Polars fit path's output.

    Pandas silently upcasts an integer column to ``float64`` whenever it
    contains a null (there's no native NaN-capable integer dtype for plain
    numpy-backed columns), so ``1`` renders as ``"1.0"`` instead of ``"1"``
    once a null is present in the batch — even though the Polars fit/apply
    paths (``cast(pl.Utf8)``) always render ``"1"`` regardless of nulls. Left
    unhandled, this causes every value to miss the category lookup whenever
    a batch's null presence differs from what a sibling batch/engine saw at
    fit time, silently producing all-zero dummy columns for every row. If the
    non-null values are all integer-valued (e.g. a nullable-float column that
    only ever held whole numbers), normalize to a nullable ``Int64`` dtype
    first so the string form matches the Polars convention.

    Nulls are preserved as actual NaN in the returned (object-dtype) series
    rather than the literal ``"<NA>"``/``"nan"`` string that ``Int64``/
    ``float64`` ``.astype(str)`` would otherwise produce — callers rely on
    ``.dropna()`` to exclude them from the learned category list, same as
    the Polars fit path's ``if c is not None`` filter.
    """
    null_mask = series.isna()
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if not non_null.empty and (non_null % 1 == 0).all():
            series = series.astype("Int64")
    return series.astype(str).mask(null_mask)


def _dummy_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
            (pl.col(col).cast(pl.Utf8) == str(cat)).cast(pl.Int8).fill_null(0).alias(f"{col}_{cat}")
            for cat in cats
        ]
        X_out = X_out.with_columns(exprs)
    return X_out.drop(valid_cols), y


def _dummy_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    categories = params.get("categories", {})
    drop_first = params.get("drop_first", False)
    X_out = X.copy()
    for col in valid_cols:
        known_cats = categories.get(col, [])
        X_out[col] = pd.Categorical(_pandas_col_to_str(X_out[col]), categories=known_cats)

    dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype=int)
    X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, dummies], axis=1), y


class DummyEncoderApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
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
    cols: list[str],
    categories: dict[str, list[str]],
    config: dict[str, Any],
) -> Mapping[str, Any]:
    return {
        "type": "dummy_encoder",
        "columns": cols,
        "categories": categories,
        "drop_first": config.get("drop_first", False),
    }


def _dummy_fit_polars(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    import polars as pl

    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "DummyEncoder", y)

    categories: dict[str, list[str]] = {}
    for col in cols:
        cats = X.select(pl.col(col).cast(pl.Utf8).unique().sort()).to_series().to_list()
        categories[col] = [str(c) for c in cats if c is not None]
    return _build_dummy_artifact(cols, categories, config)


def _dummy_fit_pandas(X: Any, y: Any, config: dict[str, Any]) -> Mapping[str, Any]:
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "DummyEncoder", y)

    categories: dict[str, list[str]] = {
        col: sorted(_pandas_col_to_str(X[col]).dropna().unique().tolist()) for col in cols
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
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> DummyEncoderArtifact:  # pylint: disable=arguments-differ
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
