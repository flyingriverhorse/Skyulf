"""Dummy Encoder node (Calculator + Applier)."""

import logging
from collections.abc import Mapping
from typing import Any, cast

import pandas as pd
import polars as pl

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
    """Drop the first category when ``drop_first`` is enabled and any category exists."""
    if drop_first and cats:
        return cats[1:]
    return cats


# A trailing ".0" is stripped from float columns so an integral value renders
# identically on both engines and in every batch. Pandas upcasts an integer
# column to float64 the moment it holds a null (no NaN-capable numpy int dtype),
# and polars renders Float64 1.0 as "1.0" where it renders Int64 1 as "1".
_INTEGRAL_FLOAT_SUFFIX = r"\.0$"


def _pandas_col_to_str(series: Any) -> Any:
    """Render a pandas Series as strings, matching the Polars path's output.

    The rendering is per value, never per batch: ``1.0`` must yield the same
    category string whether or not a fractional sibling such as ``2.5`` sits
    beside it. Stripping ``_INTEGRAL_FLOAT_SUFFIX`` from float columns only is
    what keeps an integer column upcast to ``float64`` by a null rendering as
    ``"1"`` — the string the Polars ``Int64`` path produces — without letting
    the rest of the batch decide. Non-float columns stringify untouched, so a
    string column holding the literal ``"1.0"`` keeps it.

    Nulls are preserved as actual NaN in the returned (object-dtype) series
    rather than the literal ``"<NA>"``/``"nan"`` string that ``Int64``/
    ``float64`` ``.astype(str)`` would otherwise produce — callers rely on
    ``.dropna()`` to exclude them from the learned category list, same as
    the Polars fit path's ``if c is not None`` filter.
    """
    null_mask = series.isna()
    rendered = series.astype(str)
    if pd.api.types.is_float_dtype(series):
        rendered = rendered.str.replace(_INTEGRAL_FLOAT_SUFFIX, "", regex=True)
    return rendered.mask(null_mask)


def _polars_col_to_str_expr(X: Any, col: str) -> Any:
    """Build the Polars expression rendering ``col`` to strings, matching pandas.

    Same rule as :func:`_pandas_col_to_str`: a trailing ``.0`` comes off float
    columns so ``1.0`` and ``1`` are one category on either engine, and nulls
    stay null rather than becoming a ``"null"`` string the fit path would then
    have to filter out of the category list.
    """
    expr = pl.col(col).cast(pl.Utf8)
    if X.schema[col].is_float():
        expr = expr.str.replace(_INTEGRAL_FLOAT_SUFFIX, "")
    return expr


def _dummy_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    valid_cols = _resolve_valid_cols(X, params)
    if not valid_cols:
        return X, y

    categories = params.get("categories", {})
    drop_first = params.get("drop_first", False)
    X_out = X
    for col in valid_cols:
        cats = _drop_first_if_needed(categories.get(col, []), drop_first)
        rendered = _polars_col_to_str_expr(X, col)
        exprs = [
            (rendered == str(cat)).cast(pl.Int8).fill_null(0).alias(f"{col}_{cat}") for cat in cats
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

    # Match the Polars engine's compact binary indicator dtype.
    dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype="int8")
    X_out = X_out.drop(columns=valid_cols)
    return pd.concat([X_out, dummies], axis=1), y


class DummyEncoderApplier(BaseApplier):
    """Replace each categorical column with one ``<col>_<category>`` indicator column.

    The originals are always dropped. Parity depends on both engines rendering
    a value to the *same* string before comparing it with the learned
    categories — one rule, implemented per engine by ``_pandas_col_to_str``
    and ``_polars_col_to_str_expr``, which is a function of the value alone so
    a category cannot stop matching because of what else shares its batch. A
    value unseen at fit time yields an all-zero row rather than raising.
    """

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch to the engine-specific dummy encoding, forwarding ``(X, y)`` when present."""
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _dummy_apply_polars, "pandas": _dummy_apply_pandas},
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
    cols = resolve_columns(X, config, detect_categorical_columns)
    cols = _exclude_target_column(cols, config, "DummyEncoder", y)

    categories: dict[str, list[str]] = {}
    for col in cols:
        rendered = X.select(_polars_col_to_str_expr(X, col).unique().sort()).to_series().to_list()
        categories[col] = [str(c) for c in rendered if c is not None]
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
    learns_from_data=True,
)
class DummyEncoderCalculator(BaseCalculator):
    """Learn the sorted category list for each resolved categorical column.

    The target column is excluded first: dummy encoding replaces the column it
    encodes with several derived ones, which would break a downstream
    Feature/Target Split. Nulls are dropped before the list is built, so they
    never become a category. An explicit ``columns: []`` yields an empty
    artifact so the applier no-ops.
    """

    @fit_method
    def fit(self, X: Any, y: Any, config: dict[str, Any]) -> DummyEncoderArtifact:  # pylint: disable=arguments-differ
        """Short-circuit an explicit empty column selection, else learn the categories."""
        if user_picked_no_columns(config):
            return {}
        return cast(
            DummyEncoderArtifact,
            fit_dual_engine(
                (X, y) if y is not None else X,
                config,
                {"polars": _dummy_fit_polars, "pandas": _dummy_fit_pandas},
            ),
        )


__all__ = ["DummyEncoderApplier", "DummyEncoderCalculator"]
