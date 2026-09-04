"""Drop-missing-rows node (drop rows with NaNs, y-synced)."""

from typing import Any

import numpy as np
import polars as pl

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import DropMissingRowsArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import (
    _normalize_subset,
    _pandas_filter_y_by_kept_positions,
    _polars_filter_y_by_kept_indices,
)


def _polars_missing_expr(X: Any, col: str) -> Any:
    """Return an expression that is True when ``col`` is null or (for float
    dtypes) NaN, so missing-row detection matches pandas' ``isna()``, which
    treats float NaN as missing too.
    """
    expr = pl.col(col).is_null()
    if X.schema[col].is_float():
        expr = expr | pl.col(col).is_nan()
    return expr


def _min_non_na_for_percentage(missing_threshold: float, n_cols: int) -> float:
    """Minimum non-missing count that keeps a row under a percentage threshold.

    A row is dropped when its missing share exceeds ``missing_threshold``
    percent, so it is kept when ``non_na_count >= (1 - pct/100) * n_cols``.
    """
    return (1.0 - missing_threshold / 100.0) * n_cols


def _polars_dropna_filter(
    X: Any, check_cols: list, how: str, threshold: int | None, missing_threshold: float | None
) -> Any:
    """Build the polars filter for dropna with optional threshold/how."""
    missing = [_polars_missing_expr(X, c) for c in check_cols]
    not_missing = [~m for m in missing]
    if threshold is not None:
        return X.filter(pl.sum_horizontal(not_missing) >= threshold)
    if missing_threshold is not None:
        return X.filter(
            pl.sum_horizontal(not_missing)
            >= _min_non_na_for_percentage(missing_threshold, len(check_cols))
        )
    if how == "all":
        return X.filter(~pl.all_horizontal(missing))
    return X.filter(~pl.any_horizontal(missing))


def _drop_missing_rows_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    subset = _normalize_subset(params.get("subset"), list(X.columns))
    how = params.get("how", "any")
    threshold = params.get("threshold")
    missing_threshold = params.get("missing_threshold")

    X_with_idx = X.with_row_index("__idx__")
    check_cols = subset if subset else [c for c in X.columns if c != "__idx__"]
    X_clean = _polars_dropna_filter(X_with_idx, check_cols, how, threshold, missing_threshold)
    kept = X_clean["__idx__"]
    X_out = X_clean.drop("__idx__")

    if y is None:
        return X_out, None
    return X_out, _polars_filter_y_by_kept_indices(y, kept)


def _drop_missing_rows_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    subset = _normalize_subset(params.get("subset"), list(X.columns))
    how = params.get("how", "any")
    threshold = params.get("threshold")
    missing_threshold = params.get("missing_threshold")

    # Compute the keep mask positionally (same semantics as dropna) so y can be
    # aligned by position; label-based .loc would return every row matching a
    # duplicated index label, desynchronizing y from X_clean.
    cols = subset if subset is not None else list(X.columns)
    non_na = X[cols].notna()
    if threshold is not None:
        keep_mask = non_na.sum(axis=1) >= threshold
    elif missing_threshold is not None:
        keep_mask = non_na.sum(axis=1) >= _min_non_na_for_percentage(missing_threshold, len(cols))
    elif how == "any":
        keep_mask = non_na.all(axis=1)
    else:
        keep_mask = non_na.any(axis=1)
    kept_positions = np.flatnonzero(keep_mask.to_numpy())
    X_clean = X.iloc[kept_positions]

    if y is None:
        return X_clean, None
    return X_clean, _pandas_filter_y_by_kept_positions(y, kept_positions)


class DropMissingRowsApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            {"polars": _drop_missing_rows_apply_polars, "pandas": _drop_missing_rows_apply_pandas},
        )


@NodeRegistry.register("DropMissingRows", DropMissingRowsApplier)
@node_meta(
    id="DropMissingRows",
    name="Drop Missing Rows",
    category="Cleaning",
    description="Drop rows containing missing values in specified columns.",
    params={"subset": [], "how": "any", "missing_threshold": None},
    learns_from_data=False,
)
class DropMissingRowsCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Drops rows; column set is preserved.
        return input_schema

    def fit(self, df: Any, config: dict[str, Any]) -> DropMissingRowsArtifact:
        return {
            "type": "drop_missing_rows",
            "subset": config.get("subset"),
            "how": config.get("how", "any"),
            "threshold": config.get("threshold"),
            "missing_threshold": config.get("missing_threshold"),
        }
