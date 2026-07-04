"""Drop-missing-rows node (drop rows with NaNs, y-synced)."""

from typing import Any, Dict, Optional, Tuple

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import DropMissingRowsArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import _normalize_subset, _polars_filter_y_by_kept_indices


def _polars_dropna_filter(X: Any, check_cols: list, how: str, threshold: Optional[int]) -> Any:
    """Build the polars filter for dropna with optional threshold/how."""
    import polars as pl

    if threshold is not None:
        return X.filter(pl.sum_horizontal(pl.col(check_cols).is_not_null()) >= threshold)
    if how == "all":
        return X.filter(~pl.all_horizontal(pl.col(check_cols).is_null()))
    return X.drop_nulls(subset=check_cols)


def _drop_missing_rows_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    subset = _normalize_subset(params.get("subset"), list(X.columns))
    how = params.get("how", "any")
    threshold = params.get("threshold")

    X_with_idx = X.with_row_index("__idx__")
    check_cols = subset if subset else [c for c in X.columns if c != "__idx__"]
    X_clean = _polars_dropna_filter(X_with_idx, check_cols, how, threshold)
    kept = X_clean["__idx__"]
    X_out = X_clean.drop("__idx__")

    if y is None:
        return X_out, None
    return X_out, _polars_filter_y_by_kept_indices(y, kept)


def _drop_missing_rows_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    subset = _normalize_subset(params.get("subset"), list(X.columns))
    how = params.get("how", "any")
    threshold = params.get("threshold")

    # Pandas dropna forbids both 'how' and 'thresh'; thresh takes precedence.
    if threshold is not None:
        X_clean = X.dropna(axis=0, thresh=threshold, subset=subset)
    else:
        X_clean = X.dropna(axis=0, how=how, subset=subset)

    if y is None:
        return X_clean, None
    return X_clean, y.loc[X_clean.index]


class DropMissingRowsApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            _drop_missing_rows_apply_polars,
            _drop_missing_rows_apply_pandas,
        )


@NodeRegistry.register("DropMissingRows", DropMissingRowsApplier)
@node_meta(
    id="DropMissingRows",
    name="Drop Missing Rows",
    category="Cleaning",
    description="Drop rows containing missing values in specified columns.",
    params={"subset": [], "how": "any"},
)
class DropMissingRowsCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Drops rows; column set is preserved.
        return input_schema

    def fit(self, df: Any, config: Dict[str, Any]) -> DropMissingRowsArtifact:
        return {
            "type": "drop_missing_rows",
            "subset": config.get("subset"),
            "how": config.get("how", "any"),
            "threshold": config.get("threshold"),
        }
