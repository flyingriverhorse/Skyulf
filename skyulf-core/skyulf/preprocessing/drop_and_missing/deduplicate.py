"""Deduplicate node (drop duplicate rows, y-synced)."""

from typing import Any, Dict, Tuple

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import DeduplicateArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import _normalize_subset, _polars_filter_y_by_kept_indices


def _normalize_keep(keep: Any) -> Any:
    """Map config ``"none"`` to pandas ``False`` (deduplicate keeps that semantic)."""
    return False if keep == "none" else keep


def _dedup_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    keep = _normalize_keep(params.get("keep", "first"))
    subset = _normalize_subset(params.get("subset"), list(X.columns))

    # Polars uses "none" string where pandas uses False.
    pl_keep = "none" if keep is False else keep

    if y is None:
        return X.unique(subset=subset, keep=pl_keep, maintain_order=True), None

    X_with_idx = X.with_row_index("__idx__")
    X_dedup = X_with_idx.unique(subset=subset, keep=pl_keep, maintain_order=True)
    kept = X_dedup["__idx__"]
    return X_dedup.drop("__idx__"), _polars_filter_y_by_kept_indices(y, kept)


def _dedup_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    keep = _normalize_keep(params.get("keep", "first"))
    subset = _normalize_subset(params.get("subset"), list(X.columns))

    X_dedup = X.drop_duplicates(subset=subset, keep=keep)
    if y is None:
        return X_dedup, None
    return X_dedup, y.loc[X_dedup.index]


class DeduplicateApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        # Note: dedup must propagate row drops to y, so we route X+y as a tuple
        # through apply_dual_engine which handles unpack/pack.
        return apply_dual_engine(
            (X, y) if y is not None else X,
            params,
            _dedup_apply_polars,
            _dedup_apply_pandas,
        )


@NodeRegistry.register("Deduplicate", DeduplicateApplier)
@node_meta(
    id="Deduplicate",
    name="Deduplicate",
    category="Data Operations",
    description="Drop duplicate rows.",
    params={"subset": [], "keep": "first"},
)
class DeduplicateCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Deduplication removes rows; column set is preserved.
        return input_schema

    def fit(self, df: Any, config: Dict[str, Any]) -> DeduplicateArtifact:
        return {
            "type": "deduplicate",
            "subset": config.get("subset"),
            "keep": config.get("keep", "first"),
        }
