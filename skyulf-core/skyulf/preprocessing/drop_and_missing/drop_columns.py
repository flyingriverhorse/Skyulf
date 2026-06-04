"""Drop-missing-columns node (drop columns over a missing-% threshold)."""

from typing import Any, Dict, Optional, Tuple, cast

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import DropMissingColumnsArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine, fit_dual_engine


def _drop_missing_cols_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    cols = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    return (X.drop(cols) if cols else X), y


def _drop_missing_cols_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    cols = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    return (X.drop(columns=cols) if cols else X), y


class DropMissingColumnsApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            X, params, _drop_missing_cols_apply_polars, _drop_missing_cols_apply_pandas
        )


def _resolve_threshold(raw: Any) -> Optional[float]:
    """Parse a missing-percentage threshold; ``None`` if absent or non-numeric."""
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _high_missing_cols_polars(X: Any, threshold_pct: float) -> list:
    """Polars: list of columns whose missing-% >= ``threshold_pct``."""
    null_counts = X.null_count()
    total = X.height or 1
    return [c for c in X.columns if (null_counts[c][0] / total) * 100 >= threshold_pct]


def _high_missing_cols_pandas(X: Any, threshold_pct: float) -> list:
    """Pandas: list of columns whose missing-% >= ``threshold_pct``."""
    pct = X.isna().mean() * 100
    return pct[pct >= threshold_pct].index.tolist()


def _drop_missing_cols_fit_polars(
    X: Any, _y: Any, config: Dict[str, Any]
) -> DropMissingColumnsArtifact:
    explicit = config.get("columns", []) or []
    cols = {c for c in explicit if c in X.columns}
    threshold = _resolve_threshold(config.get("missing_threshold"))
    if threshold is not None:
        cols.update(_high_missing_cols_polars(X, threshold))
    return {
        "type": "drop_missing_columns",
        "columns_to_drop": list(cols),
        "threshold": config.get("missing_threshold"),
    }


def _drop_missing_cols_fit_pandas(
    X: Any, _y: Any, config: Dict[str, Any]
) -> DropMissingColumnsArtifact:
    explicit = config.get("columns", []) or []
    cols = {c for c in explicit if c in X.columns}
    threshold = _resolve_threshold(config.get("missing_threshold"))
    if threshold is not None:
        cols.update(_high_missing_cols_pandas(X, threshold))
    return {
        "type": "drop_missing_columns",
        "columns_to_drop": list(cols),
        "threshold": config.get("missing_threshold"),
    }


@NodeRegistry.register("DropMissingColumns", DropMissingColumnsApplier)
@node_meta(
    id="DropMissingColumns",
    name="Drop Missing Columns",
    category="Cleaning",
    description="Drop columns that exceed missing value threshold.",
    params={"threshold": 0.5},
)
class DropMissingColumnsCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
        # Threshold path is data-dependent; only predictable when the user
        # supplied an explicit column list and no positive threshold.
        if _resolve_threshold(config.get("missing_threshold")) is not None:
            return None
        explicit = config.get("columns", []) or []
        return input_schema if not explicit else input_schema.drop(explicit)

    def fit(self, df: Any, config: Dict[str, Any]) -> DropMissingColumnsArtifact:
        return cast(
            DropMissingColumnsArtifact,
            fit_dual_engine(
                df, config, _drop_missing_cols_fit_polars, _drop_missing_cols_fit_pandas
            ),
        )
