"""Drop / missing-value nodes (Deduplicate / DropMissingColumns / DropMissingRows / MissingIndicator).

Appliers route through :func:`apply_dual_engine`; fits that need engine-divergent
math (column null-percentage scans) route through :func:`fit_dual_engine`. The
two splitter-friendly nodes (Deduplicate, DropMissingRows) require a
keep-indices y-sync — handled by :func:`_polars_filter_y_by_kept_indices`.
"""

from typing import Any, Dict, Optional, Tuple, cast

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from .base import BaseApplier, BaseCalculator, apply_method
from .dispatcher import apply_dual_engine, fit_dual_engine
from ._artifacts import (
    DeduplicateArtifact,
    DropMissingColumnsArtifact,
    DropMissingRowsArtifact,
    MissingIndicatorArtifact,
)
from ._schema import SkyulfSchema


# -----------------------------------------------------------------------------
# Shared polars helpers
# -----------------------------------------------------------------------------


def _polars_filter_y_by_kept_indices(y: Any, kept_indices: Any) -> Any:
    """Filter ``y`` (Polars Series / DataFrame) to the rows kept in ``X``.

    ``kept_indices`` is a Polars Series of integer row indices that survived
    a filter on ``X``. Used by Deduplicate + DropMissingRows so dropping rows
    in ``X`` propagates to a paired ``y``.
    """
    import polars as pl

    if y is None:
        return None
    if isinstance(y, pl.DataFrame):
        return (
            y.with_row_index("__idx__")
            .filter(pl.col("__idx__").is_in(kept_indices))
            .drop("__idx__")
        )
    if isinstance(y, pl.Series):
        return y.gather(kept_indices)
    return y


def _normalize_subset(subset: Any, existing_cols: list) -> Optional[list]:
    """Filter ``subset`` to columns that actually exist; return ``None`` if empty."""
    if not subset:
        return None
    filtered = [c for c in subset if c in existing_cols]
    return filtered if filtered else None


# -----------------------------------------------------------------------------
# Deduplicate
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Drop Missing Columns
# -----------------------------------------------------------------------------


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
    """Polars: list of columns whose missing-% ≥ ``threshold_pct``."""
    null_counts = X.null_count()
    total = X.height or 1
    return [c for c in X.columns if (null_counts[c][0] / total) * 100 >= threshold_pct]


def _high_missing_cols_pandas(X: Any, threshold_pct: float) -> list:
    """Pandas: list of columns whose missing-% ≥ ``threshold_pct``."""
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


# -----------------------------------------------------------------------------
# Drop Missing Rows
# -----------------------------------------------------------------------------


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
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
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


# -----------------------------------------------------------------------------
# Missing Indicator
# -----------------------------------------------------------------------------


def _missing_indicator_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    cols = params.get("columns", [])
    if not cols:
        return X, y
    exprs = [
        pl.col(c).is_null().cast(pl.Int64).alias(f"{c}_missing") for c in cols if c in X.columns
    ]
    return (X.with_columns(exprs) if exprs else X), y


def _missing_indicator_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    cols = params.get("columns", [])
    if not cols:
        return X, y
    X_out = X.copy()
    for col in cols:
        if col in X.columns:
            X_out[f"{col}_missing"] = X[col].isna().astype(int)
    return X_out, y


class MissingIndicatorApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            X, params, _missing_indicator_apply_polars, _missing_indicator_apply_pandas
        )


def _missing_cols_polars(X: Any) -> list:
    null_counts = X.null_count()
    return [c for c in X.columns if null_counts[c][0] > 0]


def _missing_cols_pandas(X: Any) -> list:
    return X.columns[X.isna().any()].tolist()


def _missing_indicator_fit_polars(
    X: Any, _y: Any, config: Dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_polars(X)
    return {"type": "missing_indicator", "columns": cols}


def _missing_indicator_fit_pandas(
    X: Any, _y: Any, config: Dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_pandas(X)
    return {"type": "missing_indicator", "columns": cols}


@NodeRegistry.register("MissingIndicator", MissingIndicatorApplier)
@node_meta(
    id="MissingIndicator",
    name="Missing Indicator",
    category="Feature Engineering",
    description="Create binary indicators for missing values.",
    params={"features": "missing-only", "sparse": "auto"},
)
class MissingIndicatorCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
        # Adds one boolean column "<col>_missing" per indicator column.
        # Only predictable when the user supplied an explicit column list;
        # otherwise the set depends on which columns actually contain NaNs.
        explicit = config.get("columns") or []
        if not explicit:
            return None
        new_schema = input_schema
        for col in explicit:
            new_schema = new_schema.add(f"{col}_missing", "bool")
        return new_schema

    def fit(self, df: Any, config: Dict[str, Any]) -> MissingIndicatorArtifact:
        return cast(
            MissingIndicatorArtifact,
            fit_dual_engine(
                df, config, _missing_indicator_fit_polars, _missing_indicator_fit_pandas
            ),
        )
