from typing import Any, Dict, Optional

from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._artifacts import (
    DeduplicateArtifact,
    DropMissingColumnsArtifact,
    DropMissingRowsArtifact,
    MissingIndicatorArtifact,
)
from ._schema import SkyulfSchema
from ..engines import EngineName, get_engine

# --- Deduplicate ---


class DeduplicateApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        subset = params.get("subset")
        keep = params.get("keep", "first")

        # Handle 'none' string from config
        if keep == "none":
            keep = False

        if subset:
            subset = [c for c in subset if c in X.columns]
            if not subset:
                subset = None

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            # Map keep parameter
            # Pandas: 'first', 'last', False
            # Polars: 'first', 'last', 'none' (if False)
            pl_keep = keep
            if keep is False:
                pl_keep = "none"

            if y is not None:
                # We need to sync X and y
                # Combine them
                # We need to ensure y name doesn't conflict
                # Assuming y is a Series or DataFrame with 1 col

                # If y is a Series/DataFrame, we can hstack
                # But we need to know which columns belong to X and which to y
                X_pl.columns

                # If y is unnamed or has name collision, rename it temporarily?
                # Or just use index? Polars has no index.
                # Best way: add a row index, filter X, get kept indices, filter y.

                X_with_idx = X_pl.with_row_index("__idx__")
                X_dedup = X_with_idx.unique(subset=subset, keep=pl_keep, maintain_order=True)
                kept_indices = X_dedup["__idx__"]

                # Filter y
                # y must be a DataFrame or Series. If it's a Series, convert to DF to filter?
                # Or use filter/take
                if isinstance(y, pl.DataFrame):
                    y_dedup = (
                        y.with_row_index("__idx__")
                        .filter(pl.col("__idx__").is_in(kept_indices))
                        .drop("__idx__")
                    )
                elif isinstance(y, pl.Series):
                    # Series doesn't have with_row_index directly in same way?
                    # Actually Series has no index. We can use take/gather.
                    y_dedup = y.gather(kept_indices)
                else:
                    # Should not happen if unpack works correctly
                    y_dedup = y

                X_out = X_dedup.drop("__idx__")
                return X_out, y_dedup

            else:
                X_out = X_pl.unique(subset=subset, keep=pl_keep, maintain_order=True)
                return X_out, y

        # Pandas Path
        X_dedup = X.drop_duplicates(subset=subset, keep=keep)

        if y is not None:
            # Align y with X
            y_dedup = y.loc[X_dedup.index]
            return X_dedup, y_dedup

        return X_dedup, y


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

    @fit_method
    def fit(
        self,
        X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> DeduplicateArtifact:
        # Config: {'subset': [...], 'keep': 'first'|'last'|False}
        # Deduplication is an operation that doesn't learn parameters from data,
        # it just applies logic. So fit just passes through the config.

        subset = config.get("subset")
        keep = config.get("keep", "first")

        return {"type": "deduplicate", "subset": subset, "keep": keep}


# --- Drop Missing Columns ---


class DropMissingColumnsApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        cols_to_drop = params.get("columns_to_drop", [])

        # Checking X.columns might be tricky if X is Any, but engine check helps
        if engine.name == EngineName.POLARS:
            # X is likely Polars, but let's be safe
            X_pl: Any = X
            cols_to_drop_X = [c for c in cols_to_drop if c in X_pl.columns]
        else:
            cols_to_drop_X = [c for c in cols_to_drop if c in X.columns]

        if cols_to_drop_X:
            if engine.name == EngineName.POLARS:
                X_pl_data: Any = X
                X = X_pl_data.drop(cols_to_drop_X)
            else:
                X = X.drop(columns=cols_to_drop_X)

        return X, y


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
        # Threshold path is data-dependent; we can only predict the schema
        # when the user supplied an explicit column list and no threshold.
        threshold = config.get("missing_threshold")
        explicit = config.get("columns", []) or []
        if threshold is not None:
            try:
                if float(threshold) > 0:
                    return None
            except (TypeError, ValueError):
                pass
        if not explicit:
            return input_schema
        return input_schema.drop(explicit)

    @fit_method
    def fit(
        self,
        X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> DropMissingColumnsArtifact:
        engine = get_engine(X)

        # Config: {'threshold': 50.0 (percent), 'columns': [...]}
        # Threshold is percentage of missing values allowed. If missing > threshold, drop.

        threshold = config.get("missing_threshold")
        explicit_cols = config.get("columns", [])

        cols_to_drop = set()

        # Handle X access depending on engine
        if engine.name == EngineName.POLARS:
            X_pl: Any = X
            if explicit_cols:
                cols_to_drop.update([c for c in explicit_cols if c in X_pl.columns])
        else:
            if explicit_cols:
                cols_to_drop.update([c for c in explicit_cols if c in X.columns])

        if threshold is not None:
            try:
                threshold_val = float(threshold)
                if threshold_val > 0:
                    if engine.name == EngineName.POLARS:
                        pass

                        X_pl_data: Any = X
                        # Calculate missing percentage for all columns
                        # null_count() returns a DF with 1 row
                        null_counts = X_pl_data.null_count()
                        total_rows = X_pl_data.height

                        for col in X_pl_data.columns:
                            # Get null count for this column
                            # null_counts[col] is a Series of length 1
                            n_null = null_counts[col][0]
                            pct = (n_null / total_rows) * 100
                            if pct >= threshold_val:
                                cols_to_drop.add(col)
                    else:
                        missing_pct = X.isna().mean() * 100
                        auto_dropped = missing_pct[missing_pct >= threshold_val].index.tolist()
                        cols_to_drop.update(auto_dropped)
            except (TypeError, ValueError):
                pass

        return {
            "type": "drop_missing_columns",
            "columns_to_drop": list(cols_to_drop),
            "threshold": threshold,
        }


# --- Drop Missing Rows ---


class DropMissingRowsApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        subset = params.get("subset")
        how = params.get("how", "any")
        threshold = params.get("threshold")

        if subset:
            if engine.name == EngineName.POLARS:
                X_pl: Any = X
                subset = [c for c in subset if c in X_pl.columns]
            else:
                subset = [c for c in subset if c in X.columns]

            if not subset:
                subset = None

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl_data: Any = X

            # We need to sync X and y, so we use the index trick
            X_with_idx = X_pl_data.with_row_index("__idx__")

            # Determine columns to check
            check_cols = subset if subset else [c for c in X_pl_data.columns if c != "__idx__"]

            if threshold is not None:
                # Keep rows with at least 'threshold' non-null values in check_cols
                # sum_horizontal of is_not_null
                X_clean = X_with_idx.filter(
                    pl.sum_horizontal(pl.col(check_cols).is_not_null()) >= threshold
                )
            elif how == "all":
                # Drop if ALL are null
                # Keep if NOT ALL are null
                X_clean = X_with_idx.filter(~pl.all_horizontal(pl.col(check_cols).is_null()))
            else:
                # how == "any" (default)
                # Drop if ANY is null
                X_clean = X_with_idx.drop_nulls(subset=check_cols)

            kept_indices = X_clean["__idx__"]

            if y is not None:
                if isinstance(y, pl.DataFrame):
                    y_clean = (
                        y.with_row_index("__idx__")
                        .filter(pl.col("__idx__").is_in(kept_indices))
                        .drop("__idx__")
                    )
                elif isinstance(y, pl.Series):
                    y_clean = y.gather(kept_indices)
                else:
                    y_clean = y

                X_out = X_clean.drop("__idx__")
                return X_out, y_clean

            X_out = X_clean.drop("__idx__")
            return X_out, y

        # Pandas Path
        # Pandas dropna forbids setting both 'how' and 'thresh'.
        # If 'thresh' is provided (not None), it takes precedence over 'how'.
        if threshold is not None:
            X_clean = X.dropna(axis=0, thresh=threshold, subset=subset)
        else:
            X_clean = X.dropna(axis=0, how=how, subset=subset)

        if y is not None:
            y_clean = y.loc[X_clean.index]
            return X_clean, y_clean

        return X_clean, y


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

    @fit_method
    def fit(
        self,
        _X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> DropMissingRowsArtifact:
        # Config: {'subset': [...], 'how': 'any'|'all', 'threshold': int}
        subset = config.get("subset")
        how = config.get("how", "any")
        threshold = config.get("threshold")

        return {
            "type": "drop_missing_rows",
            "subset": subset,
            "how": how,
            "threshold": threshold,
        }


# --- Missing Indicator ---


class MissingIndicatorApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        cols = params.get("columns", [])

        if not cols:
            return X, y

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            exprs = []
            for col in cols:
                # X_pl.columns check
                if col in X_pl.columns:
                    exprs.append(pl.col(col).is_null().cast(pl.Int64).alias(f"{col}_missing"))

            if not exprs:
                return X, y

            X_out = X_pl.with_columns(exprs)
            return X_out, y

        # Pandas Path
        X_out = X.copy()
        for col in cols:
            if col in X.columns:
                X_out[f"{col}_missing"] = X[col].isna().astype(int)

        return X_out, y


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

    @fit_method
    def fit(
        self,
        X: Any,
        _y: Any,
        config: Dict[str, Any],
    ) -> MissingIndicatorArtifact:
        engine = get_engine(X)

        explicit_cols = config.get("columns")

        if explicit_cols:
            if engine.name == EngineName.POLARS:
                X_pl_data: Any = X
                cols = [c for c in explicit_cols if c in X_pl_data.columns]
            else:
                cols = [c for c in explicit_cols if c in X.columns]
        else:
            if engine.name == EngineName.POLARS:
                pass

                X_pl: Any = X
                null_counts = X_pl.null_count()
                cols = [c for c in X_pl.columns if null_counts[c][0] > 0]
            else:
                cols = X.columns[X.isna().any()].tolist()

        return {"type": "missing_indicator", "columns": cols}
