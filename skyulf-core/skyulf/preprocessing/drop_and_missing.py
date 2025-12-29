from typing import Any, Dict, Tuple, Union

import pandas as pd

from ..registry import NodeRegistry
from ..utils import pack_pipeline_output, unpack_pipeline_input
from .base import BaseApplier, BaseCalculator
from ..engines import SkyulfDataFrame, get_engine

# --- Deduplicate ---


class DeduplicateApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
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
        if engine.name == "polars":
            import polars as pl
            
            # Map keep parameter
            # Pandas: 'first', 'last', False
            # Polars: 'first', 'last', 'none' (if False)
            pl_keep = keep
            if keep is False:
                pl_keep = "none"
            
            if is_tuple and y is not None:
                # We need to sync X and y
                # Combine them
                # We need to ensure y name doesn't conflict
                # Assuming y is a Series or DataFrame with 1 col
                
                # If y is a Series/DataFrame, we can hstack
                # But we need to know which columns belong to X and which to y
                x_cols = X.columns
                
                # If y is unnamed or has name collision, rename it temporarily?
                # Or just use index? Polars has no index.
                # Best way: add a row index, filter X, get kept indices, filter y.
                
                X_with_idx = X.with_row_index("__idx__")
                X_dedup = X_with_idx.unique(subset=subset, keep=pl_keep, maintain_order=True)
                kept_indices = X_dedup["__idx__"]
                
                # Filter y
                # y must be a DataFrame or Series. If it's a Series, convert to DF to filter?
                # Or use filter/take
                if isinstance(y, pl.DataFrame):
                    y_dedup = y.filter(pl.col("__idx__").is_in(kept_indices)) # Wait, y doesn't have __idx__
                    # We need to add row index to y too, assuming they are aligned
                    y_dedup = y.with_row_index("__idx__").filter(pl.col("__idx__").is_in(kept_indices)).drop("__idx__")
                elif isinstance(y, pl.Series):
                    # Series doesn't have with_row_index directly in same way? 
                    # Actually Series has no index. We can use take/gather.
                    y_dedup = y.gather(kept_indices)
                else:
                    # Should not happen if unpack works correctly
                    y_dedup = y
                
                X_out = X_dedup.drop("__idx__")
                return pack_pipeline_output(X_out, y_dedup, is_tuple)
            
            else:
                X_out = X.unique(subset=subset, keep=pl_keep, maintain_order=True)
                return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_dedup = X.drop_duplicates(subset=subset, keep=keep)

        if is_tuple and y is not None:
            # Align y with X
            y_dedup = y.loc[X_dedup.index]
            return pack_pipeline_output(X_dedup, y_dedup, is_tuple)

        return pack_pipeline_output(X_dedup, y, is_tuple)


@NodeRegistry.register("Deduplicate", DeduplicateApplier)
class DeduplicateCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'subset': [...], 'keep': 'first'|'last'|False}
        # Deduplication is an operation that doesn't learn parameters from data,
        # it just applies logic. So fit just passes through the config.

        subset = config.get("subset")
        keep = config.get("keep", "first")

        return {"type": "deduplicate", "subset": subset, "keep": keep}


# --- Drop Missing Columns ---


class DropMissingColumnsApplier(BaseApplier):
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols_to_drop = params.get("columns_to_drop", [])
        cols_to_drop_X = [c for c in cols_to_drop if c in X.columns]

        if cols_to_drop_X:
            if engine.name == "polars":
                X = X.drop(cols_to_drop_X)
            else:
                X = X.drop(columns=cols_to_drop_X)

        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("DropMissingColumns", DropMissingColumnsApplier)
class DropMissingColumnsCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)

        # Config: {'threshold': 50.0 (percent), 'columns': [...]}
        # Threshold is percentage of missing values allowed. If missing > threshold, drop.

        threshold = config.get("missing_threshold")
        explicit_cols = config.get("columns", [])

        cols_to_drop = set()

        if explicit_cols:
            cols_to_drop.update([c for c in explicit_cols if c in X.columns])

        if threshold is not None:
            try:
                threshold_val = float(threshold)
                if threshold_val > 0:
                    if engine.name == "polars":
                        import polars as pl
                        # Calculate missing percentage for all columns
                        # null_count() returns a DF with 1 row
                        null_counts = X.null_count()
                        total_rows = X.height
                        
                        for col in X.columns:
                            # Get null count for this column
                            # null_counts[col] is a Series of length 1
                            n_null = null_counts[col][0]
                            pct = (n_null / total_rows) * 100
                            if pct >= threshold_val:
                                cols_to_drop.add(col)
                    else:
                        missing_pct = X.isna().mean() * 100
                        auto_dropped = missing_pct[
                            missing_pct >= threshold_val
                        ].index.tolist()
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
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        subset = params.get("subset")
        how = params.get("how", "any")
        threshold = params.get("threshold")

        if subset:
            subset = [c for c in subset if c in X.columns]
            if not subset:
                subset = None

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            # We need to sync X and y, so we use the index trick
            X_with_idx = X.with_row_index("__idx__")
            
            # Determine columns to check
            check_cols = subset if subset else [c for c in X.columns if c != "__idx__"]
            
            if threshold is not None:
                # Keep rows with at least 'threshold' non-null values in check_cols
                # sum_horizontal of is_not_null
                X_clean = X_with_idx.filter(
                    pl.sum_horizontal(pl.col(check_cols).is_not_null()) >= threshold
                )
            elif how == "all":
                # Drop if ALL are null
                # Keep if NOT ALL are null
                X_clean = X_with_idx.filter(
                    ~pl.all_horizontal(pl.col(check_cols).is_null())
                )
            else:
                # how == "any" (default)
                # Drop if ANY is null
                X_clean = X_with_idx.drop_nulls(subset=check_cols)
                
            kept_indices = X_clean["__idx__"]
            
            if is_tuple and y is not None:
                if isinstance(y, pl.DataFrame):
                    y_clean = y.with_row_index("__idx__").filter(pl.col("__idx__").is_in(kept_indices)).drop("__idx__")
                elif isinstance(y, pl.Series):
                    y_clean = y.gather(kept_indices)
                else:
                    y_clean = y
                
                X_out = X_clean.drop("__idx__")
                return pack_pipeline_output(X_out, y_clean, is_tuple)
            
            X_out = X_clean.drop("__idx__")
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        # Pandas dropna forbids setting both 'how' and 'thresh'.
        # If 'thresh' is provided (not None), it takes precedence over 'how'.
        if threshold is not None:
            X_clean = X.dropna(axis=0, thresh=threshold, subset=subset)
        else:
            X_clean = X.dropna(axis=0, how=how, subset=subset)

        if is_tuple and y is not None:
            y_clean = y.loc[X_clean.index]
            return pack_pipeline_output(X_clean, y_clean, is_tuple)

        return pack_pipeline_output(X_clean, y, is_tuple)


@NodeRegistry.register("DropMissingRows", DropMissingRowsApplier)
class DropMissingRowsCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
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
    def apply(
        self,
        df: SkyulfDataFrame,
        params: Dict[str, Any],
    ) -> Union[SkyulfDataFrame, Tuple[SkyulfDataFrame, Any]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        
        if not cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            for col in cols:
                if col in X.columns:
                    exprs.append(pl.col(col).is_null().cast(pl.Int64).alias(f"{col}_missing"))
            
            if not exprs:
                 return pack_pipeline_output(X, y, is_tuple)
                 
            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        X_out = X.copy()
        for col in cols:
            if col in X.columns:
                X_out[f"{col}_missing"] = X[col].isna().astype(int)
                
        return pack_pipeline_output(X_out, y, is_tuple)


@NodeRegistry.register("MissingIndicator", MissingIndicatorApplier)
class MissingIndicatorCalculator(BaseCalculator):
    def fit(
        self,
        df: SkyulfDataFrame,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        engine = get_engine(X)
        
        explicit_cols = config.get("columns")
        
        if explicit_cols:
            cols = [c for c in explicit_cols if c in X.columns]
        else:
            if engine.name == "polars":
                import polars as pl
                null_counts = X.null_count()
                cols = [c for c in X.columns if null_counts[c][0] > 0]
            else:
                cols = X.columns[X.isna().any()].tolist()
                
        return {
            "type": "missing_indicator",
            "columns": cols
        }
