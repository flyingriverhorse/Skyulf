from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from .base import BaseCalculator, BaseApplier
from ..utils import unpack_pipeline_input, pack_pipeline_output

# --- Deduplicate ---
class DeduplicateCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'subset': [...], 'keep': 'first'|'last'|False}
        # Deduplication is an operation that doesn't learn parameters from data, 
        # it just applies logic. So fit just passes through the config.
        
        subset = config.get('subset')
        keep = config.get('keep', 'first')
        
        return {
            'type': 'deduplicate',
            'subset': subset,
            'keep': keep
        }

class DeduplicateApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        subset = params.get('subset')
        keep = params.get('keep', 'first')
        
        # Handle 'none' string from config
        if keep == 'none':
            keep = False
            
        if subset:
            subset = [c for c in subset if c in X.columns]
            if not subset:
                subset = None
        
        X_dedup = X.drop_duplicates(subset=subset, keep=keep)
        
        if is_tuple and y is not None:
            # Align y with X
            y_dedup = y.loc[X_dedup.index]
            return pack_pipeline_output(X_dedup, y_dedup, is_tuple)

        return pack_pipeline_output(X_dedup, y, is_tuple)

# --- Drop Missing Columns ---
class DropMissingColumnsCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'threshold': 50.0 (percent), 'columns': [...]}
        # Threshold is percentage of missing values allowed. If missing > threshold, drop.
        
        threshold = config.get('missing_threshold')
        explicit_cols = config.get('columns', [])
        
        cols_to_drop = set()
        
        if explicit_cols:
            cols_to_drop.update([c for c in explicit_cols if c in X.columns])
            
        if threshold is not None:
            try:
                threshold_val = float(threshold)
                # If threshold is 0, it means "Drop if missing >= 0%".
                # This drops ALL columns (since missing % is always >= 0).
                # Usually, users mean "Drop if missing > 0%" (Strict) or "Disable" if 0.
                # If the user sends 0, they likely mean "Don't use threshold" OR "Drop any missing".
                # However, to fix the "0 rows" bug where everything is dropped because default is 0:
                if threshold_val > 0:
                    missing_pct = X.isna().mean() * 100
                    auto_dropped = missing_pct[missing_pct >= threshold_val].index.tolist()
                    cols_to_drop.update(auto_dropped)
            except (TypeError, ValueError):
                pass
                
        return {
            'type': 'drop_missing_columns',
            'columns_to_drop': list(cols_to_drop),
            'threshold': threshold
        }

class DropMissingColumnsApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols_to_drop = params.get('columns_to_drop', [])
        cols_to_drop_X = [c for c in cols_to_drop if c in X.columns]
        
        if cols_to_drop_X:
            X = X.drop(columns=cols_to_drop_X)
            
        return pack_pipeline_output(X, y, is_tuple)

# --- Drop Missing Rows ---
class DropMissingRowsCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
            
        # Config: {'missing_threshold': 50.0, 'drop_if_any_missing': bool}
        # This doesn't learn from data, just passes config.
        
        return {
            'type': 'drop_missing_rows',
            'missing_threshold': config.get('missing_threshold'),
            'drop_if_any_missing': config.get('drop_if_any_missing', False)
        }

class DropMissingRowsApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        threshold = params.get('missing_threshold')
        drop_if_any = params.get('drop_if_any_missing', False)
        
        if X.empty:
            return pack_pipeline_output(X, y, is_tuple)
            
        mask = None
        if drop_if_any:
            mask = ~X.isna().any(axis=1)
        elif threshold is not None:
            try:
                threshold_val = float(threshold)
                missing_pct = X.isna().mean(axis=1) * 100
                mask = missing_pct < threshold_val
            except (TypeError, ValueError):
                pass
        
        if mask is not None:
            X = X[mask]
            if y is not None:
                y = y[mask]
                
        return pack_pipeline_output(X, y, is_tuple)

# --- Missing Indicator ---
class MissingIndicatorCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'columns': [...], 'flag_suffix': '_was_missing'}
        # If columns not provided, auto-detect columns with missing values.
        
        cols = config.get('columns')
        suffix = config.get('flag_suffix', '_was_missing')
        
        if not cols:
            # Auto-detect columns with any missing values
            cols = X.columns[X.isna().any()].tolist()
        
        cols = [c for c in cols if c in X.columns]
        
        return {
            'type': 'missing_indicator',
            'columns': cols,
            'flag_suffix': suffix
        }

class MissingIndicatorApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        suffix = params.get('flag_suffix', '_was_missing')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        for col in valid_cols:
            indicator_name = f"{col}{suffix}"
            # 1 if missing, 0 otherwise
            indicator = X_out[col].isna().astype(int)
            
            if indicator_name in X_out.columns:
                X_out[indicator_name] = indicator
            else:
                # Insert next to original column if possible
                loc = X_out.columns.get_loc(col) + 1
                X_out.insert(loc, indicator_name, indicator)
                
        return pack_pipeline_output(X_out, y, is_tuple)
