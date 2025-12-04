from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from .base import BaseCalculator, BaseApplier

# --- Deduplicate ---
class DeduplicateCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
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
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        subset = params.get('subset')
        keep = params.get('keep', 'first')
        
        # Handle 'none' string from V1 config
        if keep == 'none':
            keep = False
            
        # Validate subset columns
        if subset:
            subset = [c for c in subset if c in df.columns]
            if not subset:
                subset = None # Fallback to all columns if none of the subset cols exist
        
        return df.drop_duplicates(subset=subset, keep=keep)

# --- Drop Missing Columns ---
class DropMissingColumnsCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'threshold': 50.0 (percent), 'columns': [...]}
        # Threshold is percentage of missing values allowed. If missing > threshold, drop.
        # Or maybe threshold is "drop if missing >= threshold".
        # V1: "auto_dropped = [col for col, value in missing_pct.items() if value >= threshold_value]"
        
        threshold = config.get('missing_threshold')
        explicit_cols = config.get('columns', [])
        
        cols_to_drop = set()
        
        # Explicitly requested columns to drop (if that's the intent, or maybe these are candidates?)
        # V1 logic: "drop_candidates = {str(column) for column in configured_columns ...}"
        # So explicit columns are candidates to be checked? Or forced drops?
        # V1: "removable_columns = [col for col in drop_candidates if col in frame.columns]"
        # And drop_candidates includes explicit columns AND auto_dropped.
        # So explicit columns are ALWAYS dropped? 
        # Wait, V1 code:
        # drop_candidates = {configured_columns}
        # if threshold: auto_dropped = ...; drop_candidates.update(auto_dropped)
        # So yes, configured columns are added to drop_candidates unconditionally.
        
        if explicit_cols:
            cols_to_drop.update([c for c in explicit_cols if c in df.columns])
            
        if threshold is not None:
            try:
                threshold_val = float(threshold)
                missing_pct = df.isna().mean() * 100
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
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols_to_drop = params.get('columns_to_drop', [])
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        
        if not cols_to_drop:
            return df
            
        return df.drop(columns=cols_to_drop)

# --- Drop Missing Rows ---
class DropMissingRowsCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'missing_threshold': 50.0, 'drop_if_any_missing': bool}
        # This doesn't learn from data, just passes config.
        
        return {
            'type': 'drop_missing_rows',
            'missing_threshold': config.get('missing_threshold'),
            'drop_if_any_missing': config.get('drop_if_any_missing', False)
        }

class DropMissingRowsApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        threshold = params.get('missing_threshold')
        drop_if_any = params.get('drop_if_any_missing', False)
        
        if df.empty:
            return df
            
        if drop_if_any:
            return df.dropna()
            
        if threshold is not None:
            try:
                threshold_val = float(threshold)
                # Drop rows where missing % >= threshold
                # missing_pct per row
                missing_pct = df.isna().mean(axis=1) * 100
                # Keep rows where missing < threshold
                return df[missing_pct < threshold_val]
            except (TypeError, ValueError):
                pass
                
        return df

# --- Missing Indicator ---
class MissingIndicatorCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'columns': [...], 'flag_suffix': '_was_missing'}
        # If columns not provided, auto-detect columns with missing values.
        
        cols = config.get('columns')
        suffix = config.get('flag_suffix', '_was_missing')
        
        if not cols:
            # Auto-detect columns with any missing values
            cols = df.columns[df.isna().any()].tolist()
        
        cols = [c for c in cols if c in df.columns]
        
        return {
            'type': 'missing_indicator',
            'columns': cols,
            'flag_suffix': suffix
        }

class MissingIndicatorApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        suffix = params.get('flag_suffix', '_was_missing')
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_out = df.copy()
        
        for col in valid_cols:
            indicator_name = f"{col}{suffix}"
            # 1 if missing, 0 otherwise
            indicator = df_out[col].isna().astype(int)
            
            if indicator_name in df_out.columns:
                df_out[indicator_name] = indicator
            else:
                # Insert next to original column if possible
                loc = df_out.columns.get_loc(col) + 1
                df_out.insert(loc, indicator_name, indicator)
                
        return df_out
