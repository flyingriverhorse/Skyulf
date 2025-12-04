from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns

# --- Base Binning Applier ---
class BaseBinningApplier(BaseApplier):
    """
    Base class for applying binning transformations.
    Expects 'bin_edges' in params: Dict[str, List[float]] mapping column names to bin edges.
    """
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        bin_edges_map = params.get('bin_edges', {})
        if not bin_edges_map:
            return df
            
        output_suffix = params.get('output_suffix', '_binned')
        drop_original = params.get('drop_original', False)
        label_format = params.get('label_format', 'ordinal') # ordinal, range, bin_index
        missing_strategy = params.get('missing_strategy', 'keep') # keep, label
        missing_label = params.get('missing_label', 'Missing')
        include_lowest = params.get('include_lowest', True)
        precision = params.get('precision', 3)
        custom_labels_map = params.get('custom_labels', {})
        
        df_out = df.copy()
        processed_cols = []
        
        for col, edges in bin_edges_map.items():
            if col not in df_out.columns:
                continue
                
            processed_cols.append(col)
            
            # Determine labels for pd.cut
            labels = False # Default for ordinal (returns integers)
            
            # Check for custom labels first
            col_custom_labels = custom_labels_map.get(col)
            if col_custom_labels and len(col_custom_labels) == len(edges) - 1:
                labels = col_custom_labels
            elif label_format == 'range':
                labels = None # Returns intervals
            elif label_format == 'bin_index':
                labels = False # Returns integers 0..n-1
            
            # Apply cut
            try:
                # Ensure edges are unique and sorted
                sorted_edges = sorted(list(set(edges)))
                if len(sorted_edges) < 2:
                    continue
                    
                binned_series = pd.cut(
                    df_out[col], 
                    bins=sorted_edges, 
                    labels=labels, 
                    include_lowest=include_lowest
                )
                
                # Handle missing values
                if missing_strategy == 'label':
                    # If categorical (range or custom labels), add category
                    if isinstance(binned_series.dtype, pd.CategoricalDtype):
                        if missing_label not in binned_series.cat.categories:
                            binned_series = binned_series.cat.add_categories([missing_label])
                        binned_series = binned_series.fillna(missing_label)
                    else:
                        # If numeric (ordinal/bin_index), we convert to object/str to support "Missing" label
                        binned_series = binned_series.astype(object).fillna(missing_label)
                
                # Format ranges if needed
                if label_format == 'range' and labels is None:
                     # Convert intervals to string with precision
                     if isinstance(binned_series.dtype, pd.CategoricalDtype):
                         # It's a categorical of intervals
                         def format_interval(iv):
                             if pd.isna(iv) or isinstance(iv, str): return iv
                             return f"[{round(iv.left, precision)}, {round(iv.right, precision)}]" if include_lowest else f"({round(iv.left, precision)}, {round(iv.right, precision)}]"
                         
                         # We need to map the categories themselves
                         new_categories = [format_interval(c) for c in binned_series.cat.categories]
                         binned_series.cat.categories = new_categories
                         binned_series = binned_series.astype(str)
                     else:
                         binned_series = binned_series.astype(str)

                     if missing_strategy == 'keep':
                         # Restore NaNs if they were converted to 'nan' string
                         binned_series = binned_series.replace('nan', np.nan)

                out_col = f"{col}{output_suffix}"
                df_out[out_col] = binned_series
                
            except Exception:
                # Log error or skip
                pass
                
        if drop_original:
            df_out = df_out.drop(columns=processed_cols)
            
        return df_out

# --- General Binning Calculator (V1 Compatibility) ---
class GeneralBinningCalculator(BaseCalculator):
    """
    Master calculator that handles mixed strategies and overrides (V1 parity).
    """
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        columns = resolve_columns(df, config, detect_numeric_columns)
        
        global_strategy = config.get('strategy', 'equal_width')
        column_strategies = config.get('column_strategies', {})
        
        # Global settings
        n_bins_global = config.get('equal_width_bins', 5)
        q_bins_global = config.get('equal_frequency_bins', 5)
        duplicates_global = config.get('duplicates', 'raise')
        
        valid_cols = [c for c in columns if c in df.columns]
        bin_edges_map = {}
        custom_labels_map = {}
        
        for col in valid_cols:
            # Determine strategy and params for this column
            override = column_strategies.get(col, {})
            strategy = override.get('strategy', global_strategy)
            
            try:
                series = df[col].dropna()
                if series.empty:
                    continue
                
                edges = None
                
                if strategy == 'equal_width':
                    n_bins = override.get('equal_width_bins', n_bins_global)
                    _, edges = pd.cut(series, bins=n_bins, retbins=True)
                    
                elif strategy == 'equal_frequency':
                    n_bins = override.get('equal_frequency_bins', q_bins_global)
                    duplicates = override.get('duplicates', duplicates_global)
                    _, edges = pd.qcut(series, q=n_bins, retbins=True, duplicates=duplicates)
                    
                elif strategy == 'custom':
                    # Check override first, then global custom_bins
                    custom_bins = override.get('custom_bins')
                    if not custom_bins:
                        custom_bins = config.get('custom_bins', {}).get(col)
                    
                    if custom_bins:
                        edges = np.array(sorted(custom_bins))
                        
                    # Handle custom labels
                    labels = override.get('custom_labels')
                    if not labels:
                        labels = config.get('custom_labels', {}).get(col)
                    if labels:
                        custom_labels_map[col] = labels
                        
                elif strategy == 'kbins':
                    n_bins = override.get('kbins_n_bins', config.get('kbins_n_bins', 5))
                    k_strategy = override.get('kbins_strategy', config.get('kbins_strategy', 'quantile'))
                    
                    # Map strategy names
                    sklearn_strategy = k_strategy
                    if k_strategy == 'equal_width': sklearn_strategy = 'uniform'
                    elif k_strategy == 'equal_frequency': sklearn_strategy = 'quantile'
                    
                    est = KBinsDiscretizer(n_bins=n_bins, strategy=sklearn_strategy, encode='ordinal')
                    est.fit(series.values.reshape(-1, 1))
                    edges = est.bin_edges_[0]
                
                if edges is not None:
                    bin_edges_map[col] = edges.tolist()
                    
            except Exception:
                continue
        
        return {
            'type': 'general_binning',
            'bin_edges': bin_edges_map,
            'custom_labels': custom_labels_map,
            'output_suffix': config.get('output_suffix', '_binned'),
            'drop_original': config.get('drop_original', False),
            'label_format': config.get('label_format', 'ordinal'),
            'missing_strategy': config.get('missing_strategy', 'keep'),
            'missing_label': config.get('missing_label', 'Missing'),
            'include_lowest': config.get('include_lowest', True),
            'precision': config.get('precision', 3)
        }

class GeneralBinningApplier(BaseBinningApplier):
    pass

# --- Equal Width Binning ---
class EqualWidthBinningCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        n_bins = config.get('n_bins', 5)
        
        valid_cols = [c for c in cols if c in df.columns]
        bin_edges_map = {}
        
        for col in valid_cols:
            # Use pd.cut to find edges
            try:
                # dropna is important
                series = df[col].dropna()
                if series.empty:
                    continue
                _, edges = pd.cut(series, bins=n_bins, retbins=True)
                bin_edges_map[col] = edges.tolist()
            except Exception:
                continue
                
        return {
            'type': 'equal_width',
            'bin_edges': bin_edges_map,
            'output_suffix': config.get('output_suffix', '_binned'),
            'drop_original': config.get('drop_original', False),
            'label_format': config.get('label_format', 'ordinal'),
            'missing_strategy': config.get('missing_strategy', 'keep'),
            'missing_label': config.get('missing_label', 'Missing'),
            'include_lowest': config.get('include_lowest', True),
            'precision': config.get('precision', 3)
        }

class EqualWidthBinningApplier(BaseBinningApplier):
    pass

# --- Equal Frequency Binning ---
class EqualFrequencyBinningCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        n_bins = config.get('n_bins', 5)
        duplicates = config.get('duplicates', 'drop') # raise or drop
        
        valid_cols = [c for c in cols if c in df.columns]
        bin_edges_map = {}
        
        for col in valid_cols:
            try:
                series = df[col].dropna()
                if series.empty:
                    continue
                _, edges = pd.qcut(series, q=n_bins, retbins=True, duplicates=duplicates)
                bin_edges_map[col] = edges.tolist()
            except Exception:
                continue
                
        return {
            'type': 'equal_frequency',
            'bin_edges': bin_edges_map,
            'output_suffix': config.get('output_suffix', '_binned'),
            'drop_original': config.get('drop_original', False),
            'label_format': config.get('label_format', 'ordinal'),
            'missing_strategy': config.get('missing_strategy', 'keep'),
            'missing_label': config.get('missing_label', 'Missing'),
            'include_lowest': config.get('include_lowest', True),
            'precision': config.get('precision', 3)
        }

class EqualFrequencyBinningApplier(BaseBinningApplier):
    pass

# --- Custom Binning ---
class CustomBinningCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        bins_map = config.get('bins', {})
        labels_map = config.get('labels', {})
        
        valid_bins = {}
        valid_labels = {}
        
        for col, edges in bins_map.items():
            if col in df.columns:
                valid_bins[col] = edges
                if col in labels_map:
                    valid_labels[col] = labels_map[col]
                
        return {
            'type': 'custom_binning',
            'bin_edges': valid_bins, # Standardize key to bin_edges
            'custom_labels': valid_labels,
            'output_suffix': config.get('output_suffix', '_binned'),
            'drop_original': config.get('drop_original', False),
            'label_format': config.get('label_format', 'ordinal'),
            'missing_strategy': config.get('missing_strategy', 'keep'),
            'missing_label': config.get('missing_label', 'Missing'),
            'include_lowest': config.get('include_lowest', True),
            'precision': config.get('precision', 3)
        }

class CustomBinningApplier(BaseBinningApplier):
    pass

# --- KBins Discretizer (Sklearn Wrapper) ---
class KBinsDiscretizerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        n_bins = config.get('n_bins', 5)
        strategy = config.get('strategy', 'quantile')
        encode = config.get('encode', 'ordinal')
        
        # Map strategies if needed
        sklearn_strategy = strategy
        if strategy == 'equal_width':
            sklearn_strategy = 'uniform'
        elif strategy == 'equal_frequency':
            sklearn_strategy = 'quantile'
            
        est = KBinsDiscretizer(n_bins=n_bins, strategy=sklearn_strategy, encode=encode, subsample=None)
        
        df_clean = df[cols].dropna()
        if df_clean.empty:
            return {}
            
        est.fit(df_clean)
        
        # Extract edges
        bin_edges_map = {}
        for i, col in enumerate(cols):
            bin_edges_map[col] = est.bin_edges_[i].tolist()
            
        return {
            'type': 'kbins',
            'bin_edges': bin_edges_map,
            'n_bins': n_bins,
            'strategy': strategy,
            'encode': encode,
            'output_suffix': config.get('output_suffix', '_binned'),
            'drop_original': config.get('drop_original', False),
            'label_format': config.get('label_format', 'ordinal'),
            'missing_strategy': config.get('missing_strategy', 'keep'),
            'missing_label': config.get('missing_label', 'Missing'),
            'include_lowest': config.get('include_lowest', True),
            'precision': config.get('precision', 3)
        }

class KBinsDiscretizerApplier(BaseBinningApplier):
    pass
