from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns

logger = logging.getLogger(__name__)

# --- Standard Scaler ---
class StandardScalerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'with_mean': True, 'with_std': True, 'columns': [...]}
        with_mean = config.get('with_mean', True)
        with_std = config.get('with_std', True)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        scaler.fit(df[cols])
        
        return {
            'type': 'standard_scaler',
            'mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
            'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None,
            'var': scaler.var_.tolist() if scaler.var_ is not None else None,
            'with_mean': with_mean,
            'with_std': with_std,
            'columns': cols
        }

class StandardScalerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        mean = params.get('mean')
        scale = params.get('scale')
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        # Reconstruct scaler (lightweight) or apply manually
        # Manual application is safer for portability
        df_out = df.copy()
        
        if mean is None or scale is None:
            return df
            
        mean_arr = np.array(mean)
        scale_arr = np.array(scale)
        
        # Filter mean/scale to match valid_cols indices
        # The params['columns'] order matches mean/scale order.
        # We need to map them correctly.
        
        col_indices = [cols.index(c) for c in valid_cols]
        
        X = df_out[valid_cols].values
        
        if params.get('with_mean', True):
            X = X - mean_arr[col_indices]
            
        if params.get('with_std', True):
            # Avoid division by zero if scale is 0 (constant feature)
            # Sklearn handles this by setting scale to 1.0 for constant features usually
            safe_scale = scale_arr[col_indices]
            safe_scale[safe_scale == 0] = 1.0
            X = X / safe_scale
            
        df_out[valid_cols] = X
        return df_out

# --- MinMax Scaler ---
class MinMaxScalerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'feature_range': (0, 1), 'columns': [...]}
        feature_range = config.get('feature_range', (0, 1))
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        scaler = MinMaxScaler(feature_range=feature_range)
        scaler.fit(df[cols])
        
        return {
            'type': 'minmax_scaler',
            'min': scaler.min_.tolist(),
            'scale': scaler.scale_.tolist(),
            'data_min': scaler.data_min_.tolist(),
            'data_max': scaler.data_max_.tolist(),
            'feature_range': feature_range,
            'columns': cols
        }

class MinMaxScalerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        min_val = params.get('min')
        scale = params.get('scale')
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or min_val is None or scale is None:
            return df
            
        df_out = df.copy()
        
        min_arr = np.array(min_val)
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]
        
        X = df_out[valid_cols].values
        X = X * scale_arr[col_indices] + min_arr[col_indices]
        
        df_out[valid_cols] = X
        return df_out

# --- Robust Scaler ---
class RobustScalerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'quantile_range': (25.0, 75.0), 'with_centering': True, 'with_scaling': True, 'columns': [...]}
        quantile_range = config.get('quantile_range', (25.0, 75.0))
        with_centering = config.get('with_centering', True)
        with_scaling = config.get('with_scaling', True)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        scaler = RobustScaler(quantile_range=quantile_range, with_centering=with_centering, with_scaling=with_scaling)
        scaler.fit(df[cols])
        
        return {
            'type': 'robust_scaler',
            'center': scaler.center_.tolist() if scaler.center_ is not None else None,
            'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None,
            'quantile_range': quantile_range,
            'with_centering': with_centering,
            'with_scaling': with_scaling,
            'columns': cols
        }

class RobustScalerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        center = params.get('center')
        scale = params.get('scale')
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_out = df.copy()
        col_indices = [cols.index(c) for c in valid_cols]
        X = df_out[valid_cols].values
        
        if params.get('with_centering', True) and center is not None:
            center_arr = np.array(center)
            X = X - center_arr[col_indices]
            
        if params.get('with_scaling', True) and scale is not None:
            scale_arr = np.array(scale)
            safe_scale = scale_arr[col_indices]
            safe_scale[safe_scale == 0] = 1.0
            X = X / safe_scale
            
        df_out[valid_cols] = X
        return df_out

# --- MaxAbs Scaler ---
class MaxAbsScalerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        scaler = MaxAbsScaler()
        scaler.fit(df[cols])
        
        return {
            'type': 'maxabs_scaler',
            'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None,
            'max_abs': scaler.max_abs_.tolist() if scaler.max_abs_ is not None else None,
            'columns': cols
        }

class MaxAbsScalerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        scale = params.get('scale')
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or scale is None:
            return df
            
        df_out = df.copy()
        scale_arr = np.array(scale)
        col_indices = [cols.index(c) for c in valid_cols]
        
        X = df_out[valid_cols].values
        # MaxAbsScaler just divides by max_abs (which is stored in scale_)
        safe_scale = scale_arr[col_indices]
        safe_scale[safe_scale == 0] = 1.0
        X = X / safe_scale
        
        df_out[valid_cols] = X
        return df_out
