from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns

logger = logging.getLogger(__name__)

# --- Power Transformer (Box-Cox, Yeo-Johnson) ---
class PowerTransformerCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'method': 'yeo-johnson' | 'box-cox', 'standardize': True, 'columns': [...]}
        method = config.get('method', 'yeo-johnson')
        standardize = config.get('standardize', True)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        valid_cols = []
        if method == 'box-cox':
            for col in cols:
                # Box-Cox requires strictly positive data
                if (df[col] <= 0).any():
                    continue
                valid_cols.append(col)
        else:
            valid_cols = cols
            
        if not valid_cols:
            return {}
            
        transformer = PowerTransformer(method=method, standardize=standardize)
        transformer.fit(df[valid_cols])
        
        # Store standardization params if needed
        # PowerTransformer stores _scaler (StandardScaler) internally if standardize=True
        scaler_params = {}
        if standardize and hasattr(transformer, '_scaler'):
            scaler = transformer._scaler
            if scaler:
                scaler_params = {
                    'mean': scaler.mean_.tolist() if scaler.mean_ is not None else None,
                    'scale': scaler.scale_.tolist() if scaler.scale_ is not None else None
                }

        return {
            'type': 'power_transformer',
            'lambdas': transformer.lambdas_.tolist(),
            'method': method,
            'standardize': standardize,
            'columns': valid_cols,
            'scaler_params': scaler_params
        }

class PowerTransformerApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        cols = params.get('columns', [])
        lambdas = params.get('lambdas')
        method = params.get('method', 'yeo-johnson')
        standardize = params.get('standardize', True)
        scaler_params = params.get('scaler_params', {})
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or lambdas is None:
            return df
            
        df_out = df.copy()
        
        # We can't easily reconstruct PowerTransformer with internal scaler state via public API
        # So we might need to apply manually or hack it.
        # Manual application:
        # 1. Apply power transform (Box-Cox or Yeo-Johnson) using lambdas
        # 2. Apply standardization using scaler_params
        
        X = df_out[valid_cols].values
        lambdas_arr = np.array(lambdas)
        
        # 1. Power Transform
        # We can use sklearn's internal functions or reimplement.
        # Reimplementing Yeo-Johnson/Box-Cox is non-trivial to get exactly right with sklearn's stability checks.
        # Better to use PowerTransformer but we need to bypass fit.
        # But PowerTransformer.transform() checks check_is_fitted.
        
        # Alternative: Create a PowerTransformer, set attributes, and call transform.
        # We need to set: lambdas_, _scaler (if standardize)
        
        try:
            pt = PowerTransformer(method=method, standardize=standardize)
            pt.lambdas_ = lambdas_arr
            
            if standardize:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_params.get('mean'))
                scaler.scale_ = np.array(scaler_params.get('scale'))
                scaler.var_ = np.square(scaler.scale_) # Approximate if not stored
                pt._scaler = scaler
                
            # We need to trick sklearn into thinking it's fitted
            # Usually setting attributes is enough, but let's see.
            # PowerTransformer checks hasattr(self, "lambdas_")
            
            X_trans = pt.transform(X)
            df_out[valid_cols] = X_trans
            
        except Exception as e:
            logger.error(f"PowerTransformer application failed: {e}")
            # Fallback?
            pass
            
        return df_out

# --- Simple Transformations (Log, Sqrt, etc.) ---
class SimpleTransformationCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'transformations': [{'column': 'col1', 'method': 'log'}, ...]}
        return {
            'type': 'simple_transformation',
            'transformations': config.get('transformations', [])
        }

class SimpleTransformationApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        transformations = params.get('transformations', [])
        if not transformations:
            return df
            
        df_out = df.copy()
        
        for item in transformations:
            col = item.get('column')
            method = item.get('method')
            
            if col not in df_out.columns:
                continue
                
            series = pd.to_numeric(df_out[col], errors='coerce')
            
            if method == 'log':
                # log1p is safer for zeros
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.log1p(series)
                
            elif method == 'square_root':
                if (series < 0).any():
                    series[series < 0] = np.nan
                df_out[col] = np.sqrt(series)
                
            elif method == 'cube_root':
                df_out[col] = np.cbrt(series)
                
            elif method == 'reciprocal':
                df_out[col] = 1.0 / series.replace(0, np.nan)
                
            elif method == 'square':
                df_out[col] = np.square(series)
                
            elif method == 'exponential':
                df_out[col] = np.exp(series)
                
        return df_out
