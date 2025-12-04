from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope

from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns

logger = logging.getLogger(__name__)

# --- IQR Filter ---
class IQRCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'multiplier': 1.5, 'columns': [...]}
        multiplier = config.get('multiplier', 1.5)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        bounds = {}
        for col in cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                continue
                
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower = q1 - (multiplier * iqr)
            upper = q3 + (multiplier * iqr)
            
            bounds[col] = {'lower': lower, 'upper': upper}
            
        return {
            'type': 'iqr',
            'bounds': bounds,
            'multiplier': multiplier
        }

class IQRApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        bounds = params.get('bounds', {})
        if not bounds:
            return df
            
        mask = pd.Series(True, index=df.index)
        
        for col, bound in bounds.items():
            if col not in df.columns:
                continue
                
            lower = bound['lower']
            upper = bound['upper']
            
            series = pd.to_numeric(df[col], errors='coerce')
            
            # Keep values within bounds or NaN
            col_mask = (series >= lower) & (series <= upper)
            col_mask = col_mask | series.isna()
            
            mask = mask & col_mask
            
        return df[mask]

# --- Z-Score Filter ---
class ZScoreCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'threshold': 3.0, 'columns': [...]}
        threshold = config.get('threshold', 3.0)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        stats = {}
        for col in cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                continue
                
            mean = series.mean()
            std = series.std(ddof=0)
            
            if std == 0:
                continue
                
            stats[col] = {'mean': mean, 'std': std}
            
        return {
            'type': 'zscore',
            'stats': stats,
            'threshold': threshold
        }

class ZScoreApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        stats = params.get('stats', {})
        threshold = params.get('threshold', 3.0)
        
        if not stats:
            return df
            
        mask = pd.Series(True, index=df.index)
        
        for col, stat in stats.items():
            if col not in df.columns:
                continue
                
            mean = stat['mean']
            std = stat['std']
            
            if std == 0:
                continue
                
            series = pd.to_numeric(df[col], errors='coerce')
            z_scores = (series - mean) / std
            
            # Keep if abs(z) <= threshold
            col_mask = z_scores.abs() <= threshold
            col_mask = col_mask | series.isna()
            
            mask = mask & col_mask
            
        return df[mask]

# --- Winsorize ---
class WinsorizeCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'lower_percentile': 5.0, 'upper_percentile': 95.0, 'columns': [...]}
        lower_p = config.get('lower_percentile', 5.0)
        upper_p = config.get('upper_percentile', 95.0)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        bounds = {}
        for col in cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                continue
                
            lower_val = series.quantile(lower_p / 100.0)
            upper_val = series.quantile(upper_p / 100.0)
            
            bounds[col] = {'lower': lower_val, 'upper': upper_val}
            
        return {
            'type': 'winsorize',
            'bounds': bounds,
            'lower_percentile': lower_p,
            'upper_percentile': upper_p
        }

class WinsorizeApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        bounds = params.get('bounds', {})
        if not bounds:
            return df
            
        df_out = df.copy()
        
        for col, bound in bounds.items():
            if col not in df.columns:
                continue
                
            lower = bound['lower']
            upper = bound['upper']
            
            # Clip values
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].clip(lower=lower, upper=upper)
                
        return df_out

# --- Manual Bounds ---
class ManualBoundsCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        bounds = config.get('bounds', {})
        
        return {
            'type': 'manual_bounds',
            'bounds': bounds
        }

class ManualBoundsApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        bounds = params.get('bounds', {})
        if not bounds:
            return df
            
        mask = pd.Series(True, index=df.index)
        
        for col, bound in bounds.items():
            if col not in df.columns:
                continue
                
            lower = bound.get('lower')
            upper = bound.get('upper')
            
            series = pd.to_numeric(df[col], errors='coerce')
            col_mask = pd.Series(True, index=df.index)
            
            if lower is not None:
                col_mask = col_mask & (series >= lower)
            if upper is not None:
                col_mask = col_mask & (series <= upper)
                
            col_mask = col_mask | series.isna()
            mask = mask & col_mask
            
        return df[mask]

# --- Elliptic Envelope ---
class EllipticEnvelopeCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'contamination': 0.01, 'columns': [...]}
        contamination = config.get('contamination', 0.01)
        
        cols = resolve_columns(df, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        models = {}
        for col in cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.shape[0] < 5:
                continue
                
            try:
                model = EllipticEnvelope(contamination=contamination)
                model.fit(series.to_numpy().reshape(-1, 1))
                models[col] = model
            except Exception as e:
                logger.warning(f"EllipticEnvelope fit failed for column {col}: {e}")
                pass
                
        return {
            'type': 'elliptic_envelope',
            'models': models,
            'contamination': contamination
        }

class EllipticEnvelopeApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        models = params.get('models', {})
        if not models:
            return df
            
        mask = pd.Series(True, index=df.index)
        
        for col, model in models.items():
            if col not in df.columns:
                continue
                
            series = pd.to_numeric(df[col], errors='coerce')
            valid_mask = series.notna()
            valid_values = series[valid_mask].to_numpy().reshape(-1, 1)
            
            if valid_values.shape[0] == 0:
                continue
                
            try:
                preds = model.predict(valid_values)
                # -1 is outlier, 1 is inlier
                is_inlier = preds == 1
                
                # Create full mask for this column
                col_mask = pd.Series(True, index=df.index)
                col_mask.loc[valid_mask] = is_inlier
                
                mask = mask & col_mask
            except Exception as e:
                logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
                pass
                
        return df[mask]
