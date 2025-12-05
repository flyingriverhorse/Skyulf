from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope

from .base import BaseCalculator, BaseApplier
from ..utils import detect_numeric_columns, resolve_columns, unpack_pipeline_input, pack_pipeline_output

logger = logging.getLogger(__name__)

# --- IQR Filter ---
class IQRCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'multiplier': 1.5, 'columns': [...]}
        multiplier = config.get('multiplier', 1.5)
        
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        bounds = {}
        for col in cols:
            series = pd.to_numeric(X[col], errors='coerce').dropna()
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        bounds = params.get('bounds', {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)
            
        mask = pd.Series(True, index=X.index)
        
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
                
            lower = bound['lower']
            upper = bound['upper']
            
            series = pd.to_numeric(X[col], errors='coerce')
            
            # Keep values within bounds or NaN
            col_mask = (series >= lower) & (series <= upper)
            col_mask = col_mask | series.isna()
            
            mask = mask & col_mask
            
        X_filtered = X[mask]
        
        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)
            
        return pack_pipeline_output(X_filtered, y, is_tuple)

# --- Z-Score Filter ---
class ZScoreCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'threshold': 3.0, 'columns': [...]}
        threshold = config.get('threshold', 3.0)
        
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        stats = {}
        for col in cols:
            series = pd.to_numeric(X[col], errors='coerce').dropna()
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        stats = params.get('stats', {})
        threshold = params.get('threshold', 3.0)
        
        if not stats:
            return pack_pipeline_output(X, y, is_tuple)
            
        mask = pd.Series(True, index=X.index)
        
        for col, stat in stats.items():
            if col not in X.columns:
                continue
                
            mean = stat['mean']
            std = stat['std']
            
            if std == 0:
                continue
                
            series = pd.to_numeric(X[col], errors='coerce')
            z_scores = (series - mean) / std
            
            # Keep if abs(z) <= threshold
            col_mask = z_scores.abs() <= threshold
            col_mask = col_mask | series.isna()
            
            mask = mask & col_mask
            
        X_filtered = X[mask]
        
        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)
            
        return pack_pipeline_output(X_filtered, y, is_tuple)

# --- Winsorize ---
class WinsorizeCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'lower_percentile': 5.0, 'upper_percentile': 95.0, 'columns': [...]}
        lower_p = config.get('lower_percentile', 5.0)
        upper_p = config.get('upper_percentile', 95.0)
        
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        bounds = {}
        for col in cols:
            series = pd.to_numeric(X[col], errors='coerce').dropna()
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        bounds = params.get('bounds', {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        for col, bound in bounds.items():
            if col not in df_out.columns:
                continue
                
            lower = bound['lower']
            upper = bound['upper']
            
            # Clip values
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].clip(lower=lower, upper=upper)
                
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Manual Bounds ---
class ManualBoundsCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        bounds = config.get('bounds', {})
        
        return {
            'type': 'manual_bounds',
            'bounds': bounds
        }

class ManualBoundsApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        bounds = params.get('bounds', {})
        if not bounds:
            return pack_pipeline_output(X, y, is_tuple)
            
        mask = pd.Series(True, index=X.index)
        
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
                
            lower = bound.get('lower')
            upper = bound.get('upper')
            
            series = pd.to_numeric(X[col], errors='coerce')
            col_mask = pd.Series(True, index=X.index)
            
            if lower is not None:
                col_mask = col_mask & (series >= lower)
            if upper is not None:
                col_mask = col_mask & (series <= upper)
                
            col_mask = col_mask | series.isna()
            mask = mask & col_mask
            
        X_filtered = X[mask]
        
        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)
            
        return pack_pipeline_output(X_filtered, y, is_tuple)

# --- Elliptic Envelope ---
class EllipticEnvelopeCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'contamination': 0.01, 'columns': [...]}
        contamination = config.get('contamination', 0.01)
        
        cols = resolve_columns(X, config, detect_numeric_columns)
        
        if not cols:
            return {}
            
        models = {}
        for col in cols:
            series = pd.to_numeric(X[col], errors='coerce').dropna()
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        models = params.get('models', {})
        if not models:
            return pack_pipeline_output(X, y, is_tuple)
            
        mask = pd.Series(True, index=X.index)
        
        for col, model in models.items():
            if col not in X.columns:
                continue
                
            series = pd.to_numeric(X[col], errors='coerce')
            valid_mask = series.notna()
            valid_values = series[valid_mask].to_numpy().reshape(-1, 1)
            
            if valid_values.shape[0] == 0:
                continue
                
            try:
                preds = model.predict(valid_values)
                # -1 is outlier, 1 is inlier
                is_inlier = preds == 1
                
                # Create full mask for this column
                col_mask = pd.Series(True, index=X.index)
                col_mask.loc[valid_mask] = is_inlier
                
                mask = mask & col_mask
            except Exception as e:
                logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
                pass
                
        X_filtered = X[mask]
        
        if y is not None:
            y_filtered = y[mask]
            return pack_pipeline_output(X_filtered, y_filtered, is_tuple)
            
        return pack_pipeline_output(X_filtered, y, is_tuple)
