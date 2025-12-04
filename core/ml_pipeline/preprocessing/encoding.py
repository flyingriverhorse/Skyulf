from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from .base import BaseCalculator, BaseApplier
from ..utils import resolve_columns

logger = logging.getLogger(__name__)

def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- OneHot Encoder ---
class OneHotEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        # Config
        drop = 'first' if config.get('drop_first', False) else None
        max_categories = config.get('max_categories', 20) # Default limit to prevent explosion
        handle_unknown = 'ignore' if config.get('handle_unknown', 'ignore') == 'ignore' else 'error'
        prefix_separator = config.get('prefix_separator', '_')
        drop_original = config.get('drop_original', True)
        include_missing = config.get('include_missing', False)
        
        # Handle missing values for fit if requested
        df_fit = df[cols].copy()
        if include_missing:
            df_fit = df_fit.fillna('__mlops_missing__')
        
        # We use sklearn's OneHotEncoder
        # Note: sparse_output=False to return dense arrays for pandas
        encoder = OneHotEncoder(
            drop=drop, 
            max_categories=max_categories, 
            handle_unknown=handle_unknown, 
            sparse_output=False,
            dtype=np.int8 # Save memory
        )
        
        encoder.fit(df_fit)
        
        return {
            'type': 'onehot',
            'columns': cols,
            'encoder_object': encoder,
            'feature_names': encoder.get_feature_names_out(cols).tolist(),
            'prefix_separator': prefix_separator,
            'drop_original': drop_original,
            'include_missing': include_missing
        }

class OneHotEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        feature_names = params.get('feature_names')
        drop_original = params.get('drop_original', True)
        include_missing = params.get('include_missing', False)
        # prefix_separator is used implicitly by sklearn's get_feature_names_out during fit
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or not encoder:
            return df
            
        df_out = df.copy()
        
        # Prepare data
        X = df_out[valid_cols].copy()
        if include_missing:
            X = X.fillna('__mlops_missing__')
            
        # Transform
        # Sklearn handles unknown categories based on handle_unknown='ignore' (all zeros)
        encoded_array = encoder.transform(X)
        
        # Create DataFrame from encoded array
        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=feature_names, 
            index=df_out.index
        )
        
        # Drop original columns and concat encoded ones
        if drop_original:
            df_out = df_out.drop(columns=valid_cols)
            
        df_out = pd.concat([df_out, encoded_df], axis=1)
        
        return df_out

# --- Ordinal Encoder ---
class OrdinalEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        # Config
        # handle_unknown='use_encoded_value' is robust for production
        unknown_value = config.get('unknown_value', -1)
        encode_missing = config.get('encode_missing', False)
        
        df_fit = df[cols].copy()
        if encode_missing:
            df_fit = df_fit.fillna('__mlops_missing__')
        
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown_value,
            dtype=np.float32 # Float to support NaN/unknown_value
        )
        
        encoder.fit(df_fit)
        
        return {
            'type': 'ordinal',
            'columns': cols,
            'encoder_object': encoder,
            'output_suffix': config.get('output_suffix', ''),
            'drop_original': config.get('drop_original', True),
            'encode_missing': encode_missing
        }

class OrdinalEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        output_suffix = params.get('output_suffix', '')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or not encoder:
            return df
            
        df_out = df.copy()
        
        X = df_out[valid_cols].copy()
        if encode_missing:
            X = X.fillna('__mlops_missing__')
        
        # Transform
        transformed = encoder.transform(X)
        
        if drop_original:
            df_out[valid_cols] = transformed
        else:
            for i, col in enumerate(valid_cols):
                out_col = f"{col}{output_suffix}"
                df_out[out_col] = transformed[:, i]
        
        return df_out

# --- Target Encoder ---
class TargetEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        # Requires target column in config or passed separately?
        # BaseCalculator.fit signature is (df, config). 
        # We assume 'df' contains the target, and config specifies which column is target.
        
        target_col = config.get('target_column')
        if not target_col or target_col not in df.columns:
            logger.error(f"TargetEncoder requires target column '{target_col}' to be present in training data.")
            return {}
            
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        # Ensure target is NEVER in the columns to be encoded (leakage prevention)
        if target_col in cols:
            cols = [c for c in cols if c != target_col]
        
        if not cols:
            return {}
            
        smooth = config.get('smooth', 20.0) # Smoothing parameter
        encode_missing = config.get('encode_missing', False)
        
        df_fit = df[cols].copy()
        if encode_missing:
            df_fit = df_fit.fillna('__mlops_missing__')
        
        encoder = TargetEncoder(smooth=smooth, target_type='continuous') # Auto-detects type usually, but explicit is safer
        
        X = df_fit
        y = df[target_col]
        
        encoder.fit(X, y)
        
        return {
            'type': 'target',
            'columns': cols,
            'encoder_object': encoder,
            'output_suffix': config.get('output_suffix', '_target'),
            'drop_original': config.get('drop_original', True),
            'encode_missing': encode_missing
        }

class TargetEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        output_suffix = params.get('output_suffix', '_target')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols or not encoder:
            return df
            
        df_out = df.copy()
        
        X = df_out[valid_cols].copy()
        if encode_missing:
            X = X.fillna('__mlops_missing__')
        
        # Transform
        transformed = encoder.transform(X)
        
        if drop_original:
            df_out[valid_cols] = transformed
        else:
            for i, col in enumerate(valid_cols):
                out_col = f"{col}{output_suffix}"
                df_out[out_col] = transformed[:, i]
        
        return df_out

# --- Hash Encoder ---
class HashEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        n_features = config.get('n_features', 64)
        
        # FeatureHasher is stateless, but we wrap it to follow the pattern
        # It doesn't need 'fit' in the sklearn sense, but we configure it here
        
        return {
            'type': 'hash',
            'columns': cols,
            'n_features': n_features,
            'output_suffix': config.get('output_suffix', '_hash'),
            'drop_original': config.get('drop_original', True),
            'encode_missing': config.get('encode_missing', False)
        }

class HashEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        n_features = params.get('n_features', 64)
        output_suffix = params.get('output_suffix', '_hash')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_out = df.copy()
        
        # FeatureHasher expects an iterable of iterables (e.g. list of strings).
        # We hash each column independently to generate n_features for that specific column.
        # Each value is treated as a single string token.
        
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        
        for col in valid_cols:
            col_series = df_out[col]
            if encode_missing:
                col_series = col_series.fillna('__mlops_missing__')
                
            # Convert column to list of strings (iterable of iterables)
            # FeatureHasher expects [ ['a'], ['b'], ... ]
            # We handle NaNs by converting them to string representation
            col_data = col_series.astype(str).map(lambda x: [x]).tolist()
            
            hashed = hasher.transform(col_data)
            
            base_name = f"{col}{output_suffix}"
            
            hashed_df = pd.DataFrame(
                hashed.toarray(),
                columns=[f"{base_name}_{i}" for i in range(n_features)],
                index=df_out.index
            )
            
            df_out = pd.concat([df_out, hashed_df], axis=1)
            
        if drop_original:
            df_out = df_out.drop(columns=valid_cols)
        return df_out

# --- Dummy Encoder (Pandas get_dummies wrapper) ---
# Note: In V2, we prefer OneHotEncoder, but this is for V1 compatibility
class DummyEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        drop_first = config.get('drop_first', True)
        include_missing = config.get('include_missing', False)
        prefix_separator = config.get('prefix_separator', '_')
        
        # We need to know the categories to ensure consistent columns during apply
        categories = {}
        for col in cols:
            series = df[col]
            if include_missing:
                series = series.fillna('__mlops_missing__')
            else:
                series = series.dropna()
                
            cats = series.unique().tolist()
            cats.sort()
            if drop_first and len(cats) > 0:
                cats = cats[1:] # Drop the first one
            categories[col] = cats
            
        return {
            'type': 'dummy',
            'columns': cols,
            'categories': categories,
            'drop_first': drop_first,
            'include_missing': include_missing,
            'prefix_separator': prefix_separator,
            'drop_original': config.get('drop_original', True)
        }

class DummyEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        categories = params.get('categories', {})
        include_missing = params.get('include_missing', False)
        prefix_separator = params.get('prefix_separator', '_')
        drop_original = params.get('drop_original', True)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_out = df.copy()
        
        for col in valid_cols:
            # We manually construct the dummy columns to ensure consistency with 'fit'
            # This is safer than pd.get_dummies which depends on the data present
            known_cats = categories.get(col, [])
            
            col_data = df_out[col]
            if include_missing:
                col_data = col_data.fillna('__mlops_missing__')
            
            for cat in known_cats:
                col_name = f"{col}{prefix_separator}{cat}"
                # Create boolean column (0/1)
                df_out[col_name] = (col_data == cat).astype(int)
                
        if drop_original:
            df_out = df_out.drop(columns=valid_cols)
        return df_out

# --- Label Encoder ---
class LabelEncoderCalculator(BaseCalculator):
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        cols = resolve_columns(df, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        missing_strategy = config.get('missing_strategy', 'keep_na') # keep_na, encode
        
        # LabelEncoder is strictly 1D, so we need one per column
        encoders = {}
        for col in cols:
            le = LabelEncoder()
            series = df[col]
            
            if missing_strategy == 'encode':
                series = series.fillna('__mlops_missing__')
            
            # LabelEncoder does not handle NaNs natively.
            # We convert all values to string to treat NaNs as a distinct category 'nan'.
            le.fit(series.astype(str))
            encoders[col] = le
            
        return {
            'type': 'label',
            'columns': cols,
            'encoders': encoders,
            'output_suffix': config.get('output_suffix', ''),
            'drop_original': config.get('drop_original', True),
            'missing_strategy': missing_strategy,
            'missing_code': config.get('missing_code', -1)
        }

class LabelEncoderApplier(BaseApplier):
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        if not params or not params.get('columns'):
            return df
            
        cols = params['columns']
        encoders = params.get('encoders', {})
        output_suffix = params.get('output_suffix', '')
        drop_original = params.get('drop_original', True)
        missing_strategy = params.get('missing_strategy', 'keep_na')
        missing_code = params.get('missing_code', -1)
        
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return df
            
        df_out = df.copy()
        
        for col in valid_cols:
            le = encoders.get(col)
            if le:
                series = df_out[col]
                if missing_strategy == 'encode':
                    series = series.fillna('__mlops_missing__')
                
                # Handle unseen labels safely by mapping them to -1.
                # Standard LabelEncoder raises an error for unseen labels.
                
                # Create mapping from learned classes
                le_classes = le.classes_.tolist()
                le_map = {c: i for i, c in enumerate(le_classes)}
                
                # Apply map and fill unknowns with -1
                mapped = series.astype(str).map(le_map)
                
                if missing_strategy == 'keep_na':
                    # If we kept NaNs, they are now NaNs in mapped
                    pass
                else:
                    mapped = mapped.fillna(missing_code)
                
                # Convert to numeric if possible
                if mapped.isnull().any():
                    mapped = mapped.astype(float)
                else:
                    mapped = mapped.astype(int)
                
                if drop_original:
                    df_out[col] = mapped
                else:
                    out_col = f"{col}{output_suffix}"
                    df_out[out_col] = mapped
                
        return df_out

