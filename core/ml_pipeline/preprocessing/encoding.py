from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
from sklearn.feature_extraction import FeatureHasher

from .base import BaseCalculator, BaseApplier
from ..utils import resolve_columns, unpack_pipeline_input, pack_pipeline_output

logger = logging.getLogger(__name__)

def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

# --- OneHot Encoder ---
class OneHotEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, detect_categorical_columns)
        
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
        df_fit = X[cols].copy()
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
        
        # Check for columns that produced no features
        if hasattr(encoder, 'categories_'):
            for i, col in enumerate(cols):
                n_cats = len(encoder.categories_[i])
                # If drop='first' and n_cats == 1, we get 0 features (1-1=0)
                # If n_cats == 0, we get 0 features
                
                # We can check the actual output feature names to be sure, but checking categories is a good proxy
                # Actually, sklearn's get_feature_names_out handles the drop logic.
                
                # Let's check if this specific column generated any features in the output
                # This is a bit tricky with the bulk encoder, but we can infer.
                
                if n_cats == 0:
                     logger.warning(f"OneHotEncoder: Column '{col}' has 0 categories (empty or all missing). It will be dropped.")
                elif drop == 'first' and n_cats == 1:
                     logger.warning(f"OneHotEncoder: Column '{col}' has only 1 category ('{encoder.categories_[i][0]}') and 'Drop First' is enabled. This results in 0 encoded features. The column will be effectively dropped.")
        
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        feature_names = params.get('feature_names')
        drop_original = params.get('drop_original', True)
        include_missing = params.get('include_missing', False)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        # Ensure all expected columns are present for the encoder
        # If some columns are missing in input, we fill them with NaN
        # This allows encoder.transform to receive the correct number of features
        missing_cols = set(cols) - set(X.columns)
        if missing_cols:
            for c in missing_cols:
                X_out[c] = np.nan
        
        # Select columns in the order expected by the encoder
        X_sub = X_out[cols].copy()
        
        if include_missing:
            X_sub = X_sub.fillna('__mlops_missing__')
            
        # Handle NaNs in X_sub if not include_missing
        # If handle_unknown='ignore', OneHotEncoder treats NaNs as unknown (all zeros)
        # But we need to ensure they are not passed as float nan to string encoder if it expects strings
        # Usually OneHotEncoder handles mixed types or casts to string.
        # Let's cast to string to be safe if they were object columns
        for col in X_sub.columns:
            if X_sub[col].dtype == 'object':
                X_sub[col] = X_sub[col].astype(str)
                # "nan" string might be treated as a category if it was seen during fit.
                # If it wasn't seen, it's unknown.
                # If real NaN, astype(str) makes it "nan".
        
        try:
            encoded_array = encoder.transform(X_sub)
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=feature_names, 
                index=X_out.index
            )
            
            if drop_original:
                # Only drop columns that were actually in the original X
                cols_to_drop = [c for c in cols if c in X.columns]
                X_out = X_out.drop(columns=cols_to_drop)
                # Also drop the temporary missing columns we added? 
                # No, X_out is a copy, and we added them to X_out.
                # If we added them, we should drop them too if drop_original is True.
                # Actually, if drop_original is True, we want to remove the raw columns.
                # Since we added missing columns to X_out, we should remove them too.
                cols_to_drop_all = cols
                # But X_out might not have them if we didn't add them (if they were present).
                # Safe way: drop existing cols from 'cols' list
                existing_cols_to_drop = [c for c in cols if c in X_out.columns]
                X_out = X_out.drop(columns=existing_cols_to_drop)
                
            X_out = pd.concat([X_out, encoded_df], axis=1)
        except Exception as e:
            logger.warning(f"OneHotEncoder application failed: {e}")
            # If failed, we return original X (maybe with missing cols added? No, let's return X)
            return pack_pipeline_output(X, y, is_tuple)

        return pack_pipeline_output(X_out, y, is_tuple)

# --- Ordinal Encoder ---
class OrdinalEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        # Config
        # handle_unknown='use_encoded_value' is robust for production
        unknown_value = config.get('unknown_value', -1)
        encode_missing = config.get('encode_missing', False)
        
        df_fit = X[cols].copy()
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        output_suffix = params.get('output_suffix', '')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        X_sub = X_out[valid_cols].copy()
        if encode_missing:
            X_sub = X_sub.fillna('__mlops_missing__')
        
        transformed = encoder.transform(X_sub)
        
        if drop_original:
            X_out[valid_cols] = transformed
        else:
            for i, col in enumerate(valid_cols):
                out_col = f"{col}{output_suffix}"
                X_out[out_col] = transformed[:, i]
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Target Encoder ---
class TargetEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Requires target column in config or passed separately?
        # BaseCalculator.fit signature is (df, config). 
        # We assume 'df' contains the target, and config specifies which column is target.
        
        target_col = config.get('target_column')
        
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not is_tuple:
            if not target_col or target_col not in X.columns:
                logger.error(f"TargetEncoder requires target column '{target_col}' to be present in training data.")
                return {}
            y = X[target_col]

        cols = resolve_columns(X, config, detect_categorical_columns)
        
        # Ensure target is NEVER in the columns to be encoded (leakage prevention)
        if target_col in cols:
            cols = [c for c in cols if c != target_col]
        
        if not cols:
            return {}
            
        smooth = config.get('smooth', 20.0) # Smoothing parameter
        encode_missing = config.get('encode_missing', False)
        
        df_fit = X[cols].copy()
        if encode_missing:
            df_fit = df_fit.fillna('__mlops_missing__')
        
        encoder = TargetEncoder(smooth=smooth, target_type='continuous') # Auto-detects type usually, but explicit is safer
        
        encoder.fit(df_fit, y)
        
        return {
            'type': 'target',
            'columns': cols,
            'encoder_object': encoder,
            'output_suffix': config.get('output_suffix', '_target'),
            'drop_original': config.get('drop_original', True),
            'encode_missing': encode_missing
        }

class TargetEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        encoder = params.get('encoder_object')
        output_suffix = params.get('output_suffix', '_target')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        X_sub = X_out[valid_cols].copy()
        if encode_missing:
            X_sub = X_sub.fillna('__mlops_missing__')
        
        transformed = encoder.transform(X_sub)
        
        if drop_original:
            X_out[valid_cols] = transformed
        else:
            for i, col in enumerate(valid_cols):
                out_col = f"{col}{output_suffix}"
                X_out[out_col] = transformed[:, i]
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Hash Encoder ---
class HashEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, detect_categorical_columns)
        
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        n_features = params.get('n_features', 64)
        output_suffix = params.get('output_suffix', '_hash')
        drop_original = params.get('drop_original', True)
        encode_missing = params.get('encode_missing', False)
        
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        for col in valid_cols:
            series = X_out[col].astype(str)
            if encode_missing:
                series = series.fillna('__mlops_missing__')
            
            # FeatureHasher expects iterable of iterables (like list of strings)
            # But here we are hashing single column values.
            # Usually FeatureHasher is used for list of tokens.
            # For single column categorical, we treat value as a single token.
            # We need to wrap each value in a list: [['cat'], ['dog'], ...]
            hashed = hasher.transform([[x] for x in series])
            hashed_array = hashed.toarray()
            
            # Add hashed features
            new_cols = [f"{col}{output_suffix}_{i}" for i in range(n_features)]
            hashed_df = pd.DataFrame(hashed_array, columns=new_cols, index=X_out.index)
            
            X_out = pd.concat([X_out, hashed_df], axis=1)
            
        if drop_original:
            X_out = X_out.drop(columns=valid_cols)
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Dummy Encoder (Pandas get_dummies wrapper) ---
# Note: In V2, we prefer OneHotEncoder, but this is for V1 compatibility
class DummyEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        cols = resolve_columns(X, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        drop_first = config.get('drop_first', True)
        include_missing = config.get('include_missing', False)
        prefix_separator = config.get('prefix_separator', '_')
        
        # We need to know the categories to ensure consistent columns during apply
        categories = {}
        for col in cols:
            series = X[col]
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
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        categories = params.get('categories', {})
        include_missing = params.get('include_missing', False)
        prefix_separator = params.get('prefix_separator', '_')
        drop_original = params.get('drop_original', True)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        for col in valid_cols:
            # We manually construct the dummy columns to ensure consistency with 'fit'
            # This is safer than pd.get_dummies which depends on the data present
            known_cats = categories.get(col, [])
            
            col_data = X_out[col]
            if include_missing:
                col_data = col_data.fillna('__mlops_missing__')
            
            for cat in known_cats:
                col_name = f"{col}{prefix_separator}{cat}"
                # Create boolean column (0/1)
                X_out[col_name] = (col_data == cat).astype(int)
                
        if drop_original:
            X_out = X_out.drop(columns=valid_cols)
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Label Encoder ---
class LabelEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = resolve_columns(X, config, detect_categorical_columns)
        
        # Check if we need to encode the target variable (y)
        encode_target = False
        target_col_name = None
        
        if is_tuple and y is not None:
            target_col_name = y.name
            requested_cols = config.get('columns')
            # If target is explicitly requested but not found in X (because it's in y)
            if requested_cols and target_col_name in requested_cols and target_col_name not in cols:
                encode_target = True

        if not cols and not encode_target:
            return {}
            
        missing_strategy = config.get('missing_strategy', 'keep_na') # keep_na, encode
        
        # LabelEncoder is strictly 1D, so we need one per column
        encoders = {}
        
        # Fit columns in X
        for col in cols:
            le = LabelEncoder()
            series = X[col]
            
            if missing_strategy == 'encode':
                series = series.fillna('__mlops_missing__')
            
            # LabelEncoder does not handle NaNs natively.
            # We convert all values to string to treat NaNs as a distinct category 'nan'.
            le.fit(series.astype(str))
            encoders[col] = le
            
        # Fit target column (y)
        if encode_target and target_col_name:
            le = LabelEncoder()
            series = y
            
            if missing_strategy == 'encode':
                series = series.fillna('__mlops_missing__')
                
            le.fit(series.astype(str))
            encoders[target_col_name] = le
            
        return {
            'type': 'label',
            'columns': cols,
            'target_column': target_col_name if encode_target else None,
            'encoders': encoders,
            'output_suffix': config.get('output_suffix', ''),
            'drop_original': config.get('drop_original', True),
            'missing_strategy': missing_strategy,
            'missing_code': config.get('missing_code', -1)
        }

class LabelEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params:
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params.get('columns', [])
        target_col = params.get('target_column')
        encoders = params.get('encoders', {})
        output_suffix = params.get('output_suffix', '')
        drop_original = params.get('drop_original', True)
        missing_strategy = params.get('missing_strategy', 'keep_na')
        missing_code = params.get('missing_code', -1)
        
        # 1. Apply to X columns
        valid_cols = [c for c in cols if c in X.columns]
        X_out = X.copy()
        
        for col in valid_cols:
            le = encoders.get(col)
            if le:
                series = X_out[col]
                X_out = self._apply_encoding(X_out, col, series, le, missing_strategy, missing_code, drop_original, output_suffix)

        # 2. Apply to y (Target)
        y_out = y
        if is_tuple and y is not None and target_col == y.name:
            le = encoders.get(target_col)
            if le:
                # For y, we always "drop original" (replace) because y is a Series
                # We can't really add a suffix to y in the same way as X columns without changing it to DataFrame
                # So we just transform it.
                y_out = self._transform_series(y, le, missing_strategy, missing_code)
                
        return pack_pipeline_output(X_out, y_out, is_tuple)

    def _transform_series(self, series: pd.Series, le: LabelEncoder, missing_strategy: str, missing_code: Any) -> pd.Series:
        if missing_strategy == 'encode':
            series = series.fillna('__mlops_missing__')
        
        # Handle unseen labels safely by mapping them to -1.
        le_classes = le.classes_.tolist()
        le_map = {c: i for i, c in enumerate(le_classes)}
        
        # Apply map and fill unknowns with -1
        mapped = series.astype(str).map(le_map)
        
        if missing_strategy == 'keep_na':
            pass
        else:
            mapped = mapped.fillna(missing_code)
        
        # Convert to numeric if possible
        if mapped.isnull().any():
            mapped = mapped.astype(float)
        else:
            mapped = mapped.astype(int)
            
        return mapped

    def _apply_encoding(self, df: pd.DataFrame, col: str, series: pd.Series, le: LabelEncoder, missing_strategy: str, missing_code: Any, drop_original: bool, output_suffix: str) -> pd.DataFrame:
        mapped = self._transform_series(series, le, missing_strategy, missing_code)
        
        if drop_original:
            df[col] = mapped
        else:
            out_col = f"{col}{output_suffix}"
            df[out_col] = mapped
        return df

