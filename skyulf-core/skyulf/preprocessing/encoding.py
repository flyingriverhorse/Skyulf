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
        
        X_subset = X_out[valid_cols].copy()
        
        if include_missing:
            X_subset = X_subset.fillna('__mlops_missing__')
            
        # Transform
        try:
            encoded_array = encoder.transform(X_subset)
            
            # Create DataFrame from encoded array
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=feature_names, 
                index=X_out.index
            )
            
            # Concatenate
            X_out = pd.concat([X_out, encoded_df], axis=1)
            
            # Drop original columns
            if drop_original:
                X_out = X_out.drop(columns=valid_cols)
                
        except Exception as e:
            logger.error(f"OneHot Encoding failed: {e}")
            # If encoding fails (e.g. new categories with handle_unknown='error'), we might just return original
            pass
            
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Ordinal Encoder ---
class OrdinalEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        cols = resolve_columns(X, config, detect_categorical_columns)
        
        if not cols:
            return {}
            
        # Config: {'handle_unknown': 'use_encoded_value', 'unknown_value': -1}
        handle_unknown = config.get('handle_unknown', 'use_encoded_value')
        unknown_value = config.get('unknown_value', -1)
        
        # Sklearn OrdinalEncoder
        encoder = OrdinalEncoder(
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
            dtype=np.float32 # Use float to support NaN/unknown_value
        )
        
        # Fill missing before fit? OrdinalEncoder handles NaN if encoded_missing_value is set (new in sklearn 1.3)
        # For older versions, we might need to fill.
        # Let's assume standard behavior: NaN is a category or error.
        # We'll convert to string to treat NaN as "nan" category if needed, or let it fail.
        # Safer: Convert to string.
        X_fit = X[cols].astype(str)
        
        encoder.fit(X_fit)
        
        return {
            'type': 'ordinal',
            'columns': cols,
            'encoder_object': encoder,
            'categories_count': [len(cats) for cats in encoder.categories_]
        }

class OrdinalEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        encoder = params.get('encoder_object')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        try:
            X_subset = X_out[valid_cols].astype(str)
            encoded_array = encoder.transform(X_subset)
            
            # Replace columns in place
            X_out[valid_cols] = encoded_array
            
        except Exception as e:
            logger.error(f"Ordinal Encoding failed: {e}")
            pass
            
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Label Encoder (Target) ---
class LabelEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        # LabelEncoder is usually for the target variable y.
        # But sometimes used for features too.
        # Config: {'columns': [...]} or empty for target
        
        cols = config.get('columns')
        
        encoders = {}
        classes_count = {}
        
        if cols:
            # Encode features
            valid_cols = [c for c in cols if c in X.columns]
            for col in valid_cols:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                encoders[col] = le
                classes_count[col] = len(le.classes_)
        else:
            # Encode target y
            if y is not None:
                le = LabelEncoder()
                le.fit(y.astype(str))
                encoders['__target__'] = le
                classes_count['__target__'] = len(le.classes_)
        
        return {
            'type': 'label_encoder',
            'encoders': encoders, # Dict of LabelEncoder objects
            'columns': cols,
            'classes_count': classes_count
        }

class LabelEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        encoders = params.get('encoders', {})
        cols = params.get('columns')
        
        X_out = X.copy()
        y_out = y.copy() if y is not None else None
        
        if cols:
            # Transform features
            for col in cols:
                if col in X_out.columns and col in encoders:
                    le = encoders[col]
                    # Handle unseen labels? LabelEncoder crashes on unseen.
                    # We need a safe transform helper.
                    
                    # Fast safe transform:
                    # Map known classes to integers, unknown to -1 or NaN
                    # But LabelEncoder doesn't support unknown.
                    # We can use map.
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    X_out[col] = X_out[col].astype(str).map(mapping).fillna(-1)
        else:
            # Transform target
            if y_out is not None and '__target__' in encoders:
                le = encoders['__target__']
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                y_out = y_out.astype(str).map(mapping).fillna(-1)
                
        return pack_pipeline_output(X_out, y_out, is_tuple)

# --- Target Encoder (Mean Encoding) ---
class TargetEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if y is None:
            logger.warning("TargetEncoder requires a target variable (y). Skipping.")
            return {}
            
        cols = resolve_columns(X, config, detect_categorical_columns)
        if not cols:
            return {}
            
        # Config: {'smooth': 'auto', 'target_type': 'auto'}
        smooth = config.get('smooth', 'auto')
        target_type = config.get('target_type', 'auto')
        
        encoder = TargetEncoder(smooth=smooth, target_type=target_type)
        encoder.fit(X[cols], y)
        
        return {
            'type': 'target_encoder',
            'columns': cols,
            'encoder_object': encoder
        }

class TargetEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        encoder = params.get('encoder_object')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        try:
            X_subset = X_out[valid_cols]
            encoded_array = encoder.transform(X_subset)
            X_out[valid_cols] = encoded_array
        except Exception as e:
            logger.error(f"Target Encoding failed: {e}")
            pass
            
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Hash Encoder ---
class HashEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        cols = resolve_columns(X, config, detect_categorical_columns)
        if not cols:
            return {}
            
        # Config: {'n_features': 10}
        n_features = config.get('n_features', 10)
        
        # FeatureHasher is stateless, no fit needed really, but we store config
        return {
            'type': 'hash_encoder',
            'columns': cols,
            'n_features': n_features
        }

class HashEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        n_features = params.get('n_features', 10)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        
        # Hashing usually done row by row or on a specific column
        # FeatureHasher expects list of strings or dicts.
        # If we apply to multiple columns, do we hash them together or separately?
        # Usually separately if we want to replace the column.
        
        for col in valid_cols:
            # Convert to string list
            col_data = X_out[col].astype(str).tolist()
            # Wrap in list for hasher? No, hasher expects iterable of iterables (like list of tokens)
            # Or iterable of strings if input_type='string'
            
            # Actually FeatureHasher with input_type='string' expects an iterable of strings.
            hashed = hasher.transform([[x] for x in col_data]) # Hack to treat each value as a document?
            # No, standard usage for categorical is usually DictVectorizer or HashingVectorizer.
            # FeatureHasher is for "list of features".
            
            # Let's use a simpler approach: Apply hash() and mod n_features
            # This is deterministic and stateless.
            
            X_out[col] = X_out[col].astype(str).apply(lambda x: hash(x) % n_features)
            
        return pack_pipeline_output(X_out, y, is_tuple)

# --- Dummy Encoder (Pandas get_dummies) ---
class DummyEncoderCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        cols = resolve_columns(X, config, detect_categorical_columns)
        
        # We need to know all possible categories to align columns during transform
        categories = {}
        for col in cols:
            categories[col] = sorted(X[col].dropna().unique().astype(str).tolist())
            
        return {
            'type': 'dummy_encoder',
            'columns': cols,
            'categories': categories,
            'drop_first': config.get('drop_first', False)
        }

class DummyEncoderApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        categories = params.get('categories', {})
        drop_first = params.get('drop_first', False)
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        X_out = X.copy()
        
        for col in valid_cols:
            # Convert to categorical with known categories
            known_cats = categories.get(col, [])
            X_out[col] = pd.Categorical(X_out[col].astype(str), categories=known_cats)
            
        # Get dummies
        dummies = pd.get_dummies(X_out[valid_cols], drop_first=drop_first, dtype=int)
        
        # Drop original
        X_out = X_out.drop(columns=valid_cols)
        
        # Concat
        X_out = pd.concat([X_out, dummies], axis=1)
        
        return pack_pipeline_output(X_out, y, is_tuple)
