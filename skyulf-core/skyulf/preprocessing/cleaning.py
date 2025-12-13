from typing import Dict, Any, List, Optional, Union, Set, Tuple
import pandas as pd
import numpy as np
import re
import string
from .base import BaseCalculator, BaseApplier
from ..utils import resolve_columns, unpack_pipeline_input, pack_pipeline_output

# --- Constants ---
ALIAS_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
COMMON_BOOLEAN_ALIASES: Dict[str, str] = {
    "y": "Yes", "yes": "Yes", "true": "Yes", "1": "Yes", "on": "Yes", "t": "Yes", "affirmative": "Yes",
    "n": "No", "no": "No", "false": "No", "0": "No", "off": "No", "f": "No", "negative": "No",
}
COUNTRY_ALIAS_MAP: Dict[str, str] = {
    "usa": "USA", "us": "USA", "unitedstates": "USA", "unitedstatesofamerica": "USA", "states": "USA", "america": "USA",
    "unitedkingdom": "United Kingdom", "uk": "United Kingdom", "greatbritain": "United Kingdom", "england": "United Kingdom",
    "uae": "United Arab Emirates", "unitedarabemirates": "United Arab Emirates",
    "prc": "China", "peoplesrepublicofchina": "China",
    "southkorea": "South Korea", "republicofkorea": "South Korea", "sk": "South Korea",
}
TWO_DIGIT_YEAR_PIVOT = 50

# --- Helpers ---
def _auto_detect_text_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=['object', 'string', 'category']).columns)

def _auto_detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=['number']).columns)

def _auto_detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=['datetime', 'datetimetz']).columns)

# --- Text Cleaning ---
class TextCleaningCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config: 
        # columns: List[str]
        # operations: List[Dict] 
        #   {'op': 'trim', 'mode': 'both'|'leading'|'trailing'}
        #   {'op': 'case', 'mode': 'lower'|'upper'|'title'|'sentence'}
        #   {'op': 'remove_special', 'mode': 'keep_alphanumeric'|'keep_alphanumeric_space'|'letters_only'|'digits_only', 'replacement': ''}
        #   {'op': 'regex', 'mode': 'custom'|'collapse_whitespace'|'extract_digits'|'normalize_slash_dates', 'pattern': '...', 'repl': '...'}
        
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, _auto_detect_text_columns)
        
        if not cols:
            return {}
            
        operations = config.get('operations', [])
        
        return {
            'type': 'text_cleaning',
            'columns': cols,
            'operations': operations
        }

class TextCleaningApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        if not params or not params.get('columns'):
            return pack_pipeline_output(X, y, is_tuple)
            
        cols = params['columns']
        operations = params.get('operations', [])
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not operations:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        for col in valid_cols:
            # Ensure column is string type
            if not pd.api.types.is_string_dtype(df_out[col]):
                 df_out[col] = df_out[col].astype(str)
            
            series = df_out[col]
            
            for op in operations:
                op_type = op.get('op')
                
                if op_type == 'trim':
                    mode = op.get('mode', 'both')
                    if mode == 'leading':
                        series = series.str.lstrip()
                    elif mode == 'trailing':
                        series = series.str.rstrip()
                    else:
                        series = series.str.strip()
                        
                elif op_type == 'case':
                    mode = op.get('mode', 'lower')
                    if mode == 'upper':
                        series = series.str.upper()
                    elif mode == 'title':
                        series = series.str.title()
                    elif mode == 'sentence':
                        series = series.str.capitalize()
                    else:
                        series = series.str.lower()
                        
                elif op_type == 'remove_special':
                    mode = op.get('mode', 'keep_alphanumeric')
                    replacement = op.get('replacement', '')
                    
                    if mode == 'keep_alphanumeric':
                        # Keep letters and numbers
                        series = series.str.replace(r'[^a-zA-Z0-9]', replacement, regex=True)
                    elif mode == 'keep_alphanumeric_space':
                        # Keep letters, numbers, and spaces
                        series = series.str.replace(r'[^a-zA-Z0-9\s]', replacement, regex=True)
                    elif mode == 'letters_only':
                        series = series.str.replace(r'[^a-zA-Z]', replacement, regex=True)
                    elif mode == 'digits_only':
                        series = series.str.replace(r'[^0-9]', replacement, regex=True)
                        
                elif op_type == 'regex':
                    mode = op.get('mode', 'custom')
                    
                    if mode == 'collapse_whitespace':
                        series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
                    elif mode == 'extract_digits':
                        series = series.str.extract(r'(\d+)', expand=False)
                    elif mode == 'normalize_slash_dates':
                        # Simple attempt to fix 1/1/2020 -> 01/01/2020
                        # This is complex, maybe just a placeholder
                        pass
                    elif mode == 'custom':
                        pattern = op.get('pattern')
                        repl = op.get('repl', '')
                        if pattern:
                            series = series.str.replace(pattern, repl, regex=True)
                            
            df_out[col] = series
            
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Value Replacement ---
class ValueReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        
        # Config: {'columns': [...], 'mapping': {col: {old: new}}}
        # OR {'columns': [...], 'to_replace': '?', 'value': np.nan}
        # OR {'columns': [...], 'replacements': [{'old': 1, 'new': 100}, ...]}
        
        cols = resolve_columns(X, config)
        mapping = config.get('mapping')
        to_replace = config.get('to_replace')
        value = config.get('value')
        replacements = config.get('replacements')

        if replacements:
            # Convert list of dicts to mapping dict
            mapping = {item['old']: item['new'] for item in replacements}
        
        return {
            'type': 'value_replacement',
            'columns': cols,
            'mapping': mapping,
            'to_replace': to_replace,
            'value': value
        }

class ValueReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        mapping = params.get('mapping')
        to_replace = params.get('to_replace')
        value = params.get('value')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        if mapping:
            # Check if mapping is column-specific (nested dict) or global (simple dict)
            is_nested = any(isinstance(v, dict) for v in mapping.values())
            
            if is_nested:
                # Column-specific mapping
                for col, map_dict in mapping.items():
                    if col in valid_cols:
                        df_out[col] = df_out[col].replace(map_dict)
            else:
                # Global mapping applied to all valid columns
                for col in valid_cols:
                    df_out[col] = df_out[col].replace(mapping)

        else:
            # Global replacement
            if to_replace is not None:
                for col in valid_cols:
                    df_out[col] = df_out[col].replace(to_replace, value)
                    
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Alias Replacement ---
class AliasReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        cols = resolve_columns(X, config, _auto_detect_text_columns)
        
        # Support both 'alias_type' and 'mode'
        alias_type = config.get('alias_type') or config.get('mode', 'boolean')
        
        # Map legacy modes
        if alias_type == 'normalize_boolean':
            alias_type = 'boolean'
        elif alias_type == 'canonicalize_country_codes':
            alias_type = 'country'
        
        # Support both 'custom_map' and 'custom_pairs'
        custom_map = config.get('custom_map') or config.get('custom_pairs', {})
        
        # Normalize custom_map keys to match the cleaning logic in Applier
        if custom_map:
            normalized_map = {}
            for k, v in custom_map.items():
                if isinstance(k, str):
                    clean_k = k.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
                    normalized_map[clean_k] = v
                else:
                    normalized_map[k] = v
            custom_map = normalized_map
        
        return {
            'type': 'alias_replacement',
            'columns': cols,
            'alias_type': alias_type,
            'custom_map': custom_map
        }

class AliasReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        alias_type = params.get('alias_type', 'boolean')
        custom_map = params.get('custom_map', {})
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        # Prepare the mapping
        mapping = {}
        if alias_type == 'boolean':
            mapping = COMMON_BOOLEAN_ALIASES
        elif alias_type == 'country':
            mapping = COUNTRY_ALIAS_MAP
        elif alias_type == 'custom':
            mapping = custom_map
            
        # Normalize function
        def normalize(val):
            if not isinstance(val, str):
                return val
            # Remove punctuation and lower case for matching
            clean = val.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            return mapping.get(clean, val)

        for col in valid_cols:
            # Apply normalization
            # This is slow for large dataframes, vectorization is hard with custom fuzzy logic
            # We use map for exact matches on cleaned keys
            
            # 1. Clean the column temporarily
            clean_series = df_out[col].astype(str).str.lower().str.translate(ALIAS_PUNCTUATION_TABLE).str.replace(" ", "")
            
            # 2. Map values
            mapped_series = clean_series.map(mapping)
            
            # 3. Fill NaNs (unmapped) with original values
            df_out[col] = mapped_series.fillna(df_out[col])
            
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Invalid Value Replacement ---
class InvalidValueReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        cols = resolve_columns(X, config)
        
        # Config: {'rule': 'negative', 'replacement': np.nan}
        # Support 'mode' as alias for 'rule'
        rule = config.get('rule') or config.get('mode', 'negative')
        replacement = config.get('replacement', np.nan)
        
        # Support min_value and max_value for custom_range
        min_value = config.get('min_value')
        max_value = config.get('max_value')
        
        return {
            'type': 'invalid_value_replacement',
            'columns': cols,
            'rule': rule,
            'replacement': replacement,
            'min_value': min_value,
            'max_value': max_value
        }

class InvalidValueReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        rule = params.get('rule')
        replacement = params.get('replacement', np.nan)
        min_value = params.get('min_value')
        max_value = params.get('max_value')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        for col in valid_cols:
            series = pd.to_numeric(df_out[col], errors='coerce') # Ensure numeric
            
            mask = None
            if rule == 'negative':
                mask = series < 0
            elif rule == 'zero':
                mask = series == 0
            elif rule == 'negative_to_nan': # Alias for negative with nan replacement
                mask = series < 0
            elif rule == 'custom_range':
                if min_value is not None and max_value is not None:
                    mask = (series < min_value) | (series > max_value)
                elif min_value is not None:
                    mask = series < min_value
                elif max_value is not None:
                    mask = series > max_value
            
            if mask is not None:
                df_out.loc[mask, col] = replacement
                
        return pack_pipeline_output(df_out, y, is_tuple)
