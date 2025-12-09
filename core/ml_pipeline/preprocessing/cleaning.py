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
                        # Sentence case: first letter upper, rest lower, preserving leading whitespace
                        def to_sentence(x):
                            s = str(x)
                            if not s: return s
                            leading_len = len(s) - len(s.lstrip())
                            leading = s[:leading_len]
                            remainder = s[leading_len:]
                            if not remainder:
                                return s
                            return leading + remainder[0].upper() + remainder[1:].lower()
                        series = series.apply(to_sentence)
                    else:
                        series = series.str.lower()
                        
                elif op_type == 'remove_special':
                    mode = op.get('mode', 'keep_alphanumeric')
                    replacement = op.get('replacement', '')
                    
                    pattern = r"[^a-zA-Z0-9]" # Default keep_alphanumeric
                    if mode == 'keep_alphanumeric_space':
                        pattern = r"[^a-zA-Z0-9\s]"
                    elif mode == 'letters_only':
                        pattern = r"[^a-zA-Z]"
                    elif mode == 'digits_only':
                        pattern = r"[^0-9]"
                        
                    series = series.str.replace(pattern, replacement, regex=True)
                    
                elif op_type == 'regex':
                    mode = op.get('mode', 'custom')
                    
                    if mode == 'collapse_whitespace':
                        series = series.str.replace(r"\s+", " ", regex=True).str.strip()
                    elif mode == 'extract_digits':
                        series = series.str.replace(r"[^0-9]", "", regex=True)
                    elif mode == 'normalize_slash_dates':
                        # V1 logic: (\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4}) -> YYYY-MM-DD
                        date_pattern = re.compile(r"\b(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})\b")
                        def _replace_date(match):
                            month = int(match.group(1))
                            day = int(match.group(2))
                            year_token = match.group(3)
                            if len(year_token) == 2:
                                year_value = int(year_token)
                                year_value += 2000 if year_value < TWO_DIGIT_YEAR_PIVOT else 1900
                            else:
                                year_value = int(year_token)
                            return f"{year_value:04d}-{month:02d}-{day:02d}"
                        
                        series = series.apply(lambda x: date_pattern.sub(_replace_date, str(x)) if pd.notna(x) else x)
                        
                    else: # custom
                        pat = op.get('pattern')
                        repl = op.get('repl', '')
                        if pat:
                            series = series.str.replace(pat, repl, regex=True)
                    
            df_out[col] = series
            
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Value Replacement ---
class ValueReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = config.get('columns', [])
        cols = [c for c in cols if c in X.columns]
        
        return {
            'type': 'value_replacement',
            'columns': cols,
            'mapping': config.get('mapping', {}),
            'replacements': config.get('replacements', [])
        }

class ValueReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        mapping = params.get('mapping', {})
        replacements = params.get('replacements', [])
        
        if not mapping and not replacements:
            return pack_pipeline_output(X, y, is_tuple)
            
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        # Merge replacements into mapping
        final_mapping = mapping.copy()
        for r in replacements:
            if 'old' in r and 'new' in r:
                final_mapping[r['old']] = r['new']
        
        df_out = X.copy()
        
        for col in valid_cols:
            df_out[col] = df_out[col].replace(final_mapping)
            
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Alias Replacement ---
class AliasReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, _auto_detect_text_columns)
        
        if not cols:
            return {}
        
        return {
            'type': 'alias_replacement',
            'columns': cols,
            'mode': config.get('mode', 'custom'), # canonicalize_country_codes, normalize_boolean, punctuation, custom
            'custom_pairs': config.get('custom_pairs', {})
        }

class AliasReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        mode = params.get('mode', 'custom')
        custom_pairs = params.get('custom_pairs', {})
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        # Prepare mapping based on mode
        mapping = {}
        normalize_key = False
        strip_punct = False
        
        if mode == 'canonicalize_country_codes':
            mapping = COUNTRY_ALIAS_MAP
            normalize_key = True
            strip_punct = True
        elif mode == 'normalize_boolean':
            mapping = COMMON_BOOLEAN_ALIASES
            normalize_key = True
        elif mode == 'punctuation':
            pass # Handled separately
        elif mode == 'custom':
            mapping = custom_pairs
        
        for col in valid_cols:
            series = df_out[col].astype(str)
            
            if mode == 'punctuation':
                # Remove punctuation
                df_out[col] = series.apply(lambda x: x.translate(ALIAS_PUNCTUATION_TABLE) if pd.notna(x) else x)
            else:
                if normalize_key:
                    def _lookup(x):
                        if pd.isna(x): return x
                        s = str(x).lower().strip()
                        if strip_punct:
                            s = s.translate(ALIAS_PUNCTUATION_TABLE)
                        return mapping.get(s, x)
                    
                    df_out[col] = df_out[col].apply(_lookup)
                else:
                    # Direct replacement
                    df_out[col] = df_out[col].replace(mapping)
                    
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Invalid Value Replacement ---
class InvalidValueReplacementCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config:
        # columns: List[str]
        # mode: 'negative_to_nan', 'zero_to_nan', 'percentage_bounds', 'age_bounds', 'custom_range'
        # min_value: float (for custom_range)
        # max_value: float (for custom_range)
        
        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, _auto_detect_numeric_columns)
        
        return {
            'type': 'invalid_value_replacement',
            'columns': cols,
            'mode': config.get('mode', 'negative_to_nan'),
            'min_value': config.get('min_value'),
            'max_value': config.get('max_value')
        }

class InvalidValueReplacementApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get('columns', [])
        mode = params.get('mode', 'negative_to_nan')
        min_val = params.get('min_value')
        max_val = params.get('max_value')
        
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        # Determine bounds based on mode
        lower = None
        upper = None
        
        if mode == 'percentage_bounds':
            lower = min_val if min_val is not None else 0.0
            upper = max_val if max_val is not None else 100.0
        elif mode == 'age_bounds':
            lower = min_val if min_val is not None else 0.0
            upper = max_val if max_val is not None else 120.0
        elif mode == 'custom_range':
            lower = min_val
            upper = max_val
            
        for col in valid_cols:
            # Ensure numeric
            series = pd.to_numeric(df_out[col], errors='coerce')
            
            mask = pd.Series(False, index=series.index)
            
            if mode == 'negative_to_nan':
                mask = series < 0
            elif mode == 'zero_to_nan':
                mask = series == 0
            elif lower is not None or upper is not None:
                if lower is not None:
                    mask |= (series < lower)
                if upper is not None:
                    mask |= (series > upper)
            
            if mask.any():
                if not pd.api.types.is_float_dtype(df_out[col]):
                     df_out[col] = df_out[col].astype(float)
                
                df_out.loc[mask, col] = np.nan
                
        return pack_pipeline_output(df_out, y, is_tuple)
