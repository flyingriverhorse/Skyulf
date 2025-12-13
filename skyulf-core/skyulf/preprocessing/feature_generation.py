from typing import Dict, Any, List, Optional, Union, Iterable, Tuple
import pandas as pd
import numpy as np
import math
from difflib import SequenceMatcher
from sklearn.preprocessing import PolynomialFeatures
from .base import BaseCalculator, BaseApplier
from ..utils import unpack_pipeline_input, pack_pipeline_output

# --- Optional Dependencies ---
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    _HAS_RAPIDFUZZ = False

# --- Constants ---
DEFAULT_EPSILON = 1e-9
FEATURE_MATH_ALLOWED_TYPES = {
    "arithmetic",
    "ratio",
    "similarity",
    "datetime_extract",
    "group_agg",
    "polynomial",
}
ALLOWED_DATETIME_FEATURES = {
    "year", "quarter", "month", "month_name", "week", "day", "day_name",
    "weekday", "is_weekend", "hour", "minute", "second", "season", "time_of_day",
}

# --- Helpers ---
def _auto_detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=["number"]).columns)

def _coerce_float(value: Any) -> Optional[float]:
    if value is None: return None
    try:
        val = float(value)
        return val if not math.isnan(val) else None
    except (TypeError, ValueError):
        return None

def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    adjusted = denominator.copy()
    adjusted = adjusted.replace({0: epsilon, -0.0: epsilon})
    adjusted = adjusted.fillna(epsilon)
    # Avoid division by zero or near-zero
    mask = adjusted.abs() < epsilon
    if mask.any():
        adjusted[mask] = epsilon
    return numerator / adjusted

def _compute_similarity_score(a: Any, b: Any, method: str) -> float:
    text_a = "" if pd.isna(a) else str(a)
    text_b = "" if pd.isna(b) else str(b)
    if not text_a and not text_b: return 100.0
    if not text_a or not text_b: return 0.0

    if _HAS_RAPIDFUZZ:
        if method == "token_sort_ratio":
            return float(fuzz.token_sort_ratio(text_a, text_b))
        if method == "token_set_ratio":
            return float(fuzz.token_set_ratio(text_a, text_b))
        return float(fuzz.ratio(text_a, text_b))
    
    # Fallback
    return SequenceMatcher(None, text_a, text_b).ratio() * 100.0

# --- Polynomial Features ---
class PolynomialFeaturesCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Extract configuration parameters
        # We support standard polynomial features settings like degree and interaction_only
        
        X, _, _ = unpack_pipeline_input(df)

        cols = config.get("columns", [])
        auto_detect = config.get("auto_detect", False)
        
        if not cols and auto_detect:
            cols = _auto_detect_numeric_columns(X)
            
        cols = [c for c in cols if c in X.columns]
        
        if not cols:
            return {}
            
        degree = config.get("degree", 2)
        interaction_only = config.get("interaction_only", False)
        include_bias = config.get("include_bias", False)
        
        # We use sklearn to get feature names
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        poly.fit(X[cols])
        
        return {
            "type": "polynomial_features",
            "columns": cols,
            "degree": degree,
            "interaction_only": interaction_only,
            "include_bias": include_bias,
            "include_input_features": config.get("include_input_features", False),
            "output_prefix": config.get("output_prefix", "poly"),
            "feature_names": poly.get_feature_names_out(cols).tolist()
        }

class PolynomialFeaturesApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        cols = params.get("columns", [])
        valid_cols = [c for c in cols if c in X.columns]
        
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)
            
        degree = params.get("degree", 2)
        interaction_only = params.get("interaction_only", False)
        include_bias = params.get("include_bias", False)
        include_input_features = params.get("include_input_features", False)
        output_prefix = params.get("output_prefix", "poly")
        
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        poly.fit(X[valid_cols]) # Cheap fit
        
        transformed = poly.transform(X[valid_cols])
        feature_names = poly.get_feature_names_out(valid_cols)
        
        # Rename features to be more readable and avoid collisions
        # We convert sklearn's "col1 col2" format to "prefix_col1_col2"
        
        new_names = []
        for name in feature_names:
            # Clean up name
            clean_name = name.replace(" ", "_").replace("^", "_pow_")
            new_names.append(f"{output_prefix}_{clean_name}")
            
        df_poly = pd.DataFrame(transformed, columns=new_names, index=X.index)
        
        df_out = X.copy()
        if not include_input_features:
            # If input features are not requested, we rely on the user to drop them later if needed,
            # or we could drop them here. Currently we keep the original dataframe structure.
            pass
            
        # Concatenate
        df_out = pd.concat([df_out, df_poly], axis=1)
        
        return pack_pipeline_output(df_out, y, is_tuple)

# --- Feature Generation ---
class FeatureGenerationCalculator(BaseCalculator):
    def fit(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], config: Dict[str, Any]) -> Dict[str, Any]:
        # Config:
        # operations: List[Dict]
        # epsilon: float
        # allow_overwrite: bool
        
        return {
            "type": "feature_generation",
            "operations": config.get("operations", []),
            "epsilon": config.get("epsilon", DEFAULT_EPSILON),
            "allow_overwrite": config.get("allow_overwrite", False)
        }

class FeatureGenerationApplier(BaseApplier):
    def apply(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], params: Dict[str, Any]) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        
        operations = params.get("operations", [])
        epsilon = params.get("epsilon", DEFAULT_EPSILON)
        allow_overwrite = params.get("allow_overwrite", False)
        
        if not operations:
            return pack_pipeline_output(X, y, is_tuple)
            
        df_out = X.copy()
        
        for i, op in enumerate(operations):
            op_type = op.get("operation_type", "arithmetic")
            method = op.get("method")
            input_cols = op.get("input_columns", [])
            secondary_cols = op.get("secondary_columns", [])
            constants = op.get("constants", [])
            output_col = op.get("output_column")
            output_prefix = op.get("output_prefix")
            fillna = op.get("fillna")
            round_digits = op.get("round_digits")
            
            # Resolve output name
            if not output_col:
                # Generate fallback
                base = f"{op_type}_{i}"
                if output_prefix:
                    base = f"{output_prefix}_{base}"
                output_col = base
            
            if output_col in df_out.columns and not allow_overwrite:
                # Avoid overwriting existing columns by appending a numeric suffix
                j = 1
                while f"{output_col}_{j}" in df_out.columns:
                    j += 1
                output_col = f"{output_col}_{j}"
            
            try:
                result = None
                
                if op_type == "arithmetic":
                    # Ensure inputs exist
                    valid_inputs = [c for c in input_cols if c in df_out.columns]
                    if not valid_inputs and not constants: continue
                    
                    series_list = [pd.to_numeric(df_out[c], errors="coerce").fillna(fillna if fillna is not None else 0) for c in valid_inputs]
                    
                    if method == "add":
                        result = pd.Series(0.0, index=df_out.index)
                        for s in series_list: result = result.add(s, fill_value=0)
                        for c in constants: result = result + c
                    elif method == "subtract":
                        if series_list:
                            result = series_list[0].copy()
                            others = series_list[1:]
                        else:
                            result = pd.Series(0.0, index=df_out.index)
                            others = []
                        for s in others: result = result.subtract(s, fill_value=0)
                        for c in constants: result = result - c
                    elif method == "multiply":
                        result = pd.Series(1.0, index=df_out.index)
                        for s in series_list: result = result.multiply(s, fill_value=1)
                        for c in constants: result = result * c
                    elif method == "divide":
                        if series_list:
                            result = series_list[0].copy()
                            others = series_list[1:]
                        elif constants:
                            result = pd.Series(constants[0], index=df_out.index)
                            others = []
                            constants = constants[1:]
                        else:
                            continue
                            
                        for s in others: result = _safe_divide(result, s, epsilon)
                        for c in constants: result = result / (c if abs(c) > epsilon else epsilon)
                        
                elif op_type == "ratio":
                    # input_cols (numerator) / secondary_cols (denominator)
                    # Sum of numerators / Sum of denominators
                    nums = [pd.to_numeric(df_out[c], errors="coerce").fillna(0) for c in input_cols if c in df_out.columns]
                    dens = [pd.to_numeric(df_out[c], errors="coerce").fillna(0) for c in secondary_cols if c in df_out.columns]
                    
                    if not nums or not dens: continue
                    
                    num_sum = pd.Series(0.0, index=df_out.index)
                    for s in nums: num_sum = num_sum.add(s, fill_value=0)
                    
                    den_sum = pd.Series(0.0, index=df_out.index)
                    for s in dens: den_sum = den_sum.add(s, fill_value=0)
                    
                    result = _safe_divide(num_sum, den_sum, epsilon)
                    
                elif op_type == "similarity":
                    # input_cols[0] vs secondary_cols[0] (or input_cols[1])
                    col_a = input_cols[0] if input_cols else None
                    col_b = secondary_cols[0] if secondary_cols else (input_cols[1] if len(input_cols) > 1 else None)
                    
                    if not col_a or not col_b or col_a not in df_out.columns or col_b not in df_out.columns:
                        continue
                        
                    result = df_out.apply(lambda row: _compute_similarity_score(row[col_a], row[col_b], method), axis=1)
                    
                elif op_type == "datetime_extract":
                    valid_inputs = [c for c in input_cols if c in df_out.columns]
                    features = op.get("datetime_features", [])
                    
                    for col in valid_inputs:
                        try:
                            dt = pd.to_datetime(df_out[col], errors="coerce")
                            for feat in features:
                                feat_name = f"{col}_{feat}"
                                if feat == "year": val = dt.dt.year
                                elif feat == "month": val = dt.dt.month
                                elif feat == "day": val = dt.dt.day
                                elif feat == "hour": val = dt.dt.hour
                                elif feat == "minute": val = dt.dt.minute
                                elif feat == "second": val = dt.dt.second
                                elif feat == "quarter": val = dt.dt.quarter
                                elif feat == "dayofweek" or feat == "weekday": val = dt.dt.dayofweek
                                elif feat == "is_weekend": val = (dt.dt.dayofweek >= 5).astype(int)
                                else: continue
                                
                                df_out[feat_name] = val
                        except Exception:
                            pass
                    # datetime_extract usually generates multiple columns, so we might not set "result"
                    # unless we want to return one specific thing. V1 generates multiple.
                    continue 
                
                if result is not None:
                    if round_digits is not None:
                        result = result.round(round_digits)
                    df_out[output_col] = result
                    
            except Exception:
                pass
                
        return pack_pipeline_output(df_out, y, is_tuple)
