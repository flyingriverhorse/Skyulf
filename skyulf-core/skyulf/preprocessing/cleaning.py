import string
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import pack_pipeline_output, resolve_columns, unpack_pipeline_input
from .base import BaseApplier, BaseCalculator
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta
from ..engines import SkyulfDataFrame, get_engine

# --- Constants ---
ALIAS_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
COMMON_BOOLEAN_ALIASES: Dict[str, str] = {
    "y": "Yes",
    "yes": "Yes",
    "true": "Yes",
    "1": "Yes",
    "on": "Yes",
    "t": "Yes",
    "affirmative": "Yes",
    "n": "No",
    "no": "No",
    "false": "No",
    "0": "No",
    "off": "No",
    "f": "No",
    "negative": "No",
}
COUNTRY_ALIAS_MAP: Dict[str, str] = {
    "usa": "USA",
    "us": "USA",
    "unitedstates": "USA",
    "unitedstatesofamerica": "USA",
    "states": "USA",
    "america": "USA",
    "unitedkingdom": "United Kingdom",
    "uk": "United Kingdom",
    "greatbritain": "United Kingdom",
    "england": "United Kingdom",
    "uae": "United Arab Emirates",
    "unitedarabemirates": "United Arab Emirates",
    "prc": "China",
    "peoplesrepublicofchina": "China",
    "southkorea": "South Korea",
    "republicofkorea": "South Korea",
    "sk": "South Korea",
}
TWO_DIGIT_YEAR_PIVOT = 50

# --- Helpers ---


def _auto_detect_text_columns(df: Union[pd.DataFrame, SkyulfDataFrame]) -> List[str]:
    engine = get_engine(df)
    if engine.name == "polars":
        import polars as pl

        return [
            c
            for c, t in zip(df.columns, df.dtypes)
            if t in [pl.Utf8, pl.Categorical, pl.Object]
        ]
    return list(df.select_dtypes(include=["object", "string", "category"]).columns)


def _auto_detect_numeric_columns(df: Union[pd.DataFrame, SkyulfDataFrame]) -> List[str]:
    engine = get_engine(df)
    if engine.name == "polars":
        import polars as pl

        return [
            c
            for c, t in zip(df.columns, df.dtypes)
            if t in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        ]
    return list(df.select_dtypes(include=["number"]).columns)


def _auto_detect_datetime_columns(df: Union[pd.DataFrame, SkyulfDataFrame]) -> List[str]:
    engine = get_engine(df)
    if engine.name == "polars":
        import polars as pl

        return [
            c
            for c, t in zip(df.columns, df.dtypes)
            if t in [pl.Date, pl.Datetime] or isinstance(t, pl.Datetime)
        ]
    return list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)


# --- Text Cleaning ---


class TextCleaningApplier(BaseApplier):
    def apply(  # noqa: C901
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        if not params or not params.get("columns"):
            return pack_pipeline_output(X, y, is_tuple)

        cols = params["columns"]
        operations = params.get("operations", [])

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not operations:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            for col in valid_cols:
                expr = pl.col(col).cast(pl.String)
                
                for op in operations:
                    op_type = op.get("op")
                    
                    if op_type == "trim":
                        mode = op.get("mode", "both")
                        if mode == "leading":
                            expr = expr.str.strip_chars_start()
                        elif mode == "trailing":
                            expr = expr.str.strip_chars_end()
                        else:
                            expr = expr.str.strip_chars()
                            
                    elif op_type == "case":
                        mode = op.get("mode", "lower")
                        if mode == "upper":
                            expr = expr.str.to_uppercase()
                        elif mode == "title":
                            expr = expr.str.to_titlecase()
                        elif mode == "sentence":
                            # Capitalize first char, lower the rest
                            expr = (
                                expr.str.slice(0, 1).str.to_uppercase() + 
                                expr.str.slice(1).str.to_lowercase()
                            )
                        else:
                            expr = expr.str.to_lowercase()
                            
                    elif op_type == "remove_special":
                        mode = op.get("mode", "keep_alphanumeric")
                        replacement = op.get("replacement", "")
                        
                        pattern = ""
                        if mode == "keep_alphanumeric":
                            pattern = r"[^a-zA-Z0-9]"
                        elif mode == "keep_alphanumeric_space":
                            pattern = r"[^a-zA-Z0-9\s]"
                        elif mode == "letters_only":
                            pattern = r"[^a-zA-Z]"
                        elif mode == "digits_only":
                            pattern = r"[^0-9]"
                            
                        if pattern:
                            expr = expr.str.replace_all(pattern, replacement)
                            
                    elif op_type == "regex":
                        mode = op.get("mode", "custom")
                        
                        if mode == "collapse_whitespace":
                            expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()
                        elif mode == "extract_digits":
                            # extract first group
                            expr = expr.str.extract(r"(\d+)", 1)
                        elif mode == "normalize_slash_dates":
                            pass
                        elif mode == "custom":
                            pattern = op.get("pattern")
                            repl = op.get("repl", "")
                            if pattern:
                                expr = expr.str.replace_all(pattern, repl)
                                
                exprs.append(expr.alias(col))
            
            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

        for col in valid_cols:
            # Ensure column is string type
            if not pd.api.types.is_string_dtype(df_out[col]):
                df_out[col] = df_out[col].astype(str)

            series = df_out[col]

            for op in operations:
                op_type = op.get("op")

                if op_type == "trim":
                    mode = op.get("mode", "both")
                    if mode == "leading":
                        series = series.str.lstrip()
                    elif mode == "trailing":
                        series = series.str.rstrip()
                    else:
                        series = series.str.strip()

                elif op_type == "case":
                    mode = op.get("mode", "lower")
                    if mode == "upper":
                        series = series.str.upper()
                    elif mode == "title":
                        series = series.str.title()
                    elif mode == "sentence":
                        series = series.str.capitalize()
                    else:
                        series = series.str.lower()

                elif op_type == "remove_special":
                    mode = op.get("mode", "keep_alphanumeric")
                    replacement = op.get("replacement", "")

                    if mode == "keep_alphanumeric":
                        # Keep letters and numbers
                        series = series.str.replace(
                            r"[^a-zA-Z0-9]", replacement, regex=True
                        )
                    elif mode == "keep_alphanumeric_space":
                        # Keep letters, numbers, and spaces
                        series = series.str.replace(
                            r"[^a-zA-Z0-9\s]", replacement, regex=True
                        )
                    elif mode == "letters_only":
                        series = series.str.replace(
                            r"[^a-zA-Z]", replacement, regex=True
                        )
                    elif mode == "digits_only":
                        series = series.str.replace(r"[^0-9]", replacement, regex=True)

                elif op_type == "regex":
                    mode = op.get("mode", "custom")

                    if mode == "collapse_whitespace":
                        series = series.str.replace(r"\s+", " ", regex=True).str.strip()
                    elif mode == "extract_digits":
                        series = series.str.extract(r"(\d+)", expand=False)
                    elif mode == "normalize_slash_dates":
                        # Simple attempt to fix 1/1/2020 -> 01/01/2020
                        # This is complex, maybe just a placeholder
                        pass
                    elif mode == "custom":
                        pattern = op.get("pattern")
                        repl = op.get("repl", "")
                        if pattern:
                            series = series.str.replace(pattern, repl, regex=True)

            df_out[col] = series

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("TextCleaning", TextCleaningApplier)
@node_meta(
    id="TextCleaning",
    name="Text Cleaning",
    category="Cleaning",
    description="Clean text data (trim, case conversion, remove special chars).",
    params={"columns": [], "operations": []}
)
class TextCleaningCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Config:
        # columns: List[str]
        # operations: List[Dict]
        #   {'op': 'trim', 'mode': 'both'|'leading'|'trailing'}
        #   {'op': 'case', 'mode': 'lower'|'upper'|'title'|'sentence'}
        #   {'op': 'remove_special', 'mode': 'keep_alphanumeric'|'keep_alphanumeric_space'|'letters_only'|'digits_only',
        #    'replacement': ''}
        #   {'op': 'regex', 'mode': 'custom'|'collapse_whitespace'|'extract_digits'|'normalize_slash_dates',
        #    'pattern': '...', 'repl': '...'}

        X, _, _ = unpack_pipeline_input(df)

        cols = resolve_columns(X, config, _auto_detect_text_columns)

        if not cols:
            return {}

        operations = config.get("operations", [])

        return {"type": "text_cleaning", "columns": cols, "operations": operations}


# --- Invalid Value Replacement ---


class InvalidValueReplacementApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        
        # Params
        replace_inf = params.get("replace_inf", False)
        replace_neg_inf = params.get("replace_neg_inf", False)
        rule = params.get("rule")
        
        # Unify replacement value
        replacement = params.get("replacement", np.nan)
        value = params.get("value")
        final_replacement = value if value is not None else replacement
        
        min_value = params.get("min_value")
        max_value = params.get("max_value")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            for col in valid_cols:
                expr = pl.col(col)
                
                # 1. Handle Inf
                if replace_inf:
                    expr = pl.when(expr == float('inf')).then(final_replacement).otherwise(expr)
                if replace_neg_inf:
                    expr = pl.when(expr == float('-inf')).then(final_replacement).otherwise(expr)
                    
                # 2. Handle Rules
                if rule == "negative":
                    expr = pl.when(expr < 0).then(final_replacement).otherwise(expr)
                elif rule == "zero":
                    expr = pl.when(expr == 0).then(final_replacement).otherwise(expr)
                elif rule == "custom_range":
                    if min_value is not None and max_value is not None:
                        expr = pl.when((expr < min_value) | (expr > max_value)).then(final_replacement).otherwise(expr)
                    elif min_value is not None:
                        expr = pl.when(expr < min_value).then(final_replacement).otherwise(expr)
                    elif max_value is not None:
                        expr = pl.when(expr > max_value).then(final_replacement).otherwise(expr)
                        
                exprs.append(expr.alias(col))
            
            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()
        
        for col in valid_cols:
            # Ensure numeric
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
            
            # 1. Handle Inf
            to_replace = []
            if replace_inf:
                to_replace.append(np.inf)
            if replace_neg_inf:
                to_replace.append(-np.inf)
            
            if to_replace:
                df_out[col] = df_out[col].replace(to_replace, final_replacement)
                
            # 2. Handle Rules
            series = df_out[col]
            mask = None
            
            if rule == "negative":
                mask = series < 0
            elif rule == "zero":
                mask = series == 0
            elif rule == "negative_to_nan":
                mask = series < 0
            elif rule == "custom_range":
                if min_value is not None and max_value is not None:
                    mask = (series < min_value) | (series > max_value)
                elif min_value is not None:
                    mask = series < min_value
                elif max_value is not None:
                    mask = series > max_value
            
            if mask is not None:
                df_out.loc[mask, col] = final_replacement

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("InvalidValueReplacement", InvalidValueReplacementApplier)
@node_meta(
    id="InvalidValueReplacement",
    name="Replace Invalid Values",
    category="Cleaning",
    description="Replace specified values with nan.",
    params={"columns": [], "invalid_values": []}
)
class InvalidValueReplacementCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        cols = resolve_columns(X, config, _auto_detect_numeric_columns)
        
        return {
            "type": "invalid_value_replacement",
            "columns": cols,
            "replace_inf": config.get("replace_inf", False),
            "replace_neg_inf": config.get("replace_neg_inf", False),
            "rule": config.get("rule") or config.get("mode"),
            "replacement": config.get("replacement", np.nan),
            "value": config.get("value"),
            "min_value": config.get("min_value"),
            "max_value": config.get("max_value"),
        }


class ValueReplacementApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        mapping = params.get("mapping")
        to_replace = params.get("to_replace")
        value = params.get("value")

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            
            exprs = []
            
            if mapping:
                is_nested = any(isinstance(v, dict) for v in mapping.values())
                
                if is_nested:
                    # Column-specific mapping
                    for col in valid_cols:
                        if col in mapping:
                            map_dict = mapping[col]
                            # Polars replace: mapping dict
                            # We need to ensure types match or use strict=False
                            # map_dict keys/values should be compatible with col type
                            exprs.append(pl.col(col).replace(map_dict, default=pl.col(col)).alias(col))
                else:
                    # Global mapping
                    for col in valid_cols:
                        exprs.append(pl.col(col).replace(mapping, default=pl.col(col)).alias(col))
            
            else:
                # Global replacement
                if to_replace is not None:
                    for col in valid_cols:
                        if isinstance(to_replace, (dict, pd.Series)) or hasattr(to_replace, "items"):
                             exprs.append(pl.col(col).replace(to_replace, default=pl.col(col)).alias(col))
                        else:
                            # Replace single value
                            # pl.when(col == to_replace).then(value).otherwise(col)
                            # But replace is easier
                            exprs.append(pl.col(col).replace({to_replace: value}, default=pl.col(col)).alias(col))
            
            if exprs:
                X_out = X.with_columns(exprs)
                return pack_pipeline_output(X_out, y, is_tuple)
            return pack_pipeline_output(X, y, is_tuple)

        # Pandas Path
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
                    # Check for dict-like objects (dict, pd.Series, etc.)
                    # If to_replace is a mapping, we ignore `value` and apply the mapping.
                    if isinstance(to_replace, (dict, pd.Series)) or hasattr(to_replace, "items"):
                        df_out[col] = df_out[col].replace(to_replace)
                    else:
                        df_out[col] = df_out[col].replace(to_replace, value)

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("ValueReplacement", ValueReplacementApplier)
@node_meta(
    id="ValueReplacement",
    name="Replace Values",
    category="Cleaning",
    description="Replace specified values with new values.",
    params={"columns": [], "mapping": {}}
)
class ValueReplacementCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)

        # Config: {'columns': [...], 'mapping': {col: {old: new}}}
        # OR {'columns': [...], 'to_replace': '?', 'value': np.nan}
        # OR {'columns': [...], 'replacements': [{'old': 1, 'new': 100}, ...]}

        cols = resolve_columns(X, config)
        mapping = config.get("mapping")
        to_replace = config.get("to_replace")
        value = config.get("value")
        replacements = config.get("replacements")

        if replacements:
            # Convert list of dicts to mapping dict
            mapping = {item["old"]: item["new"] for item in replacements}

        return {
            "type": "value_replacement",
            "columns": cols,
            "mapping": mapping,
            "to_replace": to_replace,
            "value": value,
        }


# --- Alias Replacement ---


class AliasReplacementApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        engine = get_engine(X)

        cols = params.get("columns", [])
        alias_type = params.get("alias_type", "boolean")
        custom_map = params.get("custom_map", {})

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            return pack_pipeline_output(X, y, is_tuple)

        # Prepare the mapping
        mapping = {}
        if alias_type == "boolean":
            mapping = COMMON_BOOLEAN_ALIASES
        elif alias_type == "country":
            mapping = COUNTRY_ALIAS_MAP
        elif alias_type == "custom":
            mapping = custom_map

        # Polars Path
        if engine.name == "polars":
            import polars as pl
            import re
            
            # Construct regex for punctuation removal
            # We want to remove punctuation and spaces, then lower case
            # Python: val.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            
            escaped_punct = re.escape(string.punctuation)
            
            exprs = []
            for col in valid_cols:
                # 1. Clean: lower -> remove punct -> remove space
                clean_expr = (
                    pl.col(col)
                    .cast(pl.String)
                    .str.to_lowercase()
                    .str.replace_all(f"[{escaped_punct}]", "")
                    .str.replace_all(" ", "")
                )
                
                # 2. Map
                # replace(mapping, default=None) -> if not found, returns null? 
                # No, replace returns default if not found.
                # We want: if found in mapping, use mapped value. If not found, use ORIGINAL value.
                # So:
                # mapped = clean.replace(mapping, default=None)
                # result = coalesce(mapped, original)
                
                # Note: replace in Polars with a dict maps keys to values.
                # If a value is not in keys, it keeps it AS IS (if default is not set) or uses default.
                # But here 'clean_expr' is the CLEANED value (e.g. "usa").
                # If "usa" is in mapping, it becomes "USA".
                # If "foobar" is NOT in mapping, it stays "foobar".
                # But we want to revert to ORIGINAL "Foo Bar" if not found?
                # The pandas code says: mapped_series.fillna(df_out[col])
                # Pandas map returns NaN if not found.
                # Polars replace returns original value if not found (by default).
                # So we need `replace(mapping, default=None)` to get nulls for non-matches.
                
                mapped_expr = clean_expr.replace(mapping, default=None)
                
                # 3. Coalesce with original
                final_expr = pl.coalesce([mapped_expr, pl.col(col)])
                
                exprs.append(final_expr.alias(col))
            
            X_out = X.with_columns(exprs)
            return pack_pipeline_output(X_out, y, is_tuple)

        # Pandas Path
        df_out = X.copy()

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
            clean_series = (
                df_out[col]
                .astype(str)
                .str.lower()
                .str.translate(ALIAS_PUNCTUATION_TABLE)
                .str.replace(" ", "")
            )

            # 2. Map values
            mapped_series = clean_series.map(mapping)

            # 3. Fill NaNs (unmapped) with original values
            df_out[col] = mapped_series.fillna(df_out[col])

        return pack_pipeline_output(df_out, y, is_tuple)


@NodeRegistry.register("AliasReplacement", AliasReplacementApplier)
@node_meta(
    id="AliasReplacement",
    name="Standardize Values",
    category="Cleaning",
    description="Standardize common variations in text values (e.g. Yes/No, Country names).",
    params={"columns": [], "domain": "boolean"}
)
class AliasReplacementCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        cols = resolve_columns(X, config, _auto_detect_text_columns)

        # Support both 'alias_type' and 'mode'
        alias_type = config.get("alias_type") or config.get("mode", "boolean")

        # Map legacy modes
        if alias_type == "normalize_boolean":
            alias_type = "boolean"
        elif alias_type == "canonicalize_country_codes":
            alias_type = "country"

        # Support both 'custom_map' and 'custom_pairs'
        custom_map = config.get("custom_map") or config.get("custom_pairs", {})

        # Normalize custom_map keys to match the cleaning logic in Applier
        if custom_map:
            normalized_map = {}
            for k, v in custom_map.items():
                if isinstance(k, str):
                    clean_k = (
                        k.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
                    )
                    normalized_map[clean_k] = v
                else:
                    normalized_map[k] = v
            custom_map = normalized_map

        return {
            "type": "alias_replacement",
            "columns": cols,
            "alias_type": alias_type,
            "custom_map": custom_map,
        }



