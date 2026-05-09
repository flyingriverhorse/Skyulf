"""Cleaning nodes (Text / InvalidValue / ValueReplacement / AliasReplacement).

Each Applier dispatches on engine via ``apply_dual_engine`` so per-engine
logic lives in small ``_apply_polars`` / ``_apply_pandas`` static helpers.
Per-operation work is further split into module-level helpers (``_trim_*``,
``_case_*``, ``_remove_special_*``, ``_regex_*``, ``_invalid_rule_*``) plus
small dispatch dicts so individual methods stay at CCN ≤ 8.
"""

import re
import string
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import (
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import (
    AliasReplacementArtifact,
    InvalidValueReplacementArtifact,
    TextCleaningArtifact,
    ValueReplacementArtifact,
)
from ._helpers import (
    auto_detect_numeric_columns as _auto_detect_numeric_columns,
    auto_detect_text_columns as _auto_detect_text_columns,
    resolve_valid_columns,
)
from ._schema import SkyulfSchema
from ..registry import NodeRegistry
from ..core.meta.decorators import node_meta

# =============================================================================
# Constants
# =============================================================================

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

_REMOVE_SPECIAL_PATTERNS: Dict[str, str] = {
    "keep_alphanumeric": r"[^a-zA-Z0-9]",
    "keep_alphanumeric_space": r"[^a-zA-Z0-9\s]",
    "letters_only": r"[^a-zA-Z]",
    "digits_only": r"[^0-9]",
}


# =============================================================================
# Text Cleaning — per-operation helpers (Polars)
# =============================================================================


def _trim_polars(expr: Any, mode: str) -> Any:
    if mode == "leading":
        return expr.str.strip_chars_start()
    if mode == "trailing":
        return expr.str.strip_chars_end()
    return expr.str.strip_chars()


def _case_polars(expr: Any, mode: str) -> Any:
    if mode == "upper":
        return expr.str.to_uppercase()
    if mode == "title":
        return expr.str.to_titlecase()
    if mode == "sentence":
        return expr.str.slice(0, 1).str.to_uppercase() + expr.str.slice(1).str.to_lowercase()
    return expr.str.to_lowercase()


def _remove_special_polars(expr: Any, mode: str, replacement: str) -> Any:
    pattern = _REMOVE_SPECIAL_PATTERNS.get(mode, "")
    return expr.str.replace_all(pattern, replacement) if pattern else expr


def _regex_polars(expr: Any, mode: str, pattern: Optional[str], repl: str) -> Any:
    if mode == "collapse_whitespace":
        return expr.str.replace_all(r"\s+", " ").str.strip_chars()
    if mode == "extract_digits":
        return expr.str.extract(r"(\d+)", 1)
    if mode == "normalize_slash_dates":
        return expr  # placeholder
    if mode == "custom" and pattern:
        return expr.str.replace_all(pattern, repl)
    return expr


_TEXT_OPS_POLARS: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {
    "trim": lambda expr, op: _trim_polars(expr, op.get("mode", "both")),
    "case": lambda expr, op: _case_polars(expr, op.get("mode", "lower")),
    "remove_special": lambda expr, op: _remove_special_polars(
        expr, op.get("mode", "keep_alphanumeric"), op.get("replacement", "")
    ),
    "regex": lambda expr, op: _regex_polars(
        expr, op.get("mode", "custom"), op.get("pattern"), op.get("repl", "")
    ),
}


# =============================================================================
# Text Cleaning — per-operation helpers (Pandas)
# =============================================================================


def _trim_pandas(series: pd.Series, mode: str) -> pd.Series:
    if mode == "leading":
        return series.str.lstrip()
    if mode == "trailing":
        return series.str.rstrip()
    return series.str.strip()


def _case_pandas(series: pd.Series, mode: str) -> pd.Series:
    if mode == "upper":
        return series.str.upper()
    if mode == "title":
        return series.str.title()
    if mode == "sentence":
        return series.str.capitalize()
    return series.str.lower()


def _remove_special_pandas(series: pd.Series, mode: str, replacement: str) -> pd.Series:
    pattern = _REMOVE_SPECIAL_PATTERNS.get(mode, "")
    return series.str.replace(pattern, replacement, regex=True) if pattern else series


def _regex_pandas(series: pd.Series, mode: str, pattern: Optional[str], repl: str) -> pd.Series:
    if mode == "collapse_whitespace":
        return series.str.replace(r"\s+", " ", regex=True).str.strip()
    if mode == "extract_digits":
        return series.str.extract(r"(\d+)", expand=False)
    if mode == "normalize_slash_dates":
        return series  # placeholder
    if mode == "custom" and pattern:
        return series.str.replace(pattern, repl, regex=True)
    return series


_TEXT_OPS_PANDAS: Dict[str, Callable[[pd.Series, Dict[str, Any]], pd.Series]] = {
    "trim": lambda s, op: _trim_pandas(s, op.get("mode", "both")),
    "case": lambda s, op: _case_pandas(s, op.get("mode", "lower")),
    "remove_special": lambda s, op: _remove_special_pandas(
        s, op.get("mode", "keep_alphanumeric"), op.get("replacement", "")
    ),
    "regex": lambda s, op: _regex_pandas(
        s, op.get("mode", "custom"), op.get("pattern"), op.get("repl", "")
    ),
}


class TextCleaningApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        operations = params.get("operations", [])
        valid = resolve_valid_columns(X, cols)
        if not valid or not operations:
            return X, _y

        exprs = []
        for col in valid:
            expr = pl.col(col).cast(pl.String)
            for op in operations:
                handler = _TEXT_OPS_POLARS.get(op.get("op", ""))
                if handler is not None:
                    expr = handler(expr, op)
            exprs.append(expr.alias(col))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        operations = params.get("operations", [])
        valid = resolve_valid_columns(X, cols)
        if not valid or not operations:
            return X, _y

        df_out = X.copy()
        for col in valid:
            if not pd.api.types.is_string_dtype(df_out[col]):
                df_out[col] = df_out[col].astype(str)
            series = df_out[col]
            for op in operations:
                handler = _TEXT_OPS_PANDAS.get(op.get("op", ""))
                if handler is not None:
                    series = handler(series, op)
            df_out[col] = series
        return df_out, _y


@NodeRegistry.register("TextCleaning", TextCleaningApplier)
@node_meta(
    id="TextCleaning",
    name="Text Cleaning",
    category="Cleaning",
    description="Clean text data (trim, case conversion, remove special chars).",
    params={"columns": [], "operations": []},
)
class TextCleaningCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Text cleaning rewrites string values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> TextCleaningArtifact:
        if user_picked_no_columns(config):
            return {}
        cols = resolve_columns(X, config, _auto_detect_text_columns)
        if not cols:
            return {}
        return {
            "type": "text_cleaning",
            "columns": cols,
            "operations": config.get("operations", []),
        }


# =============================================================================
# Invalid Value Replacement
# =============================================================================


def _invalid_rule_polars(
    expr: Any,
    rule: Optional[str],
    final_replacement: Any,
    min_value: Any,
    max_value: Any,
) -> Any:
    """Apply a single invalid-value rule to a Polars expression."""
    import polars as pl

    if rule == "negative":
        return pl.when(expr < 0).then(final_replacement).otherwise(expr)
    if rule == "zero":
        return pl.when(expr == 0).then(final_replacement).otherwise(expr)
    if rule == "custom_range":
        if min_value is not None and max_value is not None:
            cond = (expr < min_value) | (expr > max_value)
        elif min_value is not None:
            cond = expr < min_value
        elif max_value is not None:
            cond = expr > max_value
        else:
            return expr
        return pl.when(cond).then(final_replacement).otherwise(expr)
    return expr


def _invalid_rule_pandas_mask(
    series: pd.Series,
    rule: Optional[str],
    min_value: Any,
    max_value: Any,
) -> Any:
    """Build the row-mask for an invalid-value rule on a pandas Series."""
    if rule in ("negative", "negative_to_nan"):
        return series < 0
    if rule == "zero":
        return series == 0
    if rule == "custom_range":
        if min_value is not None and max_value is not None:
            return (series < min_value) | (series > max_value)
        if min_value is not None:
            return series < min_value
        if max_value is not None:
            return series > max_value
    return None


def _invalid_inf_replacement_polars(
    expr: Any, replace_inf: bool, replace_neg_inf: bool, final_replacement: Any
) -> Any:
    import polars as pl

    if replace_inf:
        expr = pl.when(expr == float("inf")).then(final_replacement).otherwise(expr)
    if replace_neg_inf:
        expr = pl.when(expr == float("-inf")).then(final_replacement).otherwise(expr)
    return expr


def _resolve_invalid_replacement(params: Dict[str, Any]) -> Any:
    replacement = params.get("replacement", np.nan)
    value = params.get("value")
    return value if value is not None else replacement


class InvalidValueReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        replace_inf = params.get("replace_inf", False)
        replace_neg_inf = params.get("replace_neg_inf", False)
        rule = params.get("rule")
        final_replacement = _resolve_invalid_replacement(params)
        min_value = params.get("min_value")
        max_value = params.get("max_value")

        exprs = []
        for col in valid:
            expr = pl.col(col)
            expr = _invalid_inf_replacement_polars(
                expr, replace_inf, replace_neg_inf, final_replacement
            )
            expr = _invalid_rule_polars(expr, rule, final_replacement, min_value, max_value)
            exprs.append(expr.alias(col))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        replace_inf = params.get("replace_inf", False)
        replace_neg_inf = params.get("replace_neg_inf", False)
        rule = params.get("rule")
        final_replacement = _resolve_invalid_replacement(params)
        min_value = params.get("min_value")
        max_value = params.get("max_value")

        df_out = X.copy()
        for col in valid:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
            to_replace = []
            if replace_inf:
                to_replace.append(np.inf)
            if replace_neg_inf:
                to_replace.append(-np.inf)
            if to_replace:
                df_out[col] = df_out[col].replace(to_replace, final_replacement)
            mask = _invalid_rule_pandas_mask(df_out[col], rule, min_value, max_value)
            if mask is not None:
                df_out.loc[mask, col] = final_replacement
        return df_out, _y


@NodeRegistry.register("InvalidValueReplacement", InvalidValueReplacementApplier)
@node_meta(
    id="InvalidValueReplacement",
    name="Replace Invalid Values",
    category="Cleaning",
    description="Replace specified values with nan.",
    params={"columns": [], "invalid_values": []},
)
class InvalidValueReplacementCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Replaces invalid sentinel values with NaN in place; columns preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> InvalidValueReplacementArtifact:
        if user_picked_no_columns(config):
            return {}
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


# =============================================================================
# Value Replacement
# =============================================================================


def _is_mapping_like(obj: Any) -> bool:
    return isinstance(obj, (dict, pd.Series)) or hasattr(obj, "items")


def _polars_mapping_exprs(valid: List[str], mapping: Dict[str, Any]) -> List[Any]:
    import polars as pl

    is_nested = any(isinstance(v, dict) for v in mapping.values())
    exprs: List[Any] = []
    for col in valid:
        if is_nested and col in mapping:
            exprs.append(pl.col(col).replace(mapping[col], default=pl.col(col)).alias(col))
        elif not is_nested:
            exprs.append(pl.col(col).replace(mapping, default=pl.col(col)).alias(col))
    return exprs


def _polars_to_replace_exprs(valid: List[str], to_replace: Any, value: Any) -> List[Any]:
    import polars as pl

    exprs: List[Any] = []
    is_map = _is_mapping_like(to_replace)
    for col in valid:
        if is_map:
            exprs.append(pl.col(col).replace(to_replace, default=pl.col(col)).alias(col))
        else:
            exprs.append(pl.col(col).replace({to_replace: value}, default=pl.col(col)).alias(col))
    return exprs


def _value_replacement_exprs_polars(
    valid: List[str],
    mapping: Optional[Dict[str, Any]],
    to_replace: Any,
    value: Any,
) -> List[Any]:
    if mapping:
        return _polars_mapping_exprs(valid, mapping)
    if to_replace is None:
        return []
    return _polars_to_replace_exprs(valid, to_replace, value)


def _pandas_apply_mapping(
    df_out: pd.DataFrame, valid: List[str], mapping: Dict[str, Any]
) -> pd.DataFrame:
    is_nested = any(isinstance(v, dict) for v in mapping.values())
    if is_nested:
        for col, map_dict in mapping.items():
            if col in valid:
                df_out[col] = df_out[col].replace(map_dict)
    else:
        for col in valid:
            df_out[col] = df_out[col].replace(mapping)
    return df_out


def _pandas_apply_to_replace(
    df_out: pd.DataFrame, valid: List[str], to_replace: Any, value: Any
) -> pd.DataFrame:
    is_map = _is_mapping_like(to_replace)
    for col in valid:
        if is_map:
            df_out[col] = df_out[col].replace(to_replace)
        else:
            df_out[col] = df_out[col].replace(to_replace, value)
    return df_out


def _apply_value_replacement_pandas(
    df_out: pd.DataFrame,
    valid: List[str],
    mapping: Optional[Dict[str, Any]],
    to_replace: Any,
    value: Any,
) -> pd.DataFrame:
    if mapping:
        return _pandas_apply_mapping(df_out, valid, mapping)
    if to_replace is None:
        return df_out
    return _pandas_apply_to_replace(df_out, valid, to_replace, value)


class ValueReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y
        exprs = _value_replacement_exprs_polars(
            valid,
            params.get("mapping"),
            params.get("to_replace"),
            params.get("value"),
        )
        return (X.with_columns(exprs) if exprs else X), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y
        df_out = X.copy()
        df_out = _apply_value_replacement_pandas(
            df_out,
            valid,
            params.get("mapping"),
            params.get("to_replace"),
            params.get("value"),
        )
        return df_out, _y


@NodeRegistry.register("ValueReplacement", ValueReplacementApplier)
@node_meta(
    id="ValueReplacement",
    name="Replace Values",
    category="Cleaning",
    description="Replace specified values with new values.",
    params={"columns": [], "mapping": {}},
)
class ValueReplacementCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Value mapping replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> ValueReplacementArtifact:
        cols = resolve_columns(X, config)
        mapping = config.get("mapping")
        replacements = config.get("replacements")
        if replacements:
            mapping = {item["old"]: item["new"] for item in replacements}
        return {
            "type": "value_replacement",
            "columns": cols,
            "mapping": mapping,
            "to_replace": config.get("to_replace"),
            "value": config.get("value"),
        }


# =============================================================================
# Alias Replacement
# =============================================================================


def _resolve_alias_type(config: Dict[str, Any]) -> str:
    """Resolve the alias-type alias and remap legacy values."""
    alias_type = config.get("alias_type") or config.get("mode", "boolean")
    if alias_type == "normalize_boolean":
        return "boolean"
    if alias_type == "canonicalize_country_codes":
        return "country"
    return alias_type


def _normalize_alias_custom_map(custom_map: Dict[Any, Any]) -> Dict[Any, Any]:
    """Lowercase + strip punctuation/spaces from string keys to match runtime cleaning."""
    if not custom_map:
        return custom_map
    normalized: Dict[Any, Any] = {}
    for k, v in custom_map.items():
        if isinstance(k, str):
            clean_k = k.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
            normalized[clean_k] = v
        else:
            normalized[k] = v
    return normalized


def _resolve_alias_mapping(alias_type: str, custom_map: Dict[str, str]) -> Dict[str, str]:
    if alias_type == "boolean":
        return COMMON_BOOLEAN_ALIASES
    if alias_type == "country":
        return COUNTRY_ALIAS_MAP
    if alias_type == "custom":
        return custom_map
    return {}


def _normalize_alias_pandas(val: Any, mapping: Dict[str, str]) -> Any:
    if not isinstance(val, str):
        return val
    clean = val.lower().translate(ALIAS_PUNCTUATION_TABLE).replace(" ", "")
    return mapping.get(clean, val)


class AliasReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        mapping = _resolve_alias_mapping(
            params.get("alias_type", "boolean"), params.get("custom_map", {})
        )
        escaped_punct = re.escape(string.punctuation)

        exprs = []
        for col in valid:
            clean_expr = (
                pl.col(col)
                .cast(pl.String)
                .str.to_lowercase()
                .str.replace_all(f"[{escaped_punct}]", "")
                .str.replace_all(" ", "")
            )
            # Polars `replace(default=None)` returns null for non-matches, so we
            # coalesce back to the original value.
            mapped_expr = clean_expr.replace(mapping, default=None)
            final_expr = pl.coalesce([mapped_expr, pl.col(col)])
            exprs.append(final_expr.alias(col))
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y

        mapping = _resolve_alias_mapping(
            params.get("alias_type", "boolean"), params.get("custom_map", {})
        )

        df_out = X.copy()
        for col in valid:
            clean_series = (
                df_out[col]
                .astype(str)
                .str.lower()
                .str.translate(ALIAS_PUNCTUATION_TABLE)
                .str.replace(" ", "")
            )
            mapped_series = clean_series.map(mapping)
            df_out[col] = mapped_series.fillna(df_out[col])
        return df_out, _y


@NodeRegistry.register("AliasReplacement", AliasReplacementApplier)
@node_meta(
    id="AliasReplacement",
    name="Standardize Values",
    category="Cleaning",
    description="Standardize common variations in text values (e.g. Yes/No, Country names).",
    params={"columns": [], "domain": "boolean"},
)
class AliasReplacementCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Alias normalization replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> AliasReplacementArtifact:
        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, _auto_detect_text_columns)
        alias_type = _resolve_alias_type(config)
        custom_map = _normalize_alias_custom_map(
            config.get("custom_map") or config.get("custom_pairs", {})
        )

        return {
            "type": "alias_replacement",
            "columns": cols,
            "alias_type": alias_type,
            "custom_map": custom_map,
        }
