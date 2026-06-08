"""Text cleaning node (trim / case / remove-special / regex)."""

from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import TextCleaningArtifact
from .._helpers import auto_detect_text_columns as _auto_detect_text_columns
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _REMOVE_SPECIAL_PATTERNS


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
