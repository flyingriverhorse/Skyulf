"""Invalid-value replacement node."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import InvalidValueReplacementArtifact
from .._helpers import auto_detect_numeric_columns as _auto_detect_numeric_columns
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


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
