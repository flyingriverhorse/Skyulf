"""Invalid-value replacement node."""

from typing import Any

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
    rule: str | None,
    final_replacement: Any,
    min_value: Any,
    max_value: Any,
) -> Any:
    """Apply a single invalid-value rule to a Polars expression."""
    import polars as pl

    if rule in ("negative", "negative_to_nan"):
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
    rule: str | None,
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


def _resolve_invalid_replacement(params: dict[str, Any]) -> Any:
    replacement = params.get("replacement", np.nan)
    value = params.get("value")
    return value if value is not None else replacement


# The frontend's "mode" dropdown offers a few convenience presets that don't
# have a matching entry in `_invalid_rule_pandas_mask`/`_invalid_rule_polars`
# (which only understand "negative"/"negative_to_nan", "zero", and
# "custom_range"). Without this mapping, selecting "Zero to NaN",
# "Percentage Bounds", or "Age Bounds" in the UI silently did nothing on
# either engine. Normalize aliases to a canonical rule (+ default bounds when
# the user hasn't overridden them) here, once, at fit-time.
_RULE_ALIASES = {"zero_to_nan": "zero"}
_RULE_DEFAULT_BOUNDS = {
    "percentage_bounds": (0.0, 100.0),
    "age_bounds": (0.0, 120.0),
}


def _normalize_rule(
    raw_rule: str | None, min_value: Any, max_value: Any
) -> tuple[str | None, Any, Any]:
    """Map UI convenience mode aliases to a canonical rule + bounds."""
    if raw_rule in _RULE_DEFAULT_BOUNDS:
        default_min, default_max = _RULE_DEFAULT_BOUNDS[raw_rule]
        return (
            "custom_range",
            min_value if min_value is not None else default_min,
            max_value if max_value is not None else default_max,
        )
    return _RULE_ALIASES.get(raw_rule, raw_rule), min_value, max_value  # ty: ignore[no-matching-overload]


class InvalidValueReplacementApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    def _apply_pandas_column(
        df_out: Any,
        col: str,
        replace_inf: bool,
        replace_neg_inf: bool,
        rule: Any,
        final_replacement: Any,
        min_value: Any,
        max_value: Any,
    ) -> None:
        """Coerce, replace inf/-inf, and apply the invalid-value rule for a single column in-place."""
        to_replace = []
        if replace_inf:
            to_replace.append(np.inf)
        if replace_neg_inf:
            to_replace.append(-np.inf)
        # Skip entirely when no rule/inf-replacement is configured for this
        # column -- a true no-op, matching the polars path. Previously this
        # unconditionally ran pd.to_numeric(..., errors="coerce"), which
        # silently NaN'd out non-numeric columns even when nothing was
        # actually configured to change.
        if not to_replace and rule is None:
            return
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
        if to_replace:
            df_out[col] = df_out[col].replace(to_replace, final_replacement)
        mask = _invalid_rule_pandas_mask(df_out[col], rule, min_value, max_value)
        if mask is not None:
            df_out.loc[mask, col] = final_replacement

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
            InvalidValueReplacementApplier._apply_pandas_column(
                df_out,
                col,
                replace_inf,
                replace_neg_inf,
                rule,
                final_replacement,
                min_value,
                max_value,
            )
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
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Replaces invalid sentinel values with NaN in place; columns preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> InvalidValueReplacementArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return {}
        cols = resolve_columns(X, config, _auto_detect_numeric_columns)
        rule, min_value, max_value = _normalize_rule(
            config.get("rule") or config.get("mode"),
            config.get("min_value"),
            config.get("max_value"),
        )
        return {
            "type": "invalid_value_replacement",
            "columns": cols,
            "replace_inf": config.get("replace_inf", False),
            "replace_neg_inf": config.get("replace_neg_inf", False),
            "rule": rule,
            "replacement": config.get("replacement", np.nan),
            "value": config.get("value"),
            "min_value": min_value,
            "max_value": max_value,
        }
