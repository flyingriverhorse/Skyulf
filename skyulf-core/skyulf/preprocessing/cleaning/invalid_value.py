"""Invalid-value replacement node."""

from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns, user_picked_no_columns
from .._artifacts import InvalidValueReplacementArtifact
from .._fitted_validation import _columns, local_boolean, local_scalar, local_state_fields
from .._helpers import auto_detect_numeric_columns as _auto_detect_numeric_columns
from .._helpers import integer_replacement_value, resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


def _invalid_rule_polars(
    expr: Any,
    rule: str | None,
    final_replacement: Any,
    min_value: Any,
    max_value: Any,
    replacement_dtype: Any = None,
) -> Any:
    """Apply a single invalid-value rule to a Polars expression."""
    if rule in ("negative", "negative_to_nan"):
        return (
            pl.when(expr < 0)
            .then(pl.lit(final_replacement, dtype=replacement_dtype))
            .otherwise(expr)
        )
    if rule == "zero":
        return (
            pl.when(expr == 0)
            .then(pl.lit(final_replacement, dtype=replacement_dtype))
            .otherwise(expr)
        )
    if rule == "custom_range":
        if min_value is not None and max_value is not None:
            cond = (expr < min_value) | (expr > max_value)
        elif min_value is not None:
            cond = expr < min_value
        elif max_value is not None:
            cond = expr > max_value
        else:
            return expr
        return (
            pl.when(cond).then(pl.lit(final_replacement, dtype=replacement_dtype)).otherwise(expr)
        )
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
        if isinstance(series.dtype, pd.ArrowDtype) and pd.api.types.is_integer_dtype(series.dtype):
            # Arrow boxes Python bounds as signed int64, even for uint64 columns.
            series = series.convert_dtypes(dtype_backend="numpy_nullable")
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
    if replace_inf:
        expr = pl.when(expr == float("inf")).then(pl.lit(final_replacement)).otherwise(expr)
    if replace_neg_inf:
        expr = pl.when(expr == float("-inf")).then(pl.lit(final_replacement)).otherwise(expr)
    return expr


def _resolve_invalid_replacement(params: dict[str, Any]) -> Any:
    replacement = params.get("replacement", np.nan)
    value = params.get("value")
    return value if value is not None else replacement


def _is_null_replacement(value: Any) -> bool:
    """Recognize numeric missing sentinels without coercing other replacement types."""
    return (
        value is None
        or value is pd.NA
        or isinstance(value, (float, np.floating))
        and bool(np.isnan(value))
    )


def _integer_comparison_bound(value: Any) -> Any:
    """Compare integral floating bounds as integers without limiting their range."""
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value == int(value):
        return int(value)
    return value


def _integer_rule_replacement(value: Any, dtype: Any) -> Any:
    """Validate numeric integer rules while retaining native boolean and string replacements."""
    if _is_null_replacement(value):
        return None
    if pd.api.types.is_number(value) and not pd.api.types.is_bool(value):
        return integer_replacement_value(value, dtype, "InvalidValueReplacement")
    return value


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


def _has_numeric_rule(rule: Any, min_value: Any, max_value: Any) -> bool:
    """Identify configured rules that actually compare numeric values."""
    return rule in ("negative", "negative_to_nan", "zero") or (
        rule == "custom_range" and (min_value is not None or max_value is not None)
    )


def _has_numeric_operation(params: Mapping[str, Any]) -> bool:
    """Identify rules and infinity flags that actually compare numeric values."""
    return bool(
        params.get("replace_inf")
        or params.get("replace_neg_inf")
        or _has_numeric_rule(params.get("rule"), params.get("min_value"), params.get("max_value"))
    )


def _invalid_column_polars(col: str, dtype: Any, params: dict[str, Any]) -> Any:
    """Build a numeric column expression without widening exact integer rules."""
    expr = pl.col(col)
    replacement = _resolve_invalid_replacement(params)
    rule, minimum, maximum = params.get("rule"), params.get("min_value"), params.get("max_value")
    replacement_dtype = None
    if not dtype.is_integer():
        expr = _invalid_inf_replacement_polars(
            expr,
            params.get("replace_inf", False),
            params.get("replace_neg_inf", False),
            replacement,
        )
    elif _has_numeric_rule(rule, minimum, maximum):
        replacement = _integer_rule_replacement(replacement, dtype)
        minimum, maximum = _integer_comparison_bound(minimum), _integer_comparison_bound(maximum)
        if replacement is None or type(replacement) is int:
            replacement_dtype = dtype
    return _invalid_rule_polars(expr, rule, replacement, minimum, maximum, replacement_dtype).alias(
        col
    )


def _numeric_columns(X: Any) -> list[str]:
    """Find numeric columns without treating pandas durations as raw nanoseconds."""
    return [
        col for col in _auto_detect_numeric_columns(X) if getattr(X[col].dtype, "kind", None) != "m"
    ]


def _validate_numeric_columns(X: Any, columns: list[str]) -> None:
    """Reject nonnumeric selections before either engine can coerce their values."""
    if not columns:
        return
    numeric = set(_numeric_columns(X))
    non_numeric = [col for col in columns if col not in numeric]
    if non_numeric:
        raise ValueError(
            "InvalidValueReplacement requires numeric columns; "
            f"non-numeric columns: {non_numeric}. "
            "Convert these columns to a numeric type before applying numeric rules."
        )


class InvalidValueReplacementApplier(BaseApplier):
    """Replace values violating the configured rule — and optionally ±inf — with a sentinel.

    The pandas and polars paths must agree value-for-value: ``inf``/``-inf``
    are replaced first when flagged, then the rule (``negative``, ``zero`` or
    ``custom_range``, which honours a single bound when only one is given). A
    configuration without an effective rule or an inf flag is skipped outright,
    so its values and dtype survive untouched. Active operations require numeric
    columns on both engines; text must be explicitly converted before this node.
    Infinity-only cleanup preserves integer values and dtypes: integers cannot
    contain infinities and must not be widened to a floating-point sentinel.
    Active integer rules replacing values with None, NaN or pd.NA use nullable integers,
    retaining their width and exact values in full, singleton and empty requests.
    Numeric replacements must be exactly representable in the current integer dtype;
    fractional or out-of-range replacements require an explicit Casting step first.
    Integral floating range bounds are compared as integers to retain exact ordering.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect fixed numeric rules while retaining NumPy sentinels and true no-ops."""
        if not local_state_fields(
            raw,
            "invalid_value_replacement",
            {
                "type",
                "columns",
                "replace_inf",
                "replace_neg_inf",
                "rule",
                "replacement",
                "value",
                "min_value",
                "max_value",
            },
            allow_empty=True,
        ):
            return raw
        _columns(raw["columns"])
        local_boolean(raw["replace_inf"], "replace_inf")
        local_boolean(raw["replace_neg_inf"], "replace_neg_inf")
        if raw["rule"] not in (None, "negative", "negative_to_nan", "zero", "custom_range"):
            raise ValueError("Unknown fitted invalid-value rule.")
        for name in ("min_value", "max_value"):
            value = raw[name]
            if value is not None and (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            ):
                raise ValueError("Fitted invalid-value bounds must be real numbers or None.")
        local_scalar(raw["replacement"], "Invalid-value replacements")
        local_scalar(raw["value"], "Invalid-value replacements")
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe row dependency while leaving actual schema parity to the probe."""
        if engine not in ("pandas", "polars"):
            return None
        InvalidValueReplacementApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch the rule to the pandas or polars path; ``y`` passes through."""
        if _has_numeric_operation(params):
            _validate_numeric_columns(X, resolve_valid_columns(X, params.get("columns", [])))
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid or not _has_numeric_operation(params):
            return X, _y

        exprs = [_invalid_column_polars(col, X[col].dtype, params) for col in valid]
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
        """Normalize numeric data, replace infinities, and apply a rule in-place."""
        to_replace = []
        if replace_inf:
            to_replace.append(np.inf)
        if replace_neg_inf:
            to_replace.append(-np.inf)
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
        if to_replace:
            df_out[col] = df_out[col].replace(to_replace, final_replacement)
        if pd.api.types.is_integer_dtype(df_out[col]) and _has_numeric_rule(
            rule, min_value, max_value
        ):
            final_replacement = _integer_rule_replacement(final_replacement, df_out[col].dtype)
            min_value = _integer_comparison_bound(min_value)
            max_value = _integer_comparison_bound(max_value)
        mask = _invalid_rule_pandas_mask(df_out[col], rule, min_value, max_value)
        if mask is not None:
            if pd.api.types.is_integer_dtype(df_out[col]) and _is_null_replacement(
                final_replacement
            ):
                df_out[col] = df_out[col].convert_dtypes()
            df_out.loc[mask, col] = final_replacement

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid or not _has_numeric_operation(params):
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
    params={"columns": []},
    learns_from_data=False,
)
class InvalidValueReplacementCalculator(BaseCalculator):
    """Resolve invalid-value config into an artifact; nothing is learned from data."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Pass the input schema through: sentinels are rewritten in place."""
        # Replaces invalid sentinel values with NaN in place; columns preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> InvalidValueReplacementArtifact:  # pylint: disable=arguments-differ
        """Build the artifact, auto-detecting numeric columns when none are named.

        An explicit empty column selection yields an empty artifact, which
        makes the applier a no-op. UI convenience modes are canonicalised here
        rather than in each engine: ``percentage_bounds``/``age_bounds`` become
        a ``custom_range`` carrying their default bounds unless the user
        overrode them, and ``zero_to_nan`` becomes ``zero``.
        Active rules reject nonnumeric selected columns with ``ValueError``;
        the applier repeats this check when inference data changes dtype.
        """
        if user_picked_no_columns(config):
            return {}
        cols = resolve_columns(X, config, _numeric_columns)
        rule, min_value, max_value = _normalize_rule(
            config.get("rule") or config.get("mode"),
            config.get("min_value"),
            config.get("max_value"),
        )
        params: InvalidValueReplacementArtifact = {
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
        if _has_numeric_operation(params):
            _validate_numeric_columns(X, cols)
        return params
