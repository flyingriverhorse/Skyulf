"""Value replacement node (mapping / to_replace)."""

from typing import Any

import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import resolve_columns
from .._artifacts import ValueReplacementArtifact
from .._fitted_validation import _columns, local_scalar, local_state_fields
from .._helpers import resolve_valid_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

_SKIP_MAPPING_KEY = object()


def _is_mapping_like(obj: Any) -> bool:
    return isinstance(obj, (dict, pd.Series)) or hasattr(obj, "items")


def _coerce_key(key: Any, dtype_kind: str) -> Any:
    """Coerce a mapping key to its column dtype, skipping strings that cannot match."""
    if not isinstance(key, str):
        return key
    try:
        if dtype_kind in ("i", "u"):
            return int(key)
        if dtype_kind == "f":
            return float(key)
        if dtype_kind == "b":
            key_lower = key.strip().lower()
            if key_lower in {"true", "1", "false", "0"}:
                return key_lower in {"true", "1"}
            return _SKIP_MAPPING_KEY
    except (ValueError, TypeError):
        return _SKIP_MAPPING_KEY
    return key


def _coerce_mapping_keys(mapping: dict[str, Any], dtype_kind: str) -> dict[Any, Any]:
    """Coerce string-typed mapping keys (as produced by JSON configs) to a column's dtype.

    Applies to numeric/boolean columns so lookups actually match values instead
    of silently no-op'ing (pandas) or stringifying the whole column (polars
    ``replace_strict``).
    """
    if dtype_kind not in ("i", "u", "f", "b"):
        return mapping
    coerce = {}
    for key, value in mapping.items():
        coerced_key = _coerce_key(key, dtype_kind)
        if coerced_key is not _SKIP_MAPPING_KEY:
            coerce[coerced_key] = value
    return coerce


def _polars_dtype_kind(dtype: Any) -> str:
    """Map a polars dtype to a numpy-style dtype-kind character."""
    if dtype.is_integer():
        return "i"
    if dtype.is_float():
        return "f"
    if str(dtype) == "Boolean":
        return "b"
    return ""


def _polars_mapping_exprs(X: Any, valid: list[str], mapping: dict[str, Any]) -> list[Any]:
    is_nested = any(isinstance(v, dict) for v in mapping.values())
    schema = X.schema
    exprs: list[Any] = []
    for col in valid:
        if is_nested and col not in mapping:
            continue
        col_map = mapping[col] if is_nested else mapping
        col_map = _coerce_mapping_keys(col_map, _polars_dtype_kind(schema[col]))
        exprs.append(pl.col(col).replace_strict(col_map, default=pl.col(col)).alias(col))
    return exprs


def _replacement_pairs(to_replace: Any, value: Any) -> list[tuple[Any, Any]]:
    """Keep list rules ordered without merging boolean and numeric keys."""
    if not pd.api.types.is_list_like(to_replace):
        if pd.api.types.is_list_like(value):
            raise TypeError("A scalar to_replace requires a scalar replacement value.")
        return [(to_replace, value)]
    keys = list(to_replace)
    values = list(value) if pd.api.types.is_list_like(value) else [value] * len(keys)
    if len(keys) != len(values):
        raise ValueError("Replacement lists must have the same length.")
    return list(zip(keys, values, strict=True))


def _replacement_key_matches_dtype(key: Any, dtype_kind: str) -> bool:
    """Respect pandas's distinction between boolean and numeric typed columns."""
    if pd.api.types.is_bool(key):
        return dtype_kind not in ("i", "u", "f")
    return dtype_kind != "b" or not pd.api.types.is_number(key)


def _coerce_replacement_pairs(
    pairs: list[tuple[Any, Any]], dtype_kind: str
) -> list[tuple[Any, Any]]:
    """Coerce JSON string keys while retaining the order of applicable rules."""
    coerced = []
    for key, value in pairs:
        key = _coerce_key(key, dtype_kind)
        if key is not _SKIP_MAPPING_KEY and _replacement_key_matches_dtype(key, dtype_kind):
            coerced.append((key, value))
    return coerced


def _value_replacement_exprs_polars(
    X: Any,
    valid: list[str],
    mapping: dict[str, Any] | None,
    to_replace: Any,
    value: Any,
) -> list[Any]:
    if mapping:
        return _polars_mapping_exprs(X, valid, mapping)
    if to_replace is None:
        return []
    if _is_mapping_like(to_replace):
        return _polars_mapping_exprs(X, valid, dict(to_replace.items()))
    pairs = _replacement_pairs(to_replace, value)
    exprs = []
    for col in valid:
        col_map = dict(_coerce_replacement_pairs(pairs, _polars_dtype_kind(X.schema[col])))
        exprs.append(pl.col(col).replace_strict(col_map, default=pl.col(col)).alias(col))
    return exprs


def _pandas_dtype_kind(dtype: Any) -> str:
    """Map a pandas/numpy dtype to a dtype-kind character.

    Treats pandas nullable extension dtypes (``Int64``, ``Float64``,
    ``boolean``) the same as their numpy equivalents.
    """
    kind = getattr(dtype, "kind", "")
    return kind if kind in ("i", "u", "f", "b") else ""


def _pandas_apply_mapping(
    df_out: pd.DataFrame, valid: list[str], mapping: dict[str, Any]
) -> pd.DataFrame:
    is_nested = any(isinstance(v, dict) for v in mapping.values())
    # Preserve object columns instead of inferring types from request neighbors.
    with pd.option_context("future.no_silent_downcasting", True):
        if is_nested:
            for col, map_dict in mapping.items():
                if col in valid:
                    map_dict = _coerce_mapping_keys(map_dict, _pandas_dtype_kind(df_out[col].dtype))
                    df_out[col] = df_out[col].replace(map_dict)
        else:
            for col in valid:
                col_map = _coerce_mapping_keys(mapping, _pandas_dtype_kind(df_out[col].dtype))
                df_out[col] = df_out[col].replace(col_map)
    return df_out


def _pandas_replace_pairs(
    series: pd.Series, pairs: list[tuple[Any, Any]], scalar: bool
) -> pd.Series:
    """Keep native scalar null handling and ordered list replacement semantics."""
    if scalar and pairs:
        key, value = pairs[0]
        return series.replace(key, value)
    keys = [key for key, _ in pairs]
    values = [replacement for _, replacement in pairs]
    return series.replace(keys, values)


def _apply_value_replacement_pandas(
    df_out: pd.DataFrame,
    valid: list[str],
    mapping: dict[str, Any] | None,
    to_replace: Any,
    value: Any,
) -> pd.DataFrame:
    if mapping:
        return _pandas_apply_mapping(df_out, valid, mapping)
    if to_replace is None:
        return df_out
    if _is_mapping_like(to_replace):
        return _pandas_apply_mapping(df_out, valid, dict(to_replace.items()))
    pairs = _replacement_pairs(to_replace, value)
    with pd.option_context("future.no_silent_downcasting", True):
        for col in valid:
            rules = _coerce_replacement_pairs(pairs, _pandas_dtype_kind(df_out[col].dtype))
            df_out[col] = _pandas_replace_pairs(
                df_out[col], rules, scalar=not pd.api.types.is_list_like(to_replace)
            )
    return df_out


def _validate_replacement_scalar(value: Any) -> None:
    """Allow Python and NumPy scalar rules without converting null or non-finite values."""
    local_scalar(value, "ValueReplacement rules")


def _validate_replacement_mapping(mapping: Any) -> None:
    """Inspect flat or consistently column-nested maps without coercing their keys."""
    if type(mapping) is not dict:
        raise ValueError("ValueReplacement mapping must be a dictionary.")
    nested = any(type(value) is dict for value in mapping.values())
    for key, value in mapping.items():
        if nested:
            if type(key) is not str or type(value) is not dict:
                raise ValueError("ValueReplacement nested mappings need column dictionaries.")
            for old, new in value.items():
                _validate_replacement_scalar(old)
                _validate_replacement_scalar(new)
        else:
            _validate_replacement_scalar(key)
            _validate_replacement_scalar(value)


def _validate_replacement_values(value: Any) -> None:
    """Inspect scalar, list or tuple rules while retaining saved ordering and types."""
    if type(value) in (list, tuple):
        for item in value:
            _validate_replacement_scalar(item)
    else:
        _validate_replacement_scalar(value)


def _validate_replacement_rules(raw: dict) -> None:
    """Validate active scalar/list pairing and retain mapping precedence unchanged."""
    mapping, to_replace, value = raw["mapping"], raw["to_replace"], raw["value"]
    if mapping is not None:
        _validate_replacement_mapping(mapping)
    if type(to_replace) is dict:
        _validate_replacement_mapping(to_replace)
    else:
        _validate_replacement_values(to_replace)
    _validate_replacement_values(value)
    if mapping or to_replace is None or type(to_replace) is dict:
        return
    if type(value) in (list, tuple) and (
        type(to_replace) not in (list, tuple) or len(to_replace) != len(value)
    ):
        raise ValueError("ValueReplacement scalar/list rules have incompatible lengths.")


class ValueReplacementApplier(BaseApplier):
    """Replace configured values in the resolved columns, leaving other cells alone.

    The pandas and polars paths must agree value-for-value. A ``mapping`` wins
    over ``to_replace``/``value`` when both are configured, and may be flat
    (one map applied to every column) or nested (``{column: {old: new}}``).
    Mapping and ``to_replace`` keys are coerced to the column's dtype before
    lookup. A list of keys can share one replacement or have a same-length
    list of replacements. A column selection resolving to nothing is a no-op.
    """

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect scalar, list, tuple and dictionary state without coercing saved rules."""
        local_state_fields(
            raw, "value_replacement", {"type", "columns", "mapping", "to_replace", "value"}
        )
        _columns(raw["columns"])
        _validate_replacement_rules(raw)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe row dependency without claiming chunk-invariant dtype inference."""
        if engine not in ("pandas", "polars"):
            return None
        ValueReplacementApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Dispatch the replacement to the pandas or polars path; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        valid = resolve_valid_columns(X, params.get("columns", []))
        if not valid:
            return X, _y
        exprs = _value_replacement_exprs_polars(
            X,
            valid,
            params.get("mapping"),
            params.get("to_replace"),
            params.get("value"),
        )
        return (X.with_columns(exprs) if exprs else X), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    learns_from_data=False,
)
class ValueReplacementCalculator(BaseCalculator):
    """Resolve value-replacement config into an artifact; nothing is learned from data."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Pass the input schema through, since only cell values are rewritten."""
        # Value mapping replaces values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> ValueReplacementArtifact:  # pylint: disable=arguments-differ
        """Build the artifact, folding the UI's ``replacements`` pairs into a flat mapping.

        A non-empty ``replacements`` list overrides any configured ``mapping``.
        ``to_replace``/``value`` are carried through untouched for the applier
        to fall back on when no mapping survived.
        """
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
