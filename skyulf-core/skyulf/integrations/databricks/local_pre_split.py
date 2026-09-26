"""Validate fixed pre-split edits and project their saved replay contracts."""

import json
import math
from collections.abc import Callable
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from ...inference.project_code import is_registered_project_step
from ...preprocessing.casting import TYPE_ALIASES
from ._contracts import column_name


def deduplicate_columns(step: dict[str, Any]) -> tuple[str, ...]:
    """Require an explicit, lossless survivor policy for Core Deduplicate."""
    params = step["params"]
    subset = params.get("subset")
    keep = params.get("keep", "first")
    if (
        set(params) - {"subset", "keep"}
        or type(subset) is not list
        or not subset
        or any(type(name) is not str for name in subset)
        or len({name.casefold() for name in subset}) != len(subset)
        or (keep is not False and keep not in ("first", "last", "none"))
    ):
        raise ValueError(
            "pre_split_steps Deduplicate requires a distinct nonempty subset and "
            "keep first, last, none, or False."
        )
    for name in subset:
        column_name(name)
    return tuple(subset)


def custom_filter_columns(step: dict[str, Any]) -> tuple[str, ...]:
    """Validate a saved filter assertion against its isolated class registration."""
    if not is_registered_project_step(step["transformer"]):
        raise ValueError("pre_split_steps custom filter requires registered project code.")
    declaration = step.get("pre_split")
    columns = declaration.get("required_columns") if type(declaration) is dict else None
    if (
        type(declaration) is not dict
        or set(declaration) != {"effect", "required_columns", "learns_from_data"}
        or declaration["effect"] != "filter"
        or declaration["learns_from_data"] is not False
        or type(columns) is not list
        or not columns
        or any(type(name) is not str for name in columns)
        or len({name.casefold() for name in columns}) != len(columns)
        or not _json_value(step["params"])
    ):
        raise ValueError(
            "pre_split_steps custom filter needs an exact filter declaration, "
            "distinct required_columns, learns_from_data=False and JSON-safe params."
        )
    for name in columns:
        column_name(name)
    return tuple(columns)


def _json_value(value: Any) -> bool:
    """Require values whose JSON round trip preserves their type and meaning."""
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError):
        return False


def _pair_key(value: Any) -> tuple[str, Any]:
    """Group old values that can collide in Core's numeric mapping."""
    if isinstance(value, bool):
        return "number", Decimal(int(value))
    if isinstance(value, (int, float, str)):
        try:
            number = Decimal(str(value))
            if not number.is_finite():
                raise ValueError("pre_split_steps replacement old values must be finite.")
            return "number", number
        except InvalidOperation:
            return "text", value
    raise ValueError("pre_split_steps replacement old values must be scalar.")


def _explicit_columns(params: dict[str, Any], kind: str) -> tuple[str, ...]:
    """Require named write columns before validating a fixed operation."""
    columns = params.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(c, str) or not c for c in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError(f"pre_split_steps {kind} requires explicit nonempty columns.")
    return tuple(columns)


def _casting_columns(params: dict[str, Any]) -> tuple[str, ...]:
    """Validate fixed casts while preserving configured column order."""
    columns = params.get("columns")
    type_map = params.get("column_types", {})
    if not isinstance(type_map, dict) or any(not isinstance(k, str) for k in type_map):
        raise ValueError("pre_split_steps Casting requires named column_types.")
    if columns is None:
        columns = []
    if not isinstance(columns, list) or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError("pre_split_steps Casting requires explicit columns.")
    if columns and "target_type" not in params:
        raise ValueError("pre_split_steps Casting columns require target_type.")
    if "target_type" in params and not columns:
        raise ValueError("pre_split_steps Casting target_type requires columns.")
    selected = dict(type_map)
    selected.update(dict.fromkeys(columns, params.get("target_type")))
    if not selected or any(
        not isinstance(dtype, str)
        or TYPE_ALIASES.get(dtype.lower(), dtype) not in set(TYPE_ALIASES.values())
        or TYPE_ALIASES.get(dtype.lower(), dtype) == "category"
        for dtype in selected.values()
    ):
        raise ValueError("pre_split_steps Casting needs fixed noncategorical dtypes.")
    if type(params.get("coerce_on_error", True)) is not bool:
        raise ValueError("pre_split_steps Casting coerce_on_error must be boolean.")
    return tuple(dict.fromkeys(selected))


def _validate_replacement_mapping(params: dict[str, Any], columns: tuple[str, ...]) -> None:
    """Keep flat or per-column mappings lossless through saved JSON."""
    mapping = params.get("mapping")
    if mapping is not None:
        maps = mapping.values() if isinstance(mapping, dict) else ()
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(key, str) for key in mapping)
            or any(not isinstance(item, dict) and not _json_value(item) for item in maps)
        ):
            raise ValueError(
                "pre_split_steps mapping needs string keys; use replacements pairs for numeric keys."
            )
        if "value" in params or "to_replace" in params or "replacements" in params:
            raise ValueError(
                "pre_split_steps mapping cannot be combined with another replacement mode."
            )
        nested = [isinstance(item, dict) for item in maps]
        if any(nested) and (not all(nested) or any(column not in columns for column in mapping)):
            raise ValueError("pre_split_steps nested mapping must name only selected columns.")
        for item in maps:
            if isinstance(item, dict) and (
                any(not isinstance(key, str) for key in item) or not _json_value(item)
            ):
                raise ValueError(
                    "pre_split_steps mapping needs string keys; use replacements pairs for numeric keys."
                )
        for item in [mapping] if not any(nested) else maps:
            identities = [_pair_key(key) for key in item]
            if len(set(identities)) != len(identities):
                raise ValueError("pre_split_steps mapping keys collide after numeric coercion.")


def _validate_replacement_pairs(params: dict[str, Any]) -> None:
    """Reject malformed or colliding numeric replacement pairs."""
    pairs = params.get("replacements")
    if pairs is not None and (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(pair, dict) or set(pair) != {"old", "new"} or not _json_value(pair)
            for pair in pairs
        )
    ):
        raise ValueError("pre_split_steps replacements requires lossless old/new pairs.")
    if pairs is not None and ("value" in params or "to_replace" in params or "mapping" in params):
        raise ValueError("pre_split_steps replacements cannot be combined with another mode.")
    if pairs is not None:
        identities = [_pair_key(pair["old"]) for pair in pairs]
        if len(set(identities)) != len(identities):
            raise ValueError("pre_split_steps replacements pairs have colliding old values.")


def _value_replacement_columns(params: dict[str, Any]) -> tuple[str, ...]:
    """Validate one replacement mode on explicit write columns."""
    columns = _explicit_columns(params, "ValueReplacement")
    modes = sum(bool(params.get(key)) for key in ("mapping", "replacements")) + (
        "to_replace" in params
    )
    if modes != 1:
        raise ValueError("pre_split_steps ValueReplacement needs one replacement mode.")
    _validate_replacement_mapping(params, columns)
    _validate_replacement_pairs(params)
    if "to_replace" in params and (
        params["to_replace"] is None
        or not isinstance(params["to_replace"], (str, int, float, bool))
        or "value" not in params
    ):
        raise ValueError("pre_split_steps to_replace needs a scalar old value and explicit value.")
    return columns


def _validate_text_operation(op: dict[str, Any]) -> None:
    """Validate one supported text operation and its mode-specific keys."""
    modes = {
        "trim": {"both", "leading", "trailing"},
        "case": {"lower", "upper", "title", "sentence"},
        "remove_special": {
            "keep_alphanumeric",
            "keep_alphanumeric_space",
            "letters_only",
            "digits_only",
        },
        "regex": {"custom", "collapse_whitespace", "extract_digits", "normalize_slash_dates"},
    }
    defaults = {
        "trim": "both",
        "case": "lower",
        "remove_special": "keep_alphanumeric",
        "regex": "custom",
    }
    name = op["op"]
    allowed_keys = {"op", "mode"}
    if name == "remove_special":
        allowed_keys.add("replacement")
    if name == "regex":
        allowed_keys.update({"pattern", "repl"})
    if set(op) - allowed_keys or op.get("mode", defaults[name]) not in modes[name]:
        raise ValueError("pre_split_steps TextCleaning has unsupported operation parameters.")
    if name == "regex" and op.get("mode", "custom") == "custom" and not op.get("pattern"):
        raise ValueError("pre_split_steps custom regex requires a pattern.")


def _text_cleaning_columns(params: dict[str, Any]) -> tuple[str, ...]:
    """Validate explicit text cleanup without discovering source columns."""
    columns = _explicit_columns(params, "TextCleaning")
    operations = params.get("operations")
    if (
        not isinstance(operations, list)
        or not operations
        or any(
            not isinstance(op, dict)
            or op.get("op") not in {"trim", "case", "remove_special", "regex"}
            or not _json_value(op)
            for op in operations
        )
    ):
        raise ValueError("pre_split_steps TextCleaning requires supported operations.")
    for op in operations:
        _validate_text_operation(op)
    return columns


def _alias_replacement_columns(params: dict[str, Any]) -> tuple[str, ...]:
    """Validate a supported alias preset or lossless custom map."""
    columns = _explicit_columns(params, "AliasReplacement")
    alias_type = params.get("alias_type", "boolean")
    if alias_type not in {"boolean", "country", "custom", "punctuation"}:
        raise ValueError("pre_split_steps AliasReplacement requires a supported alias_type.")
    custom_map = params.get("custom_map", {})
    if (
        not isinstance(custom_map, dict)
        or any(not isinstance(k, str) for k in custom_map)
        or not _json_value(custom_map)
    ):
        raise ValueError("pre_split_steps AliasReplacement requires a JSON-safe custom_map.")
    if alias_type == "custom" and not custom_map:
        raise ValueError("pre_split_steps custom alias mode requires custom_map.")
    if alias_type != "custom" and custom_map:
        raise ValueError("pre_split_steps custom_map requires custom alias mode.")
    return columns


def _validate_range_bounds(params: dict[str, Any]) -> None:
    """Keep configured numeric bounds finite and ordered."""
    for bound in ("min_value", "max_value"):
        value = params.get(bound)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("pre_split_steps range bounds must be finite numbers.")
    if (
        params.get("min_value") is not None
        and params.get("max_value") is not None
        and params["min_value"] > params["max_value"]
    ):
        raise ValueError("pre_split_steps minimum range bound exceeds maximum.")


def _invalid_value_columns(params: dict[str, Any]) -> tuple[str, ...]:
    """Validate fixed numeric cleanup independently from learned outlier rules."""
    columns = _explicit_columns(params, "InvalidValueReplacement")
    rule = params.get("rule")
    if rule not in {None, "negative", "zero", "custom_range"} or not (
        rule or params.get("replace_inf") or params.get("replace_neg_inf")
    ):
        raise ValueError("pre_split_steps InvalidValueReplacement needs a fixed numeric rule.")
    if any(
        type(params.get(flag, False)) is not bool for flag in ("replace_inf", "replace_neg_inf")
    ):
        raise ValueError("pre_split_steps InvalidValueReplacement flags must be boolean.")
    if rule == "custom_range" and all(
        params.get(bound) is None for bound in ("min_value", "max_value")
    ):
        raise ValueError("pre_split_steps custom_range needs a bound.")
    if rule != "custom_range" and any(
        params.get(bound) is not None for bound in ("min_value", "max_value")
    ):
        raise ValueError("pre_split_steps range bounds require custom_range.")
    if "replacement" in params and "value" in params:
        raise ValueError("pre_split_steps choose replacement or value, not both.")
    _validate_range_bounds(params)
    return columns


# This is the Bundle admission contract, not the complete Core registry.
# Each entry owns its allowed parameters and column validator. Core metadata
# alone does not describe write columns, row effects or saved replay semantics.
_FIXED_RULES: dict[str, tuple[set[str], Callable[[dict[str, Any]], tuple[str, ...]]]] = {
    "ValueReplacement": (
        {"columns", "mapping", "replacements", "to_replace", "value"},
        _value_replacement_columns,
    ),
    "TextCleaning": ({"columns", "operations"}, _text_cleaning_columns),
    "AliasReplacement": ({"columns", "alias_type", "custom_map"}, _alias_replacement_columns),
    "InvalidValueReplacement": (
        {
            "columns",
            "rule",
            "replace_inf",
            "replace_neg_inf",
            "replacement",
            "value",
            "min_value",
            "max_value",
        },
        _invalid_value_columns,
    ),
    "Casting": ({"columns", "target_type", "column_types", "coerce_on_error"}, _casting_columns),
}
FIXED_TYPES = frozenset(_FIXED_RULES)


def fixed_columns(step: dict[str, Any]) -> tuple[str, ...]:
    """Return validated write columns using the node's fixed replay contract."""
    kind, params = step["transformer"], step["params"]
    if kind not in _FIXED_RULES:
        raise ValueError(f"Unsupported fixed pre_split_steps type {kind}.")
    allowed, validate = _FIXED_RULES[kind]
    if set(params) - allowed:
        raise ValueError(
            f"pre_split_steps {kind} has unsupported parameters: {sorted(set(params) - allowed)}."
        )
    columns = validate(params)
    # Casting validates a named type map directly, preserving its existing
    # contract; replacement/text/alias payloads also need lossless JSON values.
    if kind != "Casting" and not _json_value(params):
        raise ValueError(
            f"pre_split_steps {kind} parameters must round trip through JSON losslessly."
        )
    return columns


def projected_fixed_steps(
    steps: tuple[dict[str, Any], ...], selected_columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Keep ordered fixed edits only for columns present in a model or target role."""
    selected = set(selected_columns)
    projected = []
    for step in steps:
        if step["transformer"] not in FIXED_TYPES:
            continue
        keep = [column for column in fixed_columns(step) if column in selected]
        if not keep:
            continue
        clone = deepcopy(step)
        params = clone["params"]
        if clone["transformer"] == "Casting":
            params["column_types"] = {
                column: dtype
                for column, dtype in params.get("column_types", {}).items()
                if column in selected
            }
            if "columns" in params:
                params["columns"] = [column for column in params["columns"] if column in selected]
            if not params.get("columns"):
                params.pop("columns", None)
                params.pop("target_type", None)
        else:
            params["columns"] = keep
            if clone["transformer"] == "ValueReplacement" and isinstance(
                params.get("mapping"), dict
            ):
                mapping = params["mapping"]
                if any(isinstance(value, dict) for value in mapping.values()):
                    params["mapping"] = {
                        column: value for column, value in mapping.items() if column in selected
                    }
                    if not any(params["mapping"].values()):
                        continue
        projected.append(clone)
    return projected


def target_contract(steps: tuple[dict[str, Any], ...], target_column: str) -> list[dict[str, Any]]:
    """Describe ordered target edits without step labels or feature-only parameters."""
    projected = projected_fixed_steps(steps, (target_column,))
    return [{"transformer": step["transformer"], "params": step["params"]} for step in projected]
