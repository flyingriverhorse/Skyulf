"""Versioned, bounded JSON state for the initial portable preprocessing nodes.

This codec accepts learned scalar state, never estimators, sessions or arbitrary
Python objects. It performs no filesystem access and imports no optional engine.
"""

import hashlib
import json
import math
from collections.abc import Iterator
from typing import Any

import numpy as np

DEFAULT_MAX_STATE_BYTES = 8 * 1024 * 1024
_NODE_TYPES = {"StandardScaler": "standard_scaler", "SimpleImputer": "simple_imputer"}
_ENVELOPE_FIELDS = {
    "format_version",
    "codec_version",
    "node_type",
    "ordered_columns",
    "learned_parameters",
    "semantic_digest",
}
_SCALER_FIELDS = {"type", "columns", "mean", "var", "scale", "with_mean", "with_std"}
_IMPUTER_FIELDS = {"type", "columns", "strategy", "fill_values", "missing_counts", "total_missing"}


def encode_state(node_type: str, params: Any, *, max_bytes: int) -> bytes:
    """Encode supported fitted parameters with an exact UTF-8 envelope byte limit.

    NumPy scalar values normalize to their lossless Python equivalents. Dict
    insertion order is immaterial; column/list order and scalar types matter.
    Empty dictionaries remain no-op artifacts. Other objects are rejected.
    """
    _validate_limit(max_bytes)
    normalized = validate_state(node_type, params)
    document = _envelope(node_type, normalized)
    return _json_bytes(document, max_bytes)


def validate_state(node_type: str, params: Any) -> dict[str, Any]:
    """Validate and copy learned scalars without serializing bytes or imposing a wire limit."""
    _validate_node(node_type)
    normalized = _normalize(params)
    _validate_state(node_type, normalized)
    return normalized


def decode_state(
    payload: bytes, *, max_bytes: int = DEFAULT_MAX_STATE_BYTES
) -> tuple[str, dict[str, Any]]:
    """Reject oversized, unknown or corrupt state before returning parameters.

    Limits are checked before JSON parsing. The semantic checksum detects
    corruption; it is not a signature or proof of a trusted producer.
    """
    _validate_limit(max_bytes)
    if type(payload) is not bytes:
        raise TypeError("Portable state payload must be bytes.")
    if len(payload) > max_bytes:
        raise ValueError("Portable state exceeds max_bytes.")
    try:
        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant
        )
        return _decode_envelope(document)
    except (RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError("Invalid portable state encoding or nesting depth.") from exc


def _decode_envelope(document: Any) -> tuple[str, dict[str, Any]]:
    """Validate a parsed node envelope inside an already byte-bounded document."""
    _validate_envelope(document)
    node_type = document["node_type"]
    params = _unpack(document["learned_parameters"])
    _validate_state(node_type, params)
    if document["ordered_columns"] != params.get("columns", []):
        raise ValueError("ordered_columns must match the learned columns in order.")
    expected = _envelope(node_type, params)["semantic_digest"]
    if document["semantic_digest"] != expected:
        raise ValueError("Portable state semantic_digest mismatch.")
    return node_type, params


def _validate_limit(value: int) -> None:
    """Require a positive byte budget without treating booleans as integers."""
    if type(value) is not int or value <= 0:
        raise ValueError("max_bytes must be a positive integer.")


def _validate_node(node_type: str) -> None:
    """Keep the format restricted to explicitly implemented node codecs."""
    if type(node_type) is not str or node_type not in _NODE_TYPES:
        raise ValueError("Unsupported portable state node_type.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON fields rather than choosing the last interpretation."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON field in portable state.")
        result[name] = value
    return result


def _bad_constant(value: str) -> Any:
    """Require explicit tags for non-finite floats instead of JSON extensions."""
    raise ValueError("Non-finite values require portable float tags.")


def _normalize(value: Any, depth: int = 0) -> Any:
    """Copy the small scalar/container vocabulary, rejecting cycles and deep nesting."""
    if depth > 8:
        raise ValueError("Portable state exceeds supported nesting depth.")
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if type(value) is list:
        return [_normalize(item, depth + 1) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Portable state dictionary keys must be strings.")
        return {key: _normalize(item, depth + 1) for key, item in value.items()}
    raise TypeError(
        "Unsupported portable state value; only scalars, lists and dictionaries are allowed."
    )


def _validate_state(node_type: str, params: Any) -> None:
    """Validate the exact node artifact schema without inferring missing semantics."""
    if type(params) is not dict:
        raise ValueError("Learned parameters must be a dictionary.")
    if not params:
        return
    fields = _SCALER_FIELDS if node_type == "StandardScaler" else _IMPUTER_FIELDS
    if set(params) != fields or params.get("type") != _NODE_TYPES[node_type]:
        raise ValueError("Portable state fields/type do not match the node codec.")
    _validate_columns(params["columns"])
    if node_type == "StandardScaler":
        _validate_scaler(params)
    else:
        _validate_imputer(params)


def _validate_columns(columns: Any) -> None:
    """Retain ordered exact string names while rejecting ambiguous duplicates."""
    if type(columns) is not list or any(type(name) is not str for name in columns):
        raise ValueError("columns must be a list of strings.")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate columns in portable state.")


def _validate_scaler(params: dict[str, Any]) -> None:
    """Pin scaler flags and aligned per-column numeric arrays, including disabled stats."""
    for name in ("with_mean", "with_std"):
        if type(params[name]) is not bool:
            raise ValueError("Scaler flags must be booleans.")
    for name in ("mean", "var", "scale"):
        values = params[name]
        if values is not None:
            _validate_statistics(values, len(params["columns"]))
    if params["with_mean"] and params["mean"] is None:
        raise ValueError("with_mean requires mean state.")
    if params["with_std"] and (params["scale"] is None or params["var"] is None):
        raise ValueError("with_std requires scale and var state.")


def _validate_statistics(values: Any, size: int) -> None:
    """Reject shortened arrays and scalar types that would alter numeric semantics."""
    if type(values) is not list or len(values) != size:
        raise ValueError("Statistic arrays must align with ordered columns.")
    if any(type(value) not in (int, float) for value in values):
        raise ValueError("Scaler statistics must contain numeric scalars.")


def _validate_imputer(params: dict[str, Any]) -> None:
    """Validate supported strategies, scalar replacements and consistent missing counts."""
    if params["strategy"] not in ("mean", "constant"):
        raise ValueError("Portable SimpleImputer currently supports mean and constant only.")
    columns = set(params["columns"])
    for field in ("fill_values", "missing_counts"):
        if type(params[field]) is not dict or set(params[field]) != columns:
            raise ValueError(f"{field} must match the learned columns.")
    if any(
        type(value) not in (int, float, str, bool, type(None))
        for value in params["fill_values"].values()
    ):
        raise ValueError("Imputer fill values must be scalars.")
    counts = [*params["missing_counts"].values(), params["total_missing"]]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("Missing counts must be nonnegative integers.")
    if sum(params["missing_counts"].values()) != params["total_missing"]:
        raise ValueError("total_missing must equal the per-column missing counts.")


def _pack(value: Any) -> dict[str, Any]:
    """Represent each scalar type explicitly and sort unordered dictionary entries."""
    if value is None:
        return {"kind": "null"}
    if type(value) is dict:
        return {"kind": "dict", "items": [[key, _pack(value[key])] for key in sorted(value)]}
    if type(value) is list:
        return {"kind": "list", "items": [_pack(item) for item in value]}
    if type(value) is float:
        return {"kind": "float", "value": _float_token(value)}
    tags = {str: "string", int: "int", bool: "bool"}
    return {"kind": tags[type(value)], "value": str(value) if type(value) is int else value}


def _float_token(value: float) -> str:
    """Preserve finite IEEE values and signed zero; canonicalize non-finite semantics."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value).hex()


def _unpack(value: Any, depth: int = 0) -> Any:
    """Decode a closed tagged vocabulary without executing or importing user objects."""
    if depth > 8 or type(value) is not dict:
        raise ValueError("Invalid tagged state or excessive nesting.")
    kind = value.get("kind")
    if kind == "null" and set(value) == {"kind"}:
        return None
    if kind in ("dict", "list") and set(value) == {"kind", "items"}:
        return _unpack_container(kind, value["items"], depth)
    if set(value) != {"kind", "value"}:
        raise ValueError("Invalid tagged scalar fields.")
    return _unpack_scalar(kind, value["value"])


def _unpack_container(kind: str, items: Any, depth: int) -> Any:
    """Validate list/map shape and reject duplicate learned-state dictionary keys."""
    if type(items) is not list:
        raise ValueError("Tagged items must be a list.")
    if kind == "list":
        return [_unpack(item, depth + 1) for item in items]
    pairs = []
    for pair in items:
        if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str:
            raise ValueError("Invalid tagged dictionary entry.")
        pairs.append((pair[0], _unpack(pair[1], depth + 1)))
    return _unique_object(pairs)


def _unpack_scalar(kind: Any, value: Any) -> Any:
    """Require canonical scalar encodings, keeping integers separate from strings/bools."""
    if (kind == "string" and type(value) is str) or (kind == "bool" and type(value) is bool):
        return value
    if kind == "int" and type(value) is str:
        result = int(value)
        if str(result) == value:
            return result
    if kind == "float" and type(value) is str:
        result = {"nan": math.nan, "+inf": math.inf, "-inf": -math.inf}.get(value)
        if result is None:
            result = float.fromhex(value)
        if _float_token(result) == value:
            return result
    raise ValueError("Unknown or malformed portable scalar tag.")


def _validate_envelope(document: Any) -> None:
    """Reject unknown versions/nodes before interpreting tagged learned parameters."""
    if type(document) is not dict:
        raise ValueError("Portable state envelope must be a JSON object.")
    for name in ("format_version", "codec_version"):
        if type(document.get(name)) is not int or document[name] != 1:
            raise ValueError(f"Unsupported {name} in portable state.")
    if set(document) != _ENVELOPE_FIELDS:
        raise ValueError("Invalid portable state envelope fields.")
    _validate_node(document["node_type"])
    _validate_columns(document["ordered_columns"])
    if type(document["semantic_digest"]) is not str:
        raise ValueError("semantic_digest must be a string.")


def _envelope(node_type: str, params: dict[str, Any]) -> dict[str, Any]:
    """Hash only canonical versioned semantics, independently of Python object identity."""
    document = {
        "format_version": 1,
        "codec_version": 1,
        "node_type": node_type,
        "ordered_columns": params.get("columns", []),
        "learned_parameters": _pack(params),
    }
    digest = hashlib.sha256()
    for chunk in _json_chunks(document):
        digest.update(chunk)
    document["semantic_digest"] = digest.hexdigest()
    return document


def _json_bytes(document: dict, max_bytes: int) -> bytes:
    """Stop JSON output as soon as the complete wire representation exceeds its budget."""
    output = bytearray()
    for encoded in _json_chunks(document, ensure_ascii=False):
        if len(output) + len(encoded) > max_bytes:
            raise ValueError("Portable state exceeds max_bytes.")
        output.extend(encoded)
    return bytes(output)


def _json_chunks(document: dict, *, ensure_ascii: bool = True) -> Iterator[bytes]:
    """Stream canonical UTF-8 chunks for serialization and semantic hashing."""
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=ensure_ascii
    )
    for chunk in encoder.iterencode(document):
        # Lone surrogates retain their JSON escape spelling, as in the original codec.
        yield chunk.encode("utf-8", errors="backslashreplace")
