"""Bounded ordered FE envelopes composed from the existing scalar node codecs."""

import hashlib
import json
from typing import Any

from . import portable_state as codec

_FIELDS = {"kind", "format_version", "steps", "semantic_digest"}
_STEP_FIELDS = {"name", "node_type", "config", "state"}


def encode_pipeline(records: list[dict], *, max_bytes: int) -> bytes:
    """Encode resolved configurations and learned artifacts under one total wire budget."""
    codec._validate_limit(max_bytes)
    steps = []
    for record in records:
        node = record["type"]
        state = codec.validate_state(node, record["artifact"])
        params = _config(node, record["params"], state)
        if type(record["name"]) is not str:
            raise ValueError("Portable step names must be strings.")
        steps.append(
            {
                "name": record["name"],
                "node_type": node,
                "config": codec._pack(params),
                "state": codec._envelope(node, state),
            }
        )
    document = {"kind": "skyulf.feature_engineer", "format_version": 1, "steps": steps}
    document["semantic_digest"] = _digest(document)
    return codec._json_bytes(document, max_bytes)


def decode_pipeline(payload: bytes, *, max_bytes: int) -> list[dict[str, Any]]:
    """Validate the entire envelope before creating or resolving any runnable appliers."""
    codec._validate_limit(max_bytes)
    if type(payload) is not bytes:
        raise TypeError("Portable pipeline payload must be bytes.")
    if len(payload) > max_bytes:
        raise ValueError("Portable pipeline exceeds max_bytes.")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=codec._unique_object,
            parse_constant=codec._bad_constant,
        )
        _validate_envelope(document)
        content = {key: value for key, value in document.items() if key != "semantic_digest"}
        if document["semantic_digest"] != _digest(content):
            raise ValueError("Portable pipeline semantic_digest mismatch.")
        return [_decode_step(step) for step in document["steps"]]
    except (RecursionError, UnicodeError, OverflowError) as exc:
        raise ValueError("Invalid portable pipeline encoding or nesting depth.") from exc


def _decode_step(step: Any) -> dict[str, Any]:
    """Validate each node payload and its corresponding resolved configuration."""
    if type(step) is not dict or set(step) != _STEP_FIELDS or type(step["name"]) is not str:
        raise ValueError("Invalid portable pipeline step fields.")
    node, artifact = codec._decode_envelope(step["state"])
    if node != step["node_type"]:
        raise ValueError("Pipeline step node_type differs from its state.")
    params = codec._unpack(step["config"])
    resolved = _config(node, params, artifact)
    if codec._pack(resolved) != codec._pack(params):
        raise ValueError("Portable pipeline config must match resolved learned columns/defaults.")
    return {"name": step["name"], "type": node, "params": resolved, "artifact": artifact}


def _config(node: str, raw: Any, state: dict) -> dict[str, Any]:
    """Freeze learned feature selection while removing caller-specific frame context."""
    params = codec._normalize(raw)
    if type(params) is not dict:
        raise ValueError("Portable pipeline config must be a dictionary.")
    params.pop("target_column", None)
    params.pop("_auto_columns", None)
    params["columns"] = state.get("columns", [])
    defaults = (
        {"strategy": "mean", "fill_value": None}
        if node == "SimpleImputer"
        else {"with_mean": True, "with_std": True}
    )
    params = {**defaults, **params}
    keys = ("strategy",) if node == "SimpleImputer" else ("with_mean", "with_std")
    for key in keys:
        if state and (type(params[key]) is not type(state[key]) or params[key] != state[key]):
            raise ValueError("Portable config disagrees with fitted state.")
    if node == "SimpleImputer" and params["strategy"] not in ("mean", "constant"):
        raise ValueError("Unsupported portable SimpleImputer strategy.")
    if node == "StandardScaler" and any(type(params[key]) is not bool for key in keys):
        raise ValueError("Scaler flags must be booleans.")
    return params


def _validate_envelope(document: Any) -> None:
    """Reject unknown formats, versions and top-level fields before interpreting steps."""
    if type(document) is not dict or set(document) != _FIELDS:
        raise ValueError("Invalid portable pipeline envelope fields.")
    if document["kind"] != "skyulf.feature_engineer":
        raise ValueError("Unsupported portable pipeline kind.")
    if type(document["format_version"]) is not int or document["format_version"] != 1:
        raise ValueError("Unsupported portable pipeline format_version.")
    if type(document["steps"]) is not list or type(document["semantic_digest"]) is not str:
        raise ValueError("Invalid portable pipeline steps/digest.")


def _digest(document: dict) -> str:
    """Cover ordered node envelopes and configurations with a canonical content checksum."""
    digest = hashlib.sha256()
    for chunk in codec._json_chunks(document):
        digest.update(chunk)
    return digest.hexdigest()
