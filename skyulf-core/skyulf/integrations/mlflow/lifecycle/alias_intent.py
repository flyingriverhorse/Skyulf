"""Bounded registry storage for the latest recoverable alias transaction on a version."""

import hashlib
import json
from typing import Any

_MANIFEST = "alias_intent"
_PREFIX = "alias_intent_"
_CHUNK_BYTES = 256
_MAX_CHUNKS = 48


def _chunk_key(index: int) -> str:
    """Address a reusable slot without accumulating tags for each completed event."""
    return f"{_PREFIX}{index:04d}"


def read_alias_intent(tags: dict[str, str]) -> dict[str, Any]:
    """Verify the complete bounded snapshot before exposing any recovery instructions."""
    manifest = json.loads(tags[_MANIFEST])
    count = manifest["chunks"]
    if type(count) is not int or not 1 <= count <= _MAX_CHUNKS:
        raise ValueError("Alias intent chunk count is invalid.")
    chunks = [tags[_chunk_key(index)] for index in range(count)]
    if any(len(chunk.encode("ascii")) > _CHUNK_BYTES for chunk in chunks):
        raise ValueError("Alias intent chunk exceeds the tag limit.")
    raw = "".join(chunks)
    if hashlib.sha256(raw.encode("ascii")).hexdigest() != manifest["sha256"]:
        raise ValueError("Alias intent digest differs from its stored chunks.")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("format") != 1:
        raise ValueError("Unsupported alias intent format.")
    return payload


def _encoded_intent(payload: dict[str, Any]) -> dict[str, str]:
    """Split bounded ASCII JSON so every tag stays within the UC byte limit."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    chunks = [raw[index : index + _CHUNK_BYTES] for index in range(0, len(raw), _CHUNK_BYTES)]
    if len(chunks) > _MAX_CHUNKS:
        raise ValueError("Alias recovery intent exceeds its bounded registry capacity.")
    values = {_chunk_key(index): chunk for index, chunk in enumerate(chunks)}
    values[_MANIFEST] = json.dumps(
        {"chunks": len(chunks), "sha256": hashlib.sha256(raw.encode("ascii")).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return values


def write_alias_intent(
    client: Any, name: str, version: str, payload: dict[str, Any], final_keys: set[str]
) -> None:
    """Preflight UC capacity and verify intent before the caller publishes pending.

    Only the latest intent is retained per version. The caller must hold shared
    admission and verify that no pending event exists before replacing it.
    Existing compact lifecycle receipts and unrelated tags are never removed.
    """
    values = _encoded_intent(payload)
    existing = client.get_model_version(name, version).tags or {}
    reserved = {_chunk_key(index) for index in range(_MAX_CHUNKS)} | {_MANIFEST}
    if len((set(existing) - reserved) | set(values) | final_keys) > 50:
        raise ValueError("Alias transaction would exceed the 50-tag model-version limit.")
    for key in (set(existing) & reserved) - set(values):
        client.delete_model_version_tag(name, version, key)
    for key, value in values.items():
        client.set_model_version_tag(name, version, key, value)
    stored = read_alias_intent(client.get_model_version(name, version).tags or {})
    if stored != payload:
        raise ValueError("Alias recovery intent was not durably verified.")
