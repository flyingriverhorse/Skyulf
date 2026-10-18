"""Preserve alias recovery evidence under Unity Catalog's strict tag constraints."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from skyulf.integrations.mlflow.lifecycle.alias_intent import read_alias_intent, write_alias_intent


class StrictUnityCatalogTags:
    """Exercise real storage code with the cloud's byte and boundary-whitespace restrictions."""

    def __init__(self, tags=None):
        """Retain preexisting tags and expose mutations for preflight assertions."""
        self.tags = dict(tags or {})
        self.mutations = []

    def get_model_version(self, name, version):
        """Return a fresh snapshot so verification must observe persisted values."""
        return SimpleNamespace(tags=dict(self.tags))

    def set_model_version_tag(self, name, version, key, value):
        """Match the rejected cloud write rather than accepting arbitrary SQLite text."""
        if value != value.strip():
            raise ValueError("Tag value cannot contain leading or trailing whitespace.")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("Tag value exceeds 256 bytes.")
        self.tags[key] = value
        self.mutations.append((key, value))

    def delete_model_version_tag(self, name, version, key):
        """Remove only explicitly requested stale intent slots."""
        self.tags.pop(key)
        self.mutations.append((key, None))


def _raw(payload):
    """Build canonical legacy JSON independently of the storage implementation."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _legacy_tags(payload):
    """Construct a saved pre-framing intent without calling the current writer."""
    raw = _raw(payload)
    chunks = [raw[index : index + 256] for index in range(0, len(raw), 256)]
    tags = {f"alias_intent_{index:04d}": chunk for index, chunk in enumerate(chunks)}
    tags["alias_intent"] = json.dumps(
        {"chunks": len(chunks), "sha256": hashlib.sha256(raw.encode("ascii")).hexdigest()}
    )
    return tags


@pytest.mark.parametrize("boundary", [253, 254, 255, 256, 511, 512])
def test_whitespace_and_unicode_survive_every_chunk_boundary(boundary):
    """Spaces inside nested text must not cause rejected writes or change recovered evidence."""
    payload = {"format": 1, "text": ""}
    offset = _raw(payload).index('""') + 1
    payload["text"] = "x" * (boundary - offset) + ' "heldout_rmse": "İstanbul 🧪" | \t\n '
    assert _raw(payload)[boundary] == " "
    client = StrictUnityCatalogTags()
    write_alias_intent(client, "model", "1", payload, set())
    manifest = json.loads(client.tags["alias_intent"])
    assert manifest["sha256"] == hashlib.sha256(_raw(payload).encode("ascii")).hexdigest()
    assert read_alias_intent(client.tags) == payload


def test_legacy_intent_still_reads_at_its_original_maximum_capacity():
    """A recovery upgrade must not strand old intents even beyond the new framing capacity."""
    payload = {"format": 1, "text": ""}
    payload["text"] = "x" * (48 * 256 - len(_raw(payload)))
    tags = _legacy_tags(payload)
    assert json.loads(tags["alias_intent"])["chunks"] == 48
    assert read_alias_intent(tags) == payload


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_framed_capacity_is_checked_before_any_registry_mutation(extra_bytes):
    """Framing overhead must stay inside both per-tag and total recovery bounds."""
    payload = {"format": 1, "text": ""}
    payload["text"] = "x" * (48 * 254 - len(_raw(payload)) + extra_bytes)
    client = StrictUnityCatalogTags({"owner": "retained"})
    if extra_bytes:
        with pytest.raises(ValueError, match="bounded registry capacity"):
            write_alias_intent(client, "model", "1", payload, set())
        assert client.tags == {"owner": "retained"} and not client.mutations
    else:
        write_alias_intent(client, "model", "1", payload, set())
        assert len(client.tags) == 50
        assert read_alias_intent(client.tags) == payload


def test_final_transaction_tags_are_reserved_before_writing_framed_intent():
    """Encoding must not consume tag slots needed to finish an admitted alias transaction."""
    client = StrictUnityCatalogTags({f"external_{index}": "retained" for index in range(48)})
    with pytest.raises(ValueError, match="50-tag"):
        write_alias_intent(client, "model", "1", {"format": 1}, {"pending_state"})
    assert len(client.tags) == 48 and not client.mutations


@pytest.mark.parametrize("corruption", ["unknown_encoding", "missing_start", "missing_end", "text"])
def test_invalid_framed_intents_cannot_authorize_recovery(corruption):
    """Unknown formats, broken framing and altered evidence must fail closed before replay."""
    client = StrictUnityCatalogTags()
    write_alias_intent(client, "model", "1", {"format": 1, "text": "evidence"}, set())
    if corruption == "unknown_encoding":
        manifest = json.loads(client.tags["alias_intent"])
        manifest["encoding"] = "unknown"
        client.tags["alias_intent"] = json.dumps(manifest)
    elif corruption == "missing_start":
        client.tags["alias_intent_0000"] = client.tags["alias_intent_0000"][1:]
    elif corruption == "missing_end":
        client.tags["alias_intent_0000"] = client.tags["alias_intent_0000"][:-1]
    else:
        client.tags["alias_intent_0000"] = client.tags["alias_intent_0000"].replace(
            "evidence", "tampered"
        )
    with pytest.raises(ValueError, match="encoding|framing|digest"):
        read_alias_intent(client.tags)
