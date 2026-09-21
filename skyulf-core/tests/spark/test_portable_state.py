"""Portable learned-state contracts run without a Spark runtime."""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.portable_state import decode_state, encode_state
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)]
)
def test_scaler_fitted_state_preserves_apply(engine, with_mean, with_std):
    """The decoded artifact must produce exactly the same local transformation."""
    data = pd.DataFrame({"z": [1.0, 3.0, 5.0], "a": [4.0, 4.0, 4.0]})
    data = pl.from_pandas(data) if engine == "polars" else data
    params = StandardScalerCalculator().fit(
        data,
        {
            "columns": ["z", "a"],
            "with_mean": with_mean,
            "with_std": with_std,
        },
    )
    node, restored = decode_state(encode_state("StandardScaler", params, max_bytes=8192))
    assert node == "StandardScaler"
    assert artifact_digest(restored) == artifact_digest(params)
    original = StandardScalerApplier().apply(data, params)
    decoded = StandardScalerApplier().apply(data, restored)
    np.testing.assert_equal(original.to_numpy(), decoded.to_numpy())


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["mean", "constant"])
def test_imputer_fitted_state_preserves_apply(engine, strategy):
    """Encoding must retain fill values, counts and all-null column semantics."""
    data = pd.DataFrame({"x": [1.0, np.nan, 5.0], "empty": [np.nan] * 3})
    data = pl.from_pandas(data) if engine == "polars" else data
    params = SimpleImputerCalculator().fit(data, {"columns": ["x", "empty"], "strategy": strategy})
    node, restored = decode_state(encode_state("SimpleImputer", params, max_bytes=8192))
    assert node == "SimpleImputer"
    assert artifact_digest(restored) == artifact_digest(params)
    np.testing.assert_equal(
        SimpleImputerApplier().apply(data, params).to_numpy(),
        SimpleImputerApplier().apply(data, restored).to_numpy(),
    )


def imputer_state(value):
    """Create a valid one-column constant state without fitting an estimator."""
    return {
        "type": "simple_imputer",
        "columns": ["x"],
        "strategy": "constant",
        "fill_values": {"x": value},
        "missing_counts": {"x": 0},
        "total_missing": 0,
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        2**63 + 1,
        "9007199254740993",
        "λ",
        -0.0,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_scalar_types_and_nonfinite_values(value):
    """Numbers and category text must stay distinct without nonstandard JSON floats."""
    payload = encode_state("SimpleImputer", imputer_state(value), max_bytes=8192)
    document = json.loads(payload)
    assert document["ordered_columns"] == ["x"]
    assert b"NaN" not in payload and b"Infinity" not in payload
    _, restored = decode_state(payload)
    result = restored["fill_values"]["x"]
    assert type(result) is type(value)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(result)
    elif isinstance(value, float):
        assert result.hex() == value.hex()
    else:
        assert result == value


@pytest.mark.parametrize("node", ["SimpleImputer", "StandardScaler"])
def test_empty_artifact_round_trip(node):
    """No-op artifacts must not acquire learned statistics during decoding."""
    assert decode_state(encode_state(node, {}, max_bytes=8192)) == (node, {})


def test_digest_is_order_independent_but_type_sensitive():
    """A semantic digest must reflect value types rather than dict insertion order."""
    state = imputer_state(np.int64(7))
    first = encode_state("SimpleImputer", state, max_bytes=8192)
    assert first == encode_state(
        "SimpleImputer", dict(reversed(list(state.items()))), max_bytes=8192
    )
    numeric = json.loads(first)["semantic_digest"]
    textual = json.loads(encode_state("SimpleImputer", imputer_state("7"), max_bytes=8192))[
        "semantic_digest"
    ]
    assert numeric != textual


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_byte_limits_rejected(limit):
    """Invalid resource limits must not silently disable state-size enforcement."""
    with pytest.raises(ValueError, match="max_bytes"):
        encode_state("SimpleImputer", {}, max_bytes=limit)
    with pytest.raises(ValueError, match="max_bytes"):
        decode_state(b"{}", max_bytes=limit)


def test_payload_limit_is_enforced_in_both_directions():
    """Worker-side loading must reject oversized bytes before attempting JSON parsing."""
    state = imputer_state("λ" * 20)
    payload = encode_state("SimpleImputer", state, max_bytes=8192)
    assert encode_state("SimpleImputer", state, max_bytes=len(payload)) == payload
    with pytest.raises(ValueError, match="max_bytes"):
        encode_state("SimpleImputer", state, max_bytes=len(payload) - 1)
    with pytest.raises(ValueError, match="max_bytes"):
        decode_state(b"invalid-json" * 100, max_bytes=10)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"format_version":999}',
        b'{"format_version":true}',
        b"garbage",
        b"[]",
        b"{}",
        b"\xff",
        b'{"format_version":1,"format_version":1}',
    ],
)
def test_malformed_envelopes_rejected(payload):
    """Unrecognized or ambiguous wire data must fail instead of guessing semantics."""
    with pytest.raises(ValueError):
        decode_state(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec_version", 999),
        ("node_type", "Unknown"),
        ("ordered_columns", ["x", "x"]),
        ("semantic_digest", "0" * 64),
    ],
)
def test_altered_envelope_rejected(field, value):
    """A corrupt envelope must not reach an applier as apparently valid learned state."""
    document = json.loads(encode_state("SimpleImputer", imputer_state(3), max_bytes=8192))
    document[field] = value
    with pytest.raises(ValueError):
        decode_state(json.dumps(document).encode())


@pytest.mark.parametrize("value", [object(), np.array([1]), {"nested": 2}])
def test_non_scalar_fill_state_rejected(value):
    """The codec must never serialize arbitrary Python objects or nested category state."""
    with pytest.raises((TypeError, ValueError)):
        encode_state("SimpleImputer", imputer_state(value), max_bytes=8192)


def test_unknown_node_duplicate_columns_and_incomplete_state_rejected():
    """Only explicitly versioned, structurally valid node artifacts are portable."""
    with pytest.raises(ValueError, match="node"):
        encode_state("KNNImputer", {}, max_bytes=8192)
    state = imputer_state(3)
    state["columns"] = ["x", "x"]
    with pytest.raises(ValueError, match="columns"):
        encode_state("SimpleImputer", state, max_bytes=8192)
    with pytest.raises(ValueError):
        encode_state("StandardScaler", {"columns": ["x"]}, max_bytes=8192)


def test_column_order_changes_digest_and_is_retained():
    """Column ordering must remain meaningful even when dictionary order is not."""
    frame = pd.DataFrame({"z": [1.0, 2.0], "a": [3.0, 4.0]})
    digests = []
    for columns in (["z", "a"], ["a", "z"]):
        state = StandardScalerCalculator().fit(frame, {"columns": columns})
        payload = encode_state("StandardScaler", state, max_bytes=8192)
        assert decode_state(payload)[1]["columns"] == columns
        digests.append(json.loads(payload)["semantic_digest"])
    assert digests[0] != digests[1]


def resigned(document):
    """Recompute the public checksum so malformed-state tests exercise schema validation."""
    document.pop("semantic_digest", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["semantic_digest"] = hashlib.sha256(canonical.encode()).hexdigest()
    return json.dumps(document).encode()


@pytest.mark.parametrize(
    "tag",
    [
        {"kind": "object", "value": "module.Class"},
        {"kind": "int", "value": True},
        {"kind": "float", "value": "Infinity"},
        {"kind": "null", "value": None},
        {"kind": "dict", "items": [["x", {"kind": "null"}], ["x", {"kind": "null"}]]},
    ],
)
def test_malformed_tags_rejected_even_with_recomputed_checksum(tag):
    """A checksum is not authorization to bypass the closed tagged vocabulary."""
    document = json.loads(encode_state("SimpleImputer", imputer_state(1), max_bytes=8192))
    document["learned_parameters"] = tag
    with pytest.raises(ValueError):
        decode_state(resigned(document))


@pytest.mark.parametrize(
    "changes",
    [
        {"total_missing": -1},
        {"total_missing": 5},
        {"missing_counts": {"other": 0}},
        {"strategy": "median"},
        {"extra": "unknown"},
    ],
)
def test_invalid_node_fields_rejected(changes):
    """Node validation must reject inconsistent counts and unsupported semantics."""
    state = imputer_state(1) | changes
    with pytest.raises(ValueError):
        encode_state("SimpleImputer", state, max_bytes=8192)


def test_no_arbitrary_repr_or_pickle_fallback():
    """Unsupported objects must fail without invoking their serialization hooks."""

    class DangerousObject:
        """Expose hooks that a generic serializer might accidentally invoke."""

        def __repr__(self):
            """Make implicit text serialization observable."""
            pytest.fail("repr fallback")

        def __reduce__(self):
            """Make implicit pickle serialization observable."""
            pytest.fail("pickle fallback")

    with pytest.raises(TypeError, match="Unsupported portable"):
        encode_state("SimpleImputer", imputer_state(DangerousObject()), max_bytes=8192)


def test_cycles_and_deeply_nested_payload_rejected():
    """Malformed nesting must fail within the bounded codec contract."""
    cycle = {}
    cycle["cycle"] = cycle
    with pytest.raises(ValueError, match="depth"):
        encode_state("SimpleImputer", cycle, max_bytes=8192)
    with pytest.raises(ValueError):
        decode_state(b"[" * 2000 + b"0" + b"]" * 2000)


def test_plain_json_nan_rejected():
    """Nonstandard JSON constants must not bypass float tagging."""
    with pytest.raises(ValueError, match="float tags"):
        decode_state(b'{"learned_parameters":NaN}')


def test_wire_encoding_requires_utf8():
    """Workers must interpret a single documented byte encoding across runtimes."""
    payload = encode_state("SimpleImputer", {}, max_bytes=8192)
    with pytest.raises(ValueError):
        decode_state(payload.decode("utf-8").encode("utf-16"))


def test_json_layout_does_not_change_semantic_checksum():
    """Whitespace and object member ordering must not become learned semantics."""
    original = encode_state("SimpleImputer", imputer_state(9), max_bytes=8192)
    document = json.loads(original)
    reformatted = json.dumps(dict(reversed(list(document.items()))), indent=2).encode()
    assert decode_state(reformatted) == decode_state(original)


def test_decode_limit_measures_received_unicode_bytes():
    """Canonical checksum escaping must not reject a received envelope within budget."""
    payload = encode_state("SimpleImputer", imputer_state("λ" * 100), max_bytes=8192)
    compact = json.dumps(json.loads(payload), ensure_ascii=False, separators=(",", ":")).encode()
    assert len(compact) < len(payload)
    assert decode_state(compact, max_bytes=len(compact)) == decode_state(payload)
