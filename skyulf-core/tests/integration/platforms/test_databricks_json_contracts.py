"""Saved JSON identities retain their original byte encoding across shared callers."""

import hashlib
from importlib import import_module

import pytest


@pytest.fixture(
    params=[
        ("shared.json_contracts", "finite_json_digest"),
        ("observability.monitoring.monitoring_config", "json_digest"),
        ("observability.monitoring.performance.performance_policy", "_digest"),
        ("training.shared.local_training_evidence", "evidence_digest"),
    ]
)
def digest(request):
    """Exercise the shared implementation and each existing evidence entry point."""
    module, name = request.param

    def digest_value(value):
        """Resolve the selected entry point when its contract is exercised."""
        function = getattr(import_module(f"skyulf.integrations.databricks.{module}"), name)
        return function(value)

    return digest_value


def test_digest_preserves_sorted_compact_ascii_encoding(digest):
    """Existing Unicode receipts must keep their persisted hashes after extraction."""
    value = {"z": [None, True, -0.0], "a": {"city": "Vilnius \u017e", "count": 3}}
    encoded = b'{"a":{"city":"Vilnius \\u017e","count":3},"z":[null,true,-0.0]}'
    expected = hashlib.sha256(encoded).hexdigest()
    assert digest(value) == digest(dict(reversed(list(value.items())))) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_digest_rejects_nested_nonfinite_numbers(digest, value):
    """Invalid evidence must fail before nonstandard JSON can acquire an identity."""
    with pytest.raises(ValueError, match="Out of range float values"):
        digest({"nested": [value]})


def test_digest_rejects_unsupported_values(digest):
    """No fallback stringification may silently turn arbitrary objects into evidence."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        digest({"unsupported": object()})
