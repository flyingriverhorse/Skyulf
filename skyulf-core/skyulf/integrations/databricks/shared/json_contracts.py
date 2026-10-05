"""Stable finite JSON identities shared by Databricks evidence contracts."""

import hashlib
import json
from typing import Any


def finite_json_digest(value: Any) -> str:
    """Hash sorted compact ASCII-escaped JSON while rejecting nonfinite numbers."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
