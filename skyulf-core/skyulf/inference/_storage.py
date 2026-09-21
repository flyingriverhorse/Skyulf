"""Explicit local bundle I/O, separate from computation and legacy serializers."""

import json
from pathlib import Path

from ..core.execution import ExecutionOptions
from ..core.portable_state import _bad_constant, _unique_object
from ._manifest import BundleManifest


def write_payloads(path: str | Path, metadata: bytes, features: bytes, model: bytes) -> None:
    """Publish three fixed-name files into a new caller-selected directory."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in (
        ("manifest.json", metadata),
        ("features.json", features),
        ("model.pkl", model),
    ):
        with (destination / name).open("xb") as stream:
            stream.write(content)


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    """Read at most one byte beyond the wire budget, rejecting larger payloads."""
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"Bundle {path.name} exceeds {label}.")
    return payload


def read_payloads(
    path: str | Path, options: ExecutionOptions
) -> tuple[BundleManifest, bytes, bytes]:
    """Validate metadata before reading model bytes or invoking the trusted pickle loader."""
    source = Path(path)
    metadata = _read_bounded(source / "manifest.json", options.state_max_bytes, "state_max_bytes")
    document = json.loads(
        metadata.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_bad_constant
    )
    if type(document) is not dict or type(document.get("format_version")) is not int:
        raise ValueError("Invalid bundle manifest format_version.")
    manifest = BundleManifest.model_validate_json(metadata)
    state_limit = options.state_max_bytes - len(metadata)
    if state_limit <= 0:
        raise ValueError("Bundle metadata leaves no state_max_bytes budget for FE.")
    features = _read_bounded(source / "features.json", state_limit, "state_max_bytes")
    model = _read_bounded(source / "model.pkl", options.model_max_bytes, "model_max_bytes")
    return manifest, features, model
