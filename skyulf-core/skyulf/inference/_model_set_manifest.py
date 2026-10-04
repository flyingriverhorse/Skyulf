"""Immutable, canonical metadata for self-contained sets of fitted models."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._manifest import ColumnSpec, checksum

_SHA = r"^[0-9a-f]{64}$"
_KEY_DTYPES = {"string", "bool", "int64"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10))}


def canonical_json(value: object) -> str:
    """Encode portable finite JSON with deterministic key ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_branch(branch: str) -> None:
    """Require a portable directory name and an unambiguous output prefix."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", branch) or "__" in branch:
        raise ValueError("Unsafe model set branch name.")
    if branch.upper() in _RESERVED:
        raise ValueError("Reserved model set branch name.")


class ComponentReference(BaseModel):
    """Pin a concrete registry model identity without loading a registry client."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^[1-9][0-9]*$")
    digest: str = Field(pattern=_SHA)

    @model_validator(mode="after")
    def validate_name(self) -> "ComponentReference":
        """Reject paths, control characters and empty qualified model name parts."""
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in self.name.split(".")):
            raise ValueError("Unsafe component model name.")
        return self


class ComponentFile(BaseModel):
    """Bind every copied local artifact file to its exact bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: Literal["manifest.json", "pipeline.pkl", "preprocessing.py"]
    sha256: str = Field(pattern=_SHA)


class ComponentManifest(BaseModel):
    """Describe one branch using schemas derived from its fitted artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    branch: str
    reference: ComponentReference
    input_schema: tuple[ColumnSpec, ...]
    output_schema: tuple[ColumnSpec, ...]
    files: tuple[ComponentFile, ...]


class ModelSetManifest(BaseModel):
    """Freeze component identities, key schemas and captured composition rules."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    format_version: Literal[1] = 1
    components: tuple[ComponentManifest, ...] = Field(min_length=1)
    record_key_schema: tuple[ColumnSpec, ...] = Field(min_length=1)
    input_schema: tuple[ColumnSpec, ...]
    output_schema: tuple[ColumnSpec, ...]
    composition_source_sha256: str = Field(pattern=_SHA)
    composition_config_json: str
    quality_evidence_json: str | None = None
    set_sha256: str = Field(pattern=_SHA)

    @property
    def quality_evidence(self) -> dict | None:
        """Detach saved comparison pins and the expected set baseline from callers."""
        return json.loads(self.quality_evidence_json) if self.quality_evidence_json else None

    @property
    def composition_config(self) -> dict:
        """Return a detached configuration so nested mutations cannot alter identity."""
        return json.loads(self.composition_config_json)

    @property
    def record_key_columns(self) -> tuple[str, ...]:
        """Expose the ordered join keys without duplicating their schema contract."""
        return tuple(column.name for column in self.record_key_schema)


def manifest_digest(manifest: ModelSetManifest) -> str:
    """Hash the complete canonical contract, excluding the digest field itself."""
    payload = manifest.model_dump(exclude={"set_sha256"})
    if manifest.quality_evidence_json is None:
        payload.pop("quality_evidence_json")
    return checksum(canonical_json(payload).encode())


def quality_evidence_json(value: dict | None, branches: set[str]) -> str | None:
    """Bind each branch's comparison digest and preserve legacy set identities."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"expected_champion_version", "comparisons"}:
        raise ValueError("Model set quality evidence requires baseline and comparison pins.")
    baseline = value["expected_champion_version"]
    if baseline is not None and (
        not isinstance(baseline, str) or not re.fullmatch(r"[1-9][0-9]*", baseline)
    ):
        raise ValueError("Model set quality baseline must be a concrete version or None.")
    _validate_quality_pins(value["comparisons"], branches)
    return canonical_json(value)


def _validate_quality_pins(pins: object, branches: set[str]) -> None:
    """Require one concrete comparison digest for every component."""
    if not isinstance(pins, dict) or set(pins) != branches:
        raise ValueError("Model set quality evidence must identify every component.")
    if any(not isinstance(pin, str) or not re.fullmatch(_SHA, pin) for pin in pins.values()):
        raise ValueError("Model set quality comparisons require SHA256 digests.")


def validate_keys(keys: tuple[ColumnSpec, ...]) -> None:
    """Require typed, unique, nonempty join keys before any source data is read."""
    if not keys or len({key.name.casefold() for key in keys}) != len(keys):
        raise ValueError("Record key schema must contain unique keys.")
    if any(key.dtype not in _KEY_DTYPES or not key.name.strip() for key in keys):
        raise ValueError("Record keys require named string, int64 or bool columns.")


def canonical_dtype(dtype: str) -> str:
    """Normalize equivalent pandas and Polars scalar dtype names for union checks."""
    lower = dtype.lower()
    return {"utf8": "string", "str": "string", "object": "string", "boolean": "bool"}.get(
        lower, lower
    )


def combined_schemas(
    components: tuple[ComponentManifest, ...], keys: tuple[ColumnSpec, ...]
) -> tuple[tuple[ColumnSpec, ...], tuple[ColumnSpec, ...]]:
    """Build the common source schema and reject conflicting or colliding columns."""
    inputs = {column.name: column for column in keys}
    outputs = list(keys)
    for component in components:
        for column in component.input_schema:
            previous = inputs.get(column.name)
            if previous and canonical_dtype(previous.dtype) != canonical_dtype(column.dtype):
                raise ValueError(f"Model set input dtype conflict for {column.name!r}.")
            inputs.setdefault(column.name, column)
        outputs.extend(
            ColumnSpec(name=f"{component.branch}__{column.name}", dtype=column.dtype)
            for column in component.output_schema
        )
    for schema in (tuple(inputs.values()), tuple(outputs)):
        if len({column.name.casefold() for column in schema}) != len(schema):
            raise ValueError("Model set schema column collision.")
    _validate_key_inputs(components, keys)
    return tuple(inputs.values()), tuple(outputs)


def _validate_key_inputs(
    components: tuple[ComponentManifest, ...], keys: tuple[ColumnSpec, ...]
) -> None:
    """Keep record keys separate from the component features they accompany."""
    key_names = {key.name.casefold() for key in keys}
    for component in components:
        if key_names.intersection(column.name.casefold() for column in component.input_schema):
            raise ValueError("Record keys cannot also be component input columns.")


def validate_components(components: tuple[ComponentManifest, ...]) -> None:
    """Reject ambiguous branch paths and repeated concrete model identities."""
    branches = set()
    identities = set()
    for component in components:
        validate_branch(component.branch)
        identity = (component.reference.name.casefold(), component.reference.version)
        if component.branch.casefold() in branches or identity in identities:
            raise ValueError("Duplicate branch or model identity in model set.")
        branches.add(component.branch.casefold())
        identities.add(identity)
