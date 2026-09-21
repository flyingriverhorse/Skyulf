"""Strict, immutable metadata for the initial standalone inference bundle."""

import hashlib
import json
import math
import platform
from importlib.metadata import version
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.schema import SkyulfSchema
from ..pipeline.seal import artifact_digest

Label = str | int | float | bool
_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
}
_PACKAGES = ("skyulf-core", "scikit-learn", "numpy", "scipy", "pandas", "polars")


class ColumnSpec(BaseModel):
    """Describe one named scalar column without retaining any training rows."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: str = Field(min_length=1)
    dtype: str


class ThresholdProvenance(BaseModel):
    """Keep default tuning decisions distinct from explicit pipeline overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    source: Literal["estimator", "tuning", "pipeline_override"] = "estimator"
    values: tuple[float, ...] = ()
    tuning_values: tuple[float, ...] = ()
    pipeline_values: tuple[float, ...] = ()
    metric: str | None = None


class BundleManifest(BaseModel):
    """Validate the declared input stage, schemas and payload identities."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    format_version: Literal[1] = 1
    input_stage: Literal["raw", "features"]
    feature_order: tuple[str, ...]
    input_schema: tuple[ColumnSpec, ...]
    feature_schema: tuple[ColumnSpec, ...]
    output_schema: tuple[ColumnSpec, ...]
    task: Literal["regression", "classification"]
    classes: tuple[Label, ...] = ()
    positive_label: Label | None = None
    probability_columns: tuple[str, ...] = ()
    thresholds: ThresholdProvenance = ThresholdProvenance()
    requirements: tuple[tuple[str, str], ...]
    model_class: str
    model_sha256: str
    model_state_digest: str
    fe_sha256: str
    fe_semantic_digest: str
    semantic_digest: str = ""

    @model_validator(mode="after")
    def validate_contract(self) -> "BundleManifest":
        """Reject internally inconsistent schemas and decision metadata before model loading."""
        for schema in (self.input_schema, self.feature_schema, self.output_schema):
            names = tuple(col.name for col in schema)
            if not names or len(names) != len(set(names)):
                raise ValueError("Bundle schemas require distinct nonempty column names.")
        if self.feature_order != tuple(col.name for col in self.feature_schema):
            raise ValueError("feature_order must match feature_schema.")
        if self.input_stage == "features" and self.input_schema != self.feature_schema:
            raise ValueError("features input_schema must match feature_schema.")
        if any(col.dtype not in _DTYPES for col in (*self.input_schema, *self.feature_schema)):
            raise ValueError("Initial bundle inputs require primitive numeric or boolean dtypes.")
        expected = ("prediction", *self.probability_columns)
        if tuple(col.name for col in self.output_schema) != expected:
            raise ValueError("Prediction schema and probability columns disagree.")
        if self.task == "regression":
            if self.classes or self.probability_columns or self.positive_label is not None:
                raise ValueError("Regression cannot declare classes or probabilities.")
            if self.output_schema[0].dtype != "float64":
                raise ValueError("Regression prediction dtype must be float64.")
        else:
            if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
                raise ValueError("Classification requires distinct scalar classes.")
            if self.probability_columns != tuple(
                f"probability_{i}" for i in range(len(self.classes))
            ):
                raise ValueError("Probability columns must follow class positions.")
            if self.positive_label != (self.classes[1] if len(self.classes) == 2 else None):
                raise ValueError("positive_label must follow the binary class convention.")
            if self.output_schema[0].dtype != label_dtype(self.classes):
                raise ValueError("Prediction dtype must match classes.")
            if any(col.dtype != "float64" for col in self.output_schema[1:]):
                raise ValueError("Probability columns require float64.")
        self._validate_thresholds()
        names = [name for name, _ in self.requirements]
        if sorted(names) != sorted(("python", *_PACKAGES)):
            raise ValueError("Requirements must contain exactly the supported runtime packages.")
        return self

    def _validate_thresholds(self) -> None:
        """Threshold arrays are class-ordered and must agree with their active source."""
        state = self.thresholds
        for values in (state.values, state.tuning_values, state.pipeline_values):
            if values and (len(values) != len(self.classes) or not self.classes):
                raise ValueError("Threshold values must cover every class.")
            minimum_ok = all(
                value >= 0 if len(self.classes) == 2 else value > 0 for value in values
            )
            if values and (
                not all(math.isfinite(value) for value in values)
                or not minimum_ok
                or not any(values)
            ):
                raise ValueError("Invalid decision thresholds.")
        selected = {
            "estimator": (),
            "tuning": state.tuning_values,
            "pipeline_override": state.pipeline_values,
        }[state.source]
        if state.values != selected or (state.source != "estimator" and not selected):
            raise ValueError("Threshold provenance disagrees with the active decision rule.")


def schema_columns(schema: SkyulfSchema) -> tuple[ColumnSpec, ...]:
    """Normalize equivalent pandas/Polars numeric labels without casting any input data."""
    return tuple(
        ColumnSpec(
            name=name, dtype=schema.dtypes.get(name, "unknown").lower().replace("boolean", "bool")
        )
        for name in schema.columns
    )


def label_dtype(classes: tuple[Label, ...]) -> str:
    """Keep one unambiguous scalar class type suitable for tabular prediction outputs."""
    kind = type(classes[0])
    if any(type(value) is not kind for value in classes):
        raise ValueError("Class labels must share one scalar type.")
    if any(isinstance(value, float) and not math.isfinite(value) for value in classes):
        raise ValueError("Class labels must be finite.")
    if any(isinstance(value, int) and not -(2**63) <= value < 2**63 for value in classes):
        raise ValueError("Integer labels must fit int64.")
    dtypes: dict[type, str] = {str: "string", int: "int64", float: "float64", bool: "bool"}
    return dtypes[kind]


def runtime_requirements() -> tuple[tuple[str, str], ...]:
    """Record package versions only; configuration, credentials and sessions are excluded."""
    return (("python", platform.python_version()), *((name, version(name)) for name in _PACKAGES))


def check_runtime(manifest: BundleManifest) -> None:
    """Require compatible Python and exact estimator dependency versions for pickle loading."""
    current = dict(runtime_requirements())
    for name, expected in manifest.requirements:
        actual = current[name]
        if name == "python":
            compatible = actual.split(".")[:2] == expected.split(".")[:2]
        elif name in {"skyulf-core", "scikit-learn", "numpy", "scipy"}:
            compatible = actual == expected
        else:
            continue
        if not compatible:
            raise ValueError(
                f"Bundle runtime mismatch for {name}: requires {expected}, found {actual}."
            )


def semantic_digest(manifest: BundleManifest) -> str:
    """Hash semantic metadata and fitted model state independently of pickle wire encoding."""
    content = manifest.model_dump(exclude={"semantic_digest", "model_sha256", "fe_sha256"})
    return artifact_digest(content).hex()


def checksum(payload: bytes) -> str:
    """Verify exact transported bytes independently of the semantic identity."""
    return hashlib.sha256(payload).hexdigest()


def manifest_bytes(manifest: BundleManifest) -> bytes:
    """Encode deterministic, finite JSON metadata for storage and byte accounting."""
    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
