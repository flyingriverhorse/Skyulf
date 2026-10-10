"""Validated immutable selectors for optional pinned model serving."""

import re
from dataclasses import dataclass
from typing import Any, Literal

_UC_PART = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_ENDPOINT = re.compile(r"[A-Za-z][A-Za-z_0-9-]{0,62}\Z")
_VERSION = re.compile(r"[1-9][0-9]*\Z")


@dataclass(frozen=True, slots=True)
class PinnedEndpointSpec:
    """Select one concrete UC model version and one inference logging table."""

    endpoint_name: str
    model_name: str
    model_version: str
    logging_catalog: str
    logging_schema: str
    logging_table_prefix: str
    logging_mode: Literal["telemetry", "ai_gateway"] = "telemetry"

    def __post_init__(self) -> None:
        """Reject mutable selectors and unsafe service or UC identifiers."""
        _validate_selector(self)
        for label, value in (
            ("logging_catalog", self.logging_catalog),
            ("logging_schema", self.logging_schema),
            ("logging_table_prefix", self.logging_table_prefix),
        ):
            if not is_uc_identifier(value):
                raise ValueError(f"{label} must be a simple UC identifier.")
        if not isinstance(self.logging_mode, str) or self.logging_mode not in {
            "telemetry",
            "ai_gateway",
        }:
            raise ValueError("logging_mode must be telemetry or ai_gateway.")

    @property
    def model_uri(self) -> str:
        """Return the immutable MLflow registry selector."""
        return f"models:/{self.model_name}/{self.model_version}"

    @property
    def inference_table(self) -> str:
        """Return the expected inference payload table or view name."""
        return f"{self.logging_catalog}.{self.logging_schema}.{self.logging_table_prefix}_payload"

    @property
    def telemetry_logs_table(self) -> str:
        """Return the native telemetry Delta log table name."""
        return f"{self.logging_catalog}.{self.logging_schema}.{self.logging_table_prefix}_otel_logs"

    @property
    def telemetry_traces_table(self) -> str:
        """Return the native telemetry trace table name."""
        return (
            f"{self.logging_catalog}.{self.logging_schema}.{self.logging_table_prefix}_otel_spans"
        )

    @property
    def telemetry_metrics_table(self) -> str:
        """Return the native telemetry metrics table name."""
        return (
            f"{self.logging_catalog}.{self.logging_schema}.{self.logging_table_prefix}_otel_metrics"
        )


def _validate_selector(spec: PinnedEndpointSpec) -> None:
    """Validate the endpoint and concrete registered-model selector."""
    if not isinstance(spec.endpoint_name, str) or not _ENDPOINT.fullmatch(spec.endpoint_name):
        raise ValueError("endpoint_name must be a safe 1-63 character endpoint name.")
    if not isinstance(spec.model_name, str) or len(spec.model_name.split(".")) != 3:
        raise ValueError("model_name must be a concrete catalog.schema.model UC name.")
    if any(not is_uc_identifier(part) for part in spec.model_name.split(".")):
        raise ValueError("model_name must be a concrete catalog.schema.model UC name.")
    if not isinstance(spec.model_version, str) or not _VERSION.fullmatch(spec.model_version):
        raise ValueError("model_version must be a concrete positive version.")


@dataclass(frozen=True, slots=True)
class PinnedEndpointPlan:
    """A validated SDK create body and the inspected model input/output schemas."""

    spec: PinnedEndpointSpec
    config: dict[str, Any]
    input_columns: tuple[str, ...]
    input_schema: tuple[tuple[str, str], ...]
    output_schema: tuple[tuple[str, str], ...] = ()


def is_uc_identifier(value: Any) -> bool:
    """Limit UC identifiers to an unquoted, unambiguous supported form."""
    return isinstance(value, str) and bool(_UC_PART.fullmatch(value)) and len(value) <= 255
