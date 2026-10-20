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
    logging_mode: Literal["telemetry", "ai_gateway", "disabled"] = "telemetry"
    workload_type: str = "CPU"
    workload_size: str = "Small"
    scale_to_zero_enabled: bool = True
    ready_timeout_s: int = 3600

    def __post_init__(self) -> None:
        """Reject mutable selectors and unsafe service or UC identifiers."""
        _validate_selector(self)
        _validate_compute(self)
        if not isinstance(self.logging_mode, str) or self.logging_mode not in {
            "telemetry",
            "ai_gateway",
            "disabled",
        }:
            raise ValueError("logging_mode must be telemetry, ai_gateway or disabled.")
        if self.logging_mode == "disabled":
            return  # No payload table: callers own any prediction storage.
        for label, value in (
            ("logging_catalog", self.logging_catalog),
            ("logging_schema", self.logging_schema),
            ("logging_table_prefix", self.logging_table_prefix),
        ):
            if not is_uc_identifier(value):
                raise ValueError(f"{label} must be a simple UC identifier.")

    @property
    def model_uri(self) -> str:
        """Return the immutable MLflow registry selector."""
        return f"models:/{self.model_name}/{self.model_version}"

    @property
    def inference_table(self) -> str:
        """Return the expected inference payload table or view name."""
        if self.logging_mode == "disabled":
            raise ValueError(
                "Endpoint payload logging is disabled; there is no inference table. Use "
                "logging_mode telemetry or ai_gateway for rollout and serving monitoring."
            )
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


def _validate_compute(spec: PinnedEndpointSpec) -> None:
    """Validate supported CPU compute and the client-side readiness deadline."""
    for name, choices in (
        ("workload_type", ("CPU", "CPU_MEDIUM", "CPU_LARGE")),
        ("workload_size", ("Small", "Medium", "Large")),
    ):
        value = getattr(spec, name)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}.")
    if type(spec.scale_to_zero_enabled) is not bool:
        raise ValueError("scale_to_zero_enabled must be a boolean.")
    if type(spec.ready_timeout_s) is not int or spec.ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be a positive integer.")


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
    online_contract: str | None = None
    record_key_columns: tuple[str, ...] = ()


def is_uc_identifier(value: Any) -> bool:
    """Limit UC identifiers to an unquoted, unambiguous supported form."""
    return isinstance(value, str) and bool(_UC_PART.fullmatch(value)) and len(value) <= 255
