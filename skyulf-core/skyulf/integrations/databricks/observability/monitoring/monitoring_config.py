"""Explicit, bounded enrollment contracts for shared model monitoring."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Self

from ...shared._contracts import column_name, table_name
from ...shared.json_contracts import finite_json_digest
from .performance.performance_policy import validate_performance_policy

# Safety ceilings for the legacy local reader, not Spark observation row limits.
MAX_MONITOR_ROWS = 1_000_000
MAX_MONITOR_BYTES = 1024**3


def qualified_name(value: str) -> str:
    """Require a concrete Unity Catalog object rather than a session default."""
    table_name(value)
    if len(value.split(".")) != 3:
        raise ValueError("Monitoring objects require catalog.schema.name.")
    return value


def store_namespace(catalog: str, schema: str) -> str:
    """Validate central object identifiers independently of source catalogs."""
    column_name(catalog)
    column_name(schema)
    return f"{catalog}.{schema}"


def json_digest(value: Any) -> str:
    """Fingerprint finite configuration and evidence without process-specific values."""
    return finite_json_digest(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorConfig:
    """Enroll a model with independent source, result and registry locations.

    ``max_rows`` and ``max_bytes`` bound legacy local materialization. Their
    defaults are per-observation budgets, while MAX_MONITOR_* are safety ceilings.
    Spark readers keep observations distributed and do not apply these row/byte
    caps; bounded metadata and training-reference preparation have separate limits.
    """

    environment: str
    project: str
    model_name: str
    source_table: str
    prediction_table: str
    model_version: str | None = None
    model_alias: str | None = None
    label_table: str | None = None
    result_available_at_column: str | None = None
    expected_interval_hours: float = 24.0
    max_rows: int = 10000
    max_bytes: int = 67108864
    max_batches: int = 100
    enabled: bool = True
    thresholds: dict[str, float] | None = None
    model_set_name: str | None = None
    model_set_version: str | None = None
    model_set_branch: str | None = None
    performance_policy: dict[str, Any] | None = None
    execution_engine: str = "local"
    reference_namespace: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unbounded work before registry and Spark access."""
        for value in (self.environment, self.project):
            if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
                raise ValueError("Monitoring environment/project must be bounded identifiers.")
        for value in (self.model_name, self.source_table, self.prediction_table):
            qualified_name(value)
        _validate_selection(self.model_version, self.model_alias)
        _validate_label_source(self.label_table, self.result_available_at_column)
        _validate_budgets(self)
        _validate_thresholds(self.thresholds)
        _validate_model_set(self)
        _validate_performance_enrollment(self)
        _validate_execution(self)
        if type(self.enabled) is not bool:
            raise ValueError("Monitoring enabled must be a boolean.")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """Reject unknown fields instead of quietly dropping misspelled policy."""
        if type(value) is not dict or set(value) - {field.name for field in fields(cls)}:
            raise ValueError("Unknown monitoring configuration fields.")
        return cls(**value)

    @property
    def monitor_id(self) -> str:
        """Retain history across version upgrades without colliding across namespaces."""
        return json_digest([self.environment, self.project, self.model_name.lower()])

    def payload(self) -> dict[str, Any]:
        """Return the complete enrollment policy for report identity and audit."""
        payload = asdict(self)
        if self.performance_policy is None:
            payload.pop("performance_policy")
        if self.execution_engine == "local":
            payload.pop("execution_engine")
        if self.reference_namespace is None:
            payload.pop("reference_namespace")
        return payload


def _validate_execution(config: MonitorConfig) -> None:
    """Require explicit distributed reference ownership without changing legacy payloads."""
    if config.execution_engine not in {"local", "spark"}:
        raise ValueError("monitoring execution_engine must be local or spark.")
    if config.execution_engine == "spark" and config.reference_namespace is None:
        raise ValueError("Spark monitoring requires reference_namespace.")
    if config.reference_namespace is not None:
        qualified_name(f"{config.reference_namespace}.monitoring_references")


def _validate_selection(version: str | None, alias: str | None) -> None:
    """Permit exactly one pinned version or explicit alias resolved at observation time."""
    if (version is None) == (alias is None):
        raise ValueError("Set exactly one of model_version and model_alias.")
    if version is not None and (
        type(version) is not str or not re.fullmatch(r"[1-9][0-9]*", version)
    ):
        raise ValueError("model_version must be a concrete positive version string.")
    if alias is not None and (
        type(alias) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", alias)
    ):
        raise ValueError("model_alias must be a bounded registry alias.")


def _validate_label_source(table: str | None, available: str | None) -> None:
    """Labels require an explicit availability column; observation time is never fabricated."""
    if (table is None) != (available is None):
        raise ValueError("label_table and result_available_at_column must be configured together.")
    if table is not None and available is not None:
        qualified_name(table)
        column_name(available)


def _validate_budgets(config: MonitorConfig) -> None:
    """Cap driver materialization and the number of scoring snapshots replayed."""
    for name, maximum in (
        ("max_rows", MAX_MONITOR_ROWS),
        ("max_bytes", MAX_MONITOR_BYTES),
        ("max_batches", 1000),
    ):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be a positive integer at most {maximum}.")
    interval = config.expected_interval_hours
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("expected_interval_hours must be finite and positive.")


def _validate_thresholds(value: dict[str, float] | None) -> None:
    """Accept only Core drift thresholds with finite positive values."""
    if value is None:
        return
    if type(value) is not dict or set(value) - {
        "psi",
        "ks_statistic",
        "wasserstein",
        "kl_divergence",
    }:
        raise ValueError("Unknown monitoring drift thresholds.")
    for threshold in value.values():
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Drift thresholds must be finite and positive.")


def _validate_performance_enrollment(config: MonitorConfig) -> None:
    """Require explicit labels for active policy after ordinary enrollment checks."""
    if config.performance_policy is None:
        return
    if type(config.performance_policy) is not dict or not config.performance_policy:
        raise ValueError("performance_policy must be an explicit policy object.")
    policy = validate_performance_policy(config.performance_policy)
    if policy["mode"] != "off" and config.label_table is None:
        raise ValueError(
            "Active performance_policy requires label_table and result_available_at_column."
        )
    object.__setattr__(config, "performance_policy", policy)


def parse_drift_thresholds(value: str) -> dict[str, float]:
    """Parse the Bundle policy; an empty object selects Core's default thresholds."""
    try:
        thresholds = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("monitoring_drift_thresholds must be a JSON object.") from error
    if type(thresholds) is not dict:
        raise ValueError("monitoring_drift_thresholds must be a JSON object.")
    _validate_thresholds(thresholds)
    return thresholds


def parse_performance_policies(value: str) -> dict[str, dict]:
    """Validate every named model policy before any model writes predictions."""
    try:
        policies = json.loads(value, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError) as error:
        raise ValueError("monitoring_performance_policies must be a JSON object.") from error
    if type(policies) is not dict:
        raise ValueError("monitoring_performance_policies must be a JSON object.")
    for name, policy in policies.items():
        qualified_name(name)
        if type(policy) is not dict or not policy:
            raise ValueError("Each performance policy must be an explicit policy object.")
        validate_performance_policy(policy)
    return policies


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate policy fields and model keys before JSON discards them."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate monitoring performance policy JSON key.")
        result[key] = value
    return result


def _validate_model_set(config: MonitorConfig) -> None:
    """Bind projected component outputs to one explicit parent release."""
    values = (config.model_set_name, config.model_set_version, config.model_set_branch)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("Model-set monitoring requires name, version and branch together.")
    qualified_name(str(config.model_set_name))
    _validate_selection(config.model_set_version, None)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", str(config.model_set_branch)):
        raise ValueError("Invalid model-set monitoring branch.")
