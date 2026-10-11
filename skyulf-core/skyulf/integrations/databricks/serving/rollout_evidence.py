"""Trusted rollout evidence from saved comparison and existing serving metrics services."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeGuard

from ...mlflow.lifecycle.validation import (
    comparison_digest,
    evaluate_quality_gates,
    quality_gate_results,
    validate_quality_policy,
)
from ..observability.monitoring.monitoring_config import MonitorConfig
from ..observability.monitoring.serving.serving_observation import spark_dtype
from ..observability.monitoring.serving.serving_payloads import (
    REQUEST_KEY,
    ROW_KEY,
    parse_serving_payloads,
)
from ..observability.monitoring.serving.serving_sources import (
    read_serving_payload_snapshot,
    verify_serving_payload_snapshot,
)
from ..observability.monitoring.spark.spark_monitoring_metrics import build_spark_performance_report
from ..observability.monitoring.spark.spark_monitoring_sources import read_spark_labels
from ..shared.json_contracts import finite_json_digest
from ..training.shared.training_evidence import load_candidate_evidence
from .endpoints import api_transport, validate_json_rows, validate_named_rows, validate_schema_rows
from .rollout_endpoints import RolloutEndpointPlan, build_rollout_endpoint, rollout_endpoint_ready
from .rollout_policy import RolloutEvidence, RolloutState


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutHealthPolicy:
    """Require explicit operational and label coverage thresholds for live admission."""

    minimum_requests: int
    maximum_error_rate: float
    maximum_latency_ms: float
    minimum_labeled_rows: int
    minimum_label_coverage: float

    def __post_init__(self) -> None:
        """Reject booleans, nonfinite bounds and empty evidence thresholds."""
        for name, minimum in (("minimum_requests", 1), ("minimum_labeled_rows", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}.")
        for name in ("maximum_error_rate", "minimum_label_coverage"):
            value = getattr(self, name)
            if not _finite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction between zero and one.")
        if not _finite(self.maximum_latency_ms) or self.maximum_latency_ms <= 0:
            raise ValueError("maximum_latency_ms must be a positive finite number.")


@dataclass(frozen=True, slots=True)
class RolloutEvidenceResult:
    """Keep the controller verdict and complete bounded producer evidence together.

    The job must persist details and verify details_digest before delivering the
    verdict to the controller. The controller alone cannot authenticate metrics.
    """

    evidence: RolloutEvidence
    details: dict[str, Any]

    @property
    def details_digest(self) -> str:
        """Bind the finite audit record to the exact stage verdict it produced."""
        return finite_json_digest({"evidence": asdict(self.evidence), "details": self.details})


def _finite(value: Any) -> TypeGuard[int | float]:
    """Accept finite ordinary JSON numbers while rejecting booleans and overflow."""
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_stage(plan: RolloutEndpointPlan, state: RolloutState, now: datetime) -> datetime:
    """Bind admitted model selectors and request schemas before any producer reads."""
    if not isinstance(now, datetime) or now.utcoffset() != timedelta(0):
        raise ValueError("Rollout evidence requires an aware UTC now.")
    if build_rollout_endpoint(plan.champion, plan.challenger) != plan:
        raise ValueError("Rollout plan differs from its admitted model pair.")
    for role in ("champion", "challenger"):
        spec = getattr(plan, role).spec
        if (spec.endpoint_name, spec.model_name, spec.model_version) != (
            state.endpoint_name,
            getattr(state, f"{role}_model_name"),
            getattr(state, f"{role}_model_version"),
        ):
            raise ValueError("Rollout state differs from its admitted model pair.")
    start = datetime.fromisoformat(state.stage_started_at).astimezone(UTC)
    if start >= now or state.phase != "ACTIVE":
        raise ValueError("Rollout evidence requires an active stage starting before now.")
    return start


def _result(
    state: RolloutState, now: datetime, verdict: str, reason: str, details: dict
) -> RolloutEvidenceResult:
    """Attach immutable state identity and the exact observed stage window."""
    identity = asdict(state)
    identity.pop("phase")
    identity.pop("rollout_started_at")
    evidence = RolloutEvidence(
        **identity,
        window_started_at=state.stage_started_at,
        window_ended_at=now.isoformat(),
        observed_at=now.isoformat(),
        verdict=verdict,
        kind="BOOTSTRAP" if state.challenger_percentage == 0 else "LIVE",
        reason=reason,
    )
    return RolloutEvidenceResult(evidence, details)


def _gates_verdict(gates: list[dict]) -> tuple[str, str]:
    """Distinguish unavailable quality from measured threshold regression."""
    if not gates or any(not _finite(gate["value"]) for gate in gates):
        return "HOLD", "Quality metrics or configured gates are unavailable."
    if not all(gate["passed"] for gate in gates):
        return "FAIL", "Measured quality breaches configured absolute gates."
    return "PASS", "Measured quality passes configured absolute gates."


def _comparison_verdict(report: Any, state: RolloutState, digest: str) -> tuple[str, str]:
    """Re-evaluate saved numeric gates rather than trusting a caller's eligible flag."""
    if (report.model_name, report.candidate_version, report.champion_version) != (
        state.challenger_model_name,
        state.challenger_model_version,
        state.champion_model_version,
    ) or state.champion_model_name != state.challenger_model_name:
        return "HOLD", "Saved comparison differs from the pinned model pair."
    if comparison_digest(report) != digest:
        return "HOLD", "Saved comparison digest differs from the requested evidence."
    validate_quality_policy(report.metric, report.quality_threshold, report.quality_gates)
    verdict = _gates_verdict(quality_gate_results(report))
    if verdict[0] == "PASS" and report.eligible is not True:
        return "FAIL", "Saved comparison does not admit the candidate improvement."
    return verdict


def _ready(client: Any, plan: RolloutEndpointPlan, percentage: int) -> bool:
    """Read actual settled endpoint identity before and after smoke requests."""
    endpoint = api_transport(client)(
        method="GET", path=f"/api/2.0/serving-endpoints/{plan.champion.spec.endpoint_name}"
    )
    return rollout_endpoint_ready(endpoint, plan, challenger_percentage=percentage)


def _smoke(client: Any, plan: RolloutEndpointPlan, records: Sequence[Mapping[str, Any]]) -> dict:
    """Target both native routes and verify actual response entity headers and outputs."""
    rows = validate_named_rows(records, plan.champion)
    details = {}
    for role in ("champion", "challenger"):
        pinned = getattr(plan, role)
        response = api_transport(client)(
            method="POST",
            path=f"/serving-endpoints/{pinned.spec.endpoint_name}/served-models/{role}/invocations",
            body={"dataframe_records": rows},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            response_headers=["served-model-name"],
        )
        _validate_smoke_response(response, role, len(rows), pinned.output_schema)
        details[role] = {
            "model_name": pinned.spec.model_name,
            "model_version": pinned.spec.model_version,
            "rows": len(rows),
            "response_sha256": finite_json_digest(response),
        }
    return details


def _validate_smoke_response(response: Any, role: str, count: int, schema: tuple) -> None:
    """Reject wrong attribution, missing rows, unavailable predictions and malformed scalars."""
    if not isinstance(response, dict) or response.get("served-model-name") != role:
        raise ValueError("Smoke response does not attest the targeted served model.")
    rows = response.get("predictions")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Smoke response predictions differ from the requested row count.")
    _validate_smoke_rows(rows, schema)


def _validate_smoke_rows(rows: list, schema: tuple) -> None:
    """Require every saved output field to be present, finite and available."""
    names = {name for name, _ in schema}
    if any(not isinstance(row, dict) or set(row) != names for row in rows):
        raise ValueError("Smoke response predictions differ from the saved output schema.")
    if any(value is None for row in rows for value in row.values()):
        raise ValueError("Smoke response predictions are unavailable.")
    validate_json_rows(rows)
    validate_schema_rows(rows, schema)


def observe_bootstrap_rollout(
    client: Any,
    registry_client: Any,
    plan: RolloutEndpointPlan,
    state: RolloutState,
    *,
    comparison_sha256: str,
    registry_uri: str,
    records: Sequence[Mapping[str, Any]],
    now: datetime,
) -> RolloutEvidenceResult:
    """Load a real saved comparison and query both named entities at zero traffic.

    Native targeted invocations ignore traffic percentages. The SDK transport
    returns the requested served-model-name response header in the result map.
    No caller-supplied smoke-success boolean or verdict is accepted.
    """
    _validate_stage(plan, state, now)
    if state.challenger_percentage != 0:
        raise ValueError("Bootstrap observation requires a zero-percent stage.")
    details: dict[str, Any] = {"comparison_sha256": comparison_sha256}
    try:
        report, _, _, _ = load_candidate_evidence(
            registry_client,
            state.challenger_model_name,
            state.challenger_model_version,
            comparison_sha256,
            registry_uri=registry_uri,
        )
        verdict, reason = _comparison_verdict(report, state, comparison_sha256)
        details["quality_gates"] = quality_gate_results(report)
        if verdict != "PASS":
            return _result(state, now, verdict, reason, details)
        if not _ready(client, plan, 0):
            return _result(state, now, "HOLD", "Endpoint is not settled and ready.", details)
        details["smoke"] = _smoke(client, plan, records)
        if not _ready(client, plan, 0):
            return _result(state, now, "HOLD", "Endpoint changed during smoke.", details)
    except (ValueError, RuntimeError) as exc:
        return _result(state, now, "HOLD", str(exc), details)
    return _result(
        state,
        now,
        "PASS",
        "Saved comparison gates and both targeted smoke responses passed.",
        details,
    )


def _validate_live_binding(
    plan: RolloutEndpointPlan, state: RolloutState, config: MonitorConfig, artifact: Any, spec: Any
) -> None:
    """Reject unsupported model sets and wrong endpoint/source/version before native reads."""
    from ....inference.fitted_pipeline import FittedPipelineArtifact  # noqa: PLC0415
    from ....inference.pipeline_scoring import scoring_output_schema  # noqa: PLC0415

    if not isinstance(artifact, FittedPipelineArtifact) or config.model_set_name is not None:
        raise ValueError(
            "Live rollout evidence currently requires one fitted pipeline, not a model-set branch."
        )
    selected = plan.challenger
    if (config.model_name, config.model_version, config.serving_endpoint, config.source_table) != (
        state.challenger_model_name,
        state.challenger_model_version,
        state.endpoint_name,
        selected.spec.inference_table,
    ):
        raise ValueError("Serving monitoring identity differs from the admitted rollout stage.")
    if not set(spec.record_key_columns).issubset(selected.input_columns):
        raise ValueError("Serving request schema must retain the declared source record keys.")
    actual = tuple((column.name, column.dtype) for column in scoring_output_schema(artifact))
    if selected.output_schema != actual:
        raise ValueError("Serving output schema differs from the loaded fitted artifact.")


def _read_stage(
    spark: Any, plan: RolloutEndpointPlan, config: MonitorConfig, start: datetime, now: datetime
) -> tuple[Any, dict, dict, datetime | None]:
    """Use native snapshot, identity, deduplication and capture checks without reimplementing metrics."""
    payloads, source = read_serving_payload_snapshot(spark, config.source_table, now)
    selected = plan.challenger
    _, predictions, summary, observed = parse_serving_payloads(
        payloads,
        spark.table("system.serving.served_entities"),
        endpoint_name=selected.spec.endpoint_name,
        model_name=config.model_name,
        model_version=selected.spec.model_version,
        input_columns=tuple((name, spark_dtype(dtype)) for name, dtype in selected.input_schema),
        output_columns=tuple((name, spark_dtype(dtype)) for name, dtype in selected.output_schema),
        output_prefix="",
        start=start,
        end=now,
    )
    verify_serving_payload_snapshot(spark, config.source_table, source)
    finite_json_digest(summary)
    return predictions, source, summary, observed


def _health_verdict(summary: dict, policy: RolloutHealthPolicy) -> tuple[str, str]:
    """Apply only scalar operational bounds to existing native request aggregates."""
    for field in ("request_count", "success_count", "error_count"):
        if type(summary.get(field)) is not int or summary[field] < 0:
            return "HOLD", "Serving request counts are unavailable."
    count = summary["request_count"]
    if summary["success_count"] + summary["error_count"] != count:
        return "HOLD", "Serving request counts are inconsistent."
    if count < policy.minimum_requests:
        return "HOLD", "Insufficient serving requests for this stage."
    if summary["error_count"] / count > policy.maximum_error_rate:
        return "FAIL", "Serving error rate exceeds the configured maximum."
    latency = summary.get("latency_max_ms")
    if not _finite(latency) or latency < 0:
        return "HOLD", "Serving latency is unavailable."
    if latency > policy.maximum_latency_ms:
        return "FAIL", "Serving latency exceeds the configured maximum."
    return "PASS", "Serving operational thresholds passed."


def _fresh_observation(
    observed: datetime | None, start: datetime, now: datetime, config: MonitorConfig
) -> bool:
    """Reject missing, stale and future actual request timestamps."""
    if not isinstance(observed, datetime) or observed.utcoffset() != timedelta(0):
        return False
    return start <= observed <= now and now - observed <= timedelta(
        hours=config.expected_interval_hours
    )


def _performance(
    spark: Any, predictions: Any, artifact: Any, spec: Any, config: MonitorConfig, now: datetime
) -> tuple[dict, dict]:
    """Measure actual saved responses against pinned, available labels with the existing engine."""
    keys = (REQUEST_KEY, ROW_KEY)
    labels, evidence = read_spark_labels(spark, config, keys, spec.target_column, predictions, now)
    report = build_spark_performance_report(
        predictions,
        labels,
        record_key_columns=keys,
        target_column=spec.target_column,
        result_available_at_column=str(config.result_available_at_column),
        as_of=now,
        task=artifact.manifest.task,
        classes=artifact.manifest.classes,
    )
    return report, evidence


def _label_verdict(report: dict, health: RolloutHealthPolicy) -> tuple[str, str]:
    """Missing or insufficient arrived labels hold instead of creating a regression."""
    rows, coverage = report.get("labeled_rows"), report.get("label_coverage")
    if type(rows) is not int or rows < health.minimum_labeled_rows:
        return "HOLD", "Insufficient mature labeled rows for this stage."
    if not _finite(coverage) or not health.minimum_label_coverage <= coverage <= 1:
        return "HOLD", "Insufficient mature label coverage for this stage."
    return "PASS", "Mature label evidence is sufficient."


def observe_live_rollout(
    spark: Any,
    plan: RolloutEndpointPlan,
    state: RolloutState,
    config: MonitorConfig,
    artifact: Any,
    training_spec: Any,
    health_policy: RolloutHealthPolicy,
    *,
    metric: str,
    quality_threshold: float,
    quality_gates: dict[str, float] | None = None,
    now: datetime,
) -> RolloutEvidenceResult:
    """Observe one exact live stage through existing native telemetry and Core metrics.

    The caller loads artifact and training_spec with verified registry/training
    loaders for plan.challenger. Config must select that concrete version and
    endpoint payload table. LIVE v1 supports single pipelines only. Labels use
    Databricks request ID and request-row index, as in existing serving monitoring.
    No predictions are recomputed and no raw observation rows reach the driver.
    Metric names use the existing heldout_* quality gate vocabulary. Unlike
    epoch-aligned performance jobs, this observation spans stage_started_at to now.
    """
    start = _validate_stage(plan, state, now)
    if state.challenger_percentage == 0:
        raise ValueError("Live observation requires a nonzero challenger stage.")
    _validate_live_binding(plan, state, config, artifact, training_spec)
    validate_quality_policy(metric, quality_threshold, quality_gates, task=artifact.manifest.task)
    if quality_threshold is None:
        raise ValueError("Live rollout requires an explicit absolute quality threshold.")
    details: dict[str, Any] = {
        "monitor_config": config.payload(),
        "health_policy": asdict(health_policy),
    }
    try:
        predictions, source, summary, observed = _read_stage(spark, plan, config, start, now)
        details.update(
            source=source,
            serving=summary,
            latest_request_at=None if observed is None else observed.isoformat(),
        )
        if not _fresh_observation(observed, start, now, config):
            return _result(
                state, now, "HOLD", "Serving observation is missing, stale or future.", details
            )
        verdict, reason = _health_verdict(summary, health_policy)
        if verdict != "PASS":
            return _result(state, now, verdict, reason, details)
        performance, labels = _performance(spark, predictions, artifact, training_spec, config, now)
        details.update(performance=performance, labels=labels)
        verdict, reason = _label_verdict(performance, health_policy)
        if verdict == "PASS":
            values = {f"heldout_{name}": value for name, value in performance["values"].items()}
            gates = evaluate_quality_gates(values, metric, quality_threshold, quality_gates)
            details["quality_gates"] = gates
            verdict, reason = _gates_verdict(gates)
        verify_serving_payload_snapshot(spark, config.source_table, source)
        return _result(state, now, verdict, reason, details)
    except (ValueError, RuntimeError) as exc:
        return _result(state, now, "HOLD", str(exc), details)
