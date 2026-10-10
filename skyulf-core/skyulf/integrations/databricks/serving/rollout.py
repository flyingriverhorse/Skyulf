"""Durable one-step daily traffic controller for an isolated A/B endpoint.

The SDK has no atomic config compare-and-swap. All endpoint writers must share
non-expiring admission and exclusive remote mutation ACLs, including UI writers.
SingleWriterAdmission is valid only with externally enforced job serialization
and exclusive ownership. Receipts detect conflicts but cannot eliminate races
with unauthorized writers. Evidence comes from a trusted monitoring producer;
this controller does not certify metrics or turn caller PASS into cloud proof.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from ..data.admission import PublishAdmission
from ..shared.json_contracts import finite_json_digest
from .contracts import PinnedEndpointPlan, PinnedEndpointSpec
from .rollout_endpoints import (
    RolloutEndpointPlan,
    build_rollout_endpoint,
    rollout_endpoint_ready,
    rollout_traffic,
)
from .rollout_policy import (
    DailyRolloutPolicy,
    RolloutDecision,
    RolloutEvidence,
    RolloutState,
    decide_rollout,
)
from .rollout_store import MLflowRolloutStore


class RolloutOutcomeUnknownError(RuntimeError):
    """A prepared mutation may have applied; reconcile without reissuing it."""


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Expose verified state and a durable completion signal for guarded promotion."""

    state: RolloutState
    status: str
    receipt_id: str
    decision: RolloutDecision | None = None

    @property
    def promotion_pending(self) -> bool:
        """Require a separate guarded alias operation after final-stage completion."""
        return self.state.phase == "COMPLETE"


def _now() -> datetime:
    """Read wall time only where a decision or verified receipt needs it."""
    return datetime.now(UTC)


def _time(value: datetime) -> datetime:
    """Reject naive clocks and normalize durable timestamps."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Rollout clock must return an aware datetime.")
    return value.astimezone(UTC)


def _host(client: Any) -> str:
    """Bind admission to the injected authenticated workspace's canonical host."""
    return _canonical_host(getattr(getattr(client, "config", None), "host", None))


def _canonical_host(host: Any) -> str:
    """Normalize equivalent HTTPS workspace spellings into one admission identity."""
    parsed = urlsplit(host) if isinstance(host, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
    ):
        raise ValueError("A concrete HTTPS workspace host is required.")
    authority = parsed.hostname.lower().rstrip(".")
    if parsed.port not in {None, 443}:
        authority += f":{parsed.port}"
    return f"https://{authority}"


def rollout_resource_id(workspace_host: str, endpoint_name: str) -> str:
    """Share one admission key across every rollout run targeting the endpoint."""
    return "serving-rollout:" + finite_json_digest([_canonical_host(workspace_host), endpoint_name])


def _admission(admission: PublishAdmission, host: str, name: str) -> Any:
    """Require distributed or externally serialized ownership before remote work."""
    if getattr(admission, "local_only", None) is not False or not callable(
        getattr(admission, "hold", None)
    ):
        raise ValueError("Rollout requires explicit shared non-local admission.")
    return admission.hold(rollout_resource_id(host, name))


def _request(client: Any, name: str, body: dict[str, Any] | None = None) -> Any:
    """Use only the caller's authenticated SDK transport, retaining native fields."""
    transport = getattr(getattr(client, "api_client", None), "do", None)
    if not callable(transport):
        raise ValueError("Rollout requires WorkspaceClient.api_client.do transport.")
    path = f"/api/2.0/serving-endpoints/{name}"
    if body is None:
        return transport(method="GET", path=path)
    return transport(method="PUT", path=path + "/config", body=body)


def rollout_plan_from_dict(value: dict[str, Any]) -> RolloutEndpointPlan:
    """Rebuild all admitted selectors and recheck mutable configuration composition."""
    plans = []
    for role in ("champion", "challenger"):
        saved = deepcopy(value[role])
        saved["spec"] = PinnedEndpointSpec(**saved["spec"])
        for field in ("input_columns", "input_schema", "output_schema"):
            saved[field] = tuple(
                tuple(item) if isinstance(item, list) else item for item in saved[field]
            )
        plans.append(PinnedEndpointPlan(**saved))
    plan = build_rollout_endpoint(*plans)
    if finite_json_digest(asdict(plan)) != finite_json_digest(value):
        raise ValueError("Rollout plan differs from its admitted composition.")
    return plan


def _validated(
    record: dict[str, Any], host: str
) -> tuple[RolloutEndpointPlan, RolloutState, DailyRolloutPolicy]:
    """Check the entire persisted decision context before native reads or policy."""
    if record["workspace_host"] != host:
        raise ValueError("Rollout workspace identity differs from the injected client.")
    if finite_json_digest(record["plan"]) != record["plan_sha256"]:
        raise ValueError("Rollout plan digest differs from its durable record.")
    if finite_json_digest(record["promotion"]) != record["promotion_sha256"]:
        raise ValueError("Rollout promotion digest differs from its durable record.")
    plan = rollout_plan_from_dict(record["plan"])
    state = RolloutState(**record["state"])
    policy = DailyRolloutPolicy(**record["policy"])
    expected = _initial_state(plan, state.rollout_id, state.rollout_started_at)
    identity = (
        "endpoint_name",
        "champion_model_name",
        "champion_model_version",
        "challenger_model_name",
        "challenger_model_version",
    )
    if any(getattr(state, key) != getattr(expected, key) for key in identity):
        raise ValueError("Rollout state differs from its admitted model pair.")
    if record["status"] not in {"COMMITTED", "PREPARED"}:
        raise ValueError("Unknown rollout receipt status.")
    _identity(record["endpoint_id"], record["config_version"])
    if _time(datetime.fromisoformat(record["recorded_at"])) < _time(
        datetime.fromisoformat(state.stage_started_at)
    ):
        raise ValueError("Rollout receipt clock precedes its verified stage.")
    return plan, state, policy


def _initial_state(plan: RolloutEndpointPlan, rollout_id: str, timestamp: str) -> RolloutState:
    """Build a zero-exposure state from independently admitted concrete packages."""
    champion, challenger = plan.champion.spec, plan.challenger.spec
    return RolloutState(
        rollout_id,
        champion.endpoint_name,
        champion.model_name,
        champion.model_version,
        challenger.model_name,
        challenger.model_version,
        0,
        timestamp,
        timestamp,
    )


def _identity(endpoint_id: Any, revision: Any) -> None:
    """Require native identity and positive exact integer configuration revision."""
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        raise ValueError("Rollout endpoint ID is missing or invalid.")
    if type(revision) is not int or revision <= 0:
        raise ValueError("Rollout config revision must be a positive integer.")


def _verify(
    endpoint: dict[str, Any],
    record: dict[str, Any],
    plan: RolloutEndpointPlan,
    *,
    revision: int,
    percentage: int,
) -> bool:
    """Reject identity/semantic drift even while a native update is pending."""
    config = endpoint.get("config", {})
    _identity(endpoint.get("id"), config.get("config_version"))
    if endpoint["id"] != record["endpoint_id"] or config["config_version"] != revision:
        raise ValueError("Rollout endpoint identity or config revision changed unexpectedly.")
    projected = deepcopy(endpoint)
    projected["state"] = {"ready": "READY", "config_update": "NOT_UPDATING"}
    rollout_endpoint_ready(projected, plan, challenger_percentage=percentage)
    return rollout_endpoint_ready(endpoint, plan, challenger_percentage=percentage)


def _result(
    record: dict[str, Any], status: str | None = None, decision: RolloutDecision | None = None
) -> RolloutResult:
    """Expose a small immutable view of the verified durable receipt."""
    return RolloutResult(
        RolloutState(**record["state"]), status or record["status"], record["receipt_id"], decision
    )


def initialize_rollout(
    client: Any,
    plan: RolloutEndpointPlan,
    *,
    store: MLflowRolloutStore,
    policy: DailyRolloutPolicy,
    admission: PublishAdmission,
    rollout_id: str,
    clock: Callable[[], datetime] = _now,
    promotion: dict[str, Any] | None = None,
) -> RolloutResult:
    """Admit an existing isolated settled 100/0 endpoint without mutating it."""
    host = _host(client)
    plan = rollout_plan_from_dict(asdict(plan))
    policy = DailyRolloutPolicy(**asdict(policy))
    if promotion is not None and (
        not isinstance(promotion, dict) or promotion.get("auto_promote") is not True
    ):
        raise ValueError("Rollout promotion must explicitly enable auto_promote with true.")
    promotion = deepcopy(promotion)
    promotion_digest = finite_json_digest(promotion)
    name = plan.champion.spec.endpoint_name
    with _admission(admission, host, name):
        store.require_empty()
        endpoint = _request(client, name)
        _identity(endpoint.get("id"), endpoint.get("config", {}).get("config_version"))
        if endpoint.get("pending_config") or not rollout_endpoint_ready(
            endpoint, plan, challenger_percentage=0
        ):
            raise ValueError("Rollout initialization requires a settled ready endpoint.")
        timestamp = _time(clock()).isoformat()
        record = {
            "workspace_host": host,
            "endpoint_id": endpoint["id"],
            "config_version": endpoint["config"]["config_version"],
            "plan": asdict(plan),
            "plan_sha256": finite_json_digest(asdict(plan)),
            "policy": asdict(policy),
            "state": asdict(_initial_state(plan, rollout_id, timestamp)),
            "status": "COMMITTED",
            "recorded_at": timestamp,
            "decision": None,
            "evidence": None,
            "request": None,
            "promotion": promotion,
            "promotion_sha256": promotion_digest,
        }
        return _result(store.write(record, expected_receipt_id=None))


def _pending(
    record: dict[str, Any], plan: RolloutEndpointPlan, state: RolloutState
) -> dict[str, Any]:
    """Validate the full prepared intent rather than trusting mutable nested JSON."""
    request = record["request"]
    decision = RolloutDecision(**record["decision"])
    if decision.action not in {"ADVANCE", "ROLLBACK"} or state.phase != "ACTIVE":
        raise ValueError("Invalid prepared rollout action.")
    expected = deepcopy(plan.config["config"])
    expected["traffic_config"] = rollout_traffic(decision.target_percentage)
    identity = (
        request["prior_endpoint_id"],
        request["prior_config_version"],
        request["prior_percentage"],
    )
    if identity != (record["endpoint_id"], record["config_version"], state.challenger_percentage):
        raise ValueError("Prepared rollout prior identity differs from its state.")
    if (
        request["body"] != expected
        or request["target_config_version"] != record["config_version"] + 1
    ):
        raise ValueError("Prepared rollout request differs from its exact target.")
    if (
        request["action"] != decision.action
        or request["target_percentage"] != decision.target_percentage
    ):
        raise ValueError("Prepared rollout action differs from its policy decision.")
    if not isinstance(request["request_id"], str) or not request["request_id"]:
        raise ValueError("Prepared rollout request identity is missing.")
    return request


def _reconcile(
    client: Any,
    store: MLflowRolloutStore,
    record: dict[str, Any],
    plan: RolloutEndpointPlan,
    state: RolloutState,
    clock: Callable[[], datetime],
) -> RolloutResult:
    """Settle only the exact requested next revision, without another policy step."""
    request = _pending(record, plan, state)
    endpoint = _request(client, state.endpoint_name)
    revision = endpoint.get("config", {}).get("config_version")
    target = revision == request["target_config_version"]
    percentage = request["target_percentage"] if target else state.challenger_percentage
    wanted_revision = request["target_config_version"] if target else record["config_version"]
    ready = _verify(endpoint, record, plan, revision=wanted_revision, percentage=percentage)
    pending = endpoint.get("pending_config")
    if pending:
        projected = deepcopy(endpoint)
        projected["config"] = pending
        _verify(
            projected,
            record,
            plan,
            revision=request["target_config_version"],
            percentage=request["target_percentage"],
        )
        return _result(record)
    if not target or not ready:
        return _result(record)
    timestamp = _time(clock())
    if timestamp < _time(datetime.fromisoformat(record["recorded_at"])):
        raise ValueError("Rollout readiness clock moved backwards.")
    phase = "ROLLED_BACK" if request["action"] == "ROLLBACK" else "ACTIVE"
    next_state = replace(
        state, challenger_percentage=percentage, stage_started_at=timestamp.isoformat(), phase=phase
    )
    committed = record | {
        "state": asdict(next_state),
        "config_version": revision,
        "status": "COMMITTED",
        "recorded_at": timestamp.isoformat(),
    }
    return _result(
        store.write(committed, expected_receipt_id=record["receipt_id"]),
        decision=RolloutDecision(**record["decision"]),
    )


def reconcile_rollout(
    client: Any,
    *,
    store: MLflowRolloutStore,
    admission: PublishAdmission,
    clock: Callable[[], datetime] = _now,
) -> RolloutResult:
    """Resume a prepared write or verify the current state without advancing traffic."""
    host = _host(client)
    _, initial_state, _ = _validated(store.load(), host)
    with _admission(admission, host, initial_state.endpoint_name):
        record = store.load()
        plan, state, _ = _validated(record, host)
        if state.endpoint_name != initial_state.endpoint_name:
            raise ValueError("Rollout endpoint changed after admission selection.")
        if record["status"] == "PREPARED":
            return _reconcile(client, store, record, plan, state, clock)
        endpoint = _request(client, state.endpoint_name)
        ready = _verify(
            endpoint,
            record,
            plan,
            revision=record["config_version"],
            percentage=state.challenger_percentage,
        )
        if endpoint.get("pending_config"):
            raise ValueError("An unowned pending endpoint configuration blocks rollout.")
        return _result(record, None if ready else "HOLD")


@contextmanager
def hold_completed_rollout(
    client: Any, *, store: MLflowRolloutStore, admission: PublishAdmission
) -> Iterator[dict[str, Any]]:
    """Hold endpoint admission while a caller records guarded promotion or replay.

    The promotion integration validates the saved promotion pin and uses the
    existing guarded alias API. It may persist an alias receipt with store.write
    while this context retains ownership; no unverified caller state is accepted.
    """
    host = _host(client)
    _, initial_state, _ = _validated(store.load(), host)
    with _admission(admission, host, initial_state.endpoint_name):
        record = store.load()
        plan, state, _ = _validated(record, host)
        if state.endpoint_name != initial_state.endpoint_name:
            raise ValueError("Rollout endpoint changed after admission selection.")
        if record["status"] != "COMMITTED" or state.phase != "COMPLETE":
            raise ValueError("Guarded promotion requires a committed COMPLETE rollout.")
        endpoint = _request(client, state.endpoint_name)
        ready = _verify(endpoint, record, plan, revision=record["config_version"], percentage=100)
        if not ready or endpoint.get("pending_config"):
            raise ValueError("Guarded promotion requires a settled ready endpoint.")
        yield record


def _prepare(
    record: dict[str, Any],
    plan: RolloutEndpointPlan,
    decision: RolloutDecision,
    evidence: RolloutEvidence | None,
    timestamp: datetime,
) -> dict[str, Any]:
    """Bind evidence, expected prior state and exact routes to one durable intent."""
    body = deepcopy(plan.config["config"])
    body["traffic_config"] = rollout_traffic(decision.target_percentage)
    request = {
        "request_id": uuid4().hex,
        "prior_endpoint_id": record["endpoint_id"],
        "prior_config_version": record["config_version"],
        "prior_percentage": record["state"]["challenger_percentage"],
        "target_config_version": record["config_version"] + 1,
        "target_percentage": decision.target_percentage,
        "action": decision.action,
        "body": body,
    }
    return record | {
        "status": "PREPARED",
        "recorded_at": timestamp.isoformat(),
        "request": request,
        "decision": asdict(decision),
        "evidence": asdict(evidence) if evidence else None,
    }


def _apply(
    client: Any,
    store: MLflowRolloutStore,
    record: dict[str, Any],
    plan: RolloutEndpointPlan,
    state: RolloutState,
    clock: Callable[[], datetime],
) -> RolloutResult:
    """Recheck readiness immediately before a single request, then verify durability."""
    request = _pending(record, plan, state)
    endpoint = _request(client, state.endpoint_name)
    if not _verify(
        endpoint,
        record,
        plan,
        revision=record["config_version"],
        percentage=state.challenger_percentage,
    ):
        return _result(record)
    if endpoint.get("pending_config"):
        raise ValueError("An unowned pending endpoint configuration blocks rollout.")
    try:
        _request(client, state.endpoint_name, request["body"])
        return _reconcile(client, store, record, plan, state, clock)
    except Exception as exc:  # noqa: BLE001 - all post-request failures are uncertain writes
        raise RolloutOutcomeUnknownError(
            "Prepared rollout outcome is unknown; reconcile its receipt."
        ) from exc


def advance_rollout(
    client: Any,
    *,
    store: MLflowRolloutStore,
    admission: PublishAdmission,
    evidence: RolloutEvidence | None = None,
    clock: Callable[[], datetime] = _now,
) -> RolloutResult:
    """Decide at most one step; COMPLETE is durable but never mutates model aliases."""
    host = _host(client)
    initial = store.load()
    _, initial_state, _ = _validated(initial, host)
    with _admission(admission, host, initial_state.endpoint_name):
        record = store.load()
        plan, state, policy = _validated(record, host)
        if state.endpoint_name != initial_state.endpoint_name:
            raise ValueError("Rollout endpoint changed after admission selection.")
        if record["status"] == "PREPARED":
            return _reconcile(client, store, record, plan, state, clock)
        endpoint = _request(client, state.endpoint_name)
        ready = _verify(
            endpoint,
            record,
            plan,
            revision=record["config_version"],
            percentage=state.challenger_percentage,
        )
        if endpoint.get("pending_config"):
            raise ValueError("An unowned pending endpoint configuration blocks rollout.")
        if not ready:
            return _result(record, "HOLD")
        timestamp = _time(clock())
        decision = decide_rollout(policy, state, evidence, now=timestamp)
        if decision.action == "HOLD":
            return _result(record, "HOLD", decision)
        if decision.action == "COMPLETE":
            completed = record | {
                "state": asdict(replace(state, phase="COMPLETE")),
                "recorded_at": timestamp.isoformat(),
                "decision": asdict(decision),
                "evidence": asdict(evidence) if evidence else None,
                "request": None,
            }
            return _result(
                store.write(completed, expected_receipt_id=record["receipt_id"]), decision=decision
            )
        prepared = store.write(
            _prepare(record, plan, decision, evidence, timestamp),
            expected_receipt_id=record["receipt_id"],
        )
        return _apply(client, store, prepared, plan, state, clock)
