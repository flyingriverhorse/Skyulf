"""Pure daily traffic decisions; endpoint verification and persistence live outside."""

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .contracts import is_uc_identifier


@dataclass(frozen=True, slots=True)
class DailyRolloutPolicy:
    """Advance one bounded step after each fully observed minimum stage."""

    increment_percentage: int = 10
    minimum_stage_hours: float = 24.0
    evidence_max_age_hours: float = 24.0

    def __post_init__(self) -> None:
        """Reject invalid shares and nonfinite or unrepresentable durations."""
        _percentage(self.increment_percentage)
        if self.increment_percentage == 0:
            raise ValueError("increment_percentage must be positive.")
        _duration(self.minimum_stage_hours)
        _duration(self.evidence_max_age_hours)


@dataclass(frozen=True, slots=True)
class RolloutState:
    """Persist one rollout identity and its latest verified endpoint stage.

    A controller creates a new identity at zero traffic, then changes the share
    and stage clock only after verifying the endpoint update. Replacing either
    model requires a new rollout identity. Timestamps are aware ISO8601 strings
    so ``dataclasses.asdict`` yields finite JSON without a custom encoder.
    """

    rollout_id: str
    endpoint_name: str
    champion_model_name: str
    champion_model_version: str
    challenger_model_name: str
    challenger_model_version: str
    challenger_percentage: int
    rollout_started_at: str
    stage_started_at: str
    phase: str = "ACTIVE"

    def __post_init__(self) -> None:
        """Validate concrete identity, chronology, and terminal share invariants."""
        _identity(self)
        if _timestamp(self.stage_started_at) < _timestamp(self.rollout_started_at):
            raise ValueError("stage_started_at must not precede rollout_started_at.")
        _choice(self.phase, {"ACTIVE", "COMPLETE", "ROLLED_BACK"}, "phase")
        terminal_share = {"COMPLETE": 100, "ROLLED_BACK": 0}.get(self.phase)
        if terminal_share is not None and self.challenger_percentage != terminal_share:
            raise ValueError("Terminal phase has an incompatible challenger_percentage.")


@dataclass(frozen=True, slots=True)
class RolloutEvidence:
    """Carry an upstream verdict bound to one model pair and verified stage.

    BOOTSTRAP PASS certifies offline comparison plus targeted smoke and health;
    LIVE PASS certifies live quality and health. FAIL is a confirmed health or
    performance failure, not missing labels. Upstream computes these metrics.
    Failures may cover a partial stage; passing evidence must cover a complete
    minimum interval beginning at the stage start. Observation and window end
    must both be fresh, preventing old data from being reissued as new evidence.
    """

    rollout_id: str
    endpoint_name: str
    champion_model_name: str
    champion_model_version: str
    challenger_model_name: str
    challenger_model_version: str
    challenger_percentage: int
    stage_started_at: str
    window_started_at: str
    window_ended_at: str
    observed_at: str
    verdict: str
    kind: str
    reason: str

    def __post_init__(self) -> None:
        """Reject malformed identities, timestamps, verdicts and observation order."""
        _identity(self)
        _timestamp(self.stage_started_at)
        start = _timestamp(self.window_started_at)
        end = _timestamp(self.window_ended_at)
        observed = _timestamp(self.observed_at)
        if not start <= end <= observed:
            raise ValueError(
                "Evidence requires window_started_at <= window_ended_at <= observed_at."
            )
        _choice(self.verdict, {"PASS", "HOLD", "FAIL"}, "verdict")
        _choice(self.kind, {"BOOTSTRAP", "LIVE"}, "kind")
        _text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    """Describe a proposed action without mutating traffic, state, or aliases."""

    action: str
    target_percentage: int
    reason: str
    next_eligible_at: str | None

    def __post_init__(self) -> None:
        """Validate the finite JSON decision contract."""
        _choice(self.action, {"HOLD", "ADVANCE", "ROLLBACK", "COMPLETE"}, "action")
        _percentage(self.target_percentage)
        _text(self.reason, "reason")
        if self.next_eligible_at is not None:
            _timestamp(self.next_eligible_at)


def _text(value: str, label: str) -> None:
    """Require nonempty identity and explanation strings."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")


def _choice(value: str, choices: set[str], label: str) -> None:
    """Reject invalid enum values without coercing JSON scalars."""
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {sorted(choices)}.")


def _percentage(value: int) -> None:
    """Reject fractional, boolean and out-of-range traffic percentages."""
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("Traffic percentages must be integers between 0 and 100.")


def _duration(hours: float) -> timedelta:
    """Require a positive finite duration representable by datetime."""
    if type(hours) not in (int, float) or hours <= 0:
        raise ValueError("Duration hours must be positive and finite.")
    try:
        if not math.isfinite(hours):
            raise ValueError("Duration hours must be positive and finite.")
        duration = timedelta(hours=hours)
    except OverflowError as exc:
        raise ValueError("Duration hours exceed the supported range.") from exc
    if duration <= timedelta(0):
        raise ValueError("Duration hours must be at least one microsecond.")
    return duration


def _timestamp(value: str) -> datetime:
    """Parse an aware ISO8601 instant and normalize comparisons to UTC."""
    if not isinstance(value, str):
        raise ValueError("Timestamps must be aware ISO8601 strings.")
    return _utc(datetime.fromisoformat(value))


def _utc(value: datetime) -> datetime:
    """Reject naive clocks and normalize aware instants."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone offset.")
    try:
        return value.astimezone(UTC)
    except OverflowError as exc:
        raise ValueError("Timestamp exceeds the supported UTC range.") from exc


def _model(name: str, version: str) -> None:
    """Require a concrete Unity Catalog model and immutable version."""
    if not isinstance(name, str) or len(name.split(".")) != 3:
        raise ValueError("Model name must be a concrete catalog.schema.model name.")
    if not all(is_uc_identifier(part) for part in name.split(".")):
        raise ValueError("Model name must contain simple UC identifiers.")
    if not isinstance(version, str) or not re.fullmatch(r"[1-9][0-9]*", version):
        raise ValueError("Model version must be a concrete positive version string.")


def _identity(value: RolloutState | RolloutEvidence) -> None:
    """Check shared rollout identity fields without accepting mutable selectors."""
    _text(value.rollout_id, "rollout_id")
    if not isinstance(value.endpoint_name, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z_0-9-]{0,62}", value.endpoint_name
    ):
        raise ValueError("endpoint_name must be a safe 1-63 character endpoint name.")
    _model(value.champion_model_name, value.champion_model_version)
    _model(value.challenger_model_name, value.challenger_model_version)
    champion = (value.champion_model_name, value.champion_model_version)
    challenger = (value.challenger_model_name, value.challenger_model_version)
    if champion == challenger:
        raise ValueError("Champion and challenger must identify different model versions.")
    _percentage(value.challenger_percentage)


def _evidence_problem(
    policy: DailyRolloutPolicy,
    state: RolloutState,
    evidence: RolloutEvidence,
    now: datetime,
) -> str | None:
    """Explain identity or freshness failures before considering a verdict."""
    for field in (
        "rollout_id",
        "endpoint_name",
        "champion_model_name",
        "champion_model_version",
        "challenger_model_name",
        "challenger_model_version",
        "challenger_percentage",
    ):
        if getattr(state, field) != getattr(evidence, field):
            return f"Evidence {field} does not match the current rollout stage."
    stage = _timestamp(state.stage_started_at)
    if _timestamp(evidence.stage_started_at) != stage:
        return "Evidence stage_started_at does not match the verified stage."
    if _timestamp(evidence.window_started_at) < stage:
        return "Evidence window precedes the current stage."
    age = now - _timestamp(evidence.observed_at)
    if age < timedelta(0):
        return "Evidence observed_at is in the future."
    maximum = _duration(policy.evidence_max_age_hours)
    if age > maximum or now - _timestamp(evidence.window_ended_at) > maximum:
        return "Evidence observation or window is stale."
    expected = "BOOTSTRAP" if state.challenger_percentage == 0 else "LIVE"
    if evidence.kind != expected:
        return f"Current stage requires {expected} evidence."
    return None


def _passing_stage_problem(
    state: RolloutState,
    evidence: RolloutEvidence,
    eligible: datetime,
    now: datetime,
) -> str | None:
    """Require a completed, observed interval before increasing traffic."""
    if now < eligible:
        return "Minimum stage duration has not elapsed."
    if _timestamp(evidence.window_started_at) != _timestamp(state.stage_started_at):
        return "Passing evidence must begin at the verified stage start."
    if _timestamp(evidence.window_ended_at) < eligible:
        return "Passing evidence does not span a complete minimum stage."
    return None


def decide_rollout(
    policy: DailyRolloutPolicy,
    state: RolloutState,
    evidence: RolloutEvidence | None,
    *,
    now: datetime,
) -> RolloutDecision:
    """Decide one step; callers persist a new stage only after endpoint verification.

    HOLD preserves the original stage clock. Terminal states never restart.
    A fresh matching failure rolls back immediately, even during a new stage.
    An active 100-percent stage must pass its full interval before COMPLETE.
    """
    now = _utc(now)
    if state.phase != "ACTIVE":
        return RolloutDecision(
            "HOLD", state.challenger_percentage, f"Rollout is {state.phase}.", None
        )
    stage = _timestamp(state.stage_started_at)
    try:
        eligible = stage + _duration(policy.minimum_stage_hours)
    except OverflowError as exc:
        raise ValueError("Next eligible stage time exceeds the supported range.") from exc
    reason = "Evidence is missing."
    if stage > now:
        reason = "Verified stage start is in the future."
    elif evidence is not None:
        problem = _evidence_problem(policy, state, evidence, now)
        if problem is None:
            return _verified_decision(policy, state, evidence, eligible, now)
        reason = problem
    return RolloutDecision("HOLD", state.challenger_percentage, reason, eligible.isoformat())


def _verified_decision(
    policy: DailyRolloutPolicy,
    state: RolloutState,
    evidence: RolloutEvidence,
    eligible: datetime,
    now: datetime,
) -> RolloutDecision:
    """Apply a matching fresh verdict without delaying confirmed rollback."""
    if evidence.verdict == "FAIL":
        return RolloutDecision("ROLLBACK", 0, evidence.reason, None)
    if evidence.verdict == "HOLD":
        return RolloutDecision(
            "HOLD", state.challenger_percentage, evidence.reason, eligible.isoformat()
        )
    problem = _passing_stage_problem(state, evidence, eligible, now)
    if problem:
        return RolloutDecision("HOLD", state.challenger_percentage, problem, eligible.isoformat())
    if state.challenger_percentage == 100:
        return RolloutDecision("COMPLETE", 100, evidence.reason, None)
    target = min(100, state.challenger_percentage + policy.increment_percentage)
    return RolloutDecision("ADVANCE", target, evidence.reason, None)
