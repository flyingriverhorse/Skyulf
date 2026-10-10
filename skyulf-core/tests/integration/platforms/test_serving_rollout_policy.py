"""Pure daily rollout decisions bind fresh evidence to one verified stage."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from skyulf.integrations.databricks.serving.rollout_policy import (
    DailyRolloutPolicy,
    RolloutEvidence,
    RolloutState,
    decide_rollout,
)

START = datetime(2026, 12, 31, 12, tzinfo=UTC)
DAY = timedelta(days=1)


def state(share=0):
    """Bind test evidence to concrete model versions and a verified clock."""
    return RolloutState(
        "rollout-1",
        "endpoint",
        "catalog.schema.model",
        "1",
        "catalog.schema.model",
        "2",
        share,
        START.isoformat(),
        START.isoformat(),
    )


def evidence(current, now=START + DAY, **changes):
    """Build a complete stage verdict without mocking the policy."""
    fields = asdict(current)
    fields.pop("rollout_started_at")
    fields.pop("phase")
    fields.update(
        window_started_at=current.stage_started_at,
        window_ended_at=now.isoformat(),
        observed_at=now.isoformat(),
        verdict="PASS",
        kind="BOOTSTRAP" if current.challenger_percentage == 0 else "LIVE",
        reason="quality and health verified",
    )
    fields.update(changes)
    return RolloutEvidence(**fields)


@pytest.mark.parametrize("share", range(101))
def test_each_share_advances_once_or_finishes_after_a_full_stage(share):
    """Every valid share caps at 100 and final observation precedes completion."""
    current = state(share)
    decision = decide_rollout(DailyRolloutPolicy(), current, evidence(current), now=START + DAY)
    assert decision.action == ("COMPLETE" if share == 100 else "ADVANCE")
    assert decision.target_percentage == min(100, share + 10)


@pytest.mark.parametrize("gap", [1, 2, 30, 365])
def test_missed_days_never_multiply_increment(gap):
    """Scheduler downtime cannot skip unobserved traffic stages."""
    current = state()
    now = START + DAY * gap
    result = decide_rollout(DailyRolloutPolicy(), current, evidence(current, now), now=now)
    assert result.target_percentage == 10


def test_clock_boundary_and_replay_preserve_stage_clock():
    """An advance starts a new minimum interval only after controller verification."""
    current = state()
    early = START + DAY - timedelta(microseconds=1)
    held = decide_rollout(DailyRolloutPolicy(), current, evidence(current, early), now=early)
    assert held.action == "HOLD"
    assert held.next_eligible_at == (START + DAY).isoformat()
    verified = replace(
        current, challenger_percentage=10, stage_started_at=(START + DAY).isoformat()
    )
    replay = decide_rollout(DailyRolloutPolicy(), verified, evidence(current), now=START + DAY)
    assert replay.action == "HOLD"
    assert verified.stage_started_at == (START + DAY).isoformat()


@pytest.mark.parametrize(
    "change",
    [
        {"rollout_id": "new-rollout"},
        {"endpoint_name": "other"},
        {"champion_model_name": "catalog.schema.other"},
        {"champion_model_version": "3"},
        {"challenger_model_name": "catalog.schema.other"},
        {"challenger_model_version": "3"},
        {"challenger_percentage": 20},
        {"stage_started_at": (START + timedelta(seconds=1)).isoformat()},
        {"observed_at": (START + 2 * DAY).isoformat()},
        {"window_started_at": (START + timedelta(hours=1)).isoformat()},
        {"kind": "BOOTSTRAP"},
        {"verdict": "HOLD"},
    ],
)
def test_wrong_identity_incomplete_window_and_future_evidence_hold(change):
    """Only matching current-stage observations may change traffic."""
    current = state(10)
    result = decide_rollout(
        DailyRolloutPolicy(), current, evidence(current, **change), now=START + DAY
    )
    assert result.action == "HOLD"
    assert result.target_percentage == 10
    assert result.reason


@pytest.mark.parametrize("verdict", ["PASS", "FAIL"])
def test_missing_or_stale_evidence_never_mutates(verdict):
    """Stale failures cannot roll back an unrelated later condition."""
    current = state(10)
    now = START + 3 * DAY
    stale = evidence(current, verdict=verdict)
    for observation in (None, stale, replace(stale, observed_at=now.isoformat())):
        result = decide_rollout(DailyRolloutPolicy(), current, observation, now=now)
        assert result.action == "HOLD"
        assert result.target_percentage == 10


def test_failure_rolls_back_before_minimum_stage():
    """Confirmed fresh failures bypass the ordinary stage observation period."""
    current = state(70)
    now = START + timedelta(minutes=5)
    result = decide_rollout(
        DailyRolloutPolicy(), current, evidence(current, now, verdict="FAIL"), now=now
    )
    assert result.action == "ROLLBACK"
    assert result.target_percentage == 0
    assert result.next_eligible_at is None


@pytest.mark.parametrize("phase,share", [("COMPLETE", 100), ("ROLLED_BACK", 0)])
def test_terminal_rollouts_do_not_restart(phase, share):
    """Replayed triggers cannot resurrect an ended rollout."""
    current = replace(state(share), phase=phase)
    result = decide_rollout(DailyRolloutPolicy(), current, evidence(current), now=START + DAY)
    assert result.action == "HOLD"
    assert result.next_eligible_at is None


def test_bootstrap_does_not_accept_live_only_verdict():
    """Zero traffic requires the upstream offline comparison and targeted smoke gate."""
    current = state()
    result = decide_rollout(
        DailyRolloutPolicy(), current, evidence(current, kind="LIVE"), now=START + DAY
    )
    assert result.action == "HOLD"


def test_equivalent_offsets_are_the_same_stage():
    """Timestamp spelling cannot change identity when instants match."""
    current = state(10)
    alternate = START.astimezone(timezone(timedelta(hours=2))).isoformat()
    observed = evidence(current, stage_started_at=alternate, window_started_at=alternate)
    assert (
        decide_rollout(DailyRolloutPolicy(), current, observed, now=START + DAY).action == "ADVANCE"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("increment_percentage", True),
        ("increment_percentage", 0),
        ("increment_percentage", 101),
        ("increment_percentage", 1.5),
        ("minimum_stage_hours", 0),
        ("minimum_stage_hours", float("inf")),
        ("evidence_max_age_hours", float("nan")),
        ("evidence_max_age_hours", True),
        ("minimum_stage_hours", 1e300),
        ("minimum_stage_hours", 10**1000),
        ("minimum_stage_hours", 1e-20),
    ],
)
def test_invalid_policy_values_are_rejected(field, value):
    """Invalid or unrepresentable limits must not produce unsafe schedules."""
    with pytest.raises(ValueError):
        DailyRolloutPolicy(**{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"challenger_percentage": True},
        {"challenger_percentage": -1},
        {"challenger_percentage": 101},
        {"challenger_percentage": 1.5},
        {"champion_model_version": "@prod"},
        {"challenger_model_version": True},
        {"challenger_model_version": "1"},
        {"champion_model_name": "model"},
        {"stage_started_at": "2026-10-35T12:00:00+00:00"},
        {"stage_started_at": "2026-12-31T12:00:00"},
        {"rollout_started_at": (START + DAY).isoformat()},
        {"phase": "COMPLETE"},
        {"rollout_id": ""},
    ],
)
def test_invalid_states_fail_strict_reconstruction(change):
    """Malformed persisted state cannot silently default to a valid rollout."""
    with pytest.raises(ValueError):
        replace(state(), **change)


def test_finite_json_round_trip_and_custom_policy():
    """The controller can persist every state and policy field without custom codecs."""
    current = state(90)
    policy = DailyRolloutPolicy(15, 12, 6)
    restored = RolloutState(**json.loads(json.dumps(asdict(current), allow_nan=False)))
    restored_policy = DailyRolloutPolicy(**json.loads(json.dumps(asdict(policy), allow_nan=False)))
    now = START + timedelta(hours=12)
    result = decide_rollout(restored_policy, restored, evidence(restored, now), now=now)
    assert result.target_percentage == 100
    assert result.action == "ADVANCE"


def test_naive_now_is_rejected_and_future_state_holds():
    """Clock errors cannot masquerade as an elapsed stage."""
    with pytest.raises(ValueError):
        decide_rollout(DailyRolloutPolicy(), state(), None, now=datetime(2027, 1, 1))
    assert decide_rollout(DailyRolloutPolicy(), state(), None, now=START - DAY).action == "HOLD"


@pytest.mark.parametrize(
    "change",
    [
        {"observed_at": "2027-01-01T12:00:00"},
        {"observed_at": START.isoformat()},
        {"window_started_at": (START + 2 * DAY).isoformat()},
        {"stage_started_at": "2026-02-30T12:00:00Z"},
        {"verdict": "unknown"},
        {"kind": "unknown"},
        {"reason": " "},
        {"challenger_percentage": False},
    ],
)
def test_invalid_evidence_is_rejected(change):
    """Malformed persisted evidence fails before any traffic decision is made."""
    with pytest.raises(ValueError):
        evidence(state(), **change)


def test_evidence_age_limit_is_inclusive_and_does_not_retimestamp_state():
    """The maximum evidence age admits its exact boundary only."""
    current = state(10)
    observed = evidence(current)
    boundary = START + 2 * DAY
    policy = DailyRolloutPolicy()
    assert decide_rollout(policy, current, observed, now=boundary).action == "ADVANCE"
    assert (
        decide_rollout(policy, current, observed, now=boundary + timedelta(microseconds=1)).action
        == "HOLD"
    )
    assert current.stage_started_at == START.isoformat()


@pytest.mark.parametrize(
    "change",
    [
        {"rollout_id": "replacement"},
        {"stage_started_at": (START + timedelta(seconds=1)).isoformat()},
        {"challenger_percentage": 80},
        {"observed_at": (START + 2 * DAY).isoformat()},
    ],
)
def test_unbound_or_future_failure_does_not_rollback(change):
    """The emergency path still enforces rollout identity and time checks."""
    current = state(70)
    observed = evidence(current, verdict="FAIL", **change)
    result = decide_rollout(DailyRolloutPolicy(), current, observed, now=START + DAY)
    assert result.action == "HOLD"
    assert result.target_percentage == 70
