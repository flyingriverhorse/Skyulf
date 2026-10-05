"""Contract tests for pure version-bound performance policy decisions."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from skyulf.integrations.databricks.performance_policy import (
    completed_performance_window,
    evaluate_performance,
    validate_performance_policy,
)

NOW = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)
START = "2026-10-05T10:00:00Z"
END = "2026-10-05T11:00:00Z"


def policy(**changes):
    """Make an explicit active policy so each test changes one contract part."""
    value = {
        "mode": "retrain",
        "metric": "f1_weighted",
        "direction": "higher",
        "baseline": {"kind": "training_holdout", "model_version": "7"},
        "tolerance": 0.05,
        "tolerance_mode": "absolute",
        "window_hours": 1,
        "label_delay_hours": 1,
        "minimum_labeled_rows": 10,
        "minimum_label_coverage": 0.8,
        "consecutive_windows": 2,
    }
    value.update(changes)
    return value


def current(**changes):
    """Build one completed observation with explicit label cutoff and contract."""
    value = {
        "model_version": "7",
        "metric": "f1_weighted",
        "contract_digest": "c" * 64,
        "value": 0.84,
        "labeled_rows": 20,
        "label_coverage": 0.9,
        "window_start": START,
        "window_end": END,
        "as_of": "2026-10-05T12:00:00Z",
    }
    value.update(changes)
    return value


def baseline(**changes):
    """Build a comparable training holdout reference."""
    value = {
        "model_version": "7",
        "metric": "f1_weighted",
        "contract_digest": "c" * 64,
        "value": 0.9,
        "labeled_rows": 50,
        "label_coverage": 1.0,
    }
    value.update(changes)
    return value


def test_off_defaults_and_active_policy_does_not_mutate_input():
    """Enrollment stays disabled unless every active choice is explicit."""
    assert validate_performance_policy(None) == {"mode": "off"}
    assert validate_performance_policy({}) == {"mode": "off"}
    source = policy()
    snapshot = deepcopy(source)
    assert validate_performance_policy(source) == snapshot
    assert source == snapshot


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"baseline": None}, "baseline"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"tolerance": True}, "tolerance"),
        ({"window_hours": 0}, "window_hours"),
        ({"minimum_labeled_rows": True}, "minimum_labeled_rows"),
        ({"minimum_label_coverage": 1.1}, "minimum_label_coverage"),
        ({"consecutive_windows": 101}, "consecutive_windows"),
        ({"direction": "lower"}, "direction"),
        ({"metric": "mystery"}, "metric"),
        ({"surprise": 1}, "unknown"),
    ],
)
def test_invalid_active_configuration_is_rejected(change, message):
    """Invalid policy inputs cannot silently enroll or change comparisons."""
    with pytest.raises(ValueError, match=message):
        validate_performance_policy(policy(**change))


def test_production_baseline_requires_pinned_report_digest():
    """A production reference cannot drift to a different report on replay."""
    with pytest.raises(ValueError, match="report_id"):
        validate_performance_policy(
            policy(baseline={"kind": "production_window", "model_version": "7"})
        )


@pytest.mark.parametrize("version", ["0", "latest", "01", "-1", 7, True])
def test_baseline_version_is_a_concrete_positive_decimal_string(version):
    """Policy baselines cannot float with aliases or malformed registry versions."""
    with pytest.raises(ValueError, match="model_version"):
        validate_performance_policy(
            policy(baseline={"kind": "training_holdout", "model_version": version})
        )


def test_completed_window_is_epoch_aligned_and_respects_label_delay():
    """Only a full fixed observation window whose labels matured is compared."""
    start, end = completed_performance_window(NOW, policy())
    assert start.isoformat() == "2026-10-05T10:00:00+00:00"
    assert end.isoformat() == "2026-10-05T11:00:00+00:00"


def test_absolute_higher_metric_degradation_is_eligible_at_boundary():
    """An exact tolerance breach qualifies once enough windows have failed."""
    checked = evaluate_performance(
        policy(consecutive_windows=1), current(value=0.85), baseline(), [], now=NOW
    )
    assert checked["status"] == "degraded"
    assert checked["action"] == "request_eligible"
    assert checked["consecutive_failures"] == 1
    assert checked["absolute_degradation"] == pytest.approx(0.05)
    assert checked["threshold_value"] == pytest.approx(0.85)
    assert checked["as_of"] == "2026-10-05T12:00:00Z"


def test_decimal_absolute_boundary_is_degraded_despite_binary_rounding():
    """An exact decimal tolerance breach must not become healthy from float noise."""
    checked = evaluate_performance(
        policy(tolerance=0.1, consecutive_windows=1),
        current(value=0.8),
        baseline(value=0.9),
        [],
        now=NOW,
    )
    assert checked["status"] == "degraded"
    assert checked["action"] == "request_eligible"


def test_relative_lower_metric_handles_zero_baseline_as_unavailable():
    """Relative degradation cannot divide by a zero reference value."""
    setting = policy(metric="rmse", direction="lower", tolerance_mode="relative")
    checked = evaluate_performance(
        setting, current(metric="rmse", value=0.4), baseline(metric="rmse", value=0), [], now=NOW
    )
    assert checked["status"] == "unavailable"
    assert checked["action"] == "none"
    assert "zero" in checked["reason"]


def test_relative_lower_metric_reports_breach_with_pinned_production_reference():
    """Error growth uses the lower-is-better relative threshold and pinned report."""
    report_id = "a" * 64
    setting = policy(
        mode="report",
        metric="rmse",
        direction="lower",
        tolerance=0.1,
        tolerance_mode="relative",
        consecutive_windows=1,
        baseline={
            "kind": "production_window",
            "model_version": "7",
            "report_id": report_id,
        },
    )
    reference = baseline(
        metric="rmse",
        value=2.0,
        report_id=report_id,
        window_start="2026-10-01T00:00:00Z",
        window_end="2026-10-01T01:00:00Z",
        as_of="2026-10-01T02:00:00Z",
    )
    checked = evaluate_performance(
        setting, current(metric="rmse", value=2.2), reference, [], now=NOW
    )
    assert checked["status"] == "degraded"
    assert checked["action"] == "report"
    assert checked["relative_degradation"] == pytest.approx(0.1)
    assert checked["threshold_value"] == pytest.approx(2.2)
    assert checked["baseline_ref"]["report_id"] == report_id


@pytest.mark.parametrize(
    ("reference_change", "reason"),
    [
        ({"window_start": "2026-10-01T00:30:00Z"}, "window"),
        ({"window_end": "2026-10-05T11:00:00Z"}, "window"),
        ({"as_of": "2026-10-05T12:30:00Z"}, "as_of"),
    ],
)
def test_production_reference_must_precede_current_with_same_window_span(reference_change, reason):
    """A production reference cannot use overlapping, longer or future evidence."""
    report_id = "a" * 64
    setting = policy(
        baseline={"kind": "production_window", "model_version": "7", "report_id": report_id}
    )
    reference = baseline(
        **{
            "report_id": report_id,
            "window_start": "2026-10-01T00:00:00Z",
            "window_end": "2026-10-01T01:00:00Z",
            "as_of": "2026-10-01T02:00:00Z",
            **reference_change,
        }
    )
    checked = evaluate_performance(setting, current(), reference, [], now=NOW)
    assert checked["status"] == "unavailable"
    assert reason in checked["reason"]


def test_nonfinite_derived_degradation_is_unavailable_and_json_safe():
    """Finite metric inputs cannot create an infinite degradation in evidence."""
    import json

    checked = evaluate_performance(
        policy(metric="r2", tolerance=1e307),
        current(metric="r2", value=-1e308),
        baseline(metric="r2", value=1e308),
        [],
        now=NOW,
    )
    assert checked["status"] == "unavailable"
    assert "nonfinite" in checked["reason"]
    json.dumps(checked, allow_nan=False)


def test_within_tolerance_is_healthy_without_a_request():
    """A valid comparable improvement does not request retraining."""
    checked = evaluate_performance(policy(), current(value=0.91), baseline(), [], now=NOW)
    assert checked["status"] == "healthy"
    assert checked["action"] == "none"
    assert checked["consecutive_failures"] == 0


@pytest.mark.parametrize(
    ("observation", "reference", "reason"),
    [
        ({"labeled_rows": 9}, {}, "labeled_rows"),
        ({"label_coverage": 0.79}, {}, "label_coverage"),
        ({"value": float("inf")}, {}, "value"),
        ({"as_of": "2026-10-05T10:00:00Z"}, {}, "as_of"),
        ({"window_end": "2026-10-05T10:00:00Z"}, {}, "window"),
        ({"contract_digest": "d" * 64}, {}, "contract_digest"),
        ({}, {"model_version": "8"}, "model_version"),
    ],
)
def test_missing_stale_or_incomparable_evidence_is_unavailable(observation, reference, reason):
    """Bad evidence cannot be interpreted as healthy performance."""
    checked = evaluate_performance(
        policy(), current(**observation), baseline(**reference), [], now=NOW
    )
    assert checked["status"] == "unavailable"
    assert checked["action"] == "none"
    assert reason in checked["reason"]


def test_distinct_contiguous_history_and_replay_do_not_inflate_streak():
    """Only one matching result for each adjacent completed window advances streak."""
    setting = policy(consecutive_windows=2)
    first = evaluate_performance(
        setting,
        current(
            window_start="2026-10-05T09:00:00Z",
            window_end=START,
            as_of="2026-10-05T11:00:00Z",
        ),
        baseline(),
        [],
        now=datetime(2026, 10, 5, 11, 30, tzinfo=UTC),
    )
    checked = evaluate_performance(setting, current(), baseline(), [first, first], now=NOW)
    assert checked["status"] == "degraded"
    assert checked["action"] == "request_eligible"
    assert checked["consecutive_failures"] == 2
    assert len(checked["evaluated_windows"]) == 2
    replay = evaluate_performance(setting, current(), baseline(), [first, checked], now=NOW)
    assert replay["consecutive_failures"] == 2
    assert len(replay["evaluated_windows"]) == 2


def test_gap_and_policy_change_reset_streak():
    """A skipped window or changed policy cannot inherit earlier failures."""
    setting = policy(consecutive_windows=2)
    older = evaluate_performance(
        setting,
        current(
            window_start="2026-10-05T08:00:00Z",
            window_end="2026-10-05T09:00:00Z",
            as_of="2026-10-05T10:00:00Z",
        ),
        baseline(),
        [],
        now=datetime(2026, 10, 5, 10, 30, tzinfo=UTC),
    )
    checked = evaluate_performance(setting, current(), baseline(), [older], now=NOW)
    changed = evaluate_performance(
        policy(consecutive_windows=2, tolerance=0.04), current(), baseline(), [older], now=NOW
    )
    assert checked["consecutive_failures"] == 1
    assert changed["consecutive_failures"] == 1
    assert checked["action"] == changed["action"] == "report"


def test_contract_change_resets_adjacent_degraded_streak():
    """A changed label or population contract cannot inherit prior failures."""
    setting = policy(consecutive_windows=2)
    first = evaluate_performance(
        setting,
        current(
            window_start="2026-10-05T09:00:00Z",
            window_end=START,
            as_of="2026-10-05T11:00:00Z",
        ),
        baseline(),
        [],
        now=datetime(2026, 10, 5, 11, 30, tzinfo=UTC),
    )
    assert first["contract_digest"] == "c" * 64
    changed = evaluate_performance(
        setting,
        current(contract_digest="d" * 64),
        baseline(contract_digest="d" * 64),
        [first],
        now=NOW,
    )
    assert changed["status"] == "degraded"
    assert changed["contract_digest"] == "d" * 64
    assert changed["consecutive_failures"] == 1
    assert changed["action"] == "report"
