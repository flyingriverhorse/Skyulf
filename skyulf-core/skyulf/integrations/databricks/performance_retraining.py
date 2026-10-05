"""Validate persisted performance eligibility before invoking shared training guards."""

import json
from datetime import UTC, datetime

from .monitoring_config import MonitorConfig, json_digest
from .performance_policy import completed_performance_window, validate_performance_policy


def performance_decision(row: dict, config: MonitorConfig, now: datetime) -> str:
    """Require the same policy, model version and latest mature window at submission time."""
    policy = validate_performance_policy(config.performance_policy)
    if not config.enabled or policy["mode"] != "retrain":
        return "disabled" if policy["mode"] == "off" else "report_only"
    if not _matches_enrollment(row, config):
        return "superseded"
    saved = json.loads(row["report_json"]).get("performance", {})
    if saved.get("policy_digest") != json_digest(policy):
        return "superseded_policy"
    start, end = completed_performance_window(now, policy)
    if (saved.get("window_start"), saved.get("window_end")) != (start.isoformat(), end.isoformat()):
        return "stale_performance"
    if not _fresh_cutoff(saved.get("as_of"), now, policy["window_hours"]):
        return "stale_performance"
    if saved.get("model_version") != config.model_version:
        return "superseded"
    if _eligible(saved):
        return "ready"
    return saved.get("reason", "performance_unavailable")


def _eligible(saved: dict) -> bool:
    """Keep measured degradation separate from a merely reported failing window."""
    return saved.get("status") == "degraded" and saved.get("action") == "request_eligible"


def _matches_enrollment(row: dict, config: MonitorConfig) -> bool:
    """Retain project ownership and the currently enrolled concrete model version."""
    return (
        row["config_digest"] == json_digest(config.payload())
        and row["model_version"] == config.model_version
        and row["monitor_id"] == config.monitor_id
    )


def _fresh_cutoff(value: str | None, now: datetime, hours: float) -> bool:
    """Never submit from future, naive, missing or aged label evidence."""
    if not value:
        return False
    cutoff = datetime.fromisoformat(value)
    if cutoff.tzinfo is None:
        return False
    return 0 <= (now - cutoff.astimezone(UTC)).total_seconds() <= hours * 3600
