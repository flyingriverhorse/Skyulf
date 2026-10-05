"""Validate version-bound performance policies and evaluate fixed UTC windows."""

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from numbers import Real
from typing import Any

_LOWER_METRICS = {"mae", "mse", "rmse", "mape"}
_HIGHER_METRICS = {
    "r2",
    "explained_variance",
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "matthews_corrcoef",
    "g_score",
}
_FIELDS = {
    "mode",
    "metric",
    "direction",
    "baseline",
    "tolerance",
    "tolerance_mode",
    "window_hours",
    "label_delay_hours",
    "minimum_labeled_rows",
    "minimum_label_coverage",
    "consecutive_windows",
}


def _finite_number(value: Any, name: str, *, minimum: float, inclusive: bool) -> float:
    """Admit a finite real number at the requested lower bound."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (not inclusive and value == minimum):
        raise ValueError(f"{name} must be {'at least' if inclusive else 'greater than'} {minimum}.")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    """Admit a non-boolean integer inside the configured range."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _baseline_reference(value: Any) -> dict:
    """Require a concrete reference identity before admitting an active policy."""
    if not isinstance(value, dict):
        raise ValueError("baseline must be an object.")
    kind = value.get("kind")
    if kind not in {"training_holdout", "production_window"}:
        raise ValueError("baseline.kind is unsupported.")
    allowed = {"kind", "model_version"}
    if kind == "production_window":
        allowed.add("report_id")
        report_id = value.get("report_id")
        if not _is_digest(report_id):
            raise ValueError("baseline.report_id must be a SHA-256 hex digest.")
    if set(value) != allowed:
        raise ValueError("baseline has missing or unknown fields.")
    version = value.get("model_version")
    if type(version) is not str or not re.fullmatch(r"[1-9][0-9]*", version):
        raise ValueError("baseline.model_version must be a concrete positive version string.")
    return deepcopy(value)


def _is_digest(value: Any) -> bool:
    """Recognize a complete SHA-256 hexadecimal identity."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_performance_policy(value: dict | None) -> dict:
    """Return an explicit, strict policy without altering its source."""
    if value is None or value == {}:
        return {"mode": "off"}
    if not isinstance(value, dict):
        raise ValueError("performance policy must be an object.")
    mode = value.get("mode")
    if mode == "off":
        if set(value) != {"mode"}:
            raise ValueError("off policy has unknown fields.")
        return {"mode": "off"}
    if mode not in {"report", "retrain"}:
        raise ValueError("mode must be off, report or retrain.")
    _validate_active_policy(value)
    return deepcopy(value)


def _validate_active_policy(value: dict) -> None:
    """Check active policy structure, reference, direction and numeric gates."""
    if set(value) != _FIELDS:
        raise ValueError("active policy has missing or unknown fields.")
    metric = value["metric"]
    if not isinstance(metric, str) or metric not in _LOWER_METRICS | _HIGHER_METRICS:
        raise ValueError("metric is unsupported by Core performance monitoring.")
    expected_direction = "lower" if metric in _LOWER_METRICS else "higher"
    if value["direction"] != expected_direction:
        raise ValueError(f"direction for {metric} must be {expected_direction}.")
    _baseline_reference(value["baseline"])
    _validate_active_numbers(value)


def _validate_active_numbers(value: dict) -> None:
    """Check all active numeric limits without silently coalescing booleans."""
    _finite_number(value["tolerance"], "tolerance", minimum=0, inclusive=False)
    if value["tolerance_mode"] not in {"absolute", "relative"}:
        raise ValueError("tolerance_mode must be absolute or relative.")
    _finite_number(value["window_hours"], "window_hours", minimum=0, inclusive=False)
    _finite_number(value["label_delay_hours"], "label_delay_hours", minimum=0, inclusive=True)
    _integer(value["minimum_labeled_rows"], "minimum_labeled_rows", minimum=2)
    coverage = _finite_number(
        value["minimum_label_coverage"], "minimum_label_coverage", minimum=0, inclusive=True
    )
    if coverage > 1:
        raise ValueError("minimum_label_coverage must be at most 1.")
    _integer(value["consecutive_windows"], "consecutive_windows", minimum=1, maximum=100)


def _utc(value: datetime, name: str) -> datetime:
    """Require an aware UTC datetime for deterministic epoch alignment."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime.")
    return value.astimezone(UTC)


def _timestamp(value: Any, name: str) -> datetime:
    """Parse an ISO UTC evidence timestamp without local-time inference."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp.") from exc
    return _utc(parsed, name)


def completed_performance_window(now: datetime, policy: dict) -> tuple[datetime, datetime]:
    """Return the last epoch-aligned window matured by the label delay."""
    now_utc = _utc(now, "now")
    setting = validate_performance_policy(policy)
    if setting["mode"] == "off":
        raise ValueError("active policy is required for completed window.")
    span_seconds = float(setting["window_hours"]) * 3600
    cutoff = now_utc.timestamp() - float(setting["label_delay_hours"]) * 3600
    end = datetime.fromtimestamp(math.floor(cutoff / span_seconds) * span_seconds, UTC)
    return end - timedelta(seconds=span_seconds), end


def _digest(policy: dict) -> str:
    """Bind streak history to the exact normalized policy."""
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _evidence_value(value: Any, name: str) -> float:
    """Reject nonfinite measurements instead of inferring a healthy value."""
    return _finite_number(value, name, minimum=-math.inf, inclusive=True)


def _quality(evidence: dict, policy: dict, name: str) -> None:
    """Require finite metric value and the configured label evidence gates."""
    _evidence_value(evidence.get("value"), f"{name}.value")
    rows = _integer(evidence.get("labeled_rows"), f"{name}.labeled_rows", minimum=2)
    if rows < policy["minimum_labeled_rows"]:
        raise ValueError(f"{name}.labeled_rows is below minimum_labeled_rows.")
    coverage = _finite_number(
        evidence.get("label_coverage"), f"{name}.label_coverage", minimum=0, inclusive=True
    )
    if coverage > 1 or coverage < policy["minimum_label_coverage"]:
        raise ValueError(f"{name}.label_coverage is outside minimum_label_coverage.")


def _comparable(current: dict, baseline: dict, policy: dict) -> None:
    """Require the selected version, metric and eligible-population contract."""
    for name, evidence in (("current", current), ("baseline", baseline)):
        if evidence.get("model_version") != policy["baseline"]["model_version"]:
            raise ValueError(f"{name}.model_version does not match policy baseline.")
        if evidence.get("metric") != policy["metric"]:
            raise ValueError(f"{name}.metric does not match policy metric.")
        if not _is_digest(evidence.get("contract_digest")):
            raise ValueError(f"{name}.contract_digest must be a SHA-256 digest.")
        _quality(evidence, policy, name)
    if current["contract_digest"] != baseline["contract_digest"]:
        raise ValueError("contract_digest differs between current and baseline.")
    if policy["baseline"]["kind"] == "production_window":
        _production_reference(current, baseline, policy)


def _production_reference(current: dict, baseline: dict, policy: dict) -> None:
    """Require a pinned, prior production window measured by the current cutoff."""
    if baseline.get("report_id") != policy["baseline"]["report_id"]:
        raise ValueError("baseline.report_id does not match policy reference.")
    start = _timestamp(baseline.get("window_start"), "baseline.window_start")
    end = _timestamp(baseline.get("window_end"), "baseline.window_end")
    current_start = _timestamp(current["window_start"], "current.window_start")
    if end - start != timedelta(hours=policy["window_hours"]) or end > current_start:
        raise ValueError("baseline window must be one completed prior policy window.")
    as_of = _timestamp(baseline.get("as_of"), "baseline.as_of")
    current_as_of = _timestamp(current["as_of"], "current.as_of")
    if not end <= as_of <= current_as_of:
        raise ValueError("baseline.as_of must follow baseline window and precede current.as_of.")


def _current_window(current: dict, policy: dict, now: datetime) -> tuple[datetime, datetime]:
    """Require the current complete window and fresh label cutoff."""
    start, end = completed_performance_window(now, policy)
    if _timestamp(current.get("window_start"), "current.window_start") != start:
        raise ValueError("current.window_start is not the completed window.")
    if _timestamp(current.get("window_end"), "current.window_end") != end:
        raise ValueError("current.window_end is not the completed window.")
    as_of = _timestamp(current.get("as_of"), "current.as_of")
    if as_of < end or as_of > now:
        raise ValueError("current.as_of must follow window end and not exceed now.")
    if now - as_of > timedelta(hours=policy["window_hours"]):
        raise ValueError("current.as_of is stale.")
    return start, end


def _window_identity(start: datetime, end: datetime) -> dict[str, str]:
    """Represent a counted window with portable UTC timestamps."""
    return {"window_start": start.isoformat(), "window_end": end.isoformat()}


def _prior_streak(
    history: list[dict],
    policy_digest: str,
    version: str,
    contract_digest: str,
    start: datetime,
    span: timedelta,
) -> list[dict]:
    """Collect only unique adjacent degraded windows from matching history."""
    by_end = _history_by_end(history, policy_digest, version, contract_digest, start, span)
    return _walk_streak(by_end, start, span)


def _history_by_end(
    history: list[dict],
    policy_digest: str,
    version: str,
    contract_digest: str,
    start: datetime,
    span: timedelta,
) -> dict[datetime, dict | None]:
    """Index prior matching windows, treating conflicting duplicates as a break."""
    by_end: dict[datetime, dict | None] = {}
    for item in history:
        if not isinstance(item, dict) or item.get("policy_digest") != policy_digest:
            continue
        if item.get("model_version") != version:
            continue
        if item.get("contract_digest") != contract_digest:
            continue
        try:
            item_start = _timestamp(item.get("window_start"), "history.window_start")
            item_end = _timestamp(item.get("window_end"), "history.window_end")
        except ValueError:
            continue
        if item_end > start or item_end - item_start != span:
            continue
        _insert_history(by_end, item_end, item)
    return by_end


def _insert_history(by_end: dict[datetime, dict | None], end: datetime, item: dict) -> None:
    """Keep exact replay duplicates once and invalidate conflicting copies."""
    if end in by_end and by_end[end] != item:
        by_end[end] = None
    elif end not in by_end:
        by_end[end] = item


def _walk_streak(
    by_end: dict[datetime, dict | None], start: datetime, span: timedelta
) -> list[dict]:
    """Walk backward until the first gap, conflict or non-degraded verdict."""
    windows = []
    cursor = start
    while (item := by_end.get(cursor)) is not None:
        if item.get("status") != "degraded":
            break
        previous = cursor - span
        if _timestamp(item["window_start"], "history.window_start") != previous:
            break
        windows.append(_window_identity(previous, cursor))
        cursor = previous
    return list(reversed(windows))


def _empty_result(setting: dict, current: dict, digest: str) -> dict:
    """Build stable report fields even when the policy or evidence is unavailable."""
    return {
        "status": "disabled" if setting["mode"] == "off" else "unavailable",
        "action": "none",
        "reason": "policy_off" if setting["mode"] == "off" else "missing_evidence",
        "mode": setting["mode"],
        "metric": setting.get("metric"),
        "direction": setting.get("direction"),
        "baseline_kind": setting.get("baseline", {}).get("kind"),
        "baseline_ref": deepcopy(setting.get("baseline")),
        "baseline_value": None,
        "current_value": None,
        "absolute_degradation": None,
        "relative_degradation": None,
        "tolerance": setting.get("tolerance"),
        "tolerance_mode": setting.get("tolerance_mode"),
        "threshold_value": None,
        "labeled_rows": _safe_number(current.get("labeled_rows"))
        if isinstance(current, dict)
        else None,
        "label_coverage": _safe_number(current.get("label_coverage"))
        if isinstance(current, dict)
        else None,
        "as_of": None,
        "window_start": None,
        "window_end": None,
        "evaluated_windows": [],
        "consecutive_failures": 0,
        "policy_digest": digest,
        "model_version": setting.get("baseline", {}).get("model_version"),
        "contract_digest": None,
    }


def _safe_number(value: Any) -> int | float | None:
    """Avoid emitting nonfinite untrusted label metadata into JSON evidence."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return value if math.isfinite(value) else None


def _measurement(result: dict, setting: dict, current: dict, baseline: dict) -> float | None:
    """Attach signed degradation and the configured absolute threshold."""
    reference = float(baseline["value"])
    observed = float(current["value"])
    degradation = reference - observed if setting["direction"] == "higher" else observed - reference
    if not math.isfinite(degradation):
        result["reason"] = "nonfinite_degradation"
        return None
    result.update(
        baseline_value=reference, current_value=observed, absolute_degradation=degradation
    )
    return _threshold(result, setting, reference, degradation)


def _threshold(result: dict, setting: dict, reference: float, degradation: float) -> float | None:
    """Attach finite relative change and boundary for the selected tolerance."""
    if reference == 0 and setting["tolerance_mode"] == "relative":
        result["reason"] = "zero_baseline_for_relative_tolerance"
        return None
    relative = degradation / abs(reference) if reference != 0 else None
    if relative is not None and not math.isfinite(relative):
        result["reason"] = "nonfinite_relative_degradation"
        return None
    result["relative_degradation"] = relative
    delta = (
        setting["tolerance"] * abs(reference)
        if setting["tolerance_mode"] == "relative"
        else setting["tolerance"]
    )
    threshold = reference - delta if setting["direction"] == "higher" else reference + delta
    if not math.isfinite(delta) or not math.isfinite(threshold):
        result["reason"] = "nonfinite_threshold"
        return None
    result["threshold_value"] = threshold
    return delta


def _degraded_result(
    result: dict, setting: dict, history: list[dict], start: datetime, end: datetime
) -> dict:
    """Attach consecutive-window evidence and select reporting or request eligibility."""
    result["status"] = "degraded"
    result["reason"] = "tolerance_breached"
    prior = _prior_streak(
        history,
        result["policy_digest"],
        setting["baseline"]["model_version"],
        result["contract_digest"],
        start,
        end - start,
    )
    result["evaluated_windows"] = [*prior, _window_identity(start, end)]
    result["consecutive_failures"] = len(result["evaluated_windows"])
    if (
        setting["mode"] == "report"
        or result["consecutive_failures"] < setting["consecutive_windows"]
    ):
        result["action"] = "report"
        if setting["mode"] == "retrain":
            result["reason"] = "awaiting_consecutive_windows"
    else:
        result["action"] = "request_eligible"
    return result


def evaluate_performance(
    policy: dict,
    current: dict,
    baseline: dict | None,
    history: list[dict],
    *,
    now: datetime,
) -> dict:
    """Decide degradation and eligibility from comparable, completed evidence."""
    setting = validate_performance_policy(policy)
    now_utc = _utc(now, "now")
    digest = _digest(setting)
    result = _empty_result(setting, current, digest)
    if setting["mode"] == "off":
        return result
    start, end = completed_performance_window(now_utc, setting)
    result.update(_window_identity(start, end))
    if not isinstance(current, dict) or not isinstance(baseline, dict):
        return result
    try:
        _current_window(current, setting, now_utc)
        _comparable(current, baseline, setting)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    result["as_of"] = current["as_of"]
    result["contract_digest"] = current["contract_digest"]
    delta = _measurement(result, setting, current, baseline)
    if delta is None:
        return result
    if result["absolute_degradation"] < delta and not math.isclose(
        result["absolute_degradation"], delta, rel_tol=1e-12, abs_tol=0
    ):
        result["status"] = "healthy"
        result["reason"] = "within_tolerance"
        result["evaluated_windows"] = [_window_identity(start, end)]
        return result
    return _degraded_result(result, setting, history, start, end)
