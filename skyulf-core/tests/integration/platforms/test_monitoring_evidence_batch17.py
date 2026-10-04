"""Monitoring must not turn descriptive distances or insufficient evidence into retraining."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.monitoring_config import MonitorConfig, json_digest
from skyulf.integrations.databricks.monitoring_metrics import build_monitoring_report
from skyulf.integrations.databricks.retraining_task import observation_decision


def measured(engine, reference, current):
    """Build public monitoring evidence using valid saved predictions for each current record."""
    factory = pd.DataFrame if engine == "pandas" else pl.DataFrame
    size = len(next(iter(current.values())))
    return build_monitoring_report(
        factory(reference),
        factory({"id": list(range(size)), **current}),
        factory({"id": list(range(size)), "prediction": [0.0] * size}),
        None,
        feature_columns=tuple(reference),
        record_key_columns=("id",),
        target_column="target",
        result_available_at_column="available",
        as_of=datetime.now(UTC),
        task="regression",
    )


def retraining(report):
    """Ask the real automation admission path about a fresh identity-matched report."""
    config = MonitorConfig(
        environment="test",
        project="batch17",
        model_name="cat.models.model",
        model_version="1",
        source_table="cat.data.source",
        prediction_table="cat.data.output",
    )
    now = datetime.now(UTC)
    row = {
        "config_digest": json_digest(config.payload()),
        "model_version": "1",
        "monitor_id": config.monitor_id,
        "status": report["status"],
        "observed_at": now,
        "report_json": json.dumps(report),
    }
    return observation_decision(row, config, now)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_tiny_monitoring_evidence_is_explicit_and_cannot_retrain(engine):
    """Three observations cannot make even complete separation statistically significant."""
    result = measured(engine, {"x": [0.0, 1.0, 2.0]}, {"x": [100.0, 101.0, 102.0]})
    assert result["status"] == "degraded" and result["drifted_columns"] == 0
    evidence = [
        item
        for item in result["metrics"]
        if item["category"] == "drift" and item["metric_name"] == "statistical_evidence"
    ]
    assert len(evidence) == 1 and evidence[0]["status"] == "unavailable"
    assert evidence[0]["evidence"]["status"] == "insufficient_data"
    assert evidence[0]["evidence"]["reference_count"] == 3
    assert evidence[0]["evidence"]["current_count"] == 3
    assert evidence[0]["evidence"]["reason"]
    assert any("insufficient" in note.lower() for note in result["notes"])
    assert not any(item["has_issue"] for item in result["metrics"] if item["category"] == "drift")
    assert retraining(result) == "incomplete_drift"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_effect_without_statistical_support_has_no_monitoring_issue(engine):
    """A nonsignificant comparison must not reconstitute Core's former raw-distance OR alarm."""
    values = np.arange(20, dtype=float)
    result = measured(
        engine, {"x": values, "other": values}, {"x": values + 9, "other": values + 9}
    )
    assert result["drifted_columns"] == 0
    assert not any(item["has_issue"] for item in result["metrics"] if item["category"] == "drift")
    assert retraining(result) != "ready"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_one_insufficient_feature_blocks_retraining_despite_another_real_shift(engine):
    """Automation cannot train from an incomplete feature-evidence report."""
    current = {"strong": list(range(100, 164)), "sparse": [1.0, 2.0] + [None] * 62}
    reference = {"strong": list(range(64)), "sparse": [1.0, 2.0] + [None] * 62}
    result = measured(engine, reference, current)
    assert result["status"] == "drift" and result["drifted_columns"] == 1
    assert retraining(result) == "incomplete_drift"


def test_supported_shift_retains_diagnostic_ks_and_automation_admission():
    """A fully measured supported shift remains actionable while the raw p-value row stays diagnostic."""
    result = measured("pandas", {"x": list(range(64))}, {"x": list(range(100, 164))})
    p_value = next(item for item in result["metrics"] if item["metric_name"] == "ks_test_p_value")
    assert p_value["threshold"] is None and p_value["has_issue"] is False
    assert result["drifted_columns"] == 1 and retraining(result) == "ready"


@pytest.mark.parametrize(
    "case",
    [
        "legacy",
        "missing_payload",
        "duplicate",
        "wrong_column",
        "unknown_test",
        "wrong_alpha",
        "bad_probability",
        "nan_probability",
        "boolean_probability",
        "unsupported_status",
    ],
)
def test_legacy_or_malformed_saved_evidence_cannot_authorize_retraining(case):
    """Upgrading the decision policy must not leave old OR-only reports or invalid metadata actionable."""
    result = measured("pandas", {"x": list(range(64))}, {"x": list(range(100, 164))})
    statistic = next(
        item for item in result["metrics"] if item["metric_name"] == "statistical_evidence"
    )
    if case == "legacy":
        result["metrics"].remove(statistic)
    elif case == "missing_payload":
        statistic.pop("evidence")
    elif case == "duplicate":
        result["metrics"].append(dict(statistic))
    elif case == "wrong_column":
        statistic["column_name"] = "other"
    else:
        field, value = {
            "unknown_test": ("test", "invented"),
            "wrong_alpha": ("significance_level", 0.5),
            "bad_probability": ("adjusted_p_value", 0.8),
            "nan_probability": ("adjusted_p_value", float("nan")),
            "boolean_probability": ("adjusted_p_value", True),
            "unsupported_status": ("status", "not_detected"),
        }[case]
        statistic["evidence"][field] = value
    assert retraining(result) == "incomplete_drift"


def test_saved_issue_flags_need_their_own_supported_feature_evidence():
    """Another feature's significant change cannot validate an unsupported feature's historical issue flag."""
    result = measured(
        "pandas",
        {"x": list(range(64)), "stable": list(range(64))},
        {"x": list(range(100, 164)), "stable": list(range(64))},
    )
    metric = next(
        item
        for item in result["metrics"]
        if item["category"] == "drift"
        and item["column_name"] == "stable"
        and item["metric_name"] == "psi"
    )
    metric["has_issue"] = True
    assert retraining(result) == "incomplete_drift"
