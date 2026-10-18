"""One strict settings contract controls Core, Spark and bounded monitoring enrollment."""

import json

import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
    MonitorConfig,
    parse_drift_thresholds,
)
from skyulf.profiling.drift import DriftCalculator


@pytest.mark.parametrize(
    "settings",
    [
        {"categorical_top_k": True},
        {"categorical_top_k": 1.5},
        {"categorical_top_k": 0},
        {"id_like_uniqueness": 0},
        {"id_like_uniqueness": 1.1},
        {"id_like_uniqueness": float("nan")},
        {"max_category_summaries": 0},
        {"categorical_top_k": 60, "max_category_summaries": 59},
        {"psi_moderate": 0.3},
        {"psi_critical": 0.1},
        {"psi_moderate": -1},
        {"monitor_max_rows": True},
        {"monitor_max_rows": 1_000_001},
        {"monitor_max_bytes": 0},
        {"monitor_max_bytes": 1024**3 + 1},
    ],
)
def test_invalid_monitoring_options_fail_at_core_and_bundle_boundaries(settings):
    """Invalid policy must fail before data access rather than becoming a skipped comparison."""
    frame = pl.DataFrame({"x": [0.0, 1.0]})
    with pytest.raises(ValueError):
        DriftCalculator(frame, frame).calculate_drift(settings)
    with pytest.raises(ValueError):
        parse_drift_thresholds(json.dumps(settings))


def test_grouping_and_uniqueness_overrides_are_effective():
    """Configured top K and uniqueness must change actual admission and recorded test identity."""
    frame = pl.DataFrame({"zip": [f"z{i:03}" for i in range(90)]})
    settings = {"categorical_top_k": 20, "id_like_uniqueness": 1.0}
    assert parse_drift_thresholds(json.dumps(settings)) == settings
    report = DriftCalculator(frame, frame).calculate_drift(settings)
    assert report.column_drifts["zip"].evidence is not None
    assert report.column_drifts["zip"].evidence.test.endswith("_top20_other")
    assert not DriftCalculator(frame, frame).calculate_drift().column_drifts


def test_summary_cap_applies_to_both_sides_when_explicit():
    """An explicit collection cap must also protect a small reference with a large current domain."""
    reference = pl.DataFrame({"zip": ["a", "b"] * 50})
    current = pl.DataFrame({"zip": [f"z{i:03}" for i in range(90)] * 2})
    assert (
        not DriftCalculator(reference, current)
        .calculate_drift({"max_category_summaries": 60})
        .column_drifts
    )


def test_severity_bands_do_not_replace_the_psi_alarm_threshold():
    """Suggestions use configured bands while the separate PSI threshold governs the effect flag."""
    reference = pl.DataFrame({"zip": ["a"] * 90 + ["b"] * 10})
    current = pl.DataFrame({"zip": ["a"] * 10 + ["b"] * 90})
    result = (
        DriftCalculator(reference, current)
        .calculate_drift({"psi_moderate": 1.0, "psi_critical": 4.0})
        .column_drifts["zip"]
    )
    assert result.drift_detected and result.metrics[0].threshold == 0.2
    assert result.suggestions == [
        "Moderate category-distribution shift detected. Monitor model performance closely."
    ]


def test_explicit_local_budgets_override_without_changing_absent_payloads():
    """Larger bounded driver budgets require explicit enrollment, leaving old defaults intact."""
    settings = {
        "environment": "dev",
        "project": "demo",
        "model_name": "workspace.test.model",
        "source_table": "workspace.test.source",
        "prediction_table": "workspace.test.predictions",
        "model_version": "1",
    }
    original = MonitorConfig.from_dict(settings)
    assert original.max_rows == 10_000 and original.max_bytes == 64 * 1024**2
    config = MonitorConfig.from_dict(
        settings | {"thresholds": {"monitor_max_rows": 200_000, "monitor_max_bytes": 256 * 1024**2}}
    )
    assert config.max_rows == 200_000 and config.max_bytes == 256 * 1024**2
    assert original.payload()["thresholds"] is None
