"""Relative regression error remains available and policy-compatible across frame engines."""

from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_performance_report,
)
from skyulf.integrations.mlflow.lifecycle.validation import (
    evaluate_quality_gates,
    validate_quality_policy,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("zero", [False, True])
def test_monitoring_wape_is_finite_or_explicitly_unavailable(engine, zero):
    """A missing WAPE denominator must preserve other metrics and an explicit null result."""
    frame = pd.DataFrame if engine == "pandas" else pl.DataFrame
    predictions = frame({"id": [1, 2, 3], "prediction": [5.0, 12.0, 18.0]})
    labels = frame(
        {
            "id": [1, 2, 3],
            "target": [0.0, 0.0, 0.0] if zero else [0.0, 10.0, 20.0],
            "available": [datetime(2026, 1, 1, tzinfo=UTC)] * 3,
        }
    )
    report = build_performance_report(
        predictions,
        labels,
        record_key_columns=("id",),
        target_column="target",
        result_available_at_column="available",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        task="regression",
    )
    assert report["values"]["wape"] == (None if zero else pytest.approx(0.3))
    assert report["values"]["mae"] == pytest.approx(35 / 3 if zero else 3.0)
    assert report["labeled_rows"] == 3
    assert bool(report["notes"]) is zero


def test_wape_quality_gate_minimizes_and_missing_metric_fails_closed():
    """Model selection must never promote an unavailable or excessive WAPE."""
    validate_quality_policy("heldout_wape", 0.4, task="regression")
    result = evaluate_quality_gates({"heldout_wape": 0.3}, "heldout_wape", 0.4)[0]
    assert result["direction"] == "minimize" and result["passed"]
    assert not evaluate_quality_gates({}, "heldout_wape", 0.4)[0]["passed"]
