"""Monitoring distinguishes diagnostic KS p-values from decision statistics."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_monitoring_report,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("shift", [0.0, 100.0])
def test_monitoring_ks_p_value_is_diagnostic(engine, shift):
    """A p-value row must not display the KS-statistic threshold or an independent alarm."""
    factory = pd.DataFrame if engine == "pandas" else pl.DataFrame
    reference = factory({"x": np.arange(128, dtype=float)})
    current = factory({"id": range(128), "x": np.arange(128, dtype=float) + shift})
    predictions = factory({"id": range(128), "prediction": np.zeros(128)})
    report = build_monitoring_report(
        reference,
        current,
        predictions,
        None,
        feature_columns=("x",),
        record_key_columns=("id",),
        target_column="target",
        result_available_at_column="available",
        as_of=datetime.now(UTC),
        task="regression",
    )
    metrics = {item["metric_name"]: item for item in report["metrics"]}
    statistic, diagnostic = metrics["ks_statistic"], metrics["ks_test_p_value"]
    assert diagnostic["status"] == "measured"
    assert 0 <= diagnostic["value"] <= 1
    assert diagnostic["threshold"] is None
    assert diagnostic["has_issue"] is False
    assert statistic["threshold"] == 0.1
    assert statistic["has_issue"] is bool(shift)
    assert report["drifted_columns"] == int(bool(shift))
