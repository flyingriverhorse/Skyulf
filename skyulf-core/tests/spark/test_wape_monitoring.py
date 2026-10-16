"""Spark regression monitoring preserves Core's WAPE denominator and null contract."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_performance_report,
)
from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
    build_spark_performance_report,
)


@pytest.mark.parametrize("truth", [[0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [-10.0, 0.0, 20.0]])
def test_spark_wape_matches_core_for_signed_and_zero_actuals(spark, truth):
    """Distributed aggregates must agree with the in-memory evaluator without collecting raw rows."""
    predictions = pd.DataFrame({"id": [1, 2, 3], "prediction": [5.0, 12.0, 18.0]})
    labels = pd.DataFrame(
        {"id": [1, 2, 3], "target": truth, "available": [datetime(2026, 1, 1, tzinfo=UTC)] * 3}
    )
    kwargs: dict[str, Any] = {
        "record_key_columns": ("id",),
        "target_column": "target",
        "result_available_at_column": "available",
        "as_of": datetime(2026, 1, 2, tzinfo=UTC),
        "task": "regression",
    }
    expected = build_performance_report(predictions, labels, **kwargs)
    actual = build_spark_performance_report(
        spark.createDataFrame(predictions), spark.createDataFrame(labels), **kwargs
    )
    assert actual["values"] == pytest.approx(expected["values"])
    assert bool(actual["notes"]) == bool(expected["notes"])
    assert actual["labeled_rows"] == 3
