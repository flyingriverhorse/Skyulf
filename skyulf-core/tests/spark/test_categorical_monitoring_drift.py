"""Native Spark and in-memory monitoring share bounded categorical decisions."""

import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    _drift_evidence,
)
from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
    drift_evidence,
)


@pytest.mark.parametrize(
    "scenario", ["stable", "shifted", "novel", "ids", "boundary", "nulls", "low"]
)
def test_categorical_spark_core_parity(spark, scenario):
    """The engine must not change PSI, novelty, evidence, warnings or drift decisions."""
    reference = [f"zip{i:03}" for i in range(90)] * 20
    current = reference
    if scenario == "shifted":
        current = ["zip000"] * 1800
    elif scenario == "novel":
        current = ["zip089"] * 75 + ["new_zip"] * 25 + [None] * 50
    elif scenario == "ids":
        reference = [f"id{i:03}" for i in range(90)]
        current = reference
    elif scenario == "boundary":
        reference = [f"zip{i:03}" for i in range(51)] * 2
        current = reference
    elif scenario == "nulls":
        current = [None] * 20
    elif scenario == "low":
        reference = ["a"] * 90 + ["b"] * 10
        current = ["a"] * 10 + ["b"] * 90
    expected = _drift_evidence(
        pl.DataFrame({"category": pl.Series(reference, dtype=pl.String)}),
        pl.DataFrame({"category": pl.Series(current, dtype=pl.String)}),
        ("category",),
        None,
    )
    actual = drift_evidence(
        spark.createDataFrame([(value,) for value in reference], "category string"),
        spark.createDataFrame([(value,) for value in current], "category string"),
        ("category",),
        None,
    )
    assert actual[0] == [item for item in expected[0] if item["category"] == "drift"]
    assert actual[1] == expected[1]
    assert actual[3] == expected[3]
    assert bool(actual[2]) == bool(expected[2])


@pytest.mark.parametrize("side", ["reference", "current"])
def test_spark_category_summary_cap(spark, side):
    """Collecting aggregated counts must never silently truncate a large domain."""
    small = [f"cat{i:05}" for i in range(90)] * 20
    large = [f"cat{i:05}" for i in range(10001)] * 2
    reference, current = (large, small) if side == "reference" else (small, large)
    metrics, drifted, notes, unmeasured = drift_evidence(
        spark.createDataFrame([(value,) for value in reference], "category string"),
        spark.createDataFrame([(value,) for value in current], "category string"),
        ("category",),
        None,
    )
    assert [item["metric_name"] for item in metrics] == ["unavailable"]
    assert drifted == 0
    assert unmeasured and notes


@pytest.mark.parametrize(
    "settings",
    [
        {"categorical_top_k": 20, "id_like_uniqueness": 1.0},
        {"categorical_top_k": 100},
        {"categorical_top_k": 10, "max_category_summaries": 80},
    ],
)
def test_configured_category_limits_match_core(spark, settings):
    """Overrides must reach native Spark grouping, exact comparison and explicit unavailable results."""
    reference = [f"z{i:03}" for i in range(90)] * 20
    current = ["z089"] * 100 + ["unseen"] * 20
    expected = _drift_evidence(
        pl.DataFrame({"category": reference}),
        pl.DataFrame({"category": current}),
        ("category",),
        settings,
    )
    actual = drift_evidence(
        spark.createDataFrame([(value,) for value in reference], "category string"),
        spark.createDataFrame([(value,) for value in current], "category string"),
        ("category",),
        settings,
    )
    assert actual[0] == [item for item in expected[0] if item["category"] == "drift"]
    assert actual[1] == expected[1] and actual[3] == expected[3]
