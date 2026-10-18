"""Spark monitoring treats configured missing markers exactly like real nulls."""

from datetime import UTC, datetime

import pytest

AS_OF = datetime(2026, 10, 17, tzinfo=UTC)
SCHEMA = "record_id long, co_gt double, hour long, day_part string"


def _rows(marker, start, count, step):
    """Sensor rows where every step-th co_gt and hour reading is the marker or a null."""
    return [
        (
            start + i,
            marker if i % step == 0 else 1.0 + i / 10,
            None if marker is None and i % step == 0 else (-1 if i % step == 0 else i % 24),
            "night",
        )
        for i in range(count)
    ]


def _report(spark, marker, thresholds, columns=("co_gt", "hour")):
    """Build the Spark report for one reference and one fully predicted current population."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_quality import (
        monitoring_report,
    )

    reference = spark.createDataFrame(_rows(marker, 0, 40, 10), SCHEMA)
    current = spark.createDataFrame(_rows(marker, 100, 30, 3), SCHEMA)
    predictions = current.selectExpr("record_id", "1.0 AS prediction")
    return monitoring_report(
        reference,
        current,
        predictions,
        None,
        feature_columns=columns,
        record_key_columns=("record_id",),
        target_column="no2_gt",
        result_available_at_column="result_available_at",
        as_of=AS_OF,
        task="regression",
        classes=(),
        thresholds=thresholds,
    )


def test_spark_markers_measure_exactly_like_real_nulls(spark):
    """Double and long columns must report marked rows as missing, with unchanged dtypes."""
    marked = _report(spark, -200.0, {"missing_markers": {"co_gt": [-200], "hour": [-1]}})
    nulled = _report(spark, None, None)
    assert marked["metrics"] == nulled["metrics"]
    missing = {
        item["column_name"]: item
        for item in marked["metrics"]
        if item["metric_name"] == "missing_fraction"
    }
    assert missing["co_gt"]["value"] == pytest.approx(10 / 30)
    assert missing["hour"]["threshold"] == pytest.approx(4 / 40)


def test_spark_markers_reject_text_columns(spark):
    """A marker on a text feature must fail instead of comparing strings to numbers."""
    with pytest.raises(ValueError, match="floating-point"):
        _report(
            spark,
            -200.0,
            {"missing_markers": {"day_part": [-200]}},
            columns=("co_gt", "hour", "day_part"),
        )


BIG = 2**53


def _single_column(spark, values, spark_type, markers):
    """Report one Spark feature column whose reference and current rows are identical."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_quality import (
        monitoring_report,
    )

    frame = spark.createDataFrame(list(enumerate(values)), f"record_id long, x {spark_type}")
    report = monitoring_report(
        frame,
        frame,
        frame.selectExpr("record_id", "1.0 AS prediction"),
        None,
        feature_columns=("x",),
        record_key_columns=("record_id",),
        target_column="y",
        result_available_at_column="result_available_at",
        as_of=AS_OF,
        task="regression",
        classes=(),
        thresholds={"missing_markers": {"x": markers}},
    )
    return next(
        item["value"] for item in report["metrics"] if item["metric_name"] == "missing_fraction"
    )


def test_spark_large_long_marker_never_matches_its_float_neighbour(spark):
    """2**53 + 1 must null only itself in a long column, as in the in-memory engine."""
    assert _single_column(spark, [BIG, BIG + 1, BIG + 1, 7], "long", [BIG + 1]) == 0.5


@pytest.mark.parametrize(
    ("values", "spark_type", "markers", "message"),
    [
        ([1, 2], "long", [1.5], "whole numbers within Int64"),
        ([1, 2], "tinyint", [-200], "whole numbers within Int8"),
        ([1.0, 2.0], "double", [BIG + 1], "no exact Float64 value"),
        ([1.0, 2.0], "float", [2**24 + 1], "no exact Float32 value"),
        ([1.0, 2.0], "float", [0.1], "no exact Float32 value"),
    ],
)
def test_spark_markers_without_an_exact_column_value_fail(
    spark, values, spark_type, markers, message
):
    """A marker that cannot equal a stored value exactly must fail, not match approximately."""
    with pytest.raises(ValueError, match=message):
        _single_column(spark, values, spark_type, markers)


def test_spark_markers_reject_decimal_columns(spark):
    """Decimal columns are not compared, matching the in-memory engine."""
    from decimal import Decimal

    with pytest.raises(ValueError, match="integer or floating-point"):
        _single_column(spark, [Decimal("1.50"), Decimal("2.50")], "decimal(10,2)", [-200])


def test_spark_exact_float32_marker_nulls_only_itself(spark):
    """-200 and 2**24 are exact in FloatType and must not touch the neighbouring 2**24 + 2."""
    values = [-200.0, 2.0**24, 2.0**24 + 2, 5.0]
    assert _single_column(spark, values, "float", [-200, 2**24]) == 0.5
