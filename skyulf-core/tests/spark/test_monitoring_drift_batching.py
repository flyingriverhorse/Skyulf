"""Shared drift summaries preserve exact outputs without per-feature preflight scans."""

import hashlib
import json
import time
from decimal import Decimal

import pytest

from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
    drift_evidence,
)


def _measure(reference, current, columns, monkeypatch, record_property):
    """Count real terminal actions and save a lossless fingerprint for wheel comparisons."""
    actions = []
    depth = 0

    def tracked(original, name):
        """Avoid double-counting a terminal method that delegates to another method."""

        def execute(frame, *args, **kwargs):
            """Execute the original Spark action while recording the outermost call."""
            nonlocal depth
            if not depth:
                actions.append(name)
            depth += 1
            try:
                return original(frame, *args, **kwargs)
            finally:
                depth -= 1

        return execute

    started = time.monotonic()
    with monkeypatch.context() as patch:
        for name in ("collect", "count", "first"):
            original = getattr(type(reference), name)
            patch.setattr(type(reference), name, tracked(original, name))
        result = drift_evidence(reference, current, columns, None)
    serialized = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    record_property("drift_seconds", time.monotonic() - started)
    record_property("spark_actions", len(actions))
    record_property("result_sha256", hashlib.sha256(serialized).hexdigest())
    return result, actions


@pytest.mark.parametrize("width", [1, 8])
def test_numeric_drift_batches_preflight_scans(spark, monkeypatch, record_property, width):
    """Adding numeric features must add distribution work, not eight scalar scans each."""
    columns = tuple(f"feature_{index}" for index in range(width))
    schema = ", ".join(f"{column} double" for column in columns)
    reference = spark.createDataFrame(
        [tuple(float(row + index) for index in range(width)) for row in range(20)], schema
    ).repartition(3)
    current = spark.createDataFrame(
        [tuple(float(row + index + 30) for index in range(width)) for row in range(20)], schema
    ).repartition(3)
    (metrics, drifted, notes, unavailable), actions = _measure(
        reference, current, columns, monkeypatch, record_property
    )
    assert drifted == width and not notes and not unavailable
    assert sum(item["metric_name"] == "psi" for item in metrics) == width
    # Three shared summary actions; CDF, bounded levels, quantiles and two histograms
    # remain per feature. This intentionally does not impose a wall-clock deadline.
    assert 0 < len(actions) <= 3 + 5 * width, actions


def test_mixed_drift_retains_type_and_no_data_precedence(spark, monkeypatch, record_property):
    """Batching must not turn null, infinity or failed string conversion into valid drift."""
    columns = ("coerced", "bad", "nan", "infinity", "empty", "flag", "a.b`c", "missing")
    reference = spark.createDataFrame(
        [(float(i), float(i), float(i), float(i), None, bool(i % 2), float(i)) for i in range(12)],
        "coerced double, bad double, nan double, infinity double, empty double, "
        "flag boolean, `a.b``c` double",
    )
    current = spark.createDataFrame(
        [
            (
                str(i),
                "bad" if i == 0 else str(i),
                float("nan"),
                float("inf"),
                None,
                "TRUE" if i % 2 else "false",
                float(i),
            )
            for i in range(12)
        ],
        "coerced string, bad string, nan double, infinity double, empty string, "
        "flag string, `a.b``c` double",
    )
    (metrics, drifted, notes, unavailable), _ = _measure(
        reference, current, columns, monkeypatch, record_property
    )
    by_column = {}
    for item in metrics:
        by_column.setdefault(item["column_name"], {})[item["metric_name"]] = item["value"]
    assert by_column["bad"]["type_drift"] == 1.0
    assert by_column["missing"]["schema_missing"] == 1.0
    assert by_column["coerced"]["ks_statistic"] == 0.0
    assert by_column["a.b`c"]["wasserstein_distance"] == 0.0
    assert by_column["flag"]["psi_categorical"] == 0.0
    assert not any("type_drift" in by_column.get(column, {}) for column in ("empty", "nan"))
    assert drifted == 2 and unavailable and len(notes) == 4


def test_batched_drift_preserves_native_numeric_precision(spark, monkeypatch, record_property):
    """Centering before double conversion must preserve adjacent int64 and float32 values."""
    columns = ("large", "small", "decimal", "nullable")
    schema = "large long, small float, decimal decimal(12,3), nullable double"
    reference = spark.createDataFrame(
        [(2**63 - 10 + i, 1.0 + i * 2**-23, Decimal(i) / 1000, float(i)) for i in range(8)]
        + [(None, None, None, None), (None, None, None, float("nan"))],
        schema,
    )
    current = spark.createDataFrame(
        [(2**63 - 3, 1.0 + 7 * 2**-23, Decimal("0.007"), 7.0)] * 8, schema
    )
    (metrics, drifted, notes, unavailable), _ = _measure(
        reference, current, columns, monkeypatch, record_property
    )
    distances = [item["value"] for item in metrics if item["metric_name"] == "wasserstein_distance"]
    assert len(distances) == 4 and distances == pytest.approx([3.5 / (5.25**0.5)] * 4)
    assert drifted == 4 and not notes and not unavailable


@pytest.mark.parametrize("empty_side", ["reference", "current", "both"])
def test_empty_drift_population_stays_unavailable(spark, monkeypatch, record_property, empty_side):
    """Empty aggregate outputs must keep no-data precedence ahead of inferred type drift."""
    reference = spark.createDataFrame(
        [] if empty_side in {"reference", "both"} else [(None,), (1.0,)], "value double"
    )
    current = spark.createDataFrame(
        [] if empty_side in {"current", "both"} else [(None,), ("bad",)], "value string"
    )
    (metrics, drifted, notes, unavailable), _ = _measure(
        reference, current, ("value",), monkeypatch, record_property
    )
    assert drifted == 0 and unavailable and len(notes) == 1
    assert not any(item["metric_name"] == "type_drift" for item in metrics)


@pytest.mark.parametrize("columns", [(), ("missing",)])
def test_no_comparable_columns_needs_no_spark_action(spark, monkeypatch, record_property, columns):
    """Absent features should produce schema evidence without executing a meaningless aggregate."""
    frame = spark.createDataFrame([(1.0,)], "value double")
    (metrics, drifted, notes, unavailable), actions = _measure(
        frame, frame, columns, monkeypatch, record_property
    )
    assert len(metrics) == drifted == len(notes) == len(columns)
    assert not unavailable and not actions


def test_partitioned_nulls_preserve_centered_deviation(spark, monkeypatch, record_property):
    """Null CASE rows in different partitions must not affect finite counts or stable scaling."""
    import math

    import numpy as np
    from scipy.stats import wasserstein_distance

    values = [
        None if i % 5 == 0 else float("nan") if i % 7 == 0 else float(10**12 + i) for i in range(60)
    ]
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    actual = np.arange(10**12 + 80, 10**12 + 100, dtype=float)
    reference = spark.createDataFrame([(value,) for value in values], "value double").repartition(
        3, "value"
    )
    current = spark.createDataFrame([(float(value),) for value in actual], "value double")
    (metrics, drifted, notes, unavailable), _ = _measure(
        reference, current, ("value",), monkeypatch, record_property
    )
    by_name = {item["metric_name"]: item for item in metrics}
    evidence = by_name["statistical_evidence"]["evidence"]
    assert evidence["reference_count"] == len(finite)
    assert evidence["current_count"] == len(actual)
    expected = wasserstein_distance(finite, actual) / np.std(finite - finite.min())
    assert by_name["wasserstein_distance"]["value"] == pytest.approx(expected, rel=1e-14)
    assert drifted == 1 and not notes and not unavailable


def test_invalid_current_string_precedes_reference_infinity(spark, monkeypatch, record_property):
    """A failed string conversion remains type drift even when reference data is infinite."""
    reference = spark.createDataFrame([(float("inf"),), (1.0,)], "value double")
    current = spark.createDataFrame([("bad",), ("1",)], "value string")
    (metrics, drifted, notes, unavailable), _ = _measure(
        reference, current, ("value",), monkeypatch, record_property
    )
    assert len(metrics) == 1 and metrics[0]["metric_name"] == "type_drift"
    assert metrics[0]["value"] == 1.0 and metrics[0]["has_issue"]
    assert drifted == 1 and not notes and not unavailable
