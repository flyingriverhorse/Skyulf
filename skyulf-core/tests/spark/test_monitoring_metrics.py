"""Distributed monitoring preserves saved-output and eligible-label semantics."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_performance_report,
)


def test_minimum_long_key_is_finite(spark):
    """Valid minimum signed integer keys must not overflow an absolute-value check."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame(
        [(-9223372036854775808, 1.0), (1, 2.0)], "id long, prediction double"
    )
    report = build_spark_performance_report(predictions, None, task="regression", **ARGS)
    assert report["scored_rows"] == 2


def test_excluded_null_probabilities_are_not_validated_as_predictions(spark):
    """Excluded rows may have empty outputs and must remain visible as excluded counts."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame(
        [(1, None, "excluded", None, None)],
        "id long, prediction string, scoring_status string, probability_0 void, probability_1 void",
    )
    report = build_spark_performance_report(
        predictions, None, task="classification", classes=("a", "b"), **ARGS
    )
    assert report["excluded_rows"] == 1
    assert report["scored_rows"] == 0


def test_future_labels_do_not_require_eligible_record_keys(spark):
    """An entirely future label population cannot become a malformed current join."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame([(1, 1.0), (2, 2.0)], "id long, prediction double")
    labels = spark.createDataFrame([(1.0, "2099-01-01T00:00:00Z")], "y double, available string")
    report = build_spark_performance_report(predictions, labels, task="regression", **ARGS)
    assert report["labeled_rows"] == 0


NOW = datetime(2026, 1, 2, tzinfo=UTC)
ARGS: dict[str, Any] = {
    "record_key_columns": ("id",),
    "target_column": "y",
    "result_available_at_column": "available",
    "as_of": NOW,
}


def test_metric_kernel_available():
    """Distributed monitoring must expose its public entry point without a JVM."""
    import importlib

    module = importlib.import_module(
        "skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics"
    )
    assert callable(module.build_spark_performance_report)


@pytest.mark.parametrize(
    "truth,guesses", [([1.0, 3.0, 5.0], [2.0, 3.0, 4.0]), ([2.0, 2.0, 2.0], [3.0, 3.0, 3.0])]
)
def test_regression_parity(spark, truth, guesses):
    """Constant outcomes retain sklearn force-finite explained variance and R2."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = pd.DataFrame({"id": range(3), "prediction": guesses})
    labels = pd.DataFrame({"id": range(3), "y": truth, "available": [NOW.isoformat()] * 3})
    expected = build_performance_report(predictions, labels, task="regression", **ARGS)
    result = build_spark_performance_report(
        spark.createDataFrame(predictions), spark.createDataFrame(labels), task="regression", **ARGS
    )
    assert result["values"] == pytest.approx(expected["values"])
    assert result["labeled_rows"] == 3


def test_class_order_and_eligibility(spark):
    """Future duplicate outcomes and excluded predictions never enter coverage."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prediction": ["a", "z", None],
            "scoring_status": ["predicted", "predicted", "excluded"],
        }
    )
    labels = pd.DataFrame(
        {
            "id": [1, 2, 2],
            "y": ["a", "a", "z"],
            "available": [NOW.isoformat(), NOW.isoformat(), "2027-01-01T00:00:00Z"],
        }
    )
    expected = build_performance_report(
        predictions, labels, task="classification", classes=("z", "a"), **ARGS
    )
    result = build_spark_performance_report(
        spark.createDataFrame(predictions),
        spark.createDataFrame(labels),
        task="classification",
        classes=("z", "a"),
        **ARGS,
    )
    assert result["values"] == pytest.approx(expected["values"])
    assert result["excluded_rows"] == 1
    assert result["label_coverage"] == 1
    assert result["confusion_matrix"]["status"] == "measured"
    assert result["confusion_matrix"]["cells"] == [
        {
            "actual_label": "z",
            "predicted_label": "z",
            "actual_index": 0,
            "predicted_index": 0,
            "count": 0,
        },
        {
            "actual_label": "z",
            "predicted_label": "a",
            "actual_index": 0,
            "predicted_index": 1,
            "count": 0,
        },
        {
            "actual_label": "a",
            "predicted_label": "z",
            "actual_index": 1,
            "predicted_index": 0,
            "count": 1,
        },
        {
            "actual_label": "a",
            "predicted_label": "a",
            "actual_index": 1,
            "predicted_index": 1,
            "count": 1,
        },
    ]


@pytest.mark.parametrize("availability", ["2026-01-01", "invalid", None])
def test_naive_or_invalid_availability_rejected(spark, availability):
    """Unavailable timestamps cannot silently remove labels from monitoring."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame([(1, 2.0)], "id long, prediction double")
    labels = spark.createDataFrame([(1, 2.0, availability)], "id long, y double, available string")
    with pytest.raises(ValueError, match="availability"):
        build_spark_performance_report(predictions, labels, task="regression", **ARGS)


@pytest.mark.parametrize(
    "classes,truth,guess",
    [
        (("b", "a"), ["a", "a", "b"], ["a", "b", "b"]),
        (("c", "b", "a"), ["a", "b", "c"], ["a", "c", "b"]),
    ],
)
def test_confusion_summary_parity(classes, truth, guess):
    """Weighted confusion cells preserve every policy metric without expanding counts."""
    from collections import Counter

    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        _confusion_values,
    )

    indexes = {label: index for index, label in enumerate(classes)}
    counts = Counter(zip(truth, guess, strict=True))
    cells = [(indexes[first], indexes[second], count) for (first, second), count in counts.items()]
    predictions = pd.DataFrame({"id": range(len(truth)), "prediction": guess})
    labels = pd.DataFrame(
        {"id": range(len(truth)), "y": truth, "available": [NOW.isoformat()] * len(truth)}
    )
    expected = build_performance_report(
        predictions, labels, task="classification", classes=classes, **ARGS
    )
    assert _confusion_values(cells, len(classes)) == pytest.approx(expected["values"])


@pytest.mark.parametrize("partitions", [1, 3])
def test_probability_parity_and_partition_invariance(spark, partitions):
    """Exact score-frequency curves retain binary class order, ties and every AUC alias."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "prediction": ["z", "a", "a", "z"],
            "probability_0": [0.8, 0.4, 0.4, 0.9],
            "probability_1": [0.2, 0.6, 0.6, 0.1],
        }
    )
    labels = pd.DataFrame(
        {"id": [1, 2, 3, 4], "y": ["a", "a", "z", "z"], "available": [NOW.isoformat()] * 4}
    )
    expected = build_performance_report(
        predictions, labels, task="classification", classes=("z", "a"), **ARGS
    )
    result = build_spark_performance_report(
        spark.createDataFrame(predictions).repartition(partitions),
        spark.createDataFrame(labels),
        task="classification",
        classes=("z", "a"),
        **ARGS,
    )
    assert result["values"] == pytest.approx(expected["values"])


@pytest.mark.parametrize(
    "rows,message",
    [
        ([(1, 1.0), (1, 2.0)], "duplicate"),
        ([(None, 1.0)], "null record key"),
        ([(1, float("inf"))], "model output contract"),
    ],
)
def test_malformed_saved_outputs_without_labels(spark, rows, message):
    """Malformed saved evidence is rejected even before any real outcomes arrive."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame(rows, "id long, prediction double")
    with pytest.raises(ValueError, match=message):
        build_spark_performance_report(predictions, None, task="regression", **ARGS)


def test_probabilities_validated_without_labels(spark):
    """An unnormalized saved vector cannot become acceptable merely because labels are absent."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame(
        [(1, "a", 0.7, 0.7)],
        "id long, prediction string, probability_0 double, probability_1 double",
    )
    with pytest.raises(ValueError, match="sum to one"):
        build_spark_performance_report(
            predictions, None, task="classification", classes=("a", "b"), **ARGS
        )


def test_no_label_report(spark):
    """Scored predictions remain observable while real performance is explicitly unavailable."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_performance_report,
    )

    predictions = spark.createDataFrame([(1, 1.0), (2, 2.0)], "id long, prediction double")
    report = build_spark_performance_report(predictions, None, task="regression", **ARGS)
    assert report["scored_rows"] == 2
    assert report["labeled_rows"] == 0
    assert report["label_coverage"] == 0
    assert set(report["values"].values()) == {None}
