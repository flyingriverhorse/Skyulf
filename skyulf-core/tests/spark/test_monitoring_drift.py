"""Distributed drift statistics retain exact effect sizes and explicit evidence methods."""

import importlib

import pandas as pd
import pytest
from scipy.stats import ks_2samp


@pytest.mark.parametrize(
    "reference,current", [([1, 2, 3], [4, 5, 6]), ([1, 2, 2, 4], [2, 3, 4]), ([1, 1, 1], [1, 1, 1])]
)
def test_exact_small_ks_probability(reference, current):
    """The bounded count-only lattice calculation matches scipy's exact two-sided gate."""
    module = importlib.import_module(
        "skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift"
    )
    expected = ks_2samp(reference, current)
    evidence = module._ks_evidence(float(expected.statistic), len(reference), len(current))
    assert evidence.p_value == pytest.approx(expected.pvalue)
    assert evidence.test == "ks_2samp"


def test_numeric_drift_effect_parity(spark):
    """Distributed CDF integration and reference quantile bins retain Core effect values."""
    import polars as pl

    from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
        _drift_evidence,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        drift_evidence,
    )

    reference = pd.DataFrame({"x": [float(index) for index in range(40)]})
    current = pd.DataFrame({"x": [float(index + 30) for index in range(40)]})
    expected, drifted, _, _ = _drift_evidence(
        pl.from_pandas(reference), pl.from_pandas(current), ("x",), None
    )
    result, actual_drifted, _, unmeasured = drift_evidence(
        spark.createDataFrame(reference), spark.createDataFrame(current), ("x",), None
    )
    assert {item["metric_name"]: item["value"] for item in result} == pytest.approx(
        {item["metric_name"]: item["value"] for item in expected if item["category"] == "drift"}
    )
    assert actual_drifted == drifted == 1
    assert not unmeasured


def test_inconclusive_bound_is_unavailable():
    """A conservative probability bound that cannot reject must never imply healthy evidence."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        _correct_results,
        _ks_evidence,
    )
    from skyulf.profiling.drift import ColumnDrift

    evidence = _ks_evidence(0.001, 2000, 2000)
    result = ColumnDrift(column="x", metrics=[], drift_detected=False, evidence=evidence)
    _correct_results({"x": result})
    assert evidence.test == "ks_dkw_union_bound"
    assert evidence.status == "unavailable"


def test_missing_input_requires_saved_prediction(spark):
    """Stable training missingness is accepted only for rows the model actually scored."""
    from datetime import UTC, datetime

    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        build_spark_monitoring_report,
    )

    reference = spark.createDataFrame([(1, 1.0), (2, None), (3, 2.0)], "id long, x double")
    predictions = spark.createDataFrame([(1, 1.0), (3, 2.0)], "id long, prediction double")
    report = build_spark_monitoring_report(
        reference,
        reference,
        predictions,
        None,
        feature_columns=("x",),
        record_key_columns=("id",),
        target_column="y",
        result_available_at_column="available",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        task="regression",
    )
    missing = next(item for item in report["metrics"] if item["metric_name"] == "missing_fraction")
    assert missing["has_issue"]
    assert report["status"] == "degraded"
