"""Distributed drift statistics retain exact effect sizes and explicit evidence methods."""

import importlib
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp


@pytest.mark.parametrize(
    "reference,current,ref_type,cur_type",
    [
        ([0, 0, 1, 2], [0, 0, 2, 2], "long", "long"),
        ([0, 0, 1, 2], [0.5, 0.5, 1.5, 1.5], "long", "double"),
        ([0.0, 0.0, 1.0, 2.0], [-100.0, -100.0, 100.0, 100.0], "double", "double"),
        ([2**53, 2**53 + 1, 2**53 + 2], [2**53 + 2] * 3, "long", "long"),
        ([2**63 - 3, 2**63 - 2, 2**63 - 1], [2**63 - 1] * 3, "long", "long"),
        ([-(2**63), -(2**63) + 1, -(2**63) + 2], [-(2**63)] * 3, "long", "long"),
        ([2**53, 2**53 + 1, 2**53 + 2], [float(2**53)] * 3, "long", "double"),
        (
            [1.0, float(np.nextafter(1.0, np.inf))],
            [float(np.nextafter(1.0, np.inf))] * 2,
            "double",
            "double",
        ),
        (
            [1.0, float(np.nextafter(np.float32(1), np.float32(np.inf)))],
            [1.0] * 2,
            "float",
            "float",
        ),
        (
            [Decimal("0"), Decimal("1"), Decimal("2")],
            [Decimal("2")] * 3,
            "decimal(8,2)",
            "decimal(8,2)",
        ),
        ([7, 7, 7], [100, 200], "long", "long"),
        (list(range(10)), [9] * 10, "long", "long"),
        (list(range(11)), [10] * 11, "long", "long"),
        (list(range(40)), [-5, 30, 70], "long", "long"),
    ],
    ids=[
        "lost-middle-level",
        "fractional-midpoints",
        "clipped-float-tails",
        "large-integers",
        "int64-upper-edge",
        "int64-lower-edge",
        "integer-reference-float-current",
        "adjacent-doubles",
        "adjacent-floats",
        "decimals",
        "constant-reference",
        "ten-levels",
        "eleven-levels",
        "continuous-tails",
    ],
)
def test_numeric_histograms_match_core(spark, reference, current, ref_type, cur_type):
    """Spark PSI/KL must retain every discrete level and use Core's exact boundary rules."""
    import polars as pl

    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        _histogram_metrics,
    )
    from skyulf.profiling.drift import DriftCalculator

    frames = [
        spark.createDataFrame([(value,) for value in values], f"value {dtype}")
        for values, dtype in ((reference, ref_type), (current, cur_type))
    ]
    columns = [
        pl.Series("value", values).cast(pl.Float32)
        if dtype == "float"
        else pl.Series("value", values)
        for values, dtype in ((reference, ref_type), (current, cur_type))
    ]
    calculator = DriftCalculator(columns[0].to_frame(), columns[1].to_frame())
    ref, cur = (calculator._numeric_values(series) for series in columns)
    expected = (calculator._calculate_psi(ref, cur), calculator._calculate_kl(ref, cur))
    observed = _histogram_metrics(frames[0], frames[1], len(reference), len(current))
    if reference == [0, 0, 1, 2] and ref_type == cur_type == "long":
        assert expected[0] > 2
    assert observed == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("origin", [2**63 - 3, -(2**63)], ids=["positive", "negative"])
def test_integer_histograms_ignore_decimal_precision(spark, origin):
    """A caller's low Decimal precision cannot merge adjacent signed int64 levels."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        _histogram_metrics,
    )

    reference = spark.createDataFrame([(origin + index,) for index in range(3)], "value long")
    current = spark.createDataFrame([(origin + 2,)] * 3, "value long")
    with localcontext() as context:
        context.prec = 10
        observed = _histogram_metrics(reference, current, 3, 3)
    # Core's equal three-level reference versus concentration in the top level.
    assert observed == pytest.approx((6.138604569034126, 1.0965706089309952))


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


@pytest.mark.parametrize("discrete", [False, True], ids=["continuous", "discrete"])
def test_numeric_drift_effect_parity(spark, discrete):
    """Distributed CDF integration and reference quantile bins retain Core effect values."""
    import polars as pl

    from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
        _drift_evidence,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        drift_evidence,
    )

    reference = pd.DataFrame(
        {
            "x": [0.0] * 50 + [1.0] * 25 + [2.0] * 25
            if discrete
            else [float(index) for index in range(40)]
        }
    )
    current = pd.DataFrame(
        {"x": [0.0] * 50 + [2.0] * 50 if discrete else [float(index + 30) for index in range(40)]}
    )
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
