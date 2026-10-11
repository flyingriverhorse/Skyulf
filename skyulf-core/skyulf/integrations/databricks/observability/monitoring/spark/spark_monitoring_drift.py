"""Exact Spark distribution effects with bounded, explicitly named statistical evidence."""

import importlib
import math
from typing import Any

import numpy as np
from scipy.stats import entropy, kstwo

from skyulf.profiling._drift_evidence import (
    DriftEvidence,
    categorical_drift_test,
    correct_evidence,
    numeric_evidence,
)
from skyulf.profiling.drift import ColumnDrift, DriftMetric

from ..monitoring_metrics import column_drift_evidence, monitoring_metric
from .spark_monitoring_metrics import (
    finite_spark_value,
    has_spark_rows,
    is_numeric_column,
    spark_column,
    spark_functions,
)

MAX_CATEGORY_SUMMARIES = 1024
MAX_EXACT_KS_CELLS = 1_000_000


def _exact_ks_probability(statistic: float, n: int, m: int) -> float:
    """Count lattice paths inside the strict KS band using bounded integer state."""
    divisor = math.gcd(n, m)
    boundary = round(statistic * (n * m // divisor)) * divisor
    previous = [0] * (m + 1)
    for first in range(n + 1):
        row = [0] * (m + 1)
        for second in range(m + 1):
            if abs(first * m - second * n) >= boundary:
                continue
            row[second] = (previous[second] if first else 0) + (row[second - 1] if second else 0)
            if first == second == 0:
                row[second] = 1
        previous = row
    total = math.comb(n + m, n)
    return (total - previous[m]) / total


def _ks_evidence(statistic: float, n: int, m: int) -> DriftEvidence:
    """Match Core exact/asymptotic KS where bounded; otherwise use a named valid bound.

    The middle range uses the union bound on two one-sample DKW inequalities:
    P(D >= d) <= 4 exp(-2 d^2 / (1/sqrt(n) + 1/sqrt(m))^2).
    It is conservative finite-sample evidence, never an approximation labeled as KS.
    An inconclusive bound is surfaced as unavailable after family correction.
    """
    if n * m <= MAX_EXACT_KS_CELLS:
        return numeric_evidence(_exact_ks_probability(statistic, n, m), n, m)
    if max(n, m) > 10_000:
        probability = float(kstwo.sf(statistic, round(n * m / (n + m))))
        return numeric_evidence(probability, n, m)
    bound = min(1.0, 4 * math.exp(-2 * statistic**2 / (1 / math.sqrt(n) + 1 / math.sqrt(m)) ** 2))
    return DriftEvidence(
        test="ks_dkw_union_bound", p_value=bound, reference_count=n, current_count=m
    )


def _numeric_frame(frame: Any, column: str) -> Any:
    """Keep native numeric coordinates so large integer order remains exact."""
    value = spark_column(column)
    return frame.where(finite_spark_value(value)).select(value.alias("value"))


def _cdf_statistics(reference: Any, current: Any, n: int, m: int) -> tuple[float, float]:
    """Integrate exact empirical CDF gaps on executors and return two scalar summaries."""
    f = spark_functions()
    window = importlib.import_module("pyspark.sql.window").Window
    points = reference.select("value", f.lit(1).alias("r"), f.lit(0).alias("c")).unionByName(
        current.select("value", f.lit(0).alias("r"), f.lit(1).alias("c"))
    )
    points = points.groupBy("value").agg(f.sum("r").alias("r"), f.sum("c").alias("c"))
    ordered = window.orderBy("value")
    cumulative = ordered.rowsBetween(window.unboundedPreceding, window.currentRow)
    points = points.withColumn("nr", f.sum("r").over(cumulative)).withColumn(
        "nc", f.sum("c").over(cumulative)
    )
    # Decimal counts preserve the exact integer CDF numerator beyond 2**53.
    gap = f.abs(
        f.col("nr").cast("decimal(38,0)") * f.lit(m) - f.col("nc").cast("decimal(38,0)") * f.lit(n)
    )
    points = points.withColumn("gap", gap).withColumn("next", f.lead("value").over(ordered))
    delta = f.col("next") - f.col("value")
    if is_numeric_column(points, "value") and points.schema["value"].dataType.typeName() in {
        "long",
        "integer",
        "short",
        "byte",
    }:
        delta = f.col("next").cast("decimal(38,0)") - f.col("value").cast("decimal(38,0)")
    row = points.agg(
        f.max("gap").alias("ks"), f.sum(delta * f.col("gap")).alias("transport")
    ).first()
    return float(row["ks"]) / (n * m), float(row["transport"] or 0) / (n * m)


def _histogram(frame: Any, edges: list[float], size: int) -> np.ndarray:
    """Reduce clipped reference-quantile bins to at most ten counts."""
    f = spark_functions()
    value = spark_column("value")
    bucket = f.lit(0)
    for edge in edges[1:-1]:
        bucket = bucket + (value >= f.lit(edge)).cast("int")
    rows = frame.select(bucket.alias("bucket")).groupBy("bucket").count().limit(11).collect()
    counts = np.zeros(len(edges) - 1)
    for row in rows:
        counts[row["bucket"]] = row["count"] / size
    return np.where(counts == 0, 0.0001, counts)


def _histogram_metrics(reference: Any, current: Any, n: int, m: int) -> tuple[float, float]:
    """Use exact continuous quantiles, matching NumPy percentile interpolation."""
    f = spark_functions()
    percentiles = f.percentile("value", [index / 10 for index in range(11)])
    edges = sorted(set(reference.agg(percentiles.alias("edges")).first()["edges"]))
    if len(edges) < 2:
        return 0.0, 0.0
    expected, actual = _histogram(reference, edges, n), _histogram(current, edges, m)
    return float(np.sum((actual - expected) * np.log(actual / expected))), float(
        entropy(actual, expected)
    )


def _metric_result(name: str, value: float, threshold: float) -> DriftMetric:
    """Preserve the Core effect threshold and strict greater-than comparison."""
    return DriftMetric(metric=name, value=value, threshold=threshold, has_drift=value > threshold)


def _ks_p_value_metric(evidence: DriftEvidence, statistic: float, threshold: float) -> DriftMetric:
    """Require a measured probability before constructing the diagnostic KS metric."""
    if evidence.p_value is None:
        raise ValueError("KS evidence requires a p-value.")
    return DriftMetric(
        metric="ks_test_p_value",
        value=evidence.p_value,
        threshold=threshold,
        has_drift=statistic > threshold,
    )


def _numeric_result(
    reference: Any, current: Any, column: str, thresholds: dict
) -> ColumnDrift | None:
    """Calculate numeric effect statistics and a separately identified inference gate."""
    f = spark_functions()
    if has_spark_rows(
        reference.where(f.abs(spark_column(column).cast("double")) == float("inf"))
    ) or has_spark_rows(current.where(f.abs(spark_column(column).cast("double")) == float("inf"))):
        return None
    ref, cur = _numeric_frame(reference, column), _numeric_frame(current, column)
    n, m = ref.count(), cur.count()
    if not n or not m:
        return None
    statistic, distance = _cdf_statistics(ref, cur, n, m)
    # Center native integer coordinates before converting to double for stable scaling.
    origin = ref.agg(f.min("value").alias("origin")).first()["origin"]
    centered = f.col("value") - f.lit(origin)
    if ref.schema["value"].dataType.typeName() in {"long", "integer", "short", "byte"}:
        centered = f.col("value").cast("decimal(38,0)") - f.lit(origin)
    std = float(ref.agg(f.stddev_pop(centered.cast("double")).alias("std")).first()["std"] or 0)
    psi, kl = _histogram_metrics(ref, cur, n, m)
    evidence = _ks_evidence(statistic, n, m)
    metrics = [
        _metric_result(
            "wasserstein_distance", distance / std if std else distance, thresholds["wasserstein"]
        ),
        _metric_result("ks_statistic", statistic, thresholds["ks_statistic"]),
        _metric_result("psi", psi, thresholds["psi"]),
        _metric_result("kl_divergence", kl, thresholds["kl_divergence"]),
    ]
    if evidence.test == "ks_2samp":
        metrics.insert(
            2,
            _ks_p_value_metric(evidence, statistic, thresholds["ks_statistic"]),
        )
    return ColumnDrift(
        column=column,
        metrics=metrics,
        drift_detected=any(item.has_drift for item in metrics),
        evidence=evidence,
    )


def _category_counts(frame: Any, column: str) -> dict[str, int] | None:
    """Bound category summaries explicitly without ever collecting input observations."""
    rows = (
        frame.where(spark_column(column).isNotNull())
        .select(spark_column(column).cast("string").alias("value"))
        .groupBy("value")
        .count()
        .limit(MAX_CATEGORY_SUMMARIES + 1)
        .collect()
    )
    if len(rows) > MAX_CATEGORY_SUMMARIES:
        return None
    return {row["value"]: row["count"] for row in rows}


def _categorical_result(
    reference: Any, current: Any, column: str, thresholds: dict
) -> ColumnDrift | None:
    """Reuse Core exact categorical tests on bounded unsmoothed count tables."""
    ref, cur = _category_counts(reference, column), _category_counts(current, column)
    if not ref or not cur or len(ref) > 50:
        return None
    categories = sorted(ref.keys() | cur.keys())
    table = np.array(
        [[ref.get(label, 0) for label in categories], [cur.get(label, 0) for label in categories]],
        dtype=np.int64,
    )
    n, m = int(table[0].sum()), int(table[1].sum())
    expected, actual = table[0] / n, table[1] / m
    expected, actual = (
        np.where(expected == 0, 0.5 / n, expected),
        np.where(actual == 0, 0.5 / m, actual),
    )
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    test, probability, minimum = categorical_drift_test(table)
    evidence = DriftEvidence(test=test, p_value=probability, reference_count=n, current_count=m)
    evidence._minimum_p_value = minimum
    metric = _metric_result("psi_categorical", psi, thresholds["psi"])
    return ColumnDrift(
        column=column, metrics=[metric], drift_detected=metric.has_drift, evidence=evidence
    )


def _both_populated(reference: Any, current: Any, column: str) -> bool:
    """Retain the local no-data precedence even when inferred nullable types differ."""
    return has_spark_rows(reference.where(spark_column(column).isNotNull())) and has_spark_rows(
        current.where(spark_column(column).isNotNull())
    )


def _column_result(
    reference: Any, current: Any, column: str, thresholds: dict
) -> ColumnDrift | None:
    """Classify schema compatibility before selecting a distributed numeric or category kernel."""
    types = importlib.import_module("pyspark.sql.types")
    ref_type, cur_type = reference.schema[column].dataType, current.schema[column].dataType
    categorical = (types.StringType, types.BooleanType)
    if not _both_populated(reference, current, column):
        return None
    if isinstance(ref_type, types.NumericType) and isinstance(cur_type, types.StringType):
        converted = spark_column(column).try_cast("double")
        if not has_spark_rows(current.where(spark_column(column).isNotNull() & converted.isNull())):
            current = current.withColumn(column, converted)
            cur_type = current.schema[column].dataType
    if isinstance(ref_type, types.NumericType) and isinstance(cur_type, types.NumericType):
        return _numeric_result(reference, current, column, thresholds)
    if isinstance(ref_type, categorical) and isinstance(cur_type, categorical):
        return _compatible_categories(reference, current, column, thresholds)
    if ref_type != cur_type:
        return ColumnDrift(
            column=column, metrics=[_metric_result("type_drift", 1.0, 0.0)], drift_detected=True
        )
    return None


def _compatible_categories(
    reference: Any, current: Any, column: str, thresholds: dict
) -> ColumnDrift | None:
    """Normalize serialized boolean values exactly as the local category adapter does."""
    types = importlib.import_module("pyspark.sql.types")
    boolean = any(
        isinstance(frame.schema[column].dataType, types.BooleanType)
        for frame in (reference, current)
    )
    if boolean:
        value = spark_functions().lower(spark_column(column).cast("string"))
        reference, current = reference.withColumn(column, value), current.withColumn(column, value)
        if any(
            has_spark_rows(
                frame.where(
                    spark_column(column).isNotNull() & ~spark_column(column).isin("true", "false")
                )
            )
            for frame in (reference, current)
        ):
            return ColumnDrift(
                column=column, metrics=[_metric_result("type_drift", 1.0, 0.0)], drift_detected=True
            )
    return _categorical_result(reference, current, column, thresholds)


def _correct_results(results: dict[str, ColumnDrift | None]) -> None:
    """Apply one family correction before interpreting separately named bound evidence."""
    evidences = [
        result.evidence
        for result in results.values()
        if result is not None and result.evidence is not None
    ]
    correct_evidence(evidences)
    for result in results.values():
        if result is None or result.evidence is None:
            continue
        evidence = result.evidence
        if evidence.test == "ks_dkw_union_bound" and evidence.status != "supported":
            evidence.status = "unavailable"
            evidence.reason = "The conservative DKW probability bound is inconclusive."
        result.drift_detected = result.drift_detected and evidence.status == "supported"


def _result_rows(results: dict[str, ColumnDrift | None]) -> tuple[list[dict], int, list[str], bool]:
    """Adapt corrected results without concealing unsupported feature comparisons."""
    metrics, notes = [], []
    drifted = 0
    for column, result in results.items():
        metrics.extend(column_drift_evidence(column, result))
        unavailable = result is None or (
            result.evidence is not None
            and result.evidence.status in {"insufficient_data", "unavailable"}
        )
        if unavailable:
            notes.append(
                f"Feature {column} could not be measured with sufficient statistical evidence."
            )
        drifted += int(result is not None and result.drift_detected)
    return metrics, drifted, notes, bool(notes)


def drift_evidence(
    reference: Any, current: Any, columns: tuple[str, ...], thresholds: dict | None
) -> tuple[list[dict], int, list[str], bool]:
    """Return feature-only evidence with honest unavailable outcomes for unsupported types."""
    limits = {"psi": 0.2, "ks_statistic": 0.1, "wasserstein": 0.1, "kl_divergence": 0.1} | (
        thresholds or {}
    )
    missing = [
        column
        for column in columns
        if column not in reference.columns or column not in current.columns
    ]
    results = {
        column: _column_result(reference, current, column, limits)
        for column in columns
        if column not in missing
    }
    _correct_results(results)
    metrics = [
        monitoring_metric("drift", column, "schema_missing", 1.0, 0.0, True) for column in missing
    ]
    notes = [f"Feature {column} is missing from a monitoring input." for column in missing]
    measured, drifted, feature_notes, unmeasured = _result_rows(results)
    return metrics + measured, len(missing) + drifted, notes + feature_notes, unmeasured
