"""Exact Spark distribution effects with bounded, explicitly named statistical evidence."""

import importlib
import math
import warnings
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.stats import entropy, ks_2samp, kstwo

from skyulf.profiling._drift_evidence import (
    DriftEvidence,
    categorical_drift_test,
    correct_evidence,
    numeric_evidence,
)
from skyulf.profiling._drift_settings import merge_drift_settings
from skyulf.profiling.drift import ColumnDrift, DriftCalculator, DriftMetric

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


def _scipy_exact_probability(statistic: float, n: int, m: int) -> float | None:
    """Return the exact two-sample KS probability that local ``ks_2samp`` computes.

    The exact probability depends only on n, m and the statistic, so two tied samples
    with the same statistic (i values of 0 among n, j among m, |i/n - j/m| = statistic)
    give it through SciPy's public ``ks_2samp``. None means the caller falls back to the
    DKW bound: no such samples, or SciPy could not finish the exact count.
    """
    divisor = math.gcd(n, m)
    target = round(float(statistic) * (n * m // divisor)) * divisor
    for zeros in range(n + 1):
        other_zeros, remainder = divmod(zeros * m - target, n)
        if remainder == 0 and 0 <= other_zeros <= m:
            break
    else:
        return None
    reference = np.repeat([0.0, 1.0], [zeros, n - zeros])
    current = np.repeat([0.0, 1.0], [other_zeros, m - other_zeros])
    try:
        with warnings.catch_warnings():
            # SciPy warns and silently switches to the asymptotic method on failure.
            warnings.simplefilter("error", RuntimeWarning)
            probability = float(ks_2samp(reference, current, method="exact").pvalue)
    except (RuntimeWarning, ValueError, ArithmeticError):
        return None
    return probability if 0.0 <= probability <= 1.0 else None


def _ks_evidence(statistic: float, n: int, m: int) -> DriftEvidence:
    """Match Core exact/asymptotic KS where bounded; otherwise use a named valid bound.

    The middle range uses SciPy's exact count, like local ks_2samp. Only when that is
    unavailable does it fall back to the union bound on two one-sample DKW inequalities:
    P(D >= d) <= 4 exp(-2 d^2 / (1/sqrt(n) + 1/sqrt(m))^2).
    It is conservative finite-sample evidence, never an approximation labeled as KS.
    An inconclusive bound is surfaced as unavailable after family correction.
    """
    if n * m <= MAX_EXACT_KS_CELLS:
        return numeric_evidence(_exact_ks_probability(statistic, n, m), n, m)
    if max(n, m) > 10_000:
        probability = float(kstwo.sf(statistic, round(n * m / (n + m))))
        return numeric_evidence(probability, n, m)
    exact = _scipy_exact_probability(statistic, n, m)
    if exact is not None:
        return numeric_evidence(exact, n, m)
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


def _histogram(frame: Any, boundaries: list[float | Decimal], size: int) -> np.ndarray:
    """Reduce midpoint or quantile bins to at most ten counts, including tails."""
    f = spark_functions()
    value = spark_column("value")
    floating = frame.schema["value"].dataType.typeName() in {"float", "double"}
    bucket = f.lit(0)
    for edge in boundaries:
        if floating and isinstance(edge, Decimal):
            # Spark compares float/Decimal as double. Round the exact midpoint
            # upward so a representable value below it stays in the lower bin.
            rounded = float(edge)
            edge = math.nextafter(rounded, math.inf) if Decimal(rounded) < edge else rounded
        bucket = bucket + (value >= f.lit(edge)).cast("int")
    rows = frame.select(bucket.alias("bucket")).groupBy("bucket").count().limit(11).collect()
    counts = np.zeros(len(boundaries) + 1)
    for row in rows:
        counts[row["bucket"]] = row["count"] / size
    return np.where(counts == 0, 0.0001, counts)


def _histogram_boundaries(reference: Any) -> list[float | Decimal]:
    """Match Core's discrete midpoints before falling back to continuous quantiles."""
    levels = sorted(
        row["value"] for row in reference.select("value").distinct().limit(11).collect()
    )
    if len(levels) > 10:
        f = spark_functions()
        percentiles = f.percentile("value", [index / 10 for index in range(11)])
        return sorted(set(reference.agg(percentiles.alias("edges")).first()["edges"]))[1:-1]
    dtype = reference.schema["value"].dataType.typeName()
    if dtype in {"long", "integer", "short", "byte"}:
        # Parsing (left + right) * 5 at scale 1 gives the exact midpoint without
        # integer overflow or rounding by the caller's Decimal precision.
        return [
            Decimal(f"{(left + right) * 5}e-1")
            for left, right in zip(levels, levels[1:], strict=False)
        ]
    values = np.asarray(levels, dtype=np.float32 if dtype == "float" else np.float64)
    return np.maximum(values[:-1] / 2 + values[1:] / 2, np.nextafter(values[:-1], np.inf)).tolist()


def _histogram_metrics(reference: Any, current: Any, n: int, m: int) -> tuple[float, float]:
    """Preserve Core's numeric binning and smoothing for both PSI and KL."""
    # Core converts Decimal columns to float64 before numeric drift calculations.
    reference, current = (
        frame.select(spark_column("value").cast("double").alias("value"))
        if frame.schema["value"].dataType.typeName() == "decimal"
        else frame
        for frame in (reference, current)
    )
    boundaries = _histogram_boundaries(reference)
    if not boundaries:
        return 0.0, 0.0
    expected, actual = _histogram(reference, boundaries, n), _histogram(current, boundaries, m)
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
    reference: Any, current: Any, column: str, thresholds: dict, n: int, m: int, std: float
) -> ColumnDrift | None:
    """Calculate numeric effect statistics and a separately identified inference gate."""
    ref, cur = _numeric_frame(reference, column), _numeric_frame(current, column)
    statistic, distance = _cdf_statistics(ref, cur, n, m)
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


def _category_counts(
    frame: Any, column: str, limit: int = MAX_CATEGORY_SUMMARIES
) -> dict[str, int] | None:
    """Bound category summaries explicitly without ever collecting input observations."""
    rows = (
        frame.where(spark_column(column).isNotNull())
        .select(spark_column(column).cast("string").alias("value"))
        .groupBy("value")
        .count()
        .limit(limit + 1)
        .collect()
    )
    if len(rows) > limit:
        return None
    return {row["value"]: row["count"] for row in rows}


def _categorical_result(
    reference: Any, current: Any, column: str, thresholds: dict
) -> ColumnDrift | None:
    """Reuse Core exact categorical tests on bounded unsmoothed count tables."""
    cap = int(thresholds.get("max_category_summaries", DriftCalculator._MAX_CATEGORY_SUMMARIES))
    ref = _category_counts(reference, column, cap)
    if not ref:
        return None
    grouped = len(ref) > thresholds.get(
        "categorical_top_k", DriftCalculator._MAX_CATEGORICAL_CARDINALITY
    )
    configured = "max_category_summaries" in thresholds or "categorical_top_k" in thresholds
    limit = cap if grouped or configured else MAX_CATEGORY_SUMMARIES
    cur = _category_counts(current, column, limit)
    if not cur:
        return None
    if grouped:
        return DriftCalculator._grouped_categorical_drift(column, ref, cur, thresholds)
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


def _population_summaries(
    frame: Any, columns: tuple[str, ...], numeric: tuple[str, ...], *, origins: bool
) -> dict[str, dict]:
    """Collect only scalar preflight summaries in one scan, retaining native minima."""
    if not columns:
        return {}
    f = spark_functions()
    aggregates = []
    for column in columns:
        original = spark_column(column)
        fields = [f.count(original).alias("populated")]
        if column in numeric:
            value = original if is_numeric_column(frame, column) else original.try_cast("double")
            finite = f.when(finite_spark_value(value), value)
            fields.extend(
                [
                    f.count(finite).alias("count"),
                    f.max(f.abs(value.cast("double")) == float("inf")).alias("infinite"),
                    f.count(f.when(original.isNotNull() & value.isNull(), 1)).alias("invalid"),
                ]
            )
            if origins:
                fields.append(f.min(finite).alias("origin"))
        aggregates.append(f.struct(*fields).alias(column))
    return frame.agg(*aggregates).first().asDict(recursive=True)


def _has_numeric_data(summary: dict) -> bool:
    """Require finite observations and reject infinity or failed numeric conversion."""
    return bool(summary.get("count")) and not summary["infinite"] and not summary["invalid"]


def _reference_deviations(reference: Any, ref: dict, cur: dict) -> dict[str, float]:
    """Batch standard deviations after centering in each column's native coordinates."""
    f = spark_functions()
    aggregates = []
    for column, summary in ref.items():
        if not _has_numeric_data(summary) or not _has_numeric_data(cur[column]):
            continue
        value = spark_column(column)
        finite = f.when(finite_spark_value(value), value)
        if reference.schema[column].dataType.typeName() in {"long", "integer", "short", "byte"}:
            finite = finite.cast("decimal(38,0)")
        centered = finite - f.lit(summary["origin"])
        aggregates.append(f.stddev_pop(centered.cast("double")).alias(column))
    if not aggregates:
        return {}
    row = reference.agg(*aggregates).first().asDict()
    return {column: float(value or 0) for column, value in row.items()}


def _column_result(
    reference: Any,
    current: Any,
    column: str,
    thresholds: dict,
    ref_summary: dict,
    cur_summary: dict,
    std: float | None,
) -> ColumnDrift | None:
    """Classify schema compatibility before selecting a distributed numeric or category kernel."""
    types = importlib.import_module("pyspark.sql.types")
    ref_type, cur_type = reference.schema[column].dataType, current.schema[column].dataType
    categorical = (types.StringType, types.BooleanType)
    if not ref_summary["populated"] or not cur_summary["populated"]:
        return None
    if "count" in ref_summary and not cur_summary["invalid"]:
        if std is None:
            return None
        if isinstance(cur_type, types.StringType):
            current = current.withColumn(column, spark_column(column).try_cast("double"))
        return _numeric_result(
            reference, current, column, thresholds, ref_summary["count"], cur_summary["count"], std
        )
    if isinstance(ref_type, categorical) and isinstance(cur_type, categorical):
        return _compatible_categories(reference, current, column, thresholds)
    if ref_type != cur_type:
        return ColumnDrift(
            column=column, metrics=[_metric_result("type_drift", 1.0, 0.0)], drift_detected=True
        )
    return None


def _column_results(reference: Any, current: Any, columns: tuple, thresholds: dict) -> dict:
    """Share numeric preflight work while preserving column order and per-feature kernels."""
    numeric = tuple(
        column
        for column in columns
        if is_numeric_column(reference, column)
        and (
            is_numeric_column(current, column)
            or current.schema[column].dataType.typeName() == "string"
        )
    )
    ref = _population_summaries(reference, columns, numeric, origins=True)
    cur = _population_summaries(current, columns, numeric, origins=False)
    deviations = _reference_deviations(reference, ref, cur)
    return {
        column: _column_result(
            reference, current, column, thresholds, ref[column], cur[column], deviations.get(column)
        )
        for column in columns
    }


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
    limits = merge_drift_settings(thresholds)
    missing = [
        column
        for column in columns
        if column not in reference.columns or column not in current.columns
    ]
    present = tuple(dict.fromkeys(column for column in columns if column not in missing))
    results = _column_results(reference, current, present, limits)
    _correct_results(results)
    metrics = [
        monitoring_metric("drift", column, "schema_missing", 1.0, 0.0, True) for column in missing
    ]
    notes = [f"Feature {column} is missing from a monitoring input." for column in missing]
    measured, drifted, feature_notes, unmeasured = _result_rows(results)
    return metrics + measured, len(missing) + drifted, notes + feature_notes, unmeasured
