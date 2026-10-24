"""Spark feature quality and monitoring report composition from bounded counts."""

from typing import Any

from ..missing_markers import (
    Marker,
    require_marked_features,
    split_missing_markers,
    typed_markers,
)
from ..monitoring_metrics import (
    exclusion_metrics,
    missing_increase,
    monitoring_metric,
    monitoring_report_status,
    validate_monitoring_inputs,
)
from .spark_monitoring_drift import drift_evidence
from .spark_monitoring_metrics import (
    build_spark_performance_report,
    has_spark_rows,
    is_numeric_column,
    scored_predictions,
    spark_column,
    spark_functions,
    validate_record_keys,
)

# Spark integer and floating-point types named as missing_markers dtype kinds.
_SPARK_KINDS = {
    "ByteType": "Int8",
    "ShortType": "Int16",
    "IntegerType": "Int32",
    "LongType": "Int64",
    "FloatType": "Float32",
    "DoubleType": "Float64",
}


def _without_markers(frame: Any, markers: dict[str, tuple[Marker, ...]]) -> Any:
    """Replace marked values with typed nulls lazily, comparing in the column's own type."""
    f = spark_functions()
    for column, values in markers.items():
        if column not in frame.columns:
            continue
        dtype = frame.schema[column].dataType
        exact = typed_markers(column, values, _SPARK_KINDS.get(type(dtype).__name__))
        marked = spark_column(column).isin([f.lit(value).cast(dtype) for value in exact])
        null = f.lit(None).cast(dtype)
        frame = frame.withColumn(column, f.when(marked, null).otherwise(spark_column(column)))
    return frame


def _missing(frame: Any, column: str) -> Any:
    """Match Core missingness: nulls everywhere plus NaN in numeric columns."""
    missing = spark_column(column).isNull()
    return (
        missing | spark_functions().isnan(spark_column(column))
        if is_numeric_column(frame, column)
        else missing
    )


def _quality_counts(frame: Any, column: str) -> tuple[int, int, int, int]:
    """Collect a single aggregate row of population, missing, NaN and infinity counts."""
    if column not in frame.columns:
        return 0, 0, 0, 0
    f = spark_functions()
    numeric = is_numeric_column(frame, column)
    infinite = (
        f.abs(spark_column(column).cast("double")) == float("inf") if numeric else f.lit(False)
    )
    nan = f.isnan(spark_column(column)) if numeric else f.lit(False)
    row = frame.agg(
        f.count("*").alias("total"),
        f.sum(_missing(frame, column).cast("long")).alias("missing"),
        f.sum((infinite | nan).cast("long")).alias("nonfinite"),
        f.sum(infinite.cast("long")).alias("infinite"),
    ).first()
    return (
        int(row["total"] or 0),
        int(row["missing"] or 0),
        int(row["nonfinite"] or 0),
        int(row["infinite"] or 0),
    )


def _quality(reference: Any, current: Any, unscored: Any, column: str) -> list[dict]:
    """Require usable saved predictions for missing inputs even with accepted baseline rates."""
    total, missing, nonfinite, infinite = _quality_counts(current, column)
    if not total:
        return [
            monitoring_metric("quality", column, name)
            for name in ("missing_fraction", "nonfinite_fraction")
        ]
    baseline_total, baseline_missing, _, _ = _quality_counts(reference, column)
    issue = missing_increase(missing, total, baseline_missing, baseline_total)
    issue = issue or has_spark_rows(unscored.where(_missing(current, column)))
    baseline = baseline_missing / baseline_total if baseline_total else 0.0
    return [
        monitoring_metric("quality", column, "missing_fraction", missing / total, baseline, issue),
        monitoring_metric(
            "quality", column, "nonfinite_fraction", nonfinite / total, issue=bool(infinite)
        ),
    ]


def _features(
    reference: Any,
    current: Any,
    unscored: Any,
    columns: tuple,
    thresholds: dict | None,
    reference_count: int,
    current_count: int,
) -> tuple[list, int, list, bool]:
    """Retain no-data, missing-schema and unsupported-feature evidence independently."""
    quality = {column: _quality(reference, current, unscored, column) for column in columns}
    if not reference_count or not current_count:
        metrics = [
            item
            for column in columns
            for item in [monitoring_metric("drift", column, "unavailable"), *quality[column]]
        ]
        return metrics, 0, ["Reference or current feature data is empty."], True
    drift, count, notes, unavailable = drift_evidence(reference, current, columns, thresholds)
    metrics = [
        item
        for column in columns
        for item in quality[column] + [row for row in drift if row["column_name"] == column]
    ]
    return metrics, count, notes, unavailable


def _unscored(current: Any, predictions: Any, scored: Any, keys: tuple) -> Any:
    """Check prediction membership and retain absent or excluded current keys."""
    if has_spark_rows(predictions):
        if not has_spark_rows(current):
            raise ValueError("A prediction key is absent from current records.")
        known = current.select(*[spark_column(key) for key in keys])
        if has_spark_rows(predictions.join(known, list(keys), "left_anti")):
            raise ValueError("A prediction key is absent from current records.")
    if not has_spark_rows(scored):
        return current
    return current.join(
        scored.select(*[spark_column(key) for key in keys]), list(keys), "left_anti"
    )


def _report_issues(metrics: list[dict], performance: dict) -> tuple[bool, bool]:
    """Keep absent labels distinct from malformed performance or failed feature quality."""
    quality = not performance["scored_rows"] or any(
        item["has_issue"] for item in metrics if item["category"] == "quality"
    )
    bad_performance = performance["labeled_rows"] >= 2 and any(
        value is None for value in performance["values"].values()
    )
    return quality, bad_performance


def monitoring_report(
    reference: Any,
    current: Any,
    predictions: Any,
    labels: Any,
    *,
    feature_columns: tuple,
    record_key_columns: tuple,
    target_column: str,
    result_available_at_column: str,
    as_of: Any,
    task: str,
    classes: tuple,
    thresholds: dict | None,
) -> dict:
    """Build the existing monitoring result contract without materializing populations."""
    validate_monitoring_inputs(as_of, task, classes, thresholds)
    thresholds, markers = split_missing_markers(thresholds)
    require_marked_features(markers, feature_columns)
    reference = _without_markers(reference, markers)
    current = _without_markers(current, markers)
    validate_record_keys(current, record_key_columns, "current")
    validate_record_keys(predictions, record_key_columns, "predictions")
    scored, excluded = scored_predictions(predictions, task, classes)
    unscored = _unscored(current, predictions, scored, record_key_columns)
    performance = build_spark_performance_report(
        predictions,
        labels,
        record_key_columns=record_key_columns,
        target_column=target_column,
        result_available_at_column=result_available_at_column,
        as_of=as_of,
        task=task,
        classes=classes,
    )
    reference_count, current_count = reference.count(), current.count()
    metrics, drifted, notes, unavailable = _features(
        reference, current, unscored, feature_columns, thresholds, reference_count, current_count
    )
    metrics.extend(
        monitoring_metric("performance", target_column, name, value)
        for name, value in performance["values"].items()
    )
    metrics.extend(
        exclusion_metrics(
            performance["scored_rows"] + excluded,
            excluded,
            _invalid_exclusions(predictions, excluded),
        )
    )
    notes = (
        ([f"{excluded} excluded predictions were not scored."] if excluded else [])
        + notes
        + performance["notes"]
    )
    if labels is None:
        notes.append("No labels were provided; performance is unavailable.")
    if not performance["scored_rows"]:
        notes.append("No current rows have a saved predicted output.")
    quality_issue, bad_performance = _report_issues(metrics, performance)
    status = monitoring_report_status(
        reference_count, current_count, drifted, unavailable, quality_issue, bad_performance
    )
    return {
        "status": status,
        "reference_rows": reference_count,
        "current_rows": current_count,
        "scored_rows": performance["scored_rows"],
        "excluded_rows": excluded,
        "labeled_rows": performance["labeled_rows"],
        "label_coverage": performance["label_coverage"],
        "confusion_matrix": performance["confusion_matrix"],
        "drifted_columns": drifted,
        "metrics": metrics,
        "notes": notes,
    }


def _invalid_exclusions(predictions: Any, excluded: int) -> int | None:
    """Count invalid outcomes without collecting requests or discarding unknown reasons."""
    if not excluded:
        return 0
    if "exclusion_reason" not in predictions.columns:
        return None
    f = spark_functions()
    reasons = predictions.where(spark_column("scoring_status") == "excluded")
    reason = spark_column("exclusion_reason")
    counts = reasons.agg(
        f.sum(f.when(reason.isNull(), 1).otherwise(0)).alias("unknown"),
        f.sum(f.when(reason.startswith("invalid:"), 1).otherwise(0)).alias("invalid"),
    ).first()
    return None if counts.unknown else int(counts.invalid)
