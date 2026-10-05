"""Spark feature quality and monitoring report composition from bounded counts."""

from typing import Any

from scipy.stats import fisher_exact

from .monitoring_metrics import _metric, _report_status, _validate_inputs
from .spark_monitoring_drift import drift_evidence
from .spark_monitoring_metrics import (
    _column,
    _exists,
    _functions,
    _numeric,
    _scored,
    _validate_keys,
    build_spark_performance_report,
)


def _missing(frame: Any, column: str) -> Any:
    """Match Core missingness: nulls everywhere plus NaN in numeric columns."""
    missing = _column(column).isNull()
    return missing | _functions().isnan(_column(column)) if _numeric(frame, column) else missing


def _quality_counts(frame: Any, column: str) -> tuple[int, int, int, int]:
    """Collect a single aggregate row of population, missing, NaN and infinity counts."""
    if column not in frame.columns:
        return 0, 0, 0, 0
    f = _functions()
    numeric = _numeric(frame, column)
    infinite = f.abs(_column(column).cast("double")) == float("inf") if numeric else f.lit(False)
    nan = f.isnan(_column(column)) if numeric else f.lit(False)
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


def _missing_increase(missing: int, total: int, baseline_missing: int, baseline_total: int) -> bool:
    """Use the existing one-sided Fisher missingness gate on exact aggregate counts."""
    baseline = baseline_missing / baseline_total if baseline_total else 0.0
    if missing / total <= baseline:
        return False
    if baseline == 0:
        return True
    table = [[missing, total - missing], [baseline_missing, baseline_total - baseline_missing]]
    return bool(fisher_exact(table, alternative="greater").pvalue <= 0.01)


def _quality(reference: Any, current: Any, unscored: Any, column: str) -> list[dict]:
    """Require usable saved predictions for missing inputs even with accepted baseline rates."""
    total, missing, nonfinite, infinite = _quality_counts(current, column)
    if not total:
        return [
            _metric("quality", column, name) for name in ("missing_fraction", "nonfinite_fraction")
        ]
    baseline_total, baseline_missing, _, _ = _quality_counts(reference, column)
    issue = _missing_increase(missing, total, baseline_missing, baseline_total)
    issue = issue or _exists(unscored.where(_missing(current, column)))
    baseline = baseline_missing / baseline_total if baseline_total else 0.0
    return [
        _metric("quality", column, "missing_fraction", missing / total, baseline, issue),
        _metric("quality", column, "nonfinite_fraction", nonfinite / total, issue=bool(infinite)),
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
            for item in [_metric("drift", column, "unavailable"), *quality[column]]
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
    if _exists(predictions):
        if not _exists(current):
            raise ValueError("A prediction key is absent from current records.")
        known = current.select(*[_column(key) for key in keys])
        if _exists(predictions.join(known, list(keys), "left_anti")):
            raise ValueError("A prediction key is absent from current records.")
    if not _exists(scored):
        return current
    return current.join(scored.select(*[_column(key) for key in keys]), list(keys), "left_anti")


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
    _validate_inputs(as_of, task, classes, thresholds)
    _validate_keys(current, record_key_columns, "current")
    _validate_keys(predictions, record_key_columns, "predictions")
    scored, excluded = _scored(predictions, task, classes)
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
        _metric("performance", target_column, name, value)
        for name, value in performance["values"].items()
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
    status = _report_status(
        reference_count, current_count, drifted, unavailable, quality_issue, bad_performance
    )
    return {
        "status": status,
        "reference_rows": reference_count,
        "current_rows": current_count,
        "scored_rows": performance["scored_rows"],
        "labeled_rows": performance["labeled_rows"],
        "label_coverage": performance["label_coverage"],
        "drifted_columns": drifted,
        "metrics": metrics,
        "notes": notes,
    }
