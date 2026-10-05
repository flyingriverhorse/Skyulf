"""Build finite monitoring evidence from bounded saved feature and prediction rows."""

import math
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import fisher_exact

from skyulf.modeling._evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from skyulf.profiling.drift import DriftCalculator

_DRIFT_THRESHOLDS = {"psi", "ks_statistic", "wasserstein", "kl_divergence"}
_REGRESSION_METRICS = ("mae", "mse", "rmse", "r2", "mape", "explained_variance")
_CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "matthews_corrcoef",
    "g_score",
)


def _frame(value: pd.DataFrame | pl.DataFrame, name: str) -> pl.DataFrame:
    """Normalize the two supported bounded frame engines at the Core boundary."""
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, pd.DataFrame):
        return pl.from_pandas(value, nan_to_null=False)
    raise TypeError(f"{name} must be a pandas or Polars DataFrame.")


def _metric(
    category: str,
    column: str,
    name: str,
    value: float | None = None,
    threshold: float | None = None,
    issue: bool = False,
) -> dict[str, Any]:
    """Keep every scalar JSON-safe and mark nonfinite calculations unavailable."""
    finite = value is not None and math.isfinite(value)
    return {
        "category": category,
        "column_name": column,
        "metric_name": name,
        "value": float(value) if finite else None,
        "threshold": threshold,
        "has_issue": bool(issue or (value is not None and not finite)),
        "status": "measured" if finite else "unavailable",
    }


def _validate_inputs(
    as_of: datetime,
    task: str,
    classes: tuple,
    thresholds: dict | None,
) -> None:
    """Reject ambiguous policy and invalid measurement cutoffs before scoring."""
    if not isinstance(as_of, datetime) or as_of.utcoffset() != timedelta(0):
        raise ValueError("as_of must be an aware UTC timestamp.")
    if task not in {"regression", "classification"}:
        raise ValueError("Monitoring task must be regression or classification.")
    if task == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
        raise ValueError("Classification classes must be distinct and contain at least two labels.")
    _validate_thresholds(thresholds)


def _validate_thresholds(thresholds: dict | None) -> None:
    """Accept only finite positive overrides supported by Core drift."""
    if thresholds is None:
        return
    if type(thresholds) is not dict or set(thresholds) - _DRIFT_THRESHOLDS:
        raise ValueError("Unknown monitoring drift thresholds.")
    for value in thresholds.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Drift thresholds must be finite and positive.")


def _keyed_rows(frame: pl.DataFrame, columns: tuple[str, ...], name: str) -> dict[tuple, dict]:
    """Reject missing, null and duplicate composite keys before any join."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Record key columns must be distinct and nonempty.")
    if not len(frame):
        return {}
    if set(columns) - set(frame.columns):
        raise ValueError(f"{name} is missing record key columns.")
    keyed: dict[tuple, dict] = {}
    for row in frame.to_dicts():
        key = _row_key(row, columns, name)
        if key in keyed:
            raise ValueError(f"{name} has duplicate record keys.")
        keyed[key] = row
    return keyed


def _row_key(row: dict, columns: tuple[str, ...], name: str) -> tuple:
    """Extract one complete record key from a materialized row."""
    key = tuple(row[column] for column in columns)
    if any(value is None or _not_finite_number(value) for value in key):
        raise ValueError(f"{name} has a null record key.")
    return key


def _not_finite_number(value: Any) -> bool:
    """Recognize nonfinite numeric scalars without coercing string keys."""
    return isinstance(value, (int, float)) and not math.isfinite(value)


def _availability(value: Any) -> datetime:
    """Parse only valid timezone-aware availability timestamps."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid label availability timestamp.") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Invalid or naive label availability timestamp.")
    return value


def _eligible_labels(
    labels: pl.DataFrame | None,
    available_column: str,
    as_of: datetime,
    keys: tuple[str, ...],
    target: str,
) -> dict[tuple, dict]:
    """Filter future labels before enforcing unique eligible record keys."""
    if labels is None or not len(labels):
        return {}
    if target not in labels.columns or available_column not in labels.columns:
        raise ValueError("Labels need target and availability columns.")
    eligible = [row for row in labels.to_dicts() if _availability(row[available_column]) <= as_of]
    return _keyed_rows(pl.DataFrame(eligible) if eligible else pl.DataFrame(), keys, "labels")


def _probability_columns(
    predictions: pl.DataFrame, classes: tuple, task: str, scored: dict[tuple, dict]
) -> tuple[str, ...]:
    """Validate the complete saved class-ordered probability contract."""
    columns = tuple(name for name in predictions.columns if name.startswith("probability_"))
    if not columns:
        return ()
    expected = tuple(f"probability_{index}" for index in range(len(classes)))
    if task != "classification" or set(columns) != set(expected):
        raise ValueError("Prediction probabilities must match saved classes.")
    for row in scored.values():
        _validate_probability_row(tuple(row[column] for column in expected))
    return expected


def _validate_probability_row(row: tuple) -> None:
    """Reject partial, nonfinite and unnormalized probability vectors."""
    if any(value is None or _not_finite_number(value) for value in row):
        raise ValueError("Prediction probabilities must be finite.")
    if any(not 0 <= value <= 1 for value in row) or not math.isclose(sum(row), 1, abs_tol=1e-6):
        raise ValueError("Prediction probabilities must be in [0, 1] and sum to one.")


def _feature_quality(
    frame: pl.DataFrame, column: str, reference: pl.DataFrame | None = None
) -> list[dict[str, Any]]:
    """Flag significant missingness increases and infinity, keeping raw fractions."""
    if column not in frame.columns or not len(frame):
        return [
            _metric("quality", column, name) for name in ("missing_fraction", "nonfinite_fraction")
        ]
    values = frame[column].to_list()
    numeric = frame[column].dtype.is_numeric()
    missing = sum(_is_missing(value, numeric) for value in values)
    nonfinite = sum(numeric and _not_finite_number(value) for value in values)
    baseline = _reference_missing_fraction(reference, column)
    return [
        _metric(
            "quality",
            column,
            "missing_fraction",
            missing / len(values),
            baseline,
            issue=_missingness_increased(missing, len(values), reference, column),
        ),
        _metric(
            "quality",
            column,
            "nonfinite_fraction",
            nonfinite / len(values),
            issue=_has_infinite(frame[column]),
        ),
    ]


def _missingness_increased(
    missing: int, total: int, reference: pl.DataFrame | None, column: str
) -> bool:
    """Test a one-sided increase at 1%; previously absent missingness remains an issue.

    Fisher's exact test accounts for the sizes of both independent samples.
    Successful saved predictions are still required separately for missing rows.
    """
    baseline = _reference_missing_fraction(reference, column)
    if missing / total <= baseline:
        return False
    if baseline == 0 or reference is None:
        return True
    observed = round(baseline * len(reference))
    result = fisher_exact(
        [[missing, total - missing], [observed, len(reference) - observed]], alternative="greater"
    )
    return bool(result.pvalue <= 0.01)


def _reference_missing_fraction(reference: pl.DataFrame | None, column: str) -> float:
    """Use only a present, populated training column as the missingness baseline."""
    if reference is None or column not in reference.columns or not len(reference):
        return 0.0
    series = reference[column]
    return sum(_is_missing(value, series.dtype.is_numeric()) for value in series) / len(series)


def _mark_unhandled_missing(
    metrics: list[dict], current: pl.DataFrame, current_rows: dict, scored: dict
) -> None:
    """Require valid saved predictions for missing inputs before accepting the baseline."""
    unscored = [row for key, row in current_rows.items() if key not in scored]
    for item in metrics:
        if item["category"] != "quality" or item["metric_name"] != "missing_fraction":
            continue
        column = item["column_name"]
        if column not in current.columns:
            continue
        numeric = current[column].dtype.is_numeric()
        if any(_is_missing(row[column], numeric) for row in unscored):
            item["has_issue"] = True


def _is_missing(value: Any, numeric: bool) -> bool:
    """Count null and numeric NaN as absent feature observations."""
    return value is None or (numeric and isinstance(value, float) and math.isnan(value))


def _drift_evidence(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: tuple[str, ...],
    thresholds: dict | None,
) -> tuple[list[dict[str, Any]], int, list[str], bool]:
    """Adapt Core drift evidence while exposing its deliberately skipped columns."""
    common = [
        column for column in columns if column in reference.columns and column in current.columns
    ]
    missing = [
        column
        for column in columns
        if column not in reference.columns or column not in current.columns
    ]
    metrics, notes = _schema_evidence(current, missing)
    common_metrics, common_drifted, common_notes, unmeasured = _common_evidence(
        reference, current, common, thresholds
    )
    return metrics + common_metrics, len(missing) + common_drifted, notes + common_notes, unmeasured


def _schema_evidence(
    current: pl.DataFrame,
    missing: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep declared missing feature columns visible as schema drift."""
    metrics: list[dict[str, Any]] = []
    notes = []
    for column in missing:
        metrics.append(_metric("drift", column, "schema_missing", 1.0, 0.0, True))
        notes.append(f"Feature {column} is missing from a monitoring input.")
        metrics.extend(_feature_quality(current, column))
    return metrics, notes


def _common_evidence(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    common: list[str],
    thresholds: dict | None,
) -> tuple[list[dict[str, Any]], int, list[str], bool]:
    """Expose both measured Core statistics and skipped common columns."""
    comparable = [
        column
        for column in common
        if not _has_infinite(reference[column]) and not _has_infinite(current[column])
    ]
    core_report = (
        DriftCalculator(reference.select(comparable), current.select(comparable)).calculate_drift(
            thresholds
        )
        if comparable
        else None
    )
    metrics: list[dict[str, Any]] = []
    drifted = 0
    notes = []
    unmeasured = False
    for column in common:
        metrics.extend(_feature_quality(current, column, reference))
        result = core_report.column_drifts.get(column) if core_report is not None else None
        evidence = _column_drift_evidence(column, result)
        metrics.extend(evidence)
        if result is None:
            notes.append(f"Feature {column} could not be measured for drift.")
            unmeasured = True
        else:
            drifted += int(result.drift_detected)
            if result.evidence is not None and result.evidence.status in {
                "insufficient_data",
                "unavailable",
            }:
                notes.append(f"Feature {column}: {result.evidence.reason}")
                unmeasured = True
    return metrics, drifted, notes, unmeasured


def _column_drift_evidence(column: str, result: Any) -> list[dict[str, Any]]:
    """Retain Core's per-statistic verdict or an explicit unavailable row."""
    if result is None:
        return [_metric("drift", column, "unavailable")]
    metrics = [
        _metric(
            "drift",
            column,
            item.metric,
            item.value,
            None if item.metric == "ks_test_p_value" else item.threshold,
            result.drift_detected and item.has_drift and item.metric != "ks_test_p_value",
        )
        for item in result.metrics
    ]
    if result.evidence is not None:
        evidence = result.evidence
        value = (
            evidence.adjusted_p_value if evidence.status in {"supported", "not_detected"} else None
        )
        metrics.append(
            _metric("drift", column, "statistical_evidence", value, evidence.significance_level)
            | {"evidence": evidence.model_dump()}
        )
    return metrics


def _has_infinite(series: pl.Series) -> bool:
    """Prevent Core's deliberate infinity error from erasing other features."""
    return series.dtype.is_numeric() and any(
        value is not None and isinstance(value, (int, float)) and math.isinf(value)
        for value in series.to_list()
    )


def _performance_values(
    task: str,
    pairs: list[tuple[Any, Any, dict]],
    classes: tuple,
    probability_columns: tuple[str, ...],
) -> dict[str, float]:
    """Reuse Core evaluators with saved arrays and metadata, never estimator inference."""
    truth = np.asarray([pair[0] for pair in pairs])
    guesses = np.asarray([pair[1] for pair in pairs])
    features = np.empty((len(pairs), 0))
    if task == "regression":
        truth = truth.astype(float)
        guesses = guesses.astype(float)
        return calculate_regression_metrics(
            None,
            pd.DataFrame(),
            truth,
            X_np=features,
            y_np=truth,
            predictions=guesses,
        )
    # Encode against the saved order: sklearn probability metrics otherwise sort
    # string labels independently of the saved probability column order.
    indexes = {label: index for index, label in enumerate(classes)}
    truth = np.asarray([indexes[pair[0]] for pair in pairs])
    guesses = np.asarray([indexes[pair[1]] for pair in pairs])
    probabilities = (
        np.asarray([[pair[2][column] for column in probability_columns] for pair in pairs])
        if probability_columns
        else None
    )
    metadata = SimpleNamespace(classes_=np.arange(len(classes)))
    return calculate_classification_metrics(
        metadata,
        pd.DataFrame(),
        truth,
        X_np=features,
        y_np=truth,
        predictions=guesses,
        proba=probabilities,
    )


def _performance_evidence(
    task: str,
    classes: tuple,
    target: str,
    scored: dict[tuple, dict],
    labels: dict[tuple, dict],
    probability_columns: tuple[str, ...],
) -> tuple[list[dict[str, Any]], int, list[str], bool]:
    """Report an explicit unavailable row when there are too few finite pairs."""
    names = _REGRESSION_METRICS if task == "regression" else _CLASSIFICATION_METRICS
    notes: list[str] = []
    pairs = _labeled_pairs(task, classes, target, scored, labels)
    if len(pairs) < 2:
        notes.append("Performance unavailable: fewer than two finite labeled pairs.")
        return [_metric("performance", target, name) for name in names], len(pairs), notes, False
    values = _performance_values(task, pairs, classes, probability_columns)
    evidence = [
        _metric("performance", target, name, float(value)) for name, value in values.items()
    ]
    unavailable = any(item["status"] == "unavailable" for item in evidence)
    if unavailable:
        notes.append("A calculated performance metric was nonfinite.")
    return evidence, len(pairs), notes, unavailable


def _labeled_pairs(
    task: str,
    classes: tuple,
    target: str,
    scored: dict[tuple, dict],
    labels: dict[tuple, dict],
) -> list[tuple[Any, Any, dict]]:
    """Join only available labels to scored saved predictions by key."""
    pairs = []
    for key, prediction in scored.items():
        if key not in labels:
            continue
        actual, guess = labels[key][target], prediction.get("prediction")
        if _is_missing(actual, True):
            continue
        if task == "regression" and not _finite_pair(actual, guess):
            continue
        if task == "classification" and (actual not in classes or guess not in classes):
            raise ValueError("Prediction and label classes must match saved classes.")
        pairs.append((actual, guess, prediction))
    return pairs


def _finite_pair(actual: Any, guess: Any) -> bool:
    """Only measured numeric outcomes count toward regression coverage."""
    return all(
        isinstance(value, (int, float, Decimal)) and math.isfinite(value)
        for value in (actual, guess)
    )


def _scored_rows(prediction_rows: dict[tuple, dict]) -> tuple[dict[tuple, dict], int]:
    """Keep excluded rows visible without allowing them into performance."""
    scored = {}
    excluded = 0
    for key, row in prediction_rows.items():
        status = row.get("scoring_status", "predicted")
        if status not in {"predicted", "excluded"}:
            raise ValueError("scoring_status must be predicted or excluded.")
        if status == "excluded":
            excluded += 1
        else:
            scored[key] = row
    return scored, excluded


def _validate_saved_predictions(scored: dict[tuple, dict], task: str, classes: tuple) -> None:
    """Reject invalid predicted outputs even before real outcomes arrive."""
    for row in scored.values():
        value = row.get("prediction")
        valid = _finite_pair(value, value) if task == "regression" else value in classes
        if not valid:
            raise ValueError("Saved prediction does not match the model output contract.")


def _feature_report(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    columns: tuple[str, ...],
    thresholds: dict | None,
) -> tuple[list[dict[str, Any]], int, list[str], bool]:
    """Give empty populations a no-data verdict with unavailable feature rows."""
    if len(reference) and len(current):
        return _drift_evidence(reference, current, columns, thresholds)
    metrics = []
    for column in columns:
        metrics.append(_metric("drift", column, "unavailable"))
        metrics.extend(_feature_quality(current, column))
    return metrics, 0, ["Reference or current feature data is empty."], True


def _report_status(
    reference_rows: int,
    current_rows: int,
    drifted: int,
    unmeasured: bool,
    quality_issue: bool,
    bad_performance: bool,
) -> str:
    """Apply the report's documented status precedence."""
    if not reference_rows or not current_rows:
        return "no_data"
    if drifted:
        return "drift"
    if unmeasured or quality_issue or bad_performance:
        return "degraded"
    return "healthy"


def _output_or_quality_issue(metrics: list[dict], scored: dict, notes: list[str]) -> bool:
    """Require usable outputs while preserving missing labels as a separate condition."""
    if not scored:
        notes.append("No current rows have a saved predicted output.")
        return True
    return any(item["has_issue"] for item in metrics if item["category"] == "quality")


def build_monitoring_report(
    reference: pd.DataFrame | pl.DataFrame,
    current: pd.DataFrame | pl.DataFrame,
    predictions: pd.DataFrame | pl.DataFrame,
    labels: pd.DataFrame | pl.DataFrame | None,
    *,
    feature_columns: tuple[str, ...],
    record_key_columns: tuple[str, ...],
    target_column: str,
    result_available_at_column: str,
    as_of: datetime,
    task: str,
    classes: tuple = (),
    thresholds: dict | None = None,
) -> dict:
    """Compare declared features and saved outcomes at one UTC observation time.

    Existing missingness, including sampling variation, is accepted only when
    affected current rows have valid saved predictions. A one-sided Fisher test
    at 1% flags significant increases; missingness absent from training, unprocessed
    missing inputs and infinity remain quality issues. The raw nonfinite fraction
    still counts NaN, whose quality verdict comes from the missingness check.
    """
    _validate_inputs(as_of, task, classes, thresholds)
    reference = _frame(reference, "reference")
    current = _frame(current, "current")
    predictions = _frame(predictions, "predictions")
    labels = None if labels is None else _frame(labels, "labels")
    current_rows = _keyed_rows(current, record_key_columns, "current")
    prediction_rows = _keyed_rows(predictions, record_key_columns, "predictions")
    if set(prediction_rows) - set(current_rows):
        raise ValueError("A prediction key is absent from current records.")
    if prediction_rows and "prediction" not in predictions.columns:
        raise ValueError("Predictions need a prediction column.")
    scored, excluded = _scored_rows(prediction_rows)
    _validate_saved_predictions(scored, task, classes)
    probabilities = _probability_columns(predictions, classes, task, scored)
    eligible = _eligible_labels(
        labels, result_available_at_column, as_of, record_key_columns, target_column
    )
    notes: list[str] = []
    if excluded:
        notes.append(f"{excluded} excluded predictions were not scored.")
    metrics, drifted, feature_notes, unmeasured = _feature_report(
        reference, current, feature_columns, thresholds
    )
    _mark_unhandled_missing(metrics, current, current_rows, scored)
    notes.extend(feature_notes)
    performance, labeled, performance_notes, bad_performance = _performance_evidence(
        task, classes, target_column, scored, eligible, probabilities
    )
    metrics.extend(performance)
    notes.extend(performance_notes)
    if labels is None:
        notes.append("No labels were provided; performance is unavailable.")
    quality_issue = _output_or_quality_issue(metrics, scored, notes)
    status = _report_status(
        len(reference), len(current), drifted, unmeasured, quality_issue, bad_performance
    )
    return {
        "status": status,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "scored_rows": len(scored),
        "labeled_rows": labeled,
        "label_coverage": labeled / len(scored) if scored else None,
        "drifted_columns": drifted,
        "metrics": metrics,
        "notes": notes,
    }


def build_performance_report(
    predictions: pd.DataFrame | pl.DataFrame,
    labels: pd.DataFrame | pl.DataFrame | None,
    *,
    record_key_columns: tuple[str, ...],
    target_column: str,
    result_available_at_column: str,
    as_of: datetime,
    task: str,
    classes: tuple = (),
) -> dict:
    """Measure saved outcomes with the same class, eligibility and Core metric rules."""
    _validate_inputs(as_of, task, classes, None)
    predictions = _frame(predictions, "predictions")
    labels = None if labels is None else _frame(labels, "labels")
    prediction_rows = _keyed_rows(predictions, record_key_columns, "predictions")
    scored, excluded = _scored_rows(prediction_rows)
    _validate_saved_predictions(scored, task, classes)
    probabilities = _probability_columns(predictions, classes, task, scored)
    eligible = _eligible_labels(
        labels, result_available_at_column, as_of, record_key_columns, target_column
    )
    metrics, labeled, notes, _ = _performance_evidence(
        task, classes, target_column, scored, eligible, probabilities
    )
    return {
        "values": {item["metric_name"]: item["value"] for item in metrics},
        "scored_rows": len(scored),
        "excluded_rows": excluded,
        "labeled_rows": labeled,
        "label_coverage": labeled / len(scored) if scored else None,
        "notes": notes,
    }
