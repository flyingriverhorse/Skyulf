"""Distributed saved-output monitoring with bounded aggregate-only driver results."""

import importlib
import math
from datetime import datetime
from functools import reduce
from operator import and_, or_
from typing import Any

from ..local.monitoring_metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    validate_monitoring_inputs,
)

MAX_CLASSES = 256


def spark_functions() -> Any:
    """Keep Spark optional until a distributed entry point is invoked."""
    return importlib.import_module("pyspark.sql.functions")


def spark_column(name: str) -> Any:
    """Quote literal column names, including dots and embedded backticks."""
    return spark_functions().col("`" + name.replace("`", "``") + "`")


def finite_spark_value(value: Any) -> Any:
    """Recognize finite numeric Spark values without collecting observations."""
    f = spark_functions()
    return value.isNotNull() & ~f.isnan(value) & (f.abs(value.cast("double")) != float("inf"))


def is_numeric_column(frame: Any, name: str) -> bool:
    """Inspect schema metadata without coercing numeric-looking strings."""
    types = importlib.import_module("pyspark.sql.types")
    return isinstance(frame.schema[name].dataType, types.NumericType)


def has_spark_rows(frame: Any) -> bool:
    """Return existence through a bounded distributed count, never a raw row."""
    return bool(frame.limit(1).count())


def validate_record_keys(frame: Any, keys: tuple[str, ...], name: str) -> None:
    """Validate complete unique composite keys using distributed aggregates."""
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Record key columns must be distinct and nonempty.")
    if not has_spark_rows(frame):
        return
    if set(keys) - set(frame.columns):
        raise ValueError(f"{name} is missing record key columns.")
    bad = [
        ~finite_spark_value(spark_column(key))
        if is_numeric_column(frame, key)
        else spark_column(key).isNull()
        for key in keys
    ]
    if has_spark_rows(frame.where(reduce(or_, bad))):
        raise ValueError(f"{name} has a null record key.")
    key_frame = frame.select(
        *[spark_column(key).alias(f"key_{index}") for index, key in enumerate(keys)]
    )
    if has_spark_rows(key_frame.groupBy(*key_frame.columns).count().where("count > 1")):
        raise ValueError(f"{name} has duplicate record keys.")


def scored_predictions(predictions: Any, task: str, classes: tuple) -> tuple[Any, int]:
    """Reject invalid outputs independently of whether outcomes have arrived."""
    if "scoring_status" not in predictions.columns:
        scored, excluded = predictions, 0
    else:
        status = spark_column("scoring_status")
        if has_spark_rows(
            predictions.where(status.isNull() | ~status.isin("predicted", "excluded"))
        ):
            raise ValueError("scoring_status must be predicted or excluded.")
        scored = predictions.where(status == "predicted")
        excluded = predictions.where(status == "excluded").count()
    if not has_spark_rows(scored):
        return scored, excluded
    if "prediction" not in scored.columns:
        raise ValueError("Predictions need a prediction column.")
    valid = spark_column("prediction").isin(list(classes))
    if task == "regression":
        valid = (
            finite_spark_value(spark_column("prediction"))
            if is_numeric_column(scored, "prediction")
            else spark_functions().lit(False)
        )
    if has_spark_rows(
        scored.where(~spark_functions().coalesce(valid, spark_functions().lit(False)))
    ):
        raise ValueError("Saved prediction does not match the model output contract.")
    return scored, excluded


def _probability_names(scored: Any, classes: tuple, task: str) -> tuple[str, ...]:
    """Validate the saved probability schema and explicit class-index order."""
    columns = tuple(name for name in scored.columns if name.startswith("probability_"))
    if not columns:
        return ()
    expected = tuple(f"probability_{index}" for index in range(len(classes)))
    if task != "classification" or set(columns) != set(expected):
        raise ValueError("Prediction probabilities must match saved classes.")
    if has_spark_rows(scored) and any(not is_numeric_column(scored, name) for name in expected):
        raise ValueError("Prediction probabilities must be finite numbers.")
    return expected


def _probabilities(scored: Any, classes: tuple, task: str) -> tuple[str, ...]:
    """Validate all predicted probability vectors against saved class order."""
    expected = _probability_names(scored, classes, task)
    if not expected or not has_spark_rows(scored):
        return expected
    values = [spark_column(name) for name in expected]
    valid = reduce(and_, [finite_spark_value(value) & value.between(0, 1) for value in values])
    valid = valid & (spark_functions().abs(sum(values) - 1) <= 1e-6)
    if has_spark_rows(scored.where(~valid)):
        raise ValueError("Prediction probabilities must be finite, in [0, 1] and sum to one.")
    return expected


def _eligible(labels: Any, available: str, cutoff: datetime, keys: tuple, target: str) -> Any:
    """Validate availability before filtering, then enforce only eligible key uniqueness."""
    if labels is None or not has_spark_rows(labels):
        return None
    if target not in labels.columns or available not in labels.columns:
        raise ValueError("Labels need target and availability columns.")
    f = spark_functions()
    types = importlib.import_module("pyspark.sql.types")
    dtype = labels.schema[available].dataType
    value = spark_column(available)
    if isinstance(dtype, types.TimestampType):
        parsed, valid = value, value.isNotNull()
    elif isinstance(dtype, types.StringType):
        parsed = f.try_to_timestamp(value)
        valid = value.rlike(r"(?i)[T ].*(Z|[+-]\d{2}(?::?\d{2})?)$") & parsed.isNotNull()
    else:
        raise ValueError("Invalid or naive label availability timestamp.")
    if has_spark_rows(labels.where(~f.coalesce(valid, f.lit(False)))):
        raise ValueError("Invalid or naive label availability timestamp.")
    eligible = labels.where(parsed <= f.lit(cutoff))
    if not has_spark_rows(eligible):
        return None
    validate_record_keys(eligible, keys, "labels")
    return eligible


def _join_pairs(scored: Any, labels: Any, keys: tuple, target: str) -> Any:
    """Project internal join keys so user columns cannot collide with evidence aliases."""
    key_aliases = [spark_column(key).alias(f"__sm_key_{index}") for index, key in enumerate(keys)]
    truth = labels.select(*key_aliases, spark_column(target).alias("__sm_truth"))
    output_columns = [name for name in scored.columns if name.startswith("probability_")]
    outputs = scored.select(
        *key_aliases, spark_column("prediction"), *[spark_column(name) for name in output_columns]
    )
    return outputs.join(truth, [f"__sm_key_{index}" for index in range(len(keys))], "inner")


def _pairs(scored: Any, labels: Any, keys: tuple, target: str, task: str, classes: tuple) -> Any:
    """Join eligible labels with unambiguous internal aliases and finite outcome filtering."""
    if labels is None or not has_spark_rows(scored):
        return None
    pairs = _join_pairs(scored, labels, keys, target)
    actual = spark_column("__sm_truth")
    if task == "regression":
        return (
            pairs.where(finite_spark_value(actual))
            if is_numeric_column(pairs, "__sm_truth")
            else None
        )
    present = actual.isNotNull()
    if is_numeric_column(pairs, "__sm_truth"):
        present = present & ~spark_functions().isnan(actual)
    pairs = pairs.where(present)
    if has_spark_rows(pairs.where(~actual.isin(list(classes)))):
        raise ValueError("Prediction and label classes must match saved classes.")
    return pairs


def _regression_values(summary: dict) -> dict[str, float]:
    """Apply sklearn force-finite semantics to stable population variance aggregates."""
    mse, variance, residual = summary["mse"], summary["variance"], summary["residual_variance"]
    return {
        "mae": summary["mae"],
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": 1 - mse / variance if variance else float(mse == 0),
        "mape": summary["mape"],
        "explained_variance": 1 - residual / variance if variance else float(residual == 0),
    }


def _regression(pairs: Any) -> dict[str, float]:
    """Reduce regression populations to one stable finite-metric summary row."""
    f = spark_functions()
    truth, guess = (
        spark_column("__sm_truth").cast("double"),
        spark_column("prediction").cast("double"),
    )
    error = truth - guess
    summary = (
        pairs.agg(
            f.avg(f.abs(error)).alias("mae"),
            f.avg(error * error).alias("mse"),
            f.var_pop(truth).alias("variance"),
            f.var_pop(error).alias("residual_variance"),
            f.avg(f.abs(error) / f.greatest(f.abs(truth), f.lit(2.220446049250313e-16))).alias(
                "mape"
            ),
        )
        .first()
        .asDict()
    )
    return _regression_values(summary)


def _confusion_values(cells: list[tuple[int, int, int]], class_count: int) -> dict[str, float]:
    """Compute weighted metrics from bounded confusion cells without expanding observations."""
    import numpy as np  # noqa: PLC0415 - bounded metadata only
    from sklearn import metrics  # noqa: PLC0415 - weighted aggregate adapter

    from skyulf.modeling._evaluation.metrics import geometric_mean_score  # noqa: PLC0415

    truth, guesses, weights = (np.asarray(values) for values in zip(*cells, strict=True))
    kwargs = {"sample_weight": weights}
    result = {
        "accuracy": float(metrics.accuracy_score(truth, guesses, **kwargs)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(truth, guesses, **kwargs)),
        "matthews_corrcoef": float(metrics.matthews_corrcoef(truth, guesses, **kwargs)),
    }
    for name in ("precision", "recall", "f1"):
        scorer = getattr(metrics, name + "_score")
        result[name + "_weighted"] = float(
            scorer(truth, guesses, average="weighted", zero_division=0, **kwargs)
        )
        if class_count == 2:
            result[name] = float(
                scorer(truth, guesses, average="binary", pos_label=1, zero_division=0, **kwargs)
            )
    if geometric_mean_score is not None:
        result["g_score"] = float(
            geometric_mean_score(truth, guesses, average="weighted", **kwargs)
        )
    return result


def _classification(pairs: Any, classes: tuple) -> tuple[dict[str, float], dict]:
    """Collect at most the explicitly bounded class-squared confusion matrix."""
    cells = pairs.groupBy("__sm_truth", "prediction").count().limit(MAX_CLASSES**2 + 1).collect()
    if len(cells) > MAX_CLASSES**2:
        raise ValueError("Monitoring confusion matrix exceeded its metadata limit.")
    indexes = {label: index for index, label in enumerate(classes)}
    counts = [(indexes[row[0]], indexes[row[1]], row[2]) for row in cells]
    by_pair = {(actual, predicted): count for actual, predicted, count in counts}
    matrix = {
        "status": "measured",
        "cells": [
            {
                "actual_label": str(actual),
                "predicted_label": str(predicted),
                "actual_index": i,
                "predicted_index": j,
                "count": by_pair.get((i, j), 0),
            }
            for i, actual in enumerate(classes)
            for j, predicted in enumerate(classes)
        ],
    }
    return _confusion_values(counts, len(classes)), matrix


def _performance(
    scored: Any,
    labels: Any,
    keys: tuple,
    target: str,
    task: str,
    classes: tuple,
    probabilities: tuple,
) -> tuple[dict, int, list[str], dict]:
    """Measure aggregate metrics and expose unsupported probability evidence explicitly."""
    pairs = _pairs(scored, labels, keys, target, task, classes)
    count = 0 if pairs is None else pairs.count()
    if count < 2:
        names = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
        return (
            dict.fromkeys(names),
            count,
            ["Performance unavailable: fewer than two finite labeled pairs."],
            {"status": "unavailable", "reason": "insufficient_labels", "cells": []},
        )
    if task == "regression":
        values = _regression(pairs)
        matrix = {"status": "not_applicable", "reason": "regression", "cells": []}
    else:
        values, matrix = _classification(pairs, classes)
    notes = []
    if probabilities:
        from .spark_monitoring_probability import probability_values  # noqa: PLC0415

        values.update(probability_values(pairs, classes, probabilities))
    if any(not math.isfinite(value) for value in values.values()):
        notes.append("A calculated performance metric was nonfinite.")
    return (
        {name: value if math.isfinite(value) else None for name, value in values.items()},
        count,
        notes,
        matrix,
    )


def build_spark_performance_report(
    predictions: Any,
    labels: Any,
    *,
    record_key_columns: tuple[str, ...],
    target_column: str,
    result_available_at_column: str,
    as_of: datetime,
    task: str,
    classes: tuple = (),
) -> dict:
    """Measure saved Spark outputs with distributed joins and bounded summary collection."""
    validate_monitoring_inputs(as_of, task, classes, None)
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"Monitoring supports at most {MAX_CLASSES} saved classes.")
    validate_record_keys(predictions, record_key_columns, "predictions")
    scored, excluded = scored_predictions(predictions, task, classes)
    probabilities = _probabilities(scored, classes, task)
    eligible = _eligible(
        labels, result_available_at_column, as_of, record_key_columns, target_column
    )
    values, labeled, notes, matrix = _performance(
        scored, eligible, record_key_columns, target_column, task, classes, probabilities
    )
    count = scored.count()
    return {
        "values": values,
        "scored_rows": count,
        "excluded_rows": excluded,
        "labeled_rows": labeled,
        "label_coverage": labeled / count if count else None,
        "notes": notes,
        "confusion_matrix": matrix,
    }


def build_spark_monitoring_report(
    reference: Any,
    current: Any,
    predictions: Any,
    labels: Any,
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
    """Combine independent Spark feature evidence and saved-output performance."""
    from .spark_monitoring_quality import monitoring_report  # noqa: PLC0415

    return monitoring_report(
        reference,
        current,
        predictions,
        labels,
        feature_columns=feature_columns,
        record_key_columns=record_key_columns,
        target_column=target_column,
        result_available_at_column=result_available_at_column,
        as_of=as_of,
        task=task,
        classes=classes,
        thresholds=thresholds,
    )
