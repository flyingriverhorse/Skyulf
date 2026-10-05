"""Assess bounded eligible training changes before requesting an on-drift fit."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import pandas as pd

from ...jobs.lifecycle.lifecycle_tasks import phase_training_spec
from ...lifecycle.local_workflow import resolve_training_spec
from ...observability.monitoring.monitoring_config import MonitorConfig
from ...observability.monitoring.monitoring_reference import load_monitoring_reference
from ...training.fitting.local_pre_split import FIXED_TYPES
from ...training.fitting.local_retraining import (
    LocalTrainingSpec,
    eligible_training_snapshot,
    read_training_partitions,
    training_spec_payload,
    validate_pre_split_step,
)


def _number(value: Any) -> str:
    """Normalize equivalent integer and floating values without rounding to machine floats."""
    number = Decimal(str(value))
    if not number.is_finite():
        return str(number)
    numerator, denominator = number.as_integer_ratio()
    return f"{numerator}/{denominator}"


def _scalar(value: Any) -> tuple[str, Any]:
    """Encode supported scalar feature values without unstable object representations."""
    if pd.isna(value):
        return "null", None
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, (Integral, Real, Decimal)):
        return "number", _number(value)
    if isinstance(value, datetime):
        stamp = value.astimezone(UTC) if value.tzinfo is not None else value
        return "datetime", stamp.isoformat()
    if isinstance(value, date):
        return "date", value.isoformat()
    if isinstance(value, bytes):
        return "bytes", value.hex()
    if isinstance(value, str):
        return "str", value
    raise ValueError(f"Unsupported training freshness value type: {type(value).__name__}.")


def _row_counts(frame: pd.DataFrame, columns: list[str]) -> Counter[str]:
    """Hash feature-target multisets, preserving duplicates but ignoring order and dtype widening."""
    counts: Counter[str] = Counter()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        payload = json.dumps([_scalar(value) for value in row], separators=(",", ":"))
        counts[hashlib.sha256(payload.encode()).hexdigest()] += 1
    return counts


def _compatible_specs(saved: LocalTrainingSpec, current: LocalTrainingSpec) -> None:
    """Prevent automatic comparisons across unrelated training source contracts."""
    fields = (
        "table",
        "record_key_columns",
        "input_columns",
        "target_column",
        "split_strategy",
        "event_column",
        "group_column",
        "event_time_parsing",
        "result_available_at_column",
        "result_time_parsing",
        "drop_missing_labels",
    )
    if any(getattr(saved, field) != getattr(current, field) for field in fields):
        raise ValueError(
            "On-drift training source and split contract differ from the active model."
        )


def _bounded(spec: LocalTrainingSpec, monitor: MonitorConfig) -> LocalTrainingSpec:
    """Respect the stricter training or monitoring materialization limits on every read."""
    return replace(
        spec,
        max_rows=min(spec.max_rows, monitor.max_rows),
        max_bytes=min(spec.max_bytes, monitor.max_bytes),
    )


def _baseline_counts(
    spark: Any,
    monitor: MonitorConfig,
    spec: LocalTrainingSpec,
    train: pd.DataFrame,
    engine: str,
    current: LocalTrainingSpec,
    current_frame: pd.DataFrame,
) -> Counter[str]:
    """Count prior eligible values under both original and current source weights."""
    columns = [*spec.input_columns, spec.target_column]
    weights = _filter_weight_columns(spec, current)
    if spec.split_strategy == "temporal" and not weights:
        return _row_counts(train, columns)
    bounded = _bounded(spec, monitor)
    source, old_train, old_holdout, _ = read_training_partitions(
        spark, bounded, temporal_cv=False, engine=engine
    )
    baseline = _population_counts(spec, old_train, old_holdout)
    if not weights:
        return baseline
    source = _replace_source_weights(source, current_frame, spec.record_key_columns, weights)
    population = eligible_training_snapshot(source, _comparison_spec(bounded), engine=engine)
    if spec.split_strategy == "temporal":
        population = population.loc[population[spec.event_column] < spec.holdout_start]
    return baseline | _row_counts(population, columns)


def _comparison_spec(spec: LocalTrainingSpec) -> LocalTrainingSpec:
    """Keep historical label alternatives without requiring a viable model-fit population."""
    steps = deepcopy(list(spec.pre_split_steps))
    for step in steps:
        if step["transformer"] == "Deduplicate":
            subset = step["params"]["subset"]
            step["params"]["subset"] = list(dict.fromkeys([*subset, spec.target_column]))
    steps.append(
        {
            "name": "freshness_available_target",
            "transformer": "DropMissingRows",
            "params": {"subset": [spec.target_column]},
        }
    )
    return replace(
        spec,
        pre_split_steps=tuple(steps),
        weight_column=None,
        holdout_key_sha256=None,
        survivor_key_sha256=None,
        training_evidence_sha256=None,
    )


def _filter_weight_columns(saved: LocalTrainingSpec, current: LocalTrainingSpec) -> set[str]:
    """Find declared source weights read by the original eligibility recipe."""
    weights = {
        column.casefold()
        for spec in (saved, current)
        for column in (*spec.reserved_weight_columns, spec.weight_column)
        if column is not None
    }
    return {
        column
        for index, step in enumerate(saved.pre_split_steps, 1)
        if step["transformer"] not in FIXED_TYPES
        for column in validate_pre_split_step(step, index, saved.target_column, ())
        if column.casefold() in weights
    }


def _population_counts(
    spec: LocalTrainingSpec, train: pd.DataFrame, holdout: pd.DataFrame
) -> Counter[str]:
    """Treat random holdout as seen, but retain the original temporal training boundary."""
    population = train if spec.split_strategy == "temporal" else pd.concat([train, holdout])
    return _row_counts(population, [*spec.input_columns, spec.target_column])


def _replace_source_weights(
    saved: pd.DataFrame,
    current: pd.DataFrame,
    keys: tuple[str, ...],
    weights: set[str],
) -> pd.DataFrame:
    """Replace only matching records' weight metadata, preserving historical X, y and order."""
    historical = saved.set_index(list(keys))
    latest = current.set_index(list(keys))
    common = historical.index.intersection(latest.index, sort=False)
    for column in weights.intersection(current.columns):
        historical.loc[common, column] = latest.loc[common, column]
    return historical.reset_index().loc[:, saved.columns]


def assess_training_data(spark: Any, monitor: MonitorConfig, workflow: dict, now: datetime) -> dict:
    """Compare actual eligible fit values to the active model without fitting or writing.

    Random splitting treats the previous train and holdout union as already seen:
    shuffling old rows cannot justify retraining. Temporal splitting instead permits
    former holdout rows to enter training when the configured time boundary advances.
    Null and unavailable targets follow the existing training eligibility policy;
    invalid training inputs fail rather than silently changing that policy.
    """
    if monitor.execution_engine == "spark":
        from .spark_retraining_data import assess_spark_training_data  # noqa: PLC0415

        return assess_spark_training_data(spark, monitor, workflow, now)
    if workflow.get("training_version") is not None:
        raise ValueError(
            "On-drift retraining requires training_version=null for the latest snapshot."
        )
    artifact, saved, old_train, evidence = load_monitoring_reference(
        spark,
        monitor,
        tracking_uri=workflow.get("tracking_uri", "databricks"),
        registry_uri=workflow.get("registry_uri", "databricks-uc"),
    )
    engine = workflow["engine"]
    if engine != artifact.manifest.fitted_engine:
        raise ValueError("On-drift training engine differs from the active model.")
    current = resolve_training_spec(spark, workflow, now)
    _compatible_specs(saved, current)
    # Reference replay may install older custom builders; restore this job's source.
    current = phase_training_spec(
        training_spec_payload(current, engine),
        workflow.get("pipeline", {}).get("project_python_source"),
    )
    current_frame, train, _, _ = read_training_partitions(
        spark, _bounded(current, monitor), temporal_cv=False, engine=engine
    )
    saved = phase_training_spec(
        training_spec_payload(saved, engine), artifact.pipeline.config.get("project_python_source")
    )
    try:
        baseline = _baseline_counts(
            spark, monitor, saved, old_train, engine, current, current_frame
        )
    finally:
        phase_training_spec(
            training_spec_payload(current, engine),
            workflow.get("pipeline", {}).get("project_python_source"),
        )
    columns = [*current.input_columns, current.target_column]
    counts = _row_counts(train, columns)
    changed = sum((counts - baseline).values())
    payload = json.dumps(
        {"columns": columns, "rows": sorted(counts.items())}, separators=(",", ":")
    )
    return {
        "status": "ready" if changed else "no_new_training_data",
        "training_rows": len(train),
        "changed_rows": changed,
        "content_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_table": current.table,
        "source_version": current.version,
        "baseline_model_version": evidence["model_version"],
    }
