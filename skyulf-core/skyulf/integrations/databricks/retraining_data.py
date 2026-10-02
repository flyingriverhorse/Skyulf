"""Assess bounded eligible training changes before requesting an on-drift fit."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from numbers import Integral, Real
from typing import Any

import pandas as pd

from .lifecycle_tasks import phase_training_spec
from .local_retraining import (
    LocalTrainingSpec,
    read_training_partitions,
    training_spec_payload,
)
from .local_workflow import resolve_training_spec
from .monitoring_config import MonitorConfig
from .monitoring_reference import load_monitoring_reference


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


def _baseline_population(
    spark: Any,
    monitor: MonitorConfig,
    spec: LocalTrainingSpec,
    train: pd.DataFrame,
    engine: str,
) -> pd.DataFrame:
    """Exclude all previously eligible random rows, while allowing temporal holdout to age in."""
    if spec.split_strategy == "temporal":
        return train
    _, old_train, old_holdout, _ = read_training_partitions(
        spark, _bounded(spec, monitor), temporal_cv=False, engine=engine
    )
    return pd.concat([old_train, old_holdout], ignore_index=True)


def assess_training_data(spark: Any, monitor: MonitorConfig, workflow: dict, now: datetime) -> dict:
    """Compare actual eligible fit values to the active model without fitting or writing.

    Random splitting treats the previous train and holdout union as already seen:
    shuffling old rows cannot justify retraining. Temporal splitting instead permits
    former holdout rows to enter training when the configured time boundary advances.
    Null and unavailable targets follow the existing training eligibility policy;
    invalid training inputs fail rather than silently changing that policy.
    """
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
    baseline = _baseline_population(spark, monitor, saved, old_train, engine)
    # Reference replay may install older custom builders; restore this job's source.
    current = phase_training_spec(
        training_spec_payload(current, engine),
        workflow.get("pipeline", {}).get("project_python_source"),
    )
    _, train, _, _ = read_training_partitions(
        spark, _bounded(current, monitor), temporal_cv=False, engine=engine
    )
    columns = [*current.input_columns, current.target_column]
    counts = _row_counts(train, columns)
    changed = sum((counts - _row_counts(baseline, columns)).values())
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
