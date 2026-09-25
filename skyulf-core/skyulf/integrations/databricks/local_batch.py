"""Bounded Delta reads feeding whole-frame pandas or Polars workflows.

Spark only selects a pinned, filtered source snapshot. Fitted feature
engineering and model prediction stay on the recorded local engine.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...inference.local_evaluation import evaluate_local_holdout as evaluate_local_holdout
from ...inference.local_pipeline import (
    LocalPipelineArtifact,
    load_local_pipeline,
    save_local_pipeline,
)
from ...pipeline import SkyulfPipeline
from ._contracts import column_name, table_name
from .local_sdk import PreparedLocalWorkflow


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalSourceSpec:
    """Pin one monthly UC read and its driver-side resource budget."""

    table: str
    version: int
    period_start: datetime
    period_end: datetime
    record_key_columns: tuple[str, ...]
    input_columns: tuple[str, ...]
    max_rows: int
    max_bytes: int
    period_column: str = "event_time"
    business_timezone: str = "UTC"

    def __post_init__(self) -> None:
        """Reject mutable or ambiguous requests before Spark access."""
        table_name(self.table)
        if type(self.version) is not int or self.version < 0:
            raise ValueError("version must be a nonnegative Delta snapshot version.")
        for name in ("period_start", "period_end"):
            value = getattr(self, name)
            if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware.")
            if value.astimezone(UTC).astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(
                tzinfo=None
            ):
                raise ValueError(f"{name} is not a valid local instant.")
        if self.period_start >= self.period_end:
            raise ValueError("period_start must precede period_end.")
        for name in (*self.record_key_columns, *self.input_columns, self.period_column):
            column_name(name)
        if not self.record_key_columns or not self.input_columns:
            raise ValueError("record_key_columns and input_columns must be nonempty.")
        names = [*self.record_key_columns, *self.input_columns, self.period_column]
        if len({name.lower() for name in names}) != len(names):
            raise ValueError("Row keys, input columns and period column must be distinct.")
        if type(self.max_rows) is not int or self.max_rows <= 0:
            raise ValueError("max_rows must be positive.")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        try:
            ZoneInfo(self.business_timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("business_timezone must name an IANA timezone.") from exc


@dataclass(frozen=True, slots=True)
class LocalScoreResult:
    """Return predictions with source and model identity for a caller-owned sink."""

    predictions: pd.DataFrame
    diagnostics: dict[str, str | int]


def _frame_bytes(frame: pd.DataFrame | pl.DataFrame) -> int:
    """Count the actual local frame allocation used for the memory guard."""
    if isinstance(frame, pd.DataFrame):
        return int(frame.memory_usage(index=True, deep=True).sum())
    return int(frame.estimated_size())


def fit_local_workflow(
    config: dict[str, Any],
    data: SplitDataset,
    *,
    target_column: str,
    artifact_path: str | Path,
    max_rows: int,
    max_bytes: int,
) -> LocalPipelineArtifact:
    """Fit an explicit bounded split and save the existing full-pipeline format."""
    if not isinstance(data, SplitDataset):
        raise TypeError("Training requires an explicit SplitDataset.")
    if not target_column or type(target_column) is not str:
        raise ValueError("target_column must be nonempty.")
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    frames = (data.train, data.test, data.validation)
    if any(
        frame is not None and not isinstance(frame, pd.DataFrame | pl.DataFrame) for frame in frames
    ):
        raise TypeError("Training splits must be pandas or Polars DataFrames.")
    present = [frame for frame in frames if isinstance(frame, pd.DataFrame | pl.DataFrame)]
    if sum(len(frame) for frame in present) > max_rows:
        raise ValueError("Training input exceeds max_rows.")
    if sum(_frame_bytes(frame) for frame in present) > max_bytes:
        raise ValueError("Training input exceeds max_bytes.")
    pipeline = SkyulfPipeline(config)
    pipeline.fit(data, target_column=target_column)
    save_local_pipeline(pipeline, artifact_path)
    return load_local_pipeline(artifact_path)


def read_local_source(spark: Any, spec: LocalSourceSpec) -> pd.DataFrame:
    """Filter and project a fixed Delta snapshot before bounded driver transfer."""
    if not isinstance(spec, LocalSourceSpec):
        raise TypeError("spec must be a LocalSourceSpec.")
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    start = spec.period_start.astimezone(UTC).isoformat()
    end = spec.period_end.astimezone(UTC).isoformat()
    selected = source.where(
        f"{column_name(spec.period_column)} >= TIMESTAMP '{start}' AND "
        f"{column_name(spec.period_column)} < TIMESTAMP '{end}'"
    )
    names = (*spec.record_key_columns, *spec.input_columns)
    selected = selected.select(*names).orderBy(*spec.record_key_columns).limit(spec.max_rows + 1)
    records: list[dict[str, Any]] = []
    serialized_bytes = 0
    for row in selected.toLocalIterator():
        if len(records) >= spec.max_rows:
            raise ValueError("Source exceeds max_rows.")
        record = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
        # This is the decoded row payload, not a claim about Spark wire framing.
        serialized_bytes += len(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        if serialized_bytes > spec.max_bytes:
            raise ValueError("Source exceeds max_bytes.")
        records.append(record)
    frame = pd.DataFrame.from_records(records, columns=names)
    if _frame_bytes(frame) > spec.max_bytes:
        raise ValueError("Source local frame exceeds max_bytes.")
    if frame.loc[:, list(spec.record_key_columns)].isna().any().any():
        raise ValueError("Source row keys must not be null.")
    if frame.duplicated(subset=list(spec.record_key_columns)).any():
        raise ValueError("Source row keys must be unique within the requested period.")
    return frame


def score_local_source(
    spark: Any, spec: LocalSourceSpec, prepared: PreparedLocalWorkflow
) -> LocalScoreResult:
    """Score one pinned period without refitting FE or writing a Delta table."""
    if not isinstance(prepared, PreparedLocalWorkflow):
        raise TypeError("prepared must be a PreparedLocalWorkflow.")
    source = prepared.config.source
    if source.kind != "uc_table" or source.table != spec.table or source.version != spec.version:
        raise ValueError("Prepared source must match the pinned Delta table and version.")
    if spec.max_rows > source.max_rows or spec.max_bytes > source.max_bytes:
        raise ValueError("Source budget exceeds the prepared workflow budget.")
    expected = (
        prepared.artifact.manifest.input_columns
        if isinstance(prepared.artifact, LocalPipelineArtifact)
        else tuple(column.name for column in prepared.artifact.manifest.input_schema)
    )
    if spec.input_columns != expected:
        raise ValueError("Source inputs must match the saved model's raw column order.")
    frame = read_local_source(spark, spec)
    predictions = (
        pd.DataFrame(columns=pd.Index(prepared.preflight.output_columns))
        if frame.empty
        else prepared.predict(frame.loc[:, list(spec.input_columns)])
    )
    if len(predictions) != len(frame):
        raise ValueError("Prediction changed source row membership.")
    result = pd.concat(
        [
            frame.loc[:, list(spec.record_key_columns)].reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )
    if _frame_bytes(result) > spec.max_bytes:
        raise ValueError("Prediction result exceeds max_bytes.")
    diagnostics: dict[str, str | int] = {
        "source_table": spec.table,
        "source_version": spec.version,
        "period_start_utc": spec.period_start.astimezone(UTC).isoformat(),
        "period_end_utc": spec.period_end.astimezone(UTC).isoformat(),
        "business_timezone": spec.business_timezone,
        "model_digest": prepared.preflight.model_digest or "",
        "model_version": prepared.preflight.model_version or "local_path",
        "row_count": len(result),
    }
    return LocalScoreResult(result, diagnostics)
