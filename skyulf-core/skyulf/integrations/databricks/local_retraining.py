"""Bounded, label-aware local candidate training from a pinned Delta snapshot.

This adapter uses Spark only for a narrow, bounded read. Fit and evaluation run
on the recorded local engine. Alias management requires an explicit caller hook.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...inference.local_evaluation import evaluate_local_holdout
from ..mlflow.registry import ResolvedModel, register_model, resolve_model
from ..mlflow.tracking import TrackingConfig, track_run
from ..mlflow.validation import ModelComparisonReport, compare_registered_local_models
from ._contracts import column_name, table_name
from .local_batch import _frame_bytes, fit_local_workflow


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTrainingSpec:
    """Pin the source version, label cutoff, split and local memory budget."""

    table: str
    version: int
    start: datetime
    holdout_start: datetime
    cutoff: datetime
    event_column: str
    label_time_column: str
    row_keys: tuple[str, ...]
    input_columns: tuple[str, ...]
    target_column: str
    max_rows: int
    max_bytes: int

    def __post_init__(self) -> None:
        """Reject ambiguous or unsafe snapshots before opening a Spark reader."""
        table_name(self.table)
        if type(self.version) is not int or self.version < 0:
            raise ValueError("version must be a nonnegative Delta snapshot version.")
        for name in ("start", "holdout_start", "cutoff"):
            value = getattr(self, name)
            if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware.")
            if value.astimezone(UTC).astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(
                tzinfo=None
            ):
                raise ValueError(f"{name} is not a valid local instant.")
        if not self.start < self.holdout_start < self.cutoff:
            raise ValueError("Require start < holdout_start < cutoff.")
        if not self.row_keys or not self.input_columns:
            raise ValueError("row_keys and input_columns must be nonempty.")
        names = (
            *self.row_keys,
            *self.input_columns,
            self.target_column,
            self.event_column,
            self.label_time_column,
        )
        for name in names:
            column_name(name)
        if len({name.lower() for name in names}) != len(names):
            raise ValueError("Training columns must be distinct.")
        if type(self.max_rows) is not int or self.max_rows <= 0:
            raise ValueError("max_rows must be positive.")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")

    @property
    def dataset_id(self) -> str:
        """Describe the immutable evaluation rows and temporal selection."""
        return (
            f"{self.table}@{self.version}/event[{self.start.astimezone(UTC).isoformat()},"
            f"{self.cutoff.astimezone(UTC).isoformat()})/holdout>="
            f"{self.holdout_start.astimezone(UTC).isoformat()}"
            f"/label<={self.cutoff.astimezone(UTC).isoformat()}"
            f"/event_column={self.event_column}/label_column={self.label_time_column}"
            f"/target={self.target_column}/keys={','.join(self.row_keys)}"
        )


@dataclass(frozen=True, slots=True)
class LocalCandidateResult:
    """Record a concrete candidate and its read-only comparison evidence."""

    run_id: str
    model_name: str
    model_version: str
    model_digest: str
    dataset_id: str
    training_rows: int
    holdout_rows: int
    unavailable_labels: int
    engine: str
    comparison: ModelComparisonReport


def read_training_snapshot(spark: Any, spec: LocalTrainingSpec) -> pd.DataFrame:
    """Project and cap a versioned Delta read before materializing driver rows."""
    if not isinstance(spec, LocalTrainingSpec):
        raise TypeError("spec must be LocalTrainingSpec.")
    names = (
        *spec.row_keys,
        spec.event_column,
        spec.label_time_column,
        *spec.input_columns,
        spec.target_column,
    )
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    start = spec.start.astimezone(UTC).isoformat()
    cutoff = spec.cutoff.astimezone(UTC).isoformat()
    selected = (
        source.where(
            f"{column_name(spec.event_column)} >= TIMESTAMP '{start}' AND "
            f"{column_name(spec.event_column)} < TIMESTAMP '{cutoff}'"
        )
        .select(*names)
        .orderBy(spec.event_column, *spec.row_keys)
        .limit(spec.max_rows + 1)
    )
    records: list[dict[str, Any]] = []
    serialized_bytes = 0
    for row in selected.toLocalIterator():
        if len(records) >= spec.max_rows:
            raise ValueError("Training source exceeds max_rows.")
        record = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
        serialized_bytes += len(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        if serialized_bytes > spec.max_bytes:
            raise ValueError("Training source exceeds max_bytes.")
        records.append(record)
    frame = pd.DataFrame.from_records(records, columns=names)
    if _frame_bytes(frame) > spec.max_bytes:
        raise ValueError("Training frame exceeds max_bytes.")
    return frame


def split_labeled_snapshot(
    frame: pd.DataFrame, spec: LocalTrainingSpec
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Make disjoint, reproducible fit and holdout sets from available labels."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(spec, LocalTrainingSpec):
        raise TypeError("Expected a pandas frame and LocalTrainingSpec.")
    expected = (
        *spec.row_keys,
        spec.event_column,
        spec.label_time_column,
        *spec.input_columns,
        spec.target_column,
    )
    if not set(expected).issubset(frame.columns):
        raise ValueError("Training snapshot is missing required columns.")
    if frame.loc[:, list(spec.row_keys)].isna().any().any():
        raise ValueError("Training row keys must not be null.")
    if frame.duplicated(subset=list(spec.row_keys)).any():
        raise ValueError("Training row keys must be unique.")
    events = pd.to_datetime(frame[spec.event_column], utc=True, errors="raise")
    labels = pd.to_datetime(frame[spec.label_time_column], utc=True, errors="raise")
    if events.isna().any() or (events < spec.start).any() or (events >= spec.cutoff).any():
        raise ValueError("Training event time falls outside the pinned window.")
    if ((labels < events) & labels.notna()).any():
        raise ValueError("Label availability precedes event time.")
    available = labels.notna() & (labels <= spec.cutoff)
    if frame.loc[available, spec.target_column].isna().any():
        raise ValueError("Available labels must have nonnull targets.")
    selected = frame.loc[available].copy()
    selected[spec.event_column] = events.loc[available]
    selected = selected.sort_values([spec.event_column, *spec.row_keys], kind="stable")
    holdout = selected[spec.event_column] >= spec.holdout_start
    columns = [*spec.input_columns, spec.target_column]
    train_frame = selected.loc[~holdout, columns].reset_index(drop=True)
    holdout_frame = selected.loc[holdout, columns].reset_index(drop=True)
    if len(train_frame) < 2 or len(holdout_frame) < 2:
        raise ValueError("Training and temporal holdout each need at least two labeled rows.")
    return train_frame, holdout_frame, int((~available).sum())


def _log_local_model(artifact_path: str | Path, *, run_id: str, tracking_uri: str) -> str:
    """Import the optional MLflow pyfunc package only for a tracked run."""
    from ..mlflow.local_model import log_local_model  # noqa: PLC0415

    return log_local_model(
        artifact_path, run_id=run_id, artifact_path="model", tracking_uri=tracking_uri
    )


def train_local_candidate(
    spark: Any,
    spec: LocalTrainingSpec,
    config: dict[str, Any],
    *,
    model_name: str,
    tracking_uri: str,
    registry_uri: str,
    experiment_name: str,
    run_name: str,
    artifact_path: str | Path,
    metric: str,
    min_improvement: float,
    engine: Literal["pandas", "polars"] = "pandas",
    champion_version: str | None = None,
    quality_threshold: float | None = None,
    on_registered: Callable[[ResolvedModel], None] | None = None,
    risk_category: str | None = None,
) -> LocalCandidateResult:
    """Fit, register and compare; optionally notify an explicit lifecycle owner.

    By default no aliases change. The on_registered hook runs after successful
    registration and before comparison, allowing the caller to nominate a
    contender without making the generic training service an alias writer.
    """
    if not isinstance(spec, LocalTrainingSpec):
        raise TypeError("spec must be LocalTrainingSpec.")
    if engine not in ("pandas", "polars"):
        raise ValueError("engine must be pandas or polars.")
    if risk_category is not None and (
        not isinstance(risk_category, str) or len(risk_category.encode("utf-8")) > 256
    ):
        raise ValueError("risk_category must be text of at most 256 UTF-8 bytes.")
    if (
        type(min_improvement) not in (int, float)
        or not math.isfinite(min_improvement)
        or min_improvement < 0
    ):
        raise ValueError("min_improvement must be a finite nonnegative number.")
    if quality_threshold is not None and (
        type(quality_threshold) not in (int, float) or not math.isfinite(quality_threshold)
    ):
        raise ValueError("quality_threshold must be a finite number or None.")
    if champion_version is not None and (
        type(champion_version) is not str
        or not champion_version.isdecimal()
        or int(champion_version) <= 0
    ):
        raise ValueError("champion_version must be a concrete positive version.")
    champion = (
        resolve_model(
            model_name,
            version=champion_version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        if champion_version is not None
        else None
    )
    frame = read_training_snapshot(spark, spec)
    train_frame, holdout, unavailable = split_labeled_snapshot(frame, spec)
    native_train = pl.from_pandas(train_frame) if engine == "polars" else train_frame
    native_holdout = pl.from_pandas(holdout) if engine == "polars" else holdout
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native_train, test=native_train.head(0)),
        target_column=spec.target_column,
        artifact_path=artifact_path,
        max_rows=spec.max_rows,
        max_bytes=spec.max_bytes,
    )
    metrics = evaluate_local_holdout(artifact, native_holdout, target_column=spec.target_column)
    if metric not in metrics or not math.isfinite(metrics[metric]):
        raise ValueError("Selected metric is unavailable or non-finite on the holdout.")
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        failure_policy="raise",
    )
    with track_run(tracking, run_name=run_name) as run:
        if run.run_id is None:
            raise RuntimeError("MLflow did not provide a run ID.")
        run.log_config(config, artifact_file="skyulf_pipeline_config.json")
        run.log_params(
            {
                "source_table": spec.table,
                "source_version": spec.version,
                "training_rows": len(train_frame),
                "holdout_rows": len(holdout),
                "unavailable_labels": unavailable,
                "target_column": spec.target_column,
                "model_digest": artifact.manifest.pipeline_sha256,
                "engine": engine,
                "code_version": version("skyulf-core"),
            }
        )
        tags = {
            "task": "training",
            "train_data_destination": spec.table,
            "test_data_destination": spec.table,
            "train_data_version": str(spec.version),
            "test_data_version": str(spec.version),
            "train_start": spec.start.astimezone(UTC).isoformat(),
            "test_start": spec.holdout_start.astimezone(UTC).isoformat(),
            "data_end": spec.cutoff.astimezone(UTC).isoformat(),
            "model_type": str(config["modeling"]["type"]),
            "candidate_date_tag": datetime.now(UTC).date().isoformat(),
            "engine": engine,
        }
        if risk_category:
            tags["risk_category"] = risk_category
        run.set_tags(tags)
        run.client.log_dict(
            run.run_id, {"dataset_id": spec.dataset_id, **tags}, "training_data.json"
        )
        run.log_metrics(metrics)
        model_uri = _log_local_model(artifact_path, run_id=run.run_id, tracking_uri=tracking_uri)
        run_id = run.run_id
    registered = register_model(
        model_uri,
        model_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        tags={
            key: value if len(value.encode("utf-8")) <= 256 else "See training_data.json"
            for key, value in tags.items()
        },
    )
    candidate = resolve_model(
        model_name,
        version=str(registered.version),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    if on_registered is not None:
        on_registered(candidate)
    report = compare_registered_local_models(
        candidate,
        champion,
        native_holdout,
        target_column=spec.target_column,
        dataset_id=spec.dataset_id,
        metric=metric,
        min_improvement=min_improvement,
        max_rows=spec.max_rows,
        max_bytes=spec.max_bytes,
        quality_threshold=quality_threshold,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    run.client.log_dict(run_id, asdict(report), "candidate_comparison.json")
    return LocalCandidateResult(
        run_id=run_id,
        model_name=model_name,
        model_version=candidate.version,
        model_digest=candidate.digest or "",
        dataset_id=spec.dataset_id,
        training_rows=len(train_frame),
        holdout_rows=len(holdout),
        unavailable_labels=unavailable,
        engine=engine,
        comparison=report,
    )
