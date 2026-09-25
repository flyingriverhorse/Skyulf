"""Bounded, label-aware local candidate training from a pinned Delta snapshot.

Spark validates source dates before materializing a narrow, bounded read. Fit
and evaluation run on the recorded local engine. Alias management requires an
explicit caller hook.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...inference.local_evaluation import evaluate_local_holdout
from ...preprocessing.split import DataSplitter
from ...registry import NodeRegistry
from ..mlflow.registry import ResolvedModel, register_model, resolve_model
from ..mlflow.tracking import TrackingConfig, track_run
from ..mlflow.validation import ModelComparisonReport, compare_registered_local_models
from ._contracts import column_name, table_name
from .local_batch import _frame_bytes, fit_local_workflow
from .training_dates import (
    TrainingDateSpec,
    instant_from_microseconds,
    instant_microseconds,
    normalize_training_dates,
    parse_training_date,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTrainingSpec:
    """Pin the source version, label cutoff, split and local memory budget."""

    table: str
    version: int
    record_key_columns: tuple[str, ...]
    input_columns: tuple[str, ...]
    target_column: str
    max_rows: int
    max_bytes: int
    split_strategy: Literal["random", "temporal"] = "random"
    test_size: float | None = 0.2
    random_state: int | None = 42
    stratify: bool | None = False
    start: datetime | None = None
    holdout_start: datetime | None = None
    cutoff: datetime | None = None
    event_column: str | None = None
    filter_unavailable_results: bool = False
    result_available_at_column: str | None = None
    result_cutoff: datetime | None = None
    event_time_parsing: TrainingDateSpec = TrainingDateSpec()
    result_time_parsing: TrainingDateSpec = TrainingDateSpec()
    holdout_key_sha256: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or contradictory policies before opening a Spark reader."""
        table_name(self.table)
        for field in ("event_time_parsing", "result_time_parsing"):
            if not isinstance(getattr(self, field), TrainingDateSpec):
                raise TypeError(f"{field} must be TrainingDateSpec.")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("version must be a nonnegative Delta snapshot version.")
        if self.split_strategy not in ("random", "temporal"):
            raise ValueError("split_strategy must be random or temporal.")
        if type(self.filter_unavailable_results) is not bool:
            raise ValueError("filter_unavailable_results must be boolean.")
        if self.split_strategy == "random":
            if (
                any(
                    value is not None
                    for value in (self.event_column, self.start, self.holdout_start, self.cutoff)
                )
                or self.event_time_parsing != TrainingDateSpec()
            ):
                raise ValueError("Random split requires inactive event/date fields to be null.")
            if (
                self.test_size is None
                or type(self.test_size) not in (int, float)
                or not 0 < self.test_size < 1
            ):
                raise ValueError("test_size must be a proportion strictly between zero and one.")
            if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
                raise ValueError("random_state must be an integer from 0 to 2**32 - 1.")
            if type(self.stratify) is not bool:
                raise ValueError("stratify must be boolean.")
        else:
            if self.event_column is None:
                raise ValueError("Temporal split requires event_column.")
            boundaries = [
                _validate_instant(getattr(self, name), name)
                for name in ("start", "holdout_start", "cutoff")
            ]
            if not boundaries[0] < boundaries[1] < boundaries[2]:
                raise ValueError("Require start < holdout_start < cutoff.")
            if (
                self.test_size not in (None, 0.2)
                or self.random_state not in (None, 42)
                or self.stratify not in (None, False)
                or (self.random_state is not None and type(self.random_state) is not int)
                or (self.stratify is not None and type(self.stratify) is not bool)
            ):
                raise ValueError("Temporal split cannot use active random split settings.")
        if self.filter_unavailable_results:
            if self.result_available_at_column is None:
                raise ValueError("Result filtering requires result_available_at_column.")
            _validate_instant(self.result_cutoff, "result_cutoff")
        elif (
            self.result_available_at_column is not None
            or self.result_cutoff is not None
            or self.result_time_parsing != TrainingDateSpec()
        ):
            raise ValueError(
                "Disabled result filtering requires inactive result fields to be null."
            )
        if not self.record_key_columns or not self.input_columns:
            raise ValueError("record_key_columns and input_columns must be nonempty.")
        names = self.source_columns
        for name in names:
            column_name(name)
        if len({name.lower() for name in names}) != len(names):
            raise ValueError("Training columns must be distinct.")
        if type(self.max_rows) is not int or self.max_rows <= 0:
            raise ValueError("max_rows must be positive.")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        if self.holdout_key_sha256 is not None and (
            not isinstance(self.holdout_key_sha256, str)
            or len(self.holdout_key_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.holdout_key_sha256)
        ):
            raise ValueError("holdout_key_sha256 must be a SHA-256 digest.")

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Project identities, active dates and model columns without inventing source fields."""
        dates = tuple(
            name
            for name in (self.event_column, self.result_available_at_column)
            if name is not None
        )
        return (*self.record_key_columns, *dates, *self.input_columns, self.target_column)

    @property
    def dataset_id(self) -> str:
        """Pin source, selection, split and seed independently of mutable driver limits."""
        settings = asdict(self)
        for field in ("max_rows", "max_bytes"):
            settings.pop(field)
        for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
            value = getattr(self, field)
            settings[field] = None if value is None else value.astimezone(UTC).isoformat()
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        return f"{self.table}@{self.version}/{self.split_strategy}/{digest}"


def _validate_instant(value: Any, name: str) -> datetime:
    """Reject naive boundaries and nonexistent local instants instead of assuming UTC."""
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    if value.astimezone(UTC).astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(
        tzinfo=None
    ):
        raise ValueError(f"{name} is not a valid local instant.")
    return value.astimezone(UTC)


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
    holdout_key_sha256: str


def read_training_snapshot(spark: Any, spec: LocalTrainingSpec) -> pd.DataFrame:
    """Project and cap a versioned Delta read before materializing driver rows."""
    if not isinstance(spec, LocalTrainingSpec):
        raise TypeError("spec must be LocalTrainingSpec.")
    names = spec.source_columns
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    source = normalize_training_dates(
        source.select(*names),
        event_column=spec.event_column,
        result_column=spec.result_available_at_column,
        event_spec=spec.event_time_parsing,
        result_spec=spec.result_time_parsing,
    )
    if spec.split_strategy == "temporal":
        start = instant_microseconds(cast(datetime, spec.start))
        cutoff = instant_microseconds(cast(datetime, spec.cutoff))
        source = source.where(
            f"{column_name(cast(str, spec.event_column))} >= {start} AND "
            f"{column_name(cast(str, spec.event_column))} < {cutoff}"
        )
    ordering = ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)
    selected = source.orderBy(*ordering).limit(spec.max_rows + 1)
    records: list[dict[str, Any]] = []
    serialized_bytes = 0
    for row in selected.toLocalIterator():
        if len(records) >= spec.max_rows:
            raise ValueError("Training source exceeds max_rows.")
        record = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
        for name in (spec.event_column, spec.result_available_at_column):
            if name is not None:
                record[name] = instant_from_microseconds(record[name])
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
    if not set(spec.source_columns).issubset(frame.columns):
        raise ValueError("Training snapshot is missing required columns.")
    if len(frame) > spec.max_rows or _frame_bytes(frame) > spec.max_bytes:
        raise ValueError("Training snapshot exceeds max_rows or max_bytes.")
    if frame.loc[:, list(spec.record_key_columns)].isna().any().any():
        raise ValueError("Training row keys must not be null.")
    if frame.duplicated(subset=list(spec.record_key_columns)).any():
        raise ValueError("Training row keys must be unique.")
    selected = frame.copy()
    events = None
    if spec.event_column is not None:
        events = pd.Series(
            [
                parse_training_date(value, spec.event_time_parsing)
                for value in frame[spec.event_column]
            ],
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )
        if events.isna().any() or (events < spec.start).any() or (events >= spec.cutoff).any():
            raise ValueError("Training event time falls outside the pinned window.")
        selected[spec.event_column] = events
    available = pd.Series(True, index=frame.index)
    if spec.filter_unavailable_results:
        labels = pd.Series(
            [
                parse_training_date(value, spec.result_time_parsing, allow_null=True)
                for value in frame[spec.result_available_at_column]
            ],
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )
        if events is not None and ((labels < events) & labels.notna()).any():
            raise ValueError("Label availability precedes event time.")
        available = labels.notna() & (labels <= spec.result_cutoff)
    if frame.loc[available, spec.target_column].isna().any():
        raise ValueError("Available labels must have nonnull targets.")
    ordering = ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)
    selected = selected.loc[available].sort_values(ordering, kind="stable").reset_index(drop=True)
    if spec.split_strategy == "random":
        if spec.stratify:
            counts = selected[spec.target_column].value_counts()
            if counts.empty or counts.min() < 2:
                raise ValueError("Requested stratification requires at least two rows per class.")
            test_rows = math.ceil(len(selected) * cast(float, spec.test_size))
            if min(test_rows, len(selected) - test_rows) < len(counts):
                raise ValueError("Requested stratification needs each class in both partitions.")
        split = DataSplitter(
            test_size=cast(float, spec.test_size),
            random_state=cast(int, spec.random_state),
            stratify_col=spec.target_column if spec.stratify else None,
        ).split(selected)
        train, heldout = cast(pd.DataFrame, split.train), cast(pd.DataFrame, split.test)
    else:
        holdout_mask = selected[spec.event_column] >= spec.holdout_start
        train, heldout = selected.loc[~holdout_mask], selected.loc[holdout_mask]
    if len(train) < 2 or len(heldout) < 2:
        raise ValueError("Training and holdout each need at least two labeled rows.")
    # Hash the ordered identity tuples only; never include keys in model features.
    key_frame = heldout.loc[:, list(spec.record_key_columns)]
    keys = json.dumps(
        {
            "columns": list(spec.record_key_columns),
            "rows": [
                [(type(value).__name__, str(value)) for value in row]
                for row in key_frame.itertuples(index=False, name=None)
            ],
        },
        separators=(",", ":"),
    )
    digest = hashlib.sha256(keys.encode()).hexdigest()
    if spec.holdout_key_sha256 is not None and digest != spec.holdout_key_sha256:
        raise ValueError("Holdout membership differs from saved training evidence.")
    columns = [*spec.input_columns, spec.target_column]
    train_frame = train.loc[:, columns].reset_index(drop=True)
    holdout_frame = heldout.loc[:, columns].reset_index(drop=True)
    holdout_frame.attrs["holdout_key_sha256"] = digest
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
    if spec.stratify and (
        NodeRegistry.get_calculator(config["modeling"]["type"])().problem_type != "classification"
    ):
        raise ValueError("stratify requires a classification model.")
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
    spec = replace(spec, holdout_key_sha256=holdout.attrs["holdout_key_sha256"])
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
            "split_strategy": spec.split_strategy,
            "model_type": str(config["modeling"]["type"]),
            "candidate_date_tag": datetime.now(UTC).date().isoformat(),
            "engine": engine,
        }
        for tag, field in (
            ("train_start", "start"),
            ("test_start", "holdout_start"),
            ("data_end", "cutoff"),
            ("result_cutoff", "result_cutoff"),
        ):
            value = getattr(spec, field)
            if value is not None:
                tags[tag] = value.astimezone(UTC).isoformat()
        if risk_category:
            tags["risk_category"] = risk_category
        run.set_tags(tags)
        run.client.log_dict(
            run.run_id, {"dataset_id": spec.dataset_id, **tags}, "training_data.json"
        )
        saved_spec = asdict(spec)
        for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
            value = getattr(spec, field)
            saved_spec[field] = None if value is None else value.isoformat()
        run.client.log_dict(
            run.run_id,
            {
                "dataset_id": spec.dataset_id,
                "holdout_key_sha256": spec.holdout_key_sha256,
                "holdout_rows": len(holdout),
                "record_key_columns": list(spec.record_key_columns),
            },
            "holdout_membership.json",
        )
        run.client.log_dict(
            run.run_id, {**saved_spec, "engine": engine}, "candidate_training_spec.json"
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
        holdout_key_sha256=holdout.attrs["holdout_key_sha256"],
    )
