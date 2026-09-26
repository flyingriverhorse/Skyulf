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
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import polars as pl

from ...data.dataset import SplitDataset
from ...inference.local_evaluation import evaluate_local_holdout
from ...leakage import step_learns_from_data
from ...preprocessing.split import DataSplitter
from ...registry import NodeRegistry
from ..mlflow.registry import ResolvedModel, register_model, resolve_model
from ..mlflow.tracking import TrackingConfig, track_run
from ..mlflow.validation import ModelComparisonReport, compare_registered_local_models
from ._contracts import column_name, table_name
from .local_batch import _frame_bytes, fit_local_workflow
from .local_cv import CV_FIELDS, LocalCVSpec, evaluate_training_cv
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
    training_sample_rows: int | None = None
    training_sample_seed: int = 42
    sample_key_sha256: str | None = None
    pre_split_steps: tuple[dict[str, Any], ...] = ()

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
            if self.holdout_start is not None:
                raise ValueError("Random split requires inactive holdout_start to be null.")
            if self.event_column is None and (
                self.start is not None
                or self.cutoff is not None
                or self.event_time_parsing != TrainingDateSpec()
            ):
                raise ValueError("Random split requires inactive event/date fields to be null.")
            if self.event_column is not None:
                start = _validate_instant(self.start, "start")
                cutoff = _validate_instant(self.cutoff, "cutoff")
                if start >= cutoff:
                    raise ValueError("Require start < cutoff for event selection.")
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
        _pre_split_columns(self.pre_split_steps, self.target_column)
        names = self.source_columns
        for name in names:
            column_name(name)
        base_names = (
            *self.record_key_columns,
            self.event_column,
            self.result_available_at_column,
            *self.input_columns,
            self.target_column,
        )
        base_names = tuple(name for name in base_names if name is not None)
        if len({name.lower() for name in base_names}) != len(base_names):
            raise ValueError("Training columns must be distinct.")
        if type(self.max_rows) is not int or self.max_rows <= 0:
            raise ValueError("max_rows must be positive.")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        if self.training_sample_rows is not None and (
            type(self.training_sample_rows) is not int
            or not 4 <= self.training_sample_rows <= self.max_rows
        ):
            raise ValueError("training_sample_rows must be null or an integer from 4 to max_rows.")
        if type(self.training_sample_seed) is not int or not 0 <= self.training_sample_seed < 2**32:
            raise ValueError("training_sample_seed must be an integer from 0 to 2**32 - 1.")
        for field in ("holdout_key_sha256", "sample_key_sha256"):
            digest = getattr(self, field)
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"{field} must be a SHA-256 digest.")
        if self.sample_key_sha256 is not None and self.training_sample_rows is None:
            raise ValueError("sample_key_sha256 requires training_sample_rows.")

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Project identities, active dates and model columns without inventing source fields."""
        dates = tuple(
            name
            for name in (self.event_column, self.result_available_at_column)
            if name is not None
        )
        base = (*self.record_key_columns, *dates, *self.input_columns, self.target_column)
        extra = _pre_split_columns(self.pre_split_steps, self.target_column)
        return (
            *base,
            *(name for name in extra if name.casefold() not in {item.casefold() for item in base}),
        )

    @property
    def dataset_id(self) -> str:
        """Pin source, selection, split and seed independently of mutable driver limits."""
        settings = asdict(self)
        if not self.pre_split_steps:
            settings.pop("pre_split_steps")
        for field in ("max_rows", "max_bytes"):
            settings.pop(field)
        for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
            value = getattr(self, field)
            settings[field] = None if value is None else value.astimezone(UTC).isoformat()
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        return f"{self.table}@{self.version}/{self.split_strategy}/{digest}"


def _pre_split_columns(steps: Any, target_column: str) -> tuple[str, ...]:
    """Admit only explicit Core row filters and return their source dependencies."""
    if not isinstance(steps, (tuple, list)):
        raise ValueError("pre_split_steps must be an ordered sequence of Core steps.")
    columns: list[str] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or not isinstance(step.get("name"), str) or not step["name"]:
            raise ValueError(f"pre_split_steps[{index}] requires a name.")
        step_type = step.get("transformer")
        params = step.get("params", {})
        if not isinstance(params, dict) or not isinstance(step_type, str):
            raise ValueError(f"pre_split_steps[{index}] requires a Core transformer and params.")
        if step_learns_from_data(step_type, params, target_column=target_column):
            raise ValueError(f"pre_split_steps[{index}] cannot learn from data before split.")
        if step_type == "DropMissingRows":
            subset = params.get("subset")
            if (
                not isinstance(subset, list)
                or not subset
                or any(not isinstance(c, str) for c in subset)
            ):
                raise ValueError(
                    "pre_split_steps DropMissingRows requires explicit subset columns."
                )
            if params.get("how", "any") not in ("any", "all") or set(params) - {
                "subset",
                "how",
                "threshold",
                "missing_threshold",
            }:
                raise ValueError("pre_split_steps DropMissingRows has invalid row-filter params.")
            threshold = params.get("threshold")
            if threshold is not None and (
                type(threshold) is not int or not 0 <= threshold <= len(subset)
            ):
                raise ValueError(
                    "pre_split_steps threshold must be an integer from 0 to subset size."
                )
            missing_threshold = params.get("missing_threshold")
            if missing_threshold is not None and (
                type(missing_threshold) not in (int, float)
                or not math.isfinite(missing_threshold)
                or not 0 <= missing_threshold <= 100
            ):
                raise ValueError("pre_split_steps missing_threshold must be between 0 and 100.")
            columns.extend(subset)
        elif step_type == "ManualBounds":
            bounds = params.get("bounds")
            if set(params) - {"bounds"}:
                raise ValueError("pre_split_steps ManualBounds supports only bounds.")
            if not isinstance(bounds, dict) or not bounds:
                raise ValueError("pre_split_steps ManualBounds requires explicit bounds.")
            for column, limit in bounds.items():
                if not isinstance(column, str) or not isinstance(limit, dict) or not limit:
                    raise ValueError("pre_split_steps ManualBounds requires named column bounds.")
                if set(limit) - {"lower", "upper"} or not any(
                    key in limit and limit[key] is not None for key in ("lower", "upper")
                ):
                    raise ValueError("pre_split_steps ManualBounds requires lower or upper.")
                for value in limit.values():
                    if value is not None and (
                        type(value) not in (int, float) or not math.isfinite(value)
                    ):
                        raise ValueError(
                            "pre_split_steps ManualBounds requires finite numeric bounds."
                        )
                if (
                    limit.get("lower") is not None
                    and limit.get("upper") is not None
                    and limit["lower"] > limit["upper"]
                ):
                    raise ValueError("pre_split_steps ManualBounds lower must not exceed upper.")
                columns.append(column)
        else:
            raise ValueError(
                f"pre_split_steps[{index}] permits only DropMissingRows or ManualBounds."
            )
        if set(step) - {"name", "transformer", "params"}:
            raise ValueError(f"pre_split_steps[{index}] contains unsupported step fields.")
    for column in columns:
        column_name(column)
    return tuple(dict.fromkeys(columns))


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
    _pre_split_columns(spec.pre_split_steps, spec.target_column)
    names = spec.source_columns
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    source = normalize_training_dates(
        source.select(*names),
        event_column=spec.event_column,
        result_column=spec.result_available_at_column,
        event_spec=spec.event_time_parsing,
        result_spec=spec.result_time_parsing,
    )
    if spec.event_column is not None:
        start = instant_microseconds(cast(datetime, spec.start))
        cutoff = instant_microseconds(cast(datetime, spec.cutoff))
        source = source.where(
            f"{column_name(cast(str, spec.event_column))} >= {start} AND "
            f"{column_name(cast(str, spec.event_column))} < {cutoff}"
        )
    selection = None
    if spec.training_sample_rows is not None:
        source, selection = _sample_training_source(source, spec)
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
    if selection is not None:
        frame.attrs["training_selection"] = {**selection, "selected_rows": len(frame)}
    return frame


def _sample_training_source(source: Any, spec: LocalTrainingSpec) -> tuple[Any, dict[str, Any]]:
    """Select eligible keys by a seeded hash on Spark before transferring any training rows."""
    F = import_module("pyspark.sql.functions")

    keys = list(spec.record_key_columns)
    floating = {
        field.name
        for field in source.schema.fields
        if field.dataType.typeName() in {"double", "float"}
    }
    null_keys = " OR ".join(
        f"({column_name(key)} IS NULL OR isnan({column_name(key)}))"
        if key in floating
        else f"{column_name(key)} IS NULL"
        for key in keys
    )
    if source.where(null_keys).limit(1).count():
        raise ValueError("Training row keys must not be null.")
    count_name = "_sample_count"
    while count_name.casefold() in {key.casefold() for key in keys}:
        count_name += "_"
    counts = source.groupBy(*keys).agg(F.count(F.lit(1)).alias(count_name))
    if counts.where(F.col(count_name) > 1).limit(1).count():
        raise ValueError("Training row keys must be unique before sampling.")
    source_rows = source.count()
    if spec.filter_unavailable_results:
        result = column_name(cast(str, spec.result_available_at_column))
        if (
            spec.event_column is not None
            and source.where(f"{result} < {column_name(spec.event_column)}").limit(1).count()
        ):
            raise ValueError("Label availability precedes event time.")
        cutoff = instant_microseconds(cast(datetime, spec.result_cutoff))
        source = source.where(f"{result} IS NOT NULL AND {result} <= {cutoff}")
    eligible_rows = source.count() if spec.filter_unavailable_results else source_rows
    invalid_target = F.col(spec.target_column).isNull()
    if spec.target_column in floating:
        invalid_target = invalid_target | F.isnan(F.col(spec.target_column))
    target_filter = any(
        step["transformer"] == "DropMissingRows" and spec.target_column in step["params"]["subset"]
        for step in spec.pre_split_steps
    )
    if not target_filter and source.where(invalid_target).limit(1).count():
        raise ValueError("Available labels must have nonnull targets before sampling.")
    # Struct field order, UTC timestamp rendering and the tie-break keys are
    # explicit so partition layout and Spark session timezone cannot change membership.
    key_json = F.to_json(
        F.struct(
            F.lit(spec.training_sample_seed).alias("sample_seed"),
            *[F.col(key).alias(f"key_{index}") for index, key in enumerate(keys)],
        ),
        options={"timeZone": "UTC", "ignoreNullFields": "false"},
    )
    selected = source.orderBy(F.sha2(key_json, 256), *keys).limit(spec.training_sample_rows)
    return selected, {
        "method": "sha256_keys_v1",
        "seed": spec.training_sample_seed,
        "requested_rows": spec.training_sample_rows,
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "unavailable_labels": source_rows - eligible_rows,
    }


def _key_digest(frame: pd.DataFrame, keys: tuple[str, ...]) -> str:
    """Hash typed ordered identities without treating them as model inputs."""
    payload = json.dumps(
        {
            "columns": list(keys),
            "rows": [
                [(type(value).__name__, str(value)) for value in row]
                for row in frame.loc[:, list(keys)].itertuples(index=False, name=None)
            ],
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def split_labeled_snapshot(
    frame: pd.DataFrame,
    spec: LocalTrainingSpec,
    *,
    keep_training_event: bool = False,
    engine: str = "pandas",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Make disjoint, reproducible fit and holdout sets from available labels."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(spec, LocalTrainingSpec):
        raise TypeError("Expected a pandas frame and LocalTrainingSpec.")
    _pre_split_columns(spec.pre_split_steps, spec.target_column)
    if engine not in ("pandas", "polars"):
        raise ValueError("engine must be pandas or polars.")
    missing_columns = sorted(set(spec.source_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Training snapshot is missing required source columns: {missing_columns}."
        )
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
    ordering = ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)
    selected = selected.loc[available].sort_values(ordering, kind="stable").reset_index(drop=True)
    sample_digest = None
    if spec.training_sample_rows is not None:
        if len(selected) > spec.training_sample_rows:
            raise ValueError("Training snapshot was not bounded by training_sample_rows.")
        sample_digest = _key_digest(selected, spec.record_key_columns)
        if spec.sample_key_sha256 is not None and sample_digest != spec.sample_key_sha256:
            raise ValueError("Training sample membership differs from saved evidence.")
    filter_counts = []
    native = pl.from_pandas(selected) if spec.pre_split_steps and engine == "polars" else selected
    for step in spec.pre_split_steps:
        columns = (
            step["params"]["subset"]
            if step["transformer"] == "DropMissingRows"
            else step["params"]["bounds"]
        )
        if any(column not in native.columns for column in columns):
            missing = sorted(set(columns) - set(native.columns))
            raise ValueError(f"pre_split_steps {step['name']} missing source columns: {missing}.")
        if step["transformer"] == "ManualBounds":
            for column in columns:
                if not pd.api.types.is_numeric_dtype(
                    selected[column]
                ) or pd.api.types.is_bool_dtype(selected[column]):
                    raise ValueError(
                        f"pre_split_steps ManualBounds requires numeric column {column}."
                    )
        keys = list(spec.record_key_columns)
        before_keys = (
            list(native.select(keys).iter_rows())
            if isinstance(native, pl.DataFrame)
            else list(native[keys].itertuples(index=False, name=None))
        )
        before_targets = native[spec.target_column].to_list()
        artifact = NodeRegistry.get_calculator(step["transformer"])().fit(native, step["params"])
        filtered = NodeRegistry.get_applier(step["transformer"])().apply(native, artifact)
        if not isinstance(filtered, type(native)):
            raise ValueError("pre_split_steps filter did not return a frame.")
        if list(filtered.columns) != list(native.columns):
            raise ValueError("pre_split_steps must preserve all source columns.")
        after_keys = (
            list(filtered.select(keys).iter_rows())
            if isinstance(filtered, pl.DataFrame)
            else list(filtered[keys].itertuples(index=False, name=None))
        )
        positions = {key: index for index, key in enumerate(before_keys)}
        if any(key not in positions for key in after_keys) or [
            positions[key] for key in after_keys
        ] != sorted({positions[key] for key in after_keys}):
            raise ValueError("pre_split_steps must preserve row identities and order.")
        after_targets = filtered[spec.target_column].to_list()
        for key, value in zip(after_keys, after_targets, strict=True):
            expected = before_targets[positions[key]]
            if not (pd.isna(expected) and pd.isna(value)) and expected != value:
                raise ValueError("pre_split_steps must preserve target pairing.")
        filter_counts.append(
            {
                "name": step["name"],
                "transformer": step["transformer"],
                "input_rows": len(before_keys),
                "excluded_rows": len(before_keys) - len(filtered),
                "output_rows": len(filtered),
            }
        )
        native = filtered
    if spec.pre_split_steps:
        selected = native.to_pandas() if isinstance(native, pl.DataFrame) else native
        selected = selected.reset_index(drop=True)
    if selected[spec.target_column].isna().any():
        raise ValueError(
            "Available labels must have nonnull targets; add explicit DropMissingRows for the target."
        )
    if spec.pre_split_steps and len(selected) < 4:
        raise ValueError(
            "Training filters leave fewer than four eligible rows for train and holdout."
        )
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
    digest = _key_digest(heldout, spec.record_key_columns)
    if spec.holdout_key_sha256 is not None and digest != spec.holdout_key_sha256:
        raise ValueError("Holdout membership differs from saved training evidence.")
    columns = [*spec.input_columns, spec.target_column]
    train_columns = (
        [*columns, spec.event_column] if keep_training_event and spec.event_column else columns
    )
    train_frame = train.loc[:, train_columns].reset_index(drop=True)
    holdout_frame = heldout.loc[:, columns].reset_index(drop=True)
    holdout_frame.attrs["holdout_key_sha256"] = digest
    holdout_frame.attrs["sample_key_sha256"] = sample_digest
    holdout_frame.attrs["pre_split_filter_counts"] = filter_counts
    unavailable = int((~available).sum()) + frame.attrs.get("training_selection", {}).get(
        "unavailable_labels", 0
    )
    return train_frame, holdout_frame, unavailable


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
    cv: LocalCVSpec | None = None,
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
    cv = LocalCVSpec() if cv is None else cv
    if not isinstance(cv, LocalCVSpec):
        raise TypeError("cv must be LocalCVSpec.")
    _pre_split_columns(spec.pre_split_steps, spec.target_column)
    cv.validate_pipeline(config, target_column=spec.target_column, event_column=spec.event_column)
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
    temporal_cv = cv.enabled and cv.method == "time_series_split"
    train_frame, holdout, unavailable = split_labeled_snapshot(
        frame, spec, keep_training_event=temporal_cv, engine=engine
    )
    spec = replace(
        spec,
        holdout_key_sha256=holdout.attrs["holdout_key_sha256"],
        sample_key_sha256=holdout.attrs["sample_key_sha256"],
    )
    native_train = pl.from_pandas(train_frame) if engine == "polars" else train_frame
    native_holdout = pl.from_pandas(holdout) if engine == "polars" else holdout
    cv_results = evaluate_training_cv(
        native_train,
        config,
        cv,
        target_column=spec.target_column,
        event_column=spec.event_column if temporal_cv else None,
    )
    if temporal_cv:
        model_columns = [*spec.input_columns, spec.target_column]
        native_train = (
            native_train.select(model_columns)
            if isinstance(native_train, pl.DataFrame)
            else native_train.loc[:, model_columns]
        )
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
        run.log_params({key: getattr(cv, field) for key, field in CV_FIELDS.items()})
        if cv_results is not None:
            cv_results.update(
                dataset_id=spec.dataset_id, training_rows=len(train_frame), engine=engine
            )
            run.client.log_dict(run.run_id, cv_results, "cross_validation.json")
            run.log_metrics(
                {
                    f"cv_{name}_{stat}": value
                    for name, statistics in cv_results["aggregated_metrics"].items()
                    for stat, value in statistics.items()
                }
            )
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
        run.client.log_dict(
            run.run_id,
            {
                "requested_steps": list(spec.pre_split_steps),
                "counts": holdout.attrs["pre_split_filter_counts"],
                "source_rows": len(frame),
                "eligible_rows": len(train_frame) + len(holdout),
                "excluded_rows": sum(
                    item["excluded_rows"] for item in holdout.attrs["pre_split_filter_counts"]
                ),
            },
            "pre_split_filters.json",
        )
        saved_spec = asdict(spec)
        if spec.training_sample_rows is not None:
            run.client.log_dict(
                run.run_id,
                {
                    **frame.attrs.get("training_selection", {}),
                    "sample_key_sha256": spec.sample_key_sha256,
                    "dataset_id": spec.dataset_id,
                },
                "training_selection.json",
            )
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
