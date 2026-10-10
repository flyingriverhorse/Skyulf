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

from skyulf.integrations.databricks.shared._local_frames import frame_bytes

from .....data.dataset import SplitDataset
from .....inference.local_evaluation import evaluate_local_holdout
from .....inference.project_code import is_project_filter_step
from .....leakage import step_learns_from_data
from .....preprocessing.split import DataSplitter
from .....registry import NodeRegistry
from ....mlflow.lifecycle.validation import (
    ModelComparisonReport,
    compare_registered_local_models,
    comparison_digest,
    comparison_payload,
    quality_gate_results,
    validate_quality_policy,
)
from ....mlflow.registration.registry import ResolvedModel, register_model, resolve_model
from ....mlflow.runs.tracking import TrackingConfig, track_run
from ...data.training.training_dates import (
    TrainingDateSpec,
    instant_from_microseconds,
    instant_microseconds,
    normalize_training_dates,
    parse_training_date,
    training_date_spec,
)
from ...feature_store.training import (
    enrich_training_source,
    feature_log_options,
    lookup_controls,
    pin_fit_lookup,
    pin_training_lookup,
    retain_training_binding,
    training_lookup,
    validate_training_frame,
)
from ...observability.charts.evaluation_chart_data import chart_recorder, chart_settings
from ...observability.reports.local_explanations import (
    log_training_explanations,
    validate_explanation_config,
)
from ...scoring.batch.local_batch import fit_local_workflow
from ...shared._contracts import column_name, table_name
from ..shared.local_training_evidence import build_training_evidence, evidence_digest
from ..shared.preprocessing_checks import log_preprocessing_probe
from ..shared.training_parameters import log_training_parameters
from ..thresholds.decision_thresholds import threshold_policy
from ..tuning.local_cv import CV_FIELDS, LocalCVSpec, evaluate_training_cv
from ..tuning.local_search import base_model_config, prepare_search_pipeline
from ..tuning.local_search_results import (
    post_selection_cv,
    tuning_evidence,
    tuning_run_params,
    validate_search_membership,
)
from ..weights.local_weights import (
    extract_training_weights,
    training_weight_evidence,
    validate_weight_snapshot,
)
from .local_pre_split import (
    FIXED_TYPES,
    custom_filter_columns,
    deduplicate_columns,
    fixed_columns,
    projected_fixed_steps,
    target_contract,
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
    group_column: str | None = None
    weight_column: str | None = None
    reserved_weight_columns: tuple[str, ...] = ()
    weights_python_source: str | None = None
    weights_python_sha256: str | None = None
    filter_unavailable_results: bool = False
    drop_missing_labels: bool = False
    result_available_at_column: str | None = None
    result_cutoff: datetime | None = None
    event_time_parsing: TrainingDateSpec = TrainingDateSpec()
    result_time_parsing: TrainingDateSpec = TrainingDateSpec()
    holdout_key_sha256: str | None = None
    training_sample_rows: int | None = None
    training_sample_seed: int = 42
    sample_key_sha256: str | None = None
    pre_split_steps: tuple[dict[str, Any], ...] = ()
    survivor_key_sha256: str | None = None
    training_evidence_sha256: str | None = None
    feature_lookup_json: str | None = None
    feature_binding_json: str | None = None
    preprocessing_probe: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LocalTrainingSpec:
        """Restore a saved spec without mutating its JSON or interpreting engine metadata.

        Callers load verified custom source before construction so custom steps
        are registered. Older evidence may omit the empty pre-split recipe.
        """
        values = dict(payload)
        values.pop("engine", None)
        for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
            value = values[field]
            values[field] = None if value is None else datetime.fromisoformat(value)
        for field in ("record_key_columns", "input_columns"):
            values[field] = tuple(values[field])
        values["reserved_weight_columns"] = tuple(values.get("reserved_weight_columns", ()))
        values["pre_split_steps"] = tuple(values.get("pre_split_steps", ()))
        for field in ("event_time_parsing", "result_time_parsing"):
            values[field] = training_date_spec(values[field])
        return cls(**values)

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
        self._validate_label_policy()
        if self.split_strategy == "random":
            self._validate_random_split()
        else:
            self._validate_temporal_split()
        self._validate_result_filter()
        validate_weight_snapshot(asdict(self))
        self._validate_columns()
        self._validate_budgets_and_sampling()
        self._validate_evidence_digests()
        training_lookup(self)

    def _validate_label_policy(self) -> None:
        """Require explicit booleans for label eligibility and optional diagnostics."""
        for field in ("filter_unavailable_results", "drop_missing_labels", "preprocessing_probe"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean.")

    def _validate_random_split(self) -> None:
        """Check the random split and its optional event-selection window."""
        _validate_random_window(self)
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

    def _validate_temporal_split(self) -> None:
        """Require ordered temporal boundaries and inactive random split settings."""
        if self.event_column is None:
            raise ValueError("Temporal split requires event_column.")
        boundaries = [
            _validate_instant(getattr(self, name), name)
            for name in ("start", "holdout_start", "cutoff")
        ]
        if not boundaries[0] < boundaries[1] < boundaries[2]:
            raise ValueError("Require start < holdout_start < cutoff.")
        _validate_inactive_random_settings(self)

    def _validate_result_filter(self) -> None:
        """Require a cutoff only when result-availability filtering is enabled."""
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

    def _validate_columns(self) -> None:
        """Validate projected columns, pre-split steps and protected identities."""
        if not self.record_key_columns or not self.input_columns:
            raise ValueError("record_key_columns and input_columns must be nonempty.")
        _pre_split_columns(
            self.pre_split_steps,
            self.target_column,
            (
                *self.record_key_columns,
                self.event_column,
                self.result_available_at_column,
                self.group_column,
                self.weight_column,
                *self.reserved_weight_columns,
            ),
        )
        names = self.source_columns
        for name in names:
            column_name(name)
        base_names = (
            *self.record_key_columns,
            self.event_column,
            self.group_column,
            self.result_available_at_column,
            *self.input_columns,
            self.target_column,
        )
        base_names = tuple(name for name in base_names if name is not None)
        if len({name.lower() for name in base_names}) != len(base_names):
            raise ValueError("Training columns must be distinct.")

    def _validate_budgets_and_sampling(self) -> None:
        """Keep sampling within the local read budget with a reproducible seed."""
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

    def _validate_evidence_digests(self) -> None:
        """Accept only concrete membership digests with their required policies."""
        for field in (
            "holdout_key_sha256",
            "sample_key_sha256",
            "survivor_key_sha256",
            "training_evidence_sha256",
        ):
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
        metadata_columns = tuple(
            name
            for name in (
                self.event_column,
                self.result_available_at_column,
                self.group_column,
                self.weight_column,
            )
            if name is not None
        )
        base = (
            *self.record_key_columns,
            *metadata_columns,
            *self.input_columns,
            self.target_column,
            *lookup_controls(self),
        )
        extra = _pre_split_columns(
            self.pre_split_steps,
            self.target_column,
            (
                *self.record_key_columns,
                self.event_column,
                self.result_available_at_column,
                self.group_column,
                self.weight_column,
                *self.reserved_weight_columns,
            ),
        )
        return (
            *dict.fromkeys(base),
            *(name for name in extra if name.casefold() not in {item.casefold() for item in base}),
        )

    def _identity_settings(self) -> dict[str, Any]:
        """Keep optional lookup and weight controls out of older unbound identities."""
        settings = asdict(self)
        for field in ("feature_lookup_json", "feature_binding_json"):
            if settings[field] is None:
                settings.pop(field)
        return settings

    @property
    def dataset_id(self) -> str:
        """Pin source, selection, split and seed independently of mutable driver limits."""
        settings = self._identity_settings()
        if self.weight_column is None:
            for field in (
                "weight_column",
                "reserved_weight_columns",
                "weights_python_source",
                "weights_python_sha256",
            ):
                settings.pop(field)
        if not self.drop_missing_labels:
            settings.pop("drop_missing_labels")
        if self.group_column is None:
            settings.pop("group_column")
        settings.pop("survivor_key_sha256")
        if not self.training_evidence_sha256:
            settings.pop("training_evidence_sha256")
        if not self.pre_split_steps:
            settings.pop("pre_split_steps")
        for field in ("max_rows", "max_bytes", "preprocessing_probe"):
            settings.pop(field)
        for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
            value = getattr(self, field)
            settings[field] = None if value is None else value.astimezone(UTC).isoformat()
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        return f"{self.table}@{self.version}/{self.split_strategy}/{digest}"


def _missing_row_filter_columns(params: dict[str, Any]) -> list[str]:
    """Require an explicit subset and bounded missing-value filter thresholds."""
    subset = params.get("subset")
    if not isinstance(subset, list) or not subset or any(not isinstance(c, str) for c in subset):
        raise ValueError("pre_split_steps DropMissingRows requires explicit subset columns.")
    if params.get("how", "any") not in ("any", "all") or set(params) - {
        "subset",
        "how",
        "threshold",
        "missing_threshold",
    }:
        raise ValueError("pre_split_steps DropMissingRows has invalid row-filter params.")
    _validate_missing_thresholds(params, len(subset))
    return subset


def _validate_manual_limits(limit: dict[str, Any]) -> None:
    """Reject unknown, non-finite or reversed manual bounds for a named column."""
    if set(limit) - {"lower", "upper"} or not any(
        key in limit and limit[key] is not None for key in ("lower", "upper")
    ):
        raise ValueError("pre_split_steps ManualBounds requires lower or upper.")
    _validate_manual_bound_values(limit)


def _manual_bounds_columns(params: dict[str, Any]) -> list[str]:
    """Validate each explicit column bound before returning its source dependencies."""
    bounds = params.get("bounds")
    if set(params) - {"bounds"}:
        raise ValueError("pre_split_steps ManualBounds supports only bounds.")
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("pre_split_steps ManualBounds requires explicit bounds.")
    for column, limit in bounds.items():
        if not isinstance(column, str) or not isinstance(limit, dict) or not limit:
            raise ValueError("pre_split_steps ManualBounds requires named column bounds.")
        _validate_manual_limits(limit)
    return list(bounds)


def _fixed_pre_split_columns(
    step: dict[str, Any], protected: tuple[str | None, ...]
) -> tuple[str, ...]:
    """Admit fixed edits only when they leave record keys and source times intact."""
    fixed = fixed_columns(step)
    protected_names = {name.casefold() for name in protected if name is not None}
    if any(name.casefold() in protected_names for name in fixed):
        raise ValueError("pre_split_steps cannot write protected record keys or source time.")
    return fixed


def _pre_split_step_columns(
    step: dict[str, Any],
    index: int,
    *,
    custom_filter: bool,
    protected: tuple[str | None, ...],
) -> tuple[str, ...] | list[str]:
    """Apply the column contract for one admitted fixed edit or row filter."""
    step_type = step["transformer"]
    params = step.get("params", {})
    if step_type in FIXED_TYPES:
        return _fixed_pre_split_columns(step, protected)
    if step_type == "DropMissingRows":
        return _missing_row_filter_columns(params)
    if step_type == "ManualBounds":
        return _manual_bounds_columns(params)
    if step_type == "Deduplicate":
        return deduplicate_columns(step)
    if custom_filter:
        return custom_filter_columns(step)
    raise ValueError(f"pre_split_steps[{index}] permits only fixed normalization or row filters.")


def validate_pre_split_step(
    step: Any, index: int, target_column: str, protected: tuple[str | None, ...]
) -> tuple[str, ...] | list[str]:
    """Reject learned or malformed steps before accepting their source dependencies."""
    step_type, params = _pre_split_step_identity(step, index)
    custom_filter = is_project_filter_step(step)
    if (
        step_type != "Deduplicate"
        and not custom_filter
        and step_learns_from_data(step_type, params, target_column=target_column)
    ):
        raise ValueError(f"pre_split_steps[{index}] cannot learn from data before split.")
    columns = _pre_split_step_columns(step, index, custom_filter=custom_filter, protected=protected)
    allowed_fields = {"name", "transformer", "params"}
    if custom_filter:
        allowed_fields.add("pre_split")
    if set(step) - allowed_fields:
        raise ValueError(f"pre_split_steps[{index}] contains unsupported step fields.")
    return columns


def _pre_split_columns(
    steps: Any, target_column: str, protected: tuple[str | None, ...] = ()
) -> tuple[str, ...]:
    """Admit explicit fixed edits and filters, returning all source dependencies."""
    if not isinstance(steps, (tuple, list)):
        raise ValueError("pre_split_steps must be an ordered sequence of Core steps.")
    columns: list[str] = []
    for index, step in enumerate(steps, 1):
        columns.extend(validate_pre_split_step(step, index, target_column, protected))
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
    comparison_sha256: str
    holdout_key_sha256: str


def read_training_snapshot(spark: Any, spec: LocalTrainingSpec) -> pd.DataFrame:
    """Project and cap a versioned Delta read before materializing driver rows."""
    if not isinstance(spec, LocalTrainingSpec):
        raise TypeError("spec must be LocalTrainingSpec.")
    spec = pin_training_lookup(spark, spec)
    _pre_split_columns(
        spec.pre_split_steps,
        spec.target_column,
        (
            *spec.record_key_columns,
            spec.event_column,
            spec.result_available_at_column,
            spec.group_column,
            spec.weight_column,
            *spec.reserved_weight_columns,
        ),
    )
    names = spec.source_columns
    source = spark.read.format("delta").option("versionAsOf", spec.version).table(spec.table)
    source = enrich_training_source(spark, source, spec)
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
        source, selection = sample_training_source(source, spec)
    ordering = ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)
    selected = source.orderBy(*ordering).limit(spec.max_rows + 1)
    frame = _materialize_training_rows(selected, spec, names)
    retain_training_binding(spark, frame, spec)
    if selection is not None:
        frame.attrs["training_selection"] = {**selection, "selected_rows": len(frame)}
    return frame


def sample_training_source(source: Any, spec: LocalTrainingSpec) -> tuple[Any, dict[str, Any]]:
    """Select eligible keys by a seeded hash on Spark before transferring any training rows."""
    F = import_module("pyspark.sql.functions")

    keys = list(spec.record_key_columns)
    floating = {
        field.name
        for field in source.schema.fields
        if field.dataType.typeName() in {"double", "float"}
    }
    _validate_sample_keys(source, keys, floating, F)
    source_rows = source.count()
    source = eligible_training_source(source, spec, floating)
    eligible_rows = (
        source.count()
        if spec.filter_unavailable_results or spec.drop_missing_labels
        else source_rows
    )
    _validate_sample_targets(source, spec, floating, F)
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


def eligible_training_source(source: Any, spec: LocalTrainingSpec, floating: set[str]) -> Any:
    """Apply label time and missing-target policies before validating the sampling pool."""
    if spec.filter_unavailable_results:
        result = column_name(cast(str, spec.result_available_at_column))
        if (
            spec.event_column is not None
            and source.where(f"{result} < {column_name(spec.event_column)}").limit(1).count()
        ):
            raise ValueError("Label availability precedes event time.")
        cutoff = instant_microseconds(cast(datetime, spec.result_cutoff))
        source = source.where(f"{result} IS NOT NULL AND {result} <= {cutoff}")
    if spec.drop_missing_labels:
        target = column_name(spec.target_column)
        condition = f"{target} IS NOT NULL"
        if spec.target_column in floating:
            condition += f" AND NOT isnan({target})"
        source = source.where(condition)
    return source


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


def _validate_pre_split_inputs(
    native: pd.DataFrame | pl.DataFrame, step: dict[str, Any], target_column: str
) -> list[str]:
    """Check declared columns, bounds dtypes and deduplication label consistency."""
    step_type = step["transformer"]
    if step_type in FIXED_TYPES:
        columns = fixed_columns(step)
    elif step_type in ("DropMissingRows", "Deduplicate"):
        columns = step["params"]["subset"]
    elif step_type == "ManualBounds":
        columns = step["params"]["bounds"]
    else:
        columns = custom_filter_columns(step)
    if any(column not in native.columns for column in columns):
        missing = sorted(set(columns) - set(native.columns))
        raise ValueError(f"pre_split_steps {step['name']} missing source columns: {missing}.")
    _validate_filter_values(native, step, columns, target_column)
    return list(columns)


def _validate_pre_split_survivors(
    filtered: pd.DataFrame | pl.DataFrame,
    keys: list[str],
    before_keys: list[tuple[Any, ...]],
    before_columns: dict[str, list[Any]],
    *,
    target_column: str,
    allowed_edits: set[str],
) -> None:
    """Keep surviving identities ordered and undeclared source values paired."""
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
    _validate_survivor_values(
        filtered, after_keys, positions, before_columns, target_column, allowed_edits
    )


def apply_pre_split_step(
    native: pd.DataFrame | pl.DataFrame,
    step: dict[str, Any],
    *,
    keys: list[str],
    target_column: str,
) -> pd.DataFrame | pl.DataFrame:
    """Apply one fixed step with identical training and scoring survivor validation."""
    columns = _validate_pre_split_inputs(native, step, target_column)
    step_type = step["transformer"]
    original_columns = list(native.columns)
    original_dtypes = list(native.dtypes)
    before_keys = (
        list(native.select(keys).iter_rows())
        if isinstance(native, pl.DataFrame)
        else list(native[keys].itertuples(index=False, name=None))
    )
    before_columns = {column: native[column].to_list() for column in native.columns}
    artifact = NodeRegistry.get_calculator(step["transformer"])().fit(native, step["params"])
    filtered = NodeRegistry.get_applier(step["transformer"])().apply(native, artifact)
    _validate_filter_frame(
        filtered, native, is_project_filter_step(step), original_columns, original_dtypes
    )
    _validate_pre_split_survivors(
        filtered,
        keys,
        before_keys,
        before_columns,
        target_column=target_column,
        allowed_edits=set(columns) if step_type in FIXED_TYPES else set(),
    )
    return filtered


def _apply_pre_split_steps(
    selected: pd.DataFrame, spec: LocalTrainingSpec, engine: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply each ordered eligibility step once and record its survivor counts."""
    filter_counts = []
    native = pl.from_pandas(selected) if spec.pre_split_steps and engine == "polars" else selected
    for step in spec.pre_split_steps:
        before_count = len(native)
        filtered = apply_pre_split_step(
            native, step, keys=list(spec.record_key_columns), target_column=spec.target_column
        )
        filter_counts.append(
            {
                "name": step["name"],
                "transformer": step["transformer"],
                "input_rows": before_count,
                "excluded_rows": before_count - len(filtered),
                "output_rows": len(filtered),
            }
        )
        native = filtered
    if spec.pre_split_steps:
        selected = native.to_pandas() if isinstance(native, pl.DataFrame) else native
        selected = selected.reset_index(drop=True)
    return selected, filter_counts


def _available_training_snapshot(
    frame: pd.DataFrame, spec: LocalTrainingSpec, engine: str
) -> pd.DataFrame:
    """Validate and order source rows using the shared label and time eligibility policy."""
    _validate_labeled_snapshot(frame, spec, engine)
    selected = frame.copy()
    available = _label_availability(frame, selected, spec)
    ordering = ([spec.event_column] if spec.event_column else []) + list(spec.record_key_columns)
    return selected.loc[available].sort_values(ordering, kind="stable").reset_index(drop=True)


def eligible_training_snapshot(
    frame: pd.DataFrame, spec: LocalTrainingSpec, *, engine: str = "pandas"
) -> pd.DataFrame:
    """Replay eligibility for comparison without imposing a viable model-fit partition.

    Source budgets, keys, dates and sample membership remain validated. Returned
    survivors retain raw feature values, normalized labels, keys and timestamps;
    empty populations are valid and no training weight vector is validated.
    """
    selected = _available_training_snapshot(frame, spec, engine)
    _training_sample_digest(selected, spec)
    raw = selected.set_index(list(spec.record_key_columns), drop=False)
    selected, _ = _apply_pre_split_steps(selected, spec, engine)
    surviving_keys = selected.set_index(list(spec.record_key_columns)).index
    result = raw.loc[surviving_keys].reset_index(drop=True)
    result[spec.target_column] = selected[spec.target_column].to_numpy()
    return result


def split_labeled_snapshot(
    frame: pd.DataFrame,
    spec: LocalTrainingSpec,
    *,
    keep_training_event: bool = False,
    engine: str = "pandas",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split on normalized eligibility values, returning raw X and normalized y.

    Candidate training installs the saved fixed feature prefix before model fit;
    callers of this function receive untransformed model features.
    """
    selected = _available_training_snapshot(frame, spec, engine)
    available_count = len(selected)
    raw_selected = selected.copy()
    sample_digest = _training_sample_digest(selected, spec)
    pre_filter_digest = _key_digest(selected, spec.record_key_columns)
    selected, filter_counts = _apply_pre_split_steps(selected, spec, engine)
    survivor_digest = _key_digest(selected, spec.record_key_columns)
    _validate_training_survivors(selected, spec, survivor_digest)
    train, heldout = partition_training_rows(selected, spec)
    # Hash the ordered identity tuples only; never include keys in model features.
    digest = _key_digest(heldout, spec.record_key_columns)
    if spec.holdout_key_sha256 is not None and digest != spec.holdout_key_sha256:
        raise ValueError("Holdout membership differs from saved training evidence.")
    train_frame, holdout_frame = _raw_model_partitions(
        train, heldout, raw_selected, spec, keep_training_event
    )
    weight_evidence = training_weight_evidence(spec, train)
    if weight_evidence is not None:
        holdout_frame.attrs["training_weights"] = weight_evidence
    holdout_frame.attrs["holdout_key_sha256"] = digest
    holdout_frame.attrs["sample_key_sha256"] = sample_digest
    holdout_frame.attrs["pre_split_filter_counts"] = filter_counts
    holdout_frame.attrs["pre_filter_key_sha256"] = pre_filter_digest
    holdout_frame.attrs["survivor_key_sha256"] = survivor_digest
    holdout_frame.attrs["train_key_sha256"] = _key_digest(train, spec.record_key_columns)
    holdout_frame.attrs["pre_filter_rows"] = available_count
    holdout_frame.attrs["survivor_rows"] = len(selected)
    holdout_frame.attrs["training_rows"] = len(train)
    holdout_frame.attrs["source_rows"] = len(frame)
    if spec.group_column:
        holdout_frame.attrs["group_split"] = _group_split_evidence(
            train, heldout, spec.group_column
        )
    unavailable = (
        len(frame)
        - available_count
        + frame.attrs.get("training_selection", {}).get("unavailable_labels", 0)
    )
    return train_frame, holdout_frame, unavailable


def log_local_model(
    artifact_path: str | Path,
    *,
    run_id: str,
    tracking_uri: str,
    spark: Any = None,
    spec: LocalTrainingSpec | None = None,
) -> str:
    """Import the optional MLflow pyfunc package only for a tracked run."""
    from ....mlflow.models.local_model import log_local_model  # noqa: PLC0415

    if spec is not None and spec.feature_lookup_json is not None:
        from ...feature_store.training import log_training_feature_model  # noqa: PLC0415

        return log_training_feature_model(
            artifact_path, spark=spark, spec=spec, run_id=run_id, tracking_uri=tracking_uri
        )
    return log_local_model(
        artifact_path, run_id=run_id, artifact_path="model", tracking_uri=tracking_uri
    )


def training_spec_payload(spec: LocalTrainingSpec, engine: str) -> dict[str, Any]:
    """Serialize concrete selection settings before and after membership enrichment."""
    payload = asdict(spec)
    for field in ("start", "holdout_start", "cutoff", "result_cutoff"):
        value = getattr(spec, field)
        payload[field] = None if value is None else value.isoformat()
    return {**payload, "engine": engine}


@dataclass(slots=True)
class _FittedCandidate:
    """Carry fit-only state inside one process; durable callers persist its evidence."""

    artifact: Any
    spec: LocalTrainingSpec
    holdout: Any
    training_rows: int
    holdout_rows: int
    unavailable_labels: int
    tags: dict[str, str]
    evidence: dict[str, Any]
    cv_results: dict[str, Any] | None
    source_frame: pd.DataFrame
    evidence_holdout: pd.DataFrame


def candidate_config(
    spec: LocalTrainingSpec,
    config: dict[str, Any],
    *,
    engine: str,
    cv: LocalCVSpec,
    metric: str,
    min_improvement: float,
    champion_version: str | None,
    quality_threshold: float | None,
    quality_gates: dict[str, float] | None = None,
    risk_category: str | None,
) -> dict[str, Any]:
    """Validate one training request and derive its effective preprocessing recipe."""
    if not isinstance(spec, LocalTrainingSpec):
        raise TypeError("spec must be LocalTrainingSpec.")
    if engine not in ("pandas", "polars"):
        raise ValueError("engine must be pandas or polars.")
    if not isinstance(cv, LocalCVSpec):
        raise TypeError("cv must be LocalCVSpec.")
    _pre_split_columns(
        spec.pre_split_steps,
        spec.target_column,
        (
            *spec.record_key_columns,
            spec.event_column,
            spec.result_available_at_column,
            spec.group_column,
            spec.weight_column,
            *spec.reserved_weight_columns,
        ),
    )
    validate_cv_holdout_policy(spec, cv)
    validate_explanation_config(config)
    pipeline_config = prepare_search_pipeline(
        config, cv, target_column=spec.target_column, event_column=spec.event_column
    )
    feature_prefix = projected_fixed_steps(spec.pre_split_steps, spec.input_columns)
    if threshold_policy(pipeline_config)["mode"] != "off":
        pipeline_config["decision_threshold_context"] = {
            "split_strategy": spec.split_strategy,
            "event_column": spec.event_column,
            "group_column": spec.group_column,
            "input_columns": list(spec.input_columns),
            "gap": cv.gap,
        }
    pipeline_config["preprocessing"] = [*feature_prefix, *pipeline_config.get("preprocessing", [])]
    contract = target_contract(spec.pre_split_steps, spec.target_column)
    pipeline_config.pop("pre_split_target_contract", None)
    if contract:
        pipeline_config["pre_split_target_contract"] = contract
    cv.validate_pipeline(
        pipeline_config, target_column=spec.target_column, event_column=spec.event_column
    )
    if spec.stratify and (
        NodeRegistry.get_calculator(base_model_config(config)["type"])().problem_type
        != "classification"
    ):
        raise ValueError("stratify requires a classification model.")
    _validate_candidate_metadata(risk_category, min_improvement)
    validate_quality_policy(
        metric,
        quality_threshold,
        quality_gates,
        task=NodeRegistry.get_calculator(base_model_config(config)["type"])().problem_type,
    )
    _validate_champion_version(champion_version)
    return pipeline_config


def _candidate_cv(
    frame: Any,
    pipeline: dict[str, Any],
    cv: LocalCVSpec,
    spec: LocalTrainingSpec,
    *,
    search: bool,
    evaluate_cv: bool,
    sample_weight: Any = None,
) -> dict[str, Any] | None:
    """Validate search membership or run the optional fixed-model diagnostics.

    Competitions evaluate their shared objective separately, avoiding duplicate
    fixed-model diagnostics while retaining all search membership checks.
    """
    if search:
        validate_search_membership(
            frame, pipeline, cv, target_column=spec.target_column, event_column=spec.event_column
        )
        if threshold_policy(pipeline)["mode"] == "off":
            return None
    if evaluate_cv:
        return evaluate_training_cv(
            frame,
            pipeline,
            cv,
            target_column=spec.target_column,
            event_column=spec.event_column if cv.temporal else None,
            sample_weight=sample_weight,
        )
    return None


def _keep_fit_time(cv: LocalCVSpec, spec: LocalTrainingSpec, automatic: bool) -> bool:
    """Retain event metadata for temporal CV or training-only calibration."""
    return (cv.enabled and cv.temporal) or (automatic and spec.split_strategy == "temporal")


def _native_search_cv(search: bool, evaluate_cv: bool, config: dict) -> bool:
    """Avoid replacing complete decision-policy CV with native search scores."""
    return search and evaluate_cv and threshold_policy(config)["mode"] == "off"


def _candidate_training_frame(
    frame: pd.DataFrame, spec: LocalTrainingSpec, keep_time: bool
) -> pd.DataFrame:
    """Project shared prepared rows to only this candidate's feature and split roles."""
    columns = [*spec.input_columns, spec.target_column]
    if spec.group_column:
        columns.append(spec.group_column)
    if keep_time and spec.event_column:
        columns.append(spec.event_column)
    return frame.loc[:, columns]


def fit_candidate(
    spark: Any,
    spec: LocalTrainingSpec,
    config: dict[str, Any],
    *,
    run: Any,
    pipeline_config: dict[str, Any],
    artifact_path: str | Path,
    engine: str,
    cv: LocalCVSpec,
    risk_category: str | None,
    prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int] | None = None,
    evaluate_cv: bool = True,
) -> _FittedCandidate:
    """Fit CV and final training rows, persisting selection and provenance evidence."""
    spec = pin_fit_lookup(spark, spec, engine)
    run.log_config(pipeline_config, artifact_file="pipeline_config.json")
    run.client.log_dict(run.run_id, config, "training_pipeline_config.json")
    run.log_params({key: getattr(cv, field) for key, field in CV_FIELDS.items()})
    run.client.log_dict(run.run_id, training_spec_payload(spec, engine), "training_snapshot.json")
    automatic = threshold_policy(pipeline_config)["mode"] == "auto"
    temporal_cv = _keep_fit_time(cv, spec, automatic)
    frame, train_frame, holdout, unavailable = prepared_data or read_training_partitions(
        spark, spec, temporal_cv=temporal_cv, engine=engine
    )
    validate_training_frame(frame, spec)
    spec = replace(
        spec,
        holdout_key_sha256=holdout.attrs["holdout_key_sha256"],
        sample_key_sha256=holdout.attrs["sample_key_sha256"],
    )
    train_frame, sample_weight = extract_training_weights(train_frame, spec.weight_column)
    train_frame = _candidate_training_frame(train_frame, spec, temporal_cv)
    if "training_weights" in holdout.attrs:
        pipeline_config = {**pipeline_config, "training_weights": holdout.attrs["training_weights"]}
    native_train = pl.from_pandas(train_frame) if engine == "polars" else train_frame
    native_holdout = pl.from_pandas(holdout) if engine == "polars" else holdout
    search = pipeline_config["modeling"]["type"] == "hyperparameter_tuner"
    cv_results = _candidate_cv(
        native_train,
        pipeline_config,
        cv,
        spec,
        search=search,
        evaluate_cv=evaluate_cv,
        sample_weight=sample_weight,
    )
    native_train = _final_fit_frame(native_train, spec, temporal_cv, search or automatic)
    artifact = fit_local_workflow(
        pipeline_config,
        SplitDataset(
            train=native_train, test=native_train.head(0), train_sample_weight=sample_weight
        ),
        target_column=spec.target_column,
        artifact_path=artifact_path,
        max_rows=spec.max_rows,
        max_bytes=spec.max_bytes,
    )
    if _native_search_cv(search, evaluate_cv, pipeline_config):
        cv_results = post_selection_cv(
            native_train,
            artifact,
            cv,
            target_column=spec.target_column,
            event_column=spec.event_column,
            sample_weight=sample_weight,
        )
    if pipeline_config.get("explainability"):
        log_training_explanations(run, artifact, native_train)
    log_preprocessing_probe(run, artifact, native_holdout, enabled=spec.preprocessing_probe)
    evidence = build_training_evidence(
        spec, holdout, project_source_sha256=artifact.manifest.project_source_sha256
    )
    spec = replace(
        spec,
        survivor_key_sha256=holdout.attrs["survivor_key_sha256"],
        training_evidence_sha256=evidence_digest(evidence),
    )
    return _FittedCandidate(
        artifact,
        spec,
        native_holdout,
        len(train_frame),
        len(holdout),
        unavailable,
        {},
        evidence,
        cv_results,
        frame,
        holdout,
    )


def read_training_partitions(
    spark: Any, spec: LocalTrainingSpec, *, temporal_cv: bool, engine: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Retain the sequential SDK read/split path alongside prepared lifecycle data."""
    frame = read_training_snapshot(spark, spec)
    train, holdout, unavailable = split_labeled_snapshot(
        frame, spec, keep_training_event=temporal_cv, engine=engine
    )
    return frame, train, holdout, unavailable


def log_fitted_candidate(
    run: Any,
    fitted: _FittedCandidate,
    config: dict[str, Any],
    *,
    engine: str,
    risk_category: str | None,
) -> None:
    """Persist fit evidence after SDK evaluation or before a durable stage boundary."""
    from ...observability.monitoring.monitoring_source_evidence import (  # noqa: PLC0415
        source_evidence,
    )

    spec, artifact = fitted.spec, fitted.artifact
    frame, holdout = fitted.source_frame, fitted.evidence_holdout
    cv_results, evidence = fitted.cv_results, fitted.evidence
    unavailable = fitted.unavailable_labels
    _log_tuning_evidence(run, artifact, config)
    threshold_evidence = getattr(artifact.pipeline, "_decision_threshold_evidence", None)
    if threshold_evidence is not None:
        run.client.log_dict(run.run_id, threshold_evidence, "decision_threshold.json")
        run.log_params(
            {
                "decision_threshold_mode": threshold_evidence["mode"],
                "decision_threshold_fitting_rows": threshold_evidence["fitting_rows"],
            }
        )
    log_training_parameters(run, artifact, spec, config)
    if cv_results is not None:
        cv_results.update(
            dataset_id=spec.dataset_id, training_rows=fitted.training_rows, engine=engine
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
            "training_rows": fitted.training_rows,
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
        "model_type": str(base_model_config(config)["type"]),
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
    run.client.log_dict(run.run_id, {"dataset_id": spec.dataset_id, **tags}, "training_data.json")
    run.client.log_dict(
        run.run_id,
        {
            "requested_steps": list(spec.pre_split_steps),
            "counts": holdout.attrs["pre_split_filter_counts"],
            "source_rows": len(frame),
            "eligible_rows": fitted.training_rows + len(holdout),
            "excluded_rows": sum(
                item["excluded_rows"] for item in holdout.attrs["pre_split_filter_counts"]
            ),
        },
        "pre_split_filters.json",
    )
    run.client.log_dict(run.run_id, evidence, "training_filter_evidence.json")
    run.client.log_dict(
        run.run_id,
        source_evidence(frame, spec.source_columns, spec.dataset_id),
        "monitoring_source_evidence.json",
    )
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
        run.run_id, training_spec_payload(spec, engine), "candidate_training_spec.json"
    )
    fitted.tags = tags


def evaluate_candidate(
    artifact: Any,
    native_holdout: Any,
    *,
    spec: LocalTrainingSpec,
    metric: str,
    chart_run: Any = None,
    evaluation_charts: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Require a finite initial holdout metric before any registration."""
    metrics = evaluate_local_holdout(
        artifact,
        native_holdout,
        target_column=spec.target_column,
        on_predictions=chart_recorder(chart_run, artifact, spec, evaluation_charts),
    )
    if metric not in metrics or not math.isfinite(metrics[metric]):
        raise ValueError("Selected metric is unavailable or non-finite on the holdout.")
    return metrics


def compare_candidate(
    candidate: ResolvedModel,
    champion: ResolvedModel | None,
    native_holdout: Any,
    *,
    run: Any,
    spec: LocalTrainingSpec,
    model_name: str,
    metric: str,
    min_improvement: float,
    quality_threshold: float | None,
    quality_gates: dict[str, float] | None = None,
    tracking_uri: str,
    registry_uri: str,
    engine: str,
    training_rows: int,
    holdout_rows: int,
    unavailable: int,
) -> LocalCandidateResult:
    """Compare concrete model versions and save the SDK's unchanged result evidence."""
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
        quality_gates=quality_gates,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    run.client.log_dict(run.run_id, comparison_payload(report), "candidate_comparison.json")
    run.client.log_dict(run.run_id, {"gates": quality_gate_results(report)}, "quality_gates.json")
    comparison_sha256 = comparison_digest(report)
    return LocalCandidateResult(
        run_id=run.run_id,
        model_name=model_name,
        model_version=candidate.version,
        model_digest=candidate.digest or "",
        dataset_id=spec.dataset_id,
        training_rows=training_rows,
        holdout_rows=holdout_rows,
        unavailable_labels=unavailable,
        engine=engine,
        comparison=report,
        comparison_sha256=comparison_sha256,
        holdout_key_sha256=spec.holdout_key_sha256 or "",
    )


def register_candidate(
    model_uri: str,
    model_name: str,
    *,
    tracking_uri: str,
    registry_uri: str,
    tags: dict[str, str],
) -> Any:
    """Register with bounded tags, leaving receipt persistence and resolution to callers."""
    return register_model(
        model_uri,
        model_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        tags={
            key: value if len(value.encode("utf-8")) <= 256 else "See training_data.json"
            for key, value in tags.items()
        },
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
    quality_gates: dict[str, float] | None = None,
    on_registered: Callable[[ResolvedModel], None] | None = None,
    risk_category: str | None = None,
    cv: LocalCVSpec | None = None,
    run_tags: dict[str, str] | None = None,
    evaluation_charts: dict[str, Any] | None = None,
) -> LocalCandidateResult:
    """Fit, register and compare; optionally notify an explicit lifecycle owner.

    By default no aliases change. The on_registered hook runs after successful
    registration and before comparison, allowing the caller to nominate a
    contender without making the generic training service an alias writer.
    """
    chart_settings(evaluation_charts)
    cv = LocalCVSpec() if cv is None else cv
    pipeline_config = candidate_config(
        spec,
        config,
        engine=engine,
        cv=cv,
        metric=metric,
        min_improvement=min_improvement,
        champion_version=champion_version,
        quality_threshold=quality_threshold,
        quality_gates=quality_gates,
        risk_category=risk_category,
    )
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
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        failure_policy="raise",
    )
    with track_run(tracking, run_name=run_name) as run:
        if run.run_id is None:
            raise RuntimeError("MLflow did not provide a run ID.")
        run.set_tags(run_tags or {})
        fitted = fit_candidate(
            spark,
            spec,
            config,
            run=run,
            pipeline_config=pipeline_config,
            artifact_path=artifact_path,
            engine=engine,
            cv=cv,
            risk_category=risk_category,
        )
        metrics = evaluate_candidate(
            fitted.artifact,
            fitted.holdout,
            spec=fitted.spec,
            metric=metric,
            chart_run=run,
            evaluation_charts=evaluation_charts,
        )
        log_fitted_candidate(
            run,
            fitted,
            config,
            engine=engine,
            risk_category=risk_category,
        )
        run.log_metrics(metrics)
        model_uri = log_local_model(
            artifact_path,
            run_id=run.run_id,
            tracking_uri=tracking_uri,
            **feature_log_options(spark, fitted.spec),
        )
    registered = register_candidate(
        model_uri,
        model_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        tags=fitted.tags,
    )
    candidate = resolve_model(
        model_name,
        version=str(registered.version),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    if on_registered is not None:
        on_registered(candidate)
    return compare_candidate(
        candidate,
        champion,
        fitted.holdout,
        run=run,
        spec=fitted.spec,
        model_name=model_name,
        metric=metric,
        min_improvement=min_improvement,
        quality_threshold=quality_threshold,
        quality_gates=quality_gates,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        engine=engine,
        training_rows=fitted.training_rows,
        holdout_rows=fitted.holdout_rows,
        unavailable=fitted.unavailable_labels,
    )


def _validate_random_window(spec: LocalTrainingSpec) -> None:
    """Validate optional event selection independently of random partition settings."""
    if spec.holdout_start is not None:
        raise ValueError("Random split requires inactive holdout_start to be null.")
    if spec.event_column is None and (
        spec.start is not None
        or spec.cutoff is not None
        or spec.event_time_parsing != TrainingDateSpec()
    ):
        raise ValueError("Random split requires inactive event/date fields to be null.")
    if spec.event_column is not None:
        start = _validate_instant(spec.start, "start")
        cutoff = _validate_instant(spec.cutoff, "cutoff")
        if start >= cutoff:
            raise ValueError("Require start < cutoff for event selection.")


def _validate_inactive_random_settings(spec: LocalTrainingSpec) -> None:
    """Reject active or mistyped random settings for a temporal split."""
    if (
        spec.test_size not in (None, 0.2)
        or spec.random_state not in (None, 42)
        or spec.stratify not in (None, False)
        or (spec.random_state is not None and type(spec.random_state) is not int)
        or (spec.stratify is not None and type(spec.stratify) is not bool)
    ):
        raise ValueError("Temporal split cannot use active random split settings.")


def _validate_missing_thresholds(params: dict[str, Any], subset_size: int) -> None:
    """Require count and percentage thresholds within the explicitly selected subset."""
    threshold = params.get("threshold")
    if threshold is not None and (type(threshold) is not int or not 0 <= threshold <= subset_size):
        raise ValueError("pre_split_steps threshold must be an integer from 0 to subset size.")
    missing_threshold = params.get("missing_threshold")
    if missing_threshold is not None and (
        type(missing_threshold) not in (int, float)
        or not math.isfinite(missing_threshold)
        or not 0 <= missing_threshold <= 100
    ):
        raise ValueError("pre_split_steps missing_threshold must be between 0 and 100.")


def _validate_manual_bound_values(limit: dict[str, Any]) -> None:
    """Check finite numeric values before comparing lower and upper manual limits."""
    for value in limit.values():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("pre_split_steps ManualBounds requires finite numeric bounds.")
    if (
        limit.get("lower") is not None
        and limit.get("upper") is not None
        and limit["lower"] > limit["upper"]
    ):
        raise ValueError("pre_split_steps ManualBounds lower must not exceed upper.")


def _pre_split_step_identity(step: Any, index: int) -> tuple[str, dict[str, Any]]:
    """Validate a named Core transformer and its parameter object before policy checks."""
    if not isinstance(step, dict) or not isinstance(step.get("name"), str) or not step["name"]:
        raise ValueError(f"pre_split_steps[{index}] requires a name.")
    step_type = step.get("transformer")
    params = step.get("params", {})
    if not isinstance(params, dict) or not isinstance(step_type, str):
        raise ValueError(f"pre_split_steps[{index}] requires a Core transformer and params.")
    return step_type, params


def _materialize_training_rows(
    selected: Any, spec: LocalTrainingSpec, names: tuple[str, ...]
) -> pd.DataFrame:
    """Decode bounded rows and dates while enforcing serialized and local frame budgets."""
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
    if frame_bytes(frame) > spec.max_bytes:
        raise ValueError("Training frame exceeds max_bytes.")
    return frame


def _validate_sample_keys(source: Any, keys: list[str], floating: set[str], F: Any) -> None:
    """Reject missing and duplicated sampling keys before counting the source population."""
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


def _validate_sample_targets(
    source: Any, spec: LocalTrainingSpec, floating: set[str], F: Any
) -> None:
    """Require available targets unless an explicit missing-target filter owns exclusion."""
    invalid_target = F.col(spec.target_column).isNull()
    if spec.target_column in floating:
        invalid_target = invalid_target | F.isnan(F.col(spec.target_column))
    target_filter = any(
        step["transformer"] == "DropMissingRows" and spec.target_column in step["params"]["subset"]
        for step in spec.pre_split_steps
    )
    if not target_filter and source.where(invalid_target).limit(1).count():
        raise ValueError("Available labels must have nonnull targets before sampling.")


def _validate_filter_values(
    native: pd.DataFrame | pl.DataFrame, step: dict[str, Any], columns: Any, target_column: str
) -> None:
    """Check numeric bounds and label consistency after required columns exist."""
    step_type = step["transformer"]
    if step["transformer"] == "ManualBounds":
        for column in columns:
            if isinstance(native, pl.DataFrame):
                numeric = native.schema[column].is_numeric()
            else:
                dtype = native[column].dtype
                numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(
                    dtype
                )
            if not numeric:
                raise ValueError(f"pre_split_steps ManualBounds requires numeric column {column}.")
    if step_type == "Deduplicate" and target_column in native.columns:
        _validate_deduplicate_labels(native, columns, target_column)


def _validate_deduplicate_labels(
    native: pd.DataFrame | pl.DataFrame, columns: Any, target_column: str
) -> None:
    """Reject groups whose deduplication would choose among conflicting labels."""
    working = native.to_pandas() if isinstance(native, pl.DataFrame) else native
    for _, group in working.groupby(list(columns), dropna=False, sort=False):
        labels = group[target_column]
        if labels.nunique(dropna=True) > 1 or (labels.isna().any() and labels.notna().any()):
            raise ValueError(
                "pre_split_steps Deduplicate found conflicting target values "
                "within one subset group."
            )


def _validate_survivor_values(
    filtered: pd.DataFrame | pl.DataFrame,
    after_keys: list[tuple[Any, ...]],
    positions: dict[tuple[Any, ...], int],
    before_columns: dict[str, list[Any]],
    target_column: str,
    allowed_edits: set[str],
) -> None:
    """Preserve target pairing and every source value outside declared fixed edits."""
    for column in filtered.columns:
        if column in allowed_edits:
            continue
        after_values = filtered[column].to_list()
        for key, value in zip(after_keys, after_values, strict=True):
            expected = before_columns[column][positions[key]]
            expected_missing = bool(pd.isna(expected))
            value_missing = bool(pd.isna(value))
            if expected_missing != value_missing or (not expected_missing and expected != value):
                if column == target_column:
                    raise ValueError("pre_split_steps must preserve target pairing.")
                raise ValueError(f"pre_split_steps changed undeclared column {column}.")


def _validate_filter_frame(
    filtered: Any,
    native: pd.DataFrame | pl.DataFrame,
    custom_filter: bool,
    original_columns: list[str],
    original_dtypes: list[Any],
) -> None:
    """Require the same frame type and columns, plus stable custom-filter dtypes."""
    if not isinstance(filtered, type(native)):
        raise ValueError("pre_split_steps filter did not return a frame.")
    if list(filtered.columns) != original_columns:
        raise ValueError("pre_split_steps must preserve all source columns.")
    if custom_filter and list(filtered.dtypes) != original_dtypes:
        raise ValueError("pre_split_steps custom filter must preserve source dtypes.")


def _validate_labeled_snapshot(frame: pd.DataFrame, spec: LocalTrainingSpec, engine: str) -> None:
    """Validate the split request, source schema, budgets and unique identities."""
    if not isinstance(frame, pd.DataFrame) or not isinstance(spec, LocalTrainingSpec):
        raise TypeError("Expected a pandas frame and LocalTrainingSpec.")
    _pre_split_columns(
        spec.pre_split_steps,
        spec.target_column,
        (
            *spec.record_key_columns,
            spec.event_column,
            spec.result_available_at_column,
            spec.group_column,
            spec.weight_column,
            *spec.reserved_weight_columns,
        ),
    )
    if engine not in ("pandas", "polars"):
        raise ValueError("engine must be pandas or polars.")
    missing_columns = sorted(set(spec.source_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Training snapshot is missing required source columns: {missing_columns}."
        )
    if len(frame) > spec.max_rows or frame_bytes(frame) > spec.max_bytes:
        raise ValueError("Training snapshot exceeds max_rows or max_bytes.")
    if frame.loc[:, list(spec.record_key_columns)].isna().any().any():
        raise ValueError("Training row keys must not be null.")
    if frame.duplicated(subset=list(spec.record_key_columns)).any():
        raise ValueError("Training row keys must be unique.")


def _validate_training_event_window(events: pd.Series, spec: LocalTrainingSpec) -> None:
    """Require each normalized event to fall within the pinned half-open source window."""
    if events.isna().any() or (events < spec.start).any() or (events >= spec.cutoff).any():
        raise ValueError("Training event time falls outside the pinned window.")


def _label_availability(
    frame: pd.DataFrame, selected: pd.DataFrame, spec: LocalTrainingSpec
) -> pd.Series:
    """Normalize source event times before checking result availability and cutoff."""
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
        _validate_training_event_window(events, spec)
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
    if spec.drop_missing_labels:
        available &= frame[spec.target_column].notna()
    return available


def _training_sample_digest(selected: pd.DataFrame, spec: LocalTrainingSpec) -> str | None:
    """Verify bounded sample size and the saved ordered sample membership."""
    sample_digest = None
    if spec.training_sample_rows is not None:
        if len(selected) > spec.training_sample_rows:
            raise ValueError("Training snapshot was not bounded by training_sample_rows.")
        sample_digest = _key_digest(selected, spec.record_key_columns)
        if spec.sample_key_sha256 is not None and sample_digest != spec.sample_key_sha256:
            raise ValueError("Training sample membership differs from saved evidence.")
    return sample_digest


def _validate_training_survivors(
    selected: pd.DataFrame, spec: LocalTrainingSpec, survivor_digest: str
) -> None:
    """Require pinned survivor membership, available targets and enough eligible rows."""
    if spec.survivor_key_sha256 is not None and survivor_digest != spec.survivor_key_sha256:
        raise ValueError("Survivor membership differs from saved training evidence.")
    if selected[spec.target_column].isna().any():
        raise ValueError(
            "Available labels must have nonnull targets; add explicit DropMissingRows for the target."
        )
    if spec.pre_split_steps and len(selected) < 4:
        raise ValueError(
            "Training filters leave fewer than four eligible rows for train and holdout."
        )


def _group_split_evidence(
    train: pd.DataFrame, heldout: pd.DataFrame, column: str
) -> dict[str, Any]:
    """Record bounded group membership receipts without exposing raw identities."""
    train_groups = train[[column]].drop_duplicates().sort_values(column)
    heldout_groups = heldout[[column]].drop_duplicates().sort_values(column)
    return {
        "column": column,
        "training_groups": len(train_groups),
        "holdout_groups": len(heldout_groups),
        "training_groups_sha256": _key_digest(train_groups, (column,)),
        "holdout_groups_sha256": _key_digest(heldout_groups, (column,)),
    }


def validate_cv_holdout_policy(spec: LocalTrainingSpec, cv: LocalCVSpec) -> None:
    """Require final holdout boundaries that match the requested nested split policy."""
    if (
        cv.enabled
        and cv.method == "nested_cv"
        and cv.temporal
        and spec.split_strategy != "temporal"
    ):
        raise ValueError("Nested temporal CV requires a temporal final holdout.")
    if cv.group_column != spec.group_column:
        raise ValueError("cv_group_column must match the training spec group_column.")
    if spec.group_column and spec.stratify:
        raise ValueError("Group holdout uses whole groups; set stratify=false.")


def _partition_group_rows(
    selected: pd.DataFrame, spec: LocalTrainingSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Isolate complete entities in the final holdout, rejecting missing identities."""
    groups = selected[spec.group_column]
    if groups.isna().any():
        raise ValueError("Group metadata must be nonnull.")
    if spec.stratify:
        raise ValueError("Group holdout uses whole groups; set stratify=false.")
    if spec.split_strategy == "temporal":
        mask = selected[spec.event_column] >= spec.holdout_start
        train, heldout = selected.loc[~mask], selected.loc[mask]
        if set(train[spec.group_column]) & set(heldout[spec.group_column]):
            raise ValueError("Final temporal holdout must contain disjoint groups.")
        return train, heldout
    if groups.nunique() < 2:
        raise ValueError("Group holdout requires at least two distinct groups.")
    assert spec.test_size is not None and spec.random_state is not None
    splitter = DataSplitter(test_size=spec.test_size, random_state=spec.random_state)
    train, heldout = splitter.split_indices(len(selected), groups=groups)
    return selected.iloc[train], selected.iloc[heldout]


def partition_training_rows(
    selected: pd.DataFrame, spec: LocalTrainingSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build reproducible random or temporal partitions with viable labeled row counts."""
    if spec.group_column:
        train, heldout = _partition_group_rows(selected, spec)
    elif spec.split_strategy == "random":
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
    return train, heldout


def _raw_model_partitions(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    raw_selected: pd.DataFrame,
    spec: LocalTrainingSpec,
    keep_training_event: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover untouched features by immutable keys while retaining normalized target values."""
    columns = [*spec.input_columns, spec.target_column]
    train_columns = (
        [*columns, spec.event_column] if keep_training_event and spec.event_column else columns
    )
    if spec.weight_column:
        train_columns = [*train_columns, spec.weight_column]
    if spec.group_column:
        train_columns = [*train_columns, spec.group_column]
    if spec.pre_split_steps:
        raw_positions = {
            key: index
            for index, key in enumerate(
                raw_selected.loc[:, list(spec.record_key_columns)].itertuples(
                    index=False, name=None
                )
            )
        }

        def raw_partition(partition: pd.DataFrame, output_columns: list[str]) -> pd.DataFrame:
            """Recover untouched model inputs by immutable keys after normalized splitting."""
            keys = partition.loc[:, list(spec.record_key_columns)].itertuples(
                index=False, name=None
            )
            raw = (
                raw_selected.iloc[[raw_positions[key] for key in keys]]
                .loc[:, output_columns]
                .copy()
            )
            raw[spec.target_column] = partition[spec.target_column].to_numpy()
            return raw.reset_index(drop=True)

        train_frame = raw_partition(train, train_columns)
        holdout_frame = raw_partition(heldout, columns)
    else:
        train_frame = train.loc[:, train_columns].reset_index(drop=True)
        holdout_frame = heldout.loc[:, columns].reset_index(drop=True)
    return train_frame, holdout_frame


def _validate_candidate_metadata(risk_category: str | None, min_improvement: float) -> None:
    """Validate risk label size and a finite nonnegative improvement requirement."""
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


def _validate_champion_version(champion_version: str | None) -> None:
    """Require an explicitly supplied champion version to be a positive decimal string."""
    if champion_version is not None and (
        type(champion_version) is not str
        or not champion_version.isdecimal()
        or int(champion_version) <= 0
    ):
        raise ValueError("champion_version must be a concrete positive version.")


def _final_fit_frame(
    native_train: pd.DataFrame | pl.DataFrame,
    spec: LocalTrainingSpec,
    temporal_cv: bool,
    search: bool,
) -> pd.DataFrame | pl.DataFrame:
    """Remove CV-only split metadata before fitting a fixed final model."""
    if not search:
        model_columns = [*spec.input_columns, spec.target_column]
        native_train = (
            native_train.select(model_columns)
            if isinstance(native_train, pl.DataFrame)
            else native_train.loc[:, model_columns]
        )
    return native_train


def _log_tuning_evidence(run: Any, artifact: Any, config: dict[str, Any]) -> None:
    """Persist fitted search evidence and metrics before logging CV results."""
    if config["modeling"]["type"] == "hyperparameter_tuner":
        search_result = tuning_evidence(artifact)
        if search_result is None:
            raise ValueError("Fitted search artifact lacks tuning evidence.")
        run.client.log_dict(run.run_id, search_result, "tuning.json")
        run.log_params(tuning_run_params(search_result))
        run.log_metrics({"tuning_best_score": search_result["best_score"]})
