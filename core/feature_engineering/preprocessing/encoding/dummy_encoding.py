"""Dummy encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
from pandas.api import types as pd_types

from core.feature_engineering.schemas import (
    DummyEncodingAppliedColumnSignal,
    DummyEncodingNodeSignal,
)
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN

from ...shared.utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES = 20
DUMMY_ENCODING_MAX_CARDINALITY_LIMIT = 200
DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR = "_"


@dataclass
class NormalizedDummyEncodingConfig:
    columns: List[str]
    auto_detect: bool
    drop_original: bool
    drop_first: bool
    include_missing: bool
    max_categories: int
    prefix_separator: str
    skipped_columns: List[str]


@dataclass
class DummyEncodingOptions:
    drop_first: bool
    include_missing: bool
    prefix_separator: str
    drop_original: bool


@dataclass
class DummyEncodingRuntime:
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    train_mask: Optional[pd.Series]
    train_row_count: int
    split_counts: Dict[str, int]


@dataclass
class DummyFrameResult:
    frame: pd.DataFrame
    base_columns: List[str]
    final_columns: List[str]
    rename_map: Dict[str, str]
    includes_missing_dummy: bool


@dataclass
class ColumnProcessingResult:
    frame: pd.DataFrame
    detail: Optional[str]
    skip_reason: Optional[str]
    applied_signal: Optional[DummyEncodingAppliedColumnSignal]


@dataclass
class _DummyEncodingPreparation:
    working_frame: pd.DataFrame
    existing_columns: set[str]
    runtime: DummyEncodingRuntime
    candidate_columns: List[str]
    skipped_details: List[str]


@dataclass
class _DummyEncodingExecutionResult:
    frame: pd.DataFrame
    encoded_details: List[str]
    skipped_details: List[str]


def _normalize_dummy_encoding_config(config: Any) -> NormalizedDummyEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)
    drop_first = _coerce_config_boolean(config.get("drop_first"), default=True)
    include_missing = _coerce_config_boolean(config.get("include_missing"), default=False)

    raw_max_categories = config.get("max_categories")
    max_categories: int
    try:
        numeric = float(raw_max_categories)
        if math.isnan(numeric):
            raise ValueError
        max_categories = int(round(numeric))
    except (TypeError, ValueError):
        max_categories = DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES

    if max_categories < 0:
        max_categories = 0
    if max_categories > DUMMY_ENCODING_MAX_CARDINALITY_LIMIT:
        max_categories = DUMMY_ENCODING_MAX_CARDINALITY_LIMIT

    prefix_separator = str(config.get("prefix_separator") or DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR)
    if not prefix_separator:
        prefix_separator = DUMMY_ENCODING_DEFAULT_PREFIX_SEPARATOR

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedDummyEncodingConfig(
        columns=columns,
        auto_detect=auto_detect,
        drop_original=drop_original,
        drop_first=drop_first,
        include_missing=include_missing,
        max_categories=max_categories,
        prefix_separator=prefix_separator,
        skipped_columns=skipped_columns,
    )


def _resolve_unique_column_name(candidate: str, existing: set[str]) -> str:
    if candidate not in existing:
        return candidate
    counter = 2
    while True:
        next_candidate = f"{candidate}_{counter}"
        if next_candidate not in existing:
            return next_candidate
        counter += 1


def _collect_dummy_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedDummyEncodingConfig,
    signal: DummyEncodingNodeSignal,
) -> Tuple[List[str], List[str]]:
    skipped_configured = set(config.skipped_columns)
    candidate_columns: List[str] = []
    skipped_details: List[str] = []
    seen: set[str] = set()

    def register_column(raw_column: Any) -> None:
        normalized = str(raw_column or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        if normalized in skipped_configured:
            reason = f"{normalized} (skipped)"
            skipped_details.append(reason)
            signal.skipped_columns.append(reason)
            return
        candidate_columns.append(normalized)

    for column in config.columns:
        register_column(column)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            register_column(column)

    return candidate_columns, skipped_details


def _build_dummy_runtime(
    frame: pd.DataFrame,
    pipeline_id: Optional[str],
    node_id: Optional[Any],
) -> DummyEncodingRuntime:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    if not pipeline_id or not has_splits:
        return DummyEncodingRuntime(
            storage=None,
            pipeline_id=pipeline_id,
            node_id=str(node_id) if node_id is not None else None,
            train_mask=None,
            train_row_count=0,
            split_counts={},
        )

    storage = get_pipeline_store()
    split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
    train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
    train_row_count = int(split_counts.get("train", 0))

    return DummyEncodingRuntime(
        storage=storage,
        pipeline_id=pipeline_id,
        node_id=str(node_id) if node_id is not None else None,
        train_mask=train_mask,
        train_row_count=train_row_count,
        split_counts=split_counts,
    )


def _validate_dummy_column(
    column: str,
    frame: pd.DataFrame,
    config: NormalizedDummyEncodingConfig,
) -> Tuple[Optional[pd.Series], Optional[int], Optional[str]]:
    if column not in frame.columns:
        return None, None, f"{column} (missing)"

    series = frame[column]
    if pd_types.is_bool_dtype(series):
        return None, None, f"{column} (boolean column)"

    if not (
        pd_types.is_object_dtype(series)
        or pd_types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None, None, f"{column} (non-categorical dtype)"

    series_string = series.astype("string")
    unique_total = series_string.nunique(dropna=True)

    if unique_total <= 1:
        return None, None, f"{column} (single category)"

    if config.max_categories and unique_total > config.max_categories:
        return None, None, f"{column} ({unique_total} categories > {config.max_categories})"

    return series_string, int(unique_total), None


def _determine_fit_mode(
    column: str,
    runtime: DummyEncodingRuntime,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    if runtime.storage is None:
        return "fit", None, None, None

    stored_payload = runtime.storage.get_transformer(
        pipeline_id=runtime.pipeline_id,
        node_id=runtime.node_id,
        transformer_name="dummy_encoder",
        column_name=column,
    )
    stored_metadata = runtime.storage.get_metadata(
        pipeline_id=runtime.pipeline_id,
        node_id=runtime.node_id,
        transformer_name="dummy_encoder",
        column_name=column,
    )

    if runtime.train_row_count <= 0:
        if isinstance(stored_payload, dict):
            return "reuse", stored_payload, stored_metadata, None
        return "fit", None, None, f"{column} (no training data)"

    return "fit", stored_payload, stored_metadata, None


def _resolve_effective_options(
    config: NormalizedDummyEncodingConfig,
    stored_metadata: Optional[Dict[str, Any]],
) -> DummyEncodingOptions:
    options = DummyEncodingOptions(
        drop_first=config.drop_first,
        include_missing=config.include_missing,
        prefix_separator=config.prefix_separator,
        drop_original=config.drop_original,
    )

    if isinstance(stored_metadata, dict):
        if "drop_first" in stored_metadata:
            options.drop_first = bool(stored_metadata.get("drop_first"))
        if "include_missing" in stored_metadata:
            options.include_missing = bool(stored_metadata.get("include_missing"))
        if isinstance(stored_metadata.get("prefix_separator"), str):
            options.prefix_separator = str(stored_metadata.get("prefix_separator"))
        if "drop_original" in stored_metadata:
            options.drop_original = bool(stored_metadata.get("drop_original"))

    return options


def _fit_dummy_frame(
    column: str,
    series_string: pd.Series,
    options: DummyEncodingOptions,
    existing_columns: set[str],
    runtime: DummyEncodingRuntime,
) -> Tuple[Optional[DummyFrameResult], Optional[str]]:
    if runtime.storage is not None and runtime.train_mask is not None and not runtime.train_mask.any():
        return None, f"{column} (no training data)"

    train_series = series_string
    if runtime.storage is not None and runtime.train_mask is not None:
        train_series = series_string[runtime.train_mask]

    try:
        train_dummy_frame = pd.get_dummies(
            train_series,
            prefix=column,
            prefix_sep=options.prefix_separator,
            dummy_na=options.include_missing,
            drop_first=options.drop_first,
            dtype="UInt8",
        )
    except Exception:
        return None, f"{column} (failed to encode)"

    if train_dummy_frame.empty:
        return None, f"{column} (no dummy columns generated)"

    base_columns = list(train_dummy_frame.columns)

    try:
        full_dummy_frame = pd.get_dummies(
            series_string,
            prefix=column,
            prefix_sep=options.prefix_separator,
            dummy_na=options.include_missing,
            drop_first=options.drop_first,
            dtype="UInt8",
        )
    except Exception:
        return None, f"{column} (failed to encode)"

    full_dummy_frame = full_dummy_frame.reindex(columns=base_columns, fill_value=0).astype("UInt8")

    rename_map: Dict[str, str] = {}
    final_columns: List[str] = []
    for base_name in base_columns:
        target_name = _resolve_unique_column_name(base_name, existing_columns)
        rename_map[base_name] = target_name
        existing_columns.add(target_name)
        final_columns.append(target_name)

    full_dummy_frame = full_dummy_frame.rename(columns=rename_map).astype("UInt8")

    includes_missing_dummy = bool(options.include_missing and series_string.isna().any())

    return (
        DummyFrameResult(
            frame=full_dummy_frame,
            base_columns=base_columns,
            final_columns=final_columns,
            rename_map=rename_map,
            includes_missing_dummy=includes_missing_dummy,
        ),
        None,
    )


def _reuse_dummy_frame(
    column: str,
    series_string: pd.Series,
    options: DummyEncodingOptions,
    existing_columns: set[str],
    stored_payload: Optional[Dict[str, Any]],
    stored_metadata: Optional[Dict[str, Any]],
) -> Tuple[Optional[DummyFrameResult], Optional[str]]:
    if not isinstance(stored_payload, dict):
        return None, f"{column} (stored transformer invalid)"

    base_columns = [str(name) for name in stored_payload.get("base_columns", []) if str(name).strip()]
    final_columns = [str(name) for name in stored_payload.get("final_columns", []) if str(name).strip()]
    if not base_columns or not final_columns:
        return None, f"{column} (stored transformer incomplete)"

    stored_rename_map = stored_payload.get("rename_map", {})
    if isinstance(stored_rename_map, dict):
        rename_map = {str(key): str(value) for key, value in stored_rename_map.items()}
    else:
        rename_map = {name: name for name in base_columns}

    try:
        dummy_frame_current = pd.get_dummies(
            series_string,
            prefix=column,
            prefix_sep=options.prefix_separator,
            dummy_na=options.include_missing,
            drop_first=options.drop_first,
            dtype="UInt8",
        )
    except Exception:
        return None, f"{column} (failed to encode)"

    dummy_frame_current = dummy_frame_current.reindex(columns=base_columns, fill_value=0).astype("UInt8")
    dummy_frame_current = dummy_frame_current.rename(columns=rename_map)
    dummy_frame_current = dummy_frame_current.reindex(columns=final_columns, fill_value=0).astype("UInt8")

    existing_columns.update(final_columns)

    includes_missing_dummy = bool(options.include_missing and series_string.isna().any())
    if isinstance(stored_metadata, dict) and "includes_missing_dummy" in stored_metadata:
        includes_missing_dummy = bool(stored_metadata.get("includes_missing_dummy"))

    return (
        DummyFrameResult(
            frame=dummy_frame_current,
            base_columns=base_columns,
            final_columns=final_columns,
            rename_map=rename_map,
            includes_missing_dummy=includes_missing_dummy,
        ),
        None,
    )


def _build_dummy_frame(
    column: str,
    series_string: pd.Series,
    options: DummyEncodingOptions,
    existing_columns: set[str],
    runtime: DummyEncodingRuntime,
    fit_mode: str,
    stored_payload: Optional[Dict[str, Any]],
    stored_metadata: Optional[Dict[str, Any]],
) -> Tuple[Optional[DummyFrameResult], Optional[str]]:
    if fit_mode == "fit":
        return _fit_dummy_frame(column, series_string, options, existing_columns, runtime)
    return _reuse_dummy_frame(column, series_string, options, existing_columns, stored_payload, stored_metadata)


def _apply_dummy_frame(
    working_frame: pd.DataFrame,
    column: str,
    frame_result: DummyFrameResult,
    options: DummyEncodingOptions,
    existing_columns: set[str],
) -> pd.DataFrame:
    insert_at = working_frame.columns.get_loc(column) + (0 if options.drop_original else 1)
    for offset, dummy_column in enumerate(frame_result.final_columns):
        if dummy_column in working_frame.columns:
            working_frame[dummy_column] = frame_result.frame[dummy_column]
        else:
            working_frame.insert(insert_at + offset, dummy_column, frame_result.frame[dummy_column])
            existing_columns.add(dummy_column)

    if options.drop_original:
        working_frame = working_frame.drop(columns=[column])
        existing_columns.discard(column)

    return working_frame


def _record_dummy_transformer(
    column: str,
    frame_result: DummyFrameResult,
    runtime: DummyEncodingRuntime,
    options: DummyEncodingOptions,
    unique_total: int,
    fit_mode: str,
) -> None:
    if runtime.storage is None or not runtime.pipeline_id or not runtime.node_id:
        return

    storage = runtime.storage

    if fit_mode == "fit":
        transformer_payload = {
            "base_columns": [str(name) for name in frame_result.base_columns],
            "final_columns": [str(name) for name in frame_result.final_columns],
            "rename_map": {str(key): str(value) for key, value in frame_result.rename_map.items()},
        }
        metadata: Dict[str, Any] = {
            "dummy_columns": [str(name) for name in frame_result.final_columns],
            "base_columns": [str(name) for name in frame_result.base_columns],
            "rename_map": {str(key): str(value) for key, value in frame_result.rename_map.items()},
            "drop_first": options.drop_first,
            "include_missing": options.include_missing,
            "prefix_separator": options.prefix_separator,
            "drop_original": options.drop_original,
            "includes_missing_dummy": bool(frame_result.includes_missing_dummy),
            "category_count": int(unique_total),
            "train_rows": runtime.train_row_count,
        }

        storage.store_transformer(
            pipeline_id=runtime.pipeline_id,
            node_id=runtime.node_id,
            transformer_name="dummy_encoder",
            transformer=transformer_payload,
            column_name=column,
            metadata=metadata,
        )

        storage.record_split_activity(
            pipeline_id=runtime.pipeline_id,
            node_id=runtime.node_id,
            transformer_name="dummy_encoder",
            column_name=column,
            split_name="train",
            action="fit_transform",
            row_count=runtime.train_row_count,
        )

        for split_name in ("test", "validation"):
            rows_processed = int(runtime.split_counts.get(split_name, 0))
            storage.record_split_activity(
                pipeline_id=runtime.pipeline_id,
                node_id=runtime.node_id,
                transformer_name="dummy_encoder",
                column_name=column,
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )
    else:
        storage.record_split_activity(
            pipeline_id=runtime.pipeline_id,
            node_id=runtime.node_id,
            transformer_name="dummy_encoder",
            column_name=column,
            split_name="train",
            action="not_available",
            row_count=runtime.train_row_count,
        )

        for split_name in ("test", "validation"):
            rows_processed = int(runtime.split_counts.get(split_name, 0))
            storage.record_split_activity(
                pipeline_id=runtime.pipeline_id,
                node_id=runtime.node_id,
                transformer_name="dummy_encoder",
                column_name=column,
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )


def _process_dummy_column(
    column: str,
    working_frame: pd.DataFrame,
    config: NormalizedDummyEncodingConfig,
    signal: DummyEncodingNodeSignal,
    runtime: DummyEncodingRuntime,
    existing_columns: set[str],
) -> ColumnProcessingResult:
    series_string, unique_total, skip_reason = _validate_dummy_column(column, working_frame, config)
    if skip_reason or series_string is None or unique_total is None:
        return ColumnProcessingResult(working_frame, None, skip_reason, None)

    fit_mode, stored_payload, stored_metadata, fit_skip_reason = _determine_fit_mode(column, runtime)
    if fit_skip_reason:
        return ColumnProcessingResult(working_frame, None, fit_skip_reason, None)

    options = _resolve_effective_options(config, stored_metadata)
    signal.drop_first = options.drop_first
    signal.include_missing = options.include_missing
    signal.prefix_separator = options.prefix_separator
    signal.drop_original = options.drop_original

    frame_result, frame_error = _build_dummy_frame(
        column,
        series_string,
        options,
        existing_columns,
        runtime,
        fit_mode,
        stored_payload,
        stored_metadata,
    )
    if frame_error or frame_result is None:
        return ColumnProcessingResult(working_frame, None, frame_error, None)

    working_frame = _apply_dummy_frame(working_frame, column, frame_result, options, existing_columns)

    preview_values = series_string.dropna().value_counts().head(3).index.tolist()
    preview = ", ".join(str(value) for value in preview_values)
    detail = (
        f"{column} → {len(frame_result.final_columns)} column"
        f"{'s' if len(frame_result.final_columns) != 1 else ''}"
        + (f" ({preview})" if preview else "")
    )

    applied_signal = DummyEncodingAppliedColumnSignal(
        source_column=column,
        dummy_columns=list(frame_result.final_columns),
        replaced_original=options.drop_original,
        category_count=int(unique_total),
        includes_missing_dummy=bool(frame_result.includes_missing_dummy),
        preview_categories=[str(value) for value in preview_values],
    )

    _record_dummy_transformer(
        column=column,
        frame_result=frame_result,
        runtime=runtime,
        options=options,
        unique_total=unique_total,
        fit_mode=fit_mode,
    )

    return ColumnProcessingResult(working_frame, detail, None, applied_signal)


def _initialize_dummy_signal(
    node: Dict[str, Any],
    config: NormalizedDummyEncodingConfig,
) -> Tuple[DummyEncodingNodeSignal, Optional[str]]:
    node_id = node.get("id") if isinstance(node, dict) else None
    signal = DummyEncodingNodeSignal(node_id=str(node_id) if node_id is not None else None)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.drop_first = config.drop_first
    signal.include_missing = config.include_missing
    signal.max_categories = config.max_categories
    signal.prefix_separator = config.prefix_separator

    return signal, node_id


def _prepare_dummy_encoding(
    frame: pd.DataFrame,
    config: NormalizedDummyEncodingConfig,
    signal: DummyEncodingNodeSignal,
    pipeline_id: Optional[str],
    node_id: Optional[Any],
) -> Tuple[Optional[_DummyEncodingPreparation], Optional[str]]:
    if frame.empty:
        signal.evaluated_columns = []
        return None, "Dummy encoding: no data available"

    candidate_columns, skipped_details = _collect_dummy_candidate_columns(frame, config, signal)

    if not candidate_columns:
        signal.evaluated_columns = []
        return None, "Dummy encoding: no categorical columns selected"

    working_frame = frame.copy()
    existing_columns = set(working_frame.columns)
    runtime = _build_dummy_runtime(working_frame, pipeline_id, node_id)

    signal.evaluated_columns = list(candidate_columns)

    preparation = _DummyEncodingPreparation(
        working_frame=working_frame,
        existing_columns=existing_columns,
        runtime=runtime,
        candidate_columns=candidate_columns,
        skipped_details=skipped_details,
    )

    return preparation, None


def _execute_dummy_encoding(
    preparation: _DummyEncodingPreparation,
    config: NormalizedDummyEncodingConfig,
    signal: DummyEncodingNodeSignal,
) -> _DummyEncodingExecutionResult:
    working_frame = preparation.working_frame
    encoded_details: List[str] = []
    skipped_details = list(preparation.skipped_details)

    for column in preparation.candidate_columns:
        result = _process_dummy_column(
            column=column,
            working_frame=working_frame,
            config=config,
            signal=signal,
            runtime=preparation.runtime,
            existing_columns=preparation.existing_columns,
        )

        working_frame = result.frame

        if result.skip_reason:
            skipped_details.append(result.skip_reason)
            signal.skipped_columns.append(result.skip_reason)
            continue

        if result.detail:
            encoded_details.append(result.detail)

        if result.applied_signal:
            signal.encoded_columns.append(result.applied_signal)

    return _DummyEncodingExecutionResult(working_frame, encoded_details, skipped_details)


def _summarize_dummy_encoding(result: _DummyEncodingExecutionResult) -> str:
    if not result.encoded_details:
        summary = "Dummy encoding: no columns encoded"
    else:
        preview = ", ".join(result.encoded_details[:3])
        if len(result.encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Dummy encoding: expanded {len(result.encoded_details)} column(s) ({preview})"

    if result.skipped_details:
        preview = ", ".join(result.skipped_details[:3])
        if len(result.skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return summary


def apply_dummy_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, DummyEncodingNodeSignal]:
    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_dummy_encoding_config(config_payload)
    signal, node_id = _initialize_dummy_signal(node, config)

    preparation, early_summary = _prepare_dummy_encoding(frame, config, signal, pipeline_id, node_id)
    if early_summary is not None or preparation is None:
        return frame, early_summary or "Dummy encoding: no columns encoded", signal

    execution_result = _execute_dummy_encoding(preparation, config, signal)
    summary = _summarize_dummy_encoding(execution_result)

    return execution_result.frame, summary, signal


__all__ = [
    "DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES",
    "apply_dummy_encoding",
]

