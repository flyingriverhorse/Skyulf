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
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

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


def apply_dummy_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, DummyEncodingNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = DummyEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_dummy_encoding_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.drop_first = config.drop_first
    signal.include_missing = config.include_missing
    signal.max_categories = config.max_categories
    signal.prefix_separator = config.prefix_separator

    if frame.empty:
        signal.evaluated_columns = []
        return frame, "Dummy encoding: no data available", signal

    skipped_configured = set(config.skipped_columns)
    candidate_columns: List[str] = []
    seen: set[str] = set()
    skipped_details: List[str] = []

    for column in config.columns:
        normalized = str(column or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in skipped_configured:
            skipped_details.append(f"{normalized} (skipped)")
            signal.skipped_columns.append(f"{normalized} (skipped)")
            continue
        candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured:
                skipped_details.append(f"{column} (skipped)")
                signal.skipped_columns.append(f"{column} (skipped)")
                continue
            candidate_columns.append(column)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Dummy encoding: no categorical columns selected", signal

    working_frame = frame.copy()
    encoded_details: List[str] = []

    existing_columns = set(working_frame.columns)

    signal.evaluated_columns = list(candidate_columns)

    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None

    split_counts: Dict[str, int] = {}
    train_mask: Optional[pd.Series] = None
    train_row_count = 0
    if has_splits:
        split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_row_count = int(split_counts.get("train", 0))
        train_mask = working_frame[SPLIT_TYPE_COLUMN] == "train"

    for column in candidate_columns:
        if column not in working_frame.columns:
            skipped_details.append(f"{column} (missing)")
            signal.skipped_columns.append(f"{column} (missing)")
            continue

        series = working_frame[column]
        if pd_types.is_bool_dtype(series):
            skipped_details.append(f"{column} (boolean column)")
            signal.skipped_columns.append(f"{column} (boolean column)")
            continue

        if not (
            pd_types.is_object_dtype(series)
            or pd_types.is_string_dtype(series)
            or pd_types.is_categorical_dtype(series)
        ):
            skipped_details.append(f"{column} (non-categorical dtype)")
            signal.skipped_columns.append(f"{column} (non-categorical dtype)")
            continue

        series_string = series.astype("string")
        unique_total = series_string.nunique(dropna=True)

        if unique_total <= 1:
            skipped_details.append(f"{column} (single category)")
            signal.skipped_columns.append(f"{column} (single category)")
            continue

        if config.max_categories and unique_total > config.max_categories:
            skipped_details.append(
                f"{column} ({unique_total} categories > {config.max_categories})"
            )
            signal.skipped_columns.append(
                f"{column} ({unique_total} categories > {config.max_categories})"
            )
            continue

        stored_payload: Optional[Dict[str, Any]] = None
        stored_metadata: Optional[Dict[str, Any]] = None
        if storage is not None:
            stored_payload = storage.get_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="dummy_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="dummy_encoder",
                column_name=column,
            )

        fit_mode = "fit"
        if storage is not None and pipeline_id and train_row_count <= 0:
            if isinstance(stored_payload, dict):
                fit_mode = "reuse"
            else:
                skipped_details.append(f"{column} (no training data)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

        effective_drop_first = config.drop_first
        effective_include_missing = config.include_missing
        effective_prefix_separator = config.prefix_separator
        effective_drop_original = config.drop_original
        includes_missing_dummy = bool(config.include_missing and series.isna().any())
        rename_map: Dict[str, str] = {}
        final_columns: List[str] = []
        base_columns: List[str] = []

        if fit_mode == "reuse" and isinstance(stored_metadata, dict):
            stored_drop_first = stored_metadata.get("drop_first")
            if stored_drop_first is not None:
                effective_drop_first = bool(stored_drop_first)
                signal.drop_first = effective_drop_first
            stored_include_missing = stored_metadata.get("include_missing")
            if stored_include_missing is not None:
                effective_include_missing = bool(stored_include_missing)
                signal.include_missing = effective_include_missing
            stored_prefix = stored_metadata.get("prefix_separator")
            if isinstance(stored_prefix, str):
                effective_prefix_separator = stored_prefix
                signal.prefix_separator = stored_prefix
            stored_drop_original = stored_metadata.get("drop_original")
            if stored_drop_original is not None:
                effective_drop_original = bool(stored_drop_original)
                signal.drop_original = effective_drop_original
            includes_missing_dummy = bool(stored_metadata.get("includes_missing_dummy", False))

        full_dummy_frame: Optional[pd.DataFrame] = None

        if fit_mode == "fit":
            if storage is not None and train_mask is not None and not train_mask.any():
                skipped_details.append(f"{column} (no training data)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

            train_series = series_string
            if storage is not None and train_mask is not None:
                train_series = series_string[train_mask]

            try:
                train_dummy_frame = pd.get_dummies(
                    train_series,
                    prefix=column,
                    prefix_sep=effective_prefix_separator,
                    dummy_na=effective_include_missing,
                    drop_first=effective_drop_first,
                    dtype="UInt8",
                )
            except Exception:
                skipped_details.append(f"{column} (failed to encode)")
                signal.skipped_columns.append(f"{column} (failed to encode)")
                continue

            if train_dummy_frame.empty:
                skipped_details.append(f"{column} (no dummy columns generated)")
                signal.skipped_columns.append(f"{column} (no dummy columns generated)")
                continue

            base_columns = list(train_dummy_frame.columns)

            try:
                full_dummy_frame = pd.get_dummies(
                    series_string,
                    prefix=column,
                    prefix_sep=effective_prefix_separator,
                    dummy_na=effective_include_missing,
                    drop_first=effective_drop_first,
                    dtype="UInt8",
                )
            except Exception:
                skipped_details.append(f"{column} (failed to encode)")
                signal.skipped_columns.append(f"{column} (failed to encode)")
                continue

            full_dummy_frame = full_dummy_frame.reindex(columns=base_columns, fill_value=0).astype("UInt8")

            renamed_columns: List[str] = []
            rename_map = {}
            for base_name in base_columns:
                target_name = _resolve_unique_column_name(base_name, existing_columns)
                rename_map[base_name] = target_name
                existing_columns.add(target_name)
                renamed_columns.append(target_name)

            full_dummy_frame = full_dummy_frame.rename(columns=rename_map).astype("UInt8")
            final_columns = list(full_dummy_frame.columns)

        else:  # reuse
            if not isinstance(stored_payload, dict):
                skipped_details.append(f"{column} (stored transformer invalid)")
                signal.skipped_columns.append(f"{column} (stored transformer invalid)")
                continue

            base_columns = [str(name) for name in stored_payload.get("base_columns", []) if str(name).strip()]
            stored_rename_map = stored_payload.get("rename_map", {})
            if isinstance(stored_rename_map, dict):
                rename_map = {str(key): str(value) for key, value in stored_rename_map.items()}
            else:
                rename_map = {name: name for name in base_columns}

            final_columns = [str(name) for name in stored_payload.get("final_columns", []) if str(name).strip()]
            if not base_columns or not final_columns:
                skipped_details.append(f"{column} (stored transformer incomplete)")
                signal.skipped_columns.append(f"{column} (stored transformer incomplete)")
                continue

            try:
                dummy_frame_current = pd.get_dummies(
                    series_string,
                    prefix=column,
                    prefix_sep=effective_prefix_separator,
                    dummy_na=effective_include_missing,
                    drop_first=effective_drop_first,
                    dtype="UInt8",
                )
            except Exception:
                skipped_details.append(f"{column} (failed to encode)")
                signal.skipped_columns.append(f"{column} (failed to encode)")
                continue

            dummy_frame_current = dummy_frame_current.reindex(columns=base_columns, fill_value=0).astype("UInt8")
            dummy_frame_current = dummy_frame_current.rename(columns=rename_map)
            dummy_frame_current = dummy_frame_current.reindex(columns=final_columns, fill_value=0).astype("UInt8")

            full_dummy_frame = dummy_frame_current

        if full_dummy_frame is None or full_dummy_frame.empty:
            skipped_details.append(f"{column} (no dummy columns generated)")
            signal.skipped_columns.append(f"{column} (no dummy columns generated)")
            continue

        insert_at = working_frame.columns.get_loc(column) + (0 if effective_drop_original else 1)
        for offset, dummy_column in enumerate(final_columns):
            if dummy_column in working_frame.columns:
                working_frame[dummy_column] = full_dummy_frame[dummy_column]
            else:
                working_frame.insert(insert_at + offset, dummy_column, full_dummy_frame[dummy_column])
                existing_columns.add(dummy_column)

        if effective_drop_original:
            working_frame = working_frame.drop(columns=[column])
            existing_columns.discard(column)

        preview_values = series_string.dropna().value_counts().head(3).index.tolist()
        preview = ", ".join(str(value) for value in preview_values)
        encoded_details.append(
            f"{column} → {len(final_columns)} column{'s' if len(final_columns) != 1 else ''}"
            + (f" ({preview})" if preview else "")
        )

        signal.encoded_columns.append(
            DummyEncodingAppliedColumnSignal(
                source_column=column,
                dummy_columns=list(final_columns),
                replaced_original=effective_drop_original,
                category_count=int(unique_total),
                includes_missing_dummy=bool(includes_missing_dummy),
                preview_categories=[str(value) for value in preview_values],
            )
        )

        if storage is not None:
            if fit_mode == "fit":
                transformer_payload = {
                    "base_columns": [str(name) for name in base_columns],
                    "final_columns": [str(name) for name in final_columns],
                    "rename_map": {str(key): str(value) for key, value in rename_map.items()},
                }
                metadata: Dict[str, Any] = {
                    "dummy_columns": [str(name) for name in final_columns],
                    "base_columns": [str(name) for name in base_columns],
                    "rename_map": {str(key): str(value) for key, value in rename_map.items()},
                    "drop_first": effective_drop_first,
                    "include_missing": effective_include_missing,
                    "prefix_separator": effective_prefix_separator,
                    "drop_original": effective_drop_original,
                    "includes_missing_dummy": bool(includes_missing_dummy),
                    "category_count": int(unique_total),
                    "train_rows": train_row_count,
                }

                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="dummy_encoder",
                    transformer=transformer_payload,
                    column_name=column,
                    metadata=metadata,
                )

                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="dummy_encoder",
                    column_name=column,
                    split_name="train",
                    action="fit_transform",
                    row_count=train_row_count,
                )

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="dummy_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )
            else:
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="dummy_encoder",
                    column_name=column,
                    split_name="train",
                    action="not_available",
                    row_count=train_row_count,
                )

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="dummy_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )

    if not encoded_details:
        summary = "Dummy encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Dummy encoding: expanded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "DUMMY_ENCODING_DEFAULT_MAX_CATEGORIES",
    "apply_dummy_encoding",
]
