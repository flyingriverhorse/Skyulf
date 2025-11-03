"""Target encoding helper utilities for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
from pandas.api import types as pd_types

from core.feature_engineering.recommendations.categorical import format_category_label
from core.feature_engineering.schemas import (
    TargetEncodingAppliedColumnSignal,
    TargetEncodingCategoryPreview,
    TargetEncodingNodeSignal,
)
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

TARGET_ENCODING_DEFAULT_SUFFIX = "_target"
TARGET_ENCODING_DEFAULT_MAX_CATEGORIES = 50
TARGET_ENCODING_MAX_CARDINALITY_LIMIT = 1000
TARGET_ENCODING_DEFAULT_SMOOTHING = 20.0
TARGET_ENCODING_MAX_SMOOTHING = 1_000_000.0
TARGET_ENCODING_HANDLE_UNKNOWN_VALUES = {"global_mean", "error"}
TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN = "global_mean"


@dataclass
class NormalizedTargetEncodingConfig:
    columns: List[str]
    auto_detect: bool
    max_categories: int
    output_suffix: str
    drop_original: bool
    target_column: str
    smoothing: float
    encode_missing: bool
    handle_unknown: str
    skipped_columns: List[str]


def _normalize_target_encoding_config(config: Any) -> NormalizedTargetEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)
    encode_missing = _coerce_config_boolean(config.get("encode_missing"), default=False)

    raw_suffix = str(config.get("output_suffix") or TARGET_ENCODING_DEFAULT_SUFFIX).strip()
    output_suffix = raw_suffix or TARGET_ENCODING_DEFAULT_SUFFIX

    raw_target_column = str(config.get("target_column") or "").strip()
    target_column = raw_target_column

    raw_max_categories = config.get("max_categories")
    try:
        numeric = float(raw_max_categories)
        if math.isnan(numeric):
            raise ValueError
        max_categories = int(round(numeric))
    except (TypeError, ValueError):
        max_categories = TARGET_ENCODING_DEFAULT_MAX_CATEGORIES

    if max_categories < 0:
        max_categories = 0
    if max_categories > TARGET_ENCODING_MAX_CARDINALITY_LIMIT:
        max_categories = TARGET_ENCODING_MAX_CARDINALITY_LIMIT

    raw_smoothing = config.get("smoothing", TARGET_ENCODING_DEFAULT_SMOOTHING)
    try:
        smoothing_value = float(raw_smoothing)
        if math.isnan(smoothing_value):
            raise ValueError
    except (TypeError, ValueError):
        smoothing_value = TARGET_ENCODING_DEFAULT_SMOOTHING

    if smoothing_value < 0.0:
        smoothing_value = 0.0
    if smoothing_value > TARGET_ENCODING_MAX_SMOOTHING:
        smoothing_value = TARGET_ENCODING_MAX_SMOOTHING

    raw_handle_unknown = str(config.get("handle_unknown") or TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN).strip().lower()
    handle_unknown = (
        raw_handle_unknown if raw_handle_unknown in TARGET_ENCODING_HANDLE_UNKNOWN_VALUES else TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN
    )

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedTargetEncodingConfig(
        columns=columns,
        auto_detect=auto_detect,
        max_categories=max_categories,
        output_suffix=output_suffix,
        drop_original=drop_original,
        target_column=target_column,
        smoothing=smoothing_value,
        encode_missing=encode_missing,
        handle_unknown=handle_unknown,
        skipped_columns=skipped_columns,
    )


def _resolve_encoded_column_name(base: str, suffix: str, existing: set[str]) -> str:
    candidate = f"{base}{suffix}"
    if candidate not in existing:
        return candidate
    counter = 2
    while True:
        numbered = f"{candidate}_{counter}"
        if numbered not in existing:
            return numbered
        counter += 1


def apply_target_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, TargetEncodingNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = TargetEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_target_encoding_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.target_column = config.target_column or None
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.output_suffix = config.output_suffix
    signal.max_categories = config.max_categories
    signal.smoothing = config.smoothing
    signal.encode_missing = config.encode_missing
    signal.handle_unknown = config.handle_unknown

    if frame.empty:
        return frame, "Target encoding: no data available", signal

    if not config.target_column:
        return frame, "Target encoding: target column not specified", signal

    if config.target_column not in frame.columns:
        signal.skipped_columns.append(f"target column '{config.target_column}' missing")
        return frame, f"Target encoding: target column '{config.target_column}' missing", signal

    target_series = pd.to_numeric(frame[config.target_column], errors="coerce")
    if target_series.notna().sum() == 0:
        signal.skipped_columns.append("target column non-numeric")
        return frame, f"Target encoding: target column '{config.target_column}' must be numeric", signal

    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None

    split_counts: Dict[str, int] = {}
    train_mask: Optional[pd.Series] = None
    train_row_count = 0
    if has_splits:
        split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
        train_row_count = int(split_counts.get("train", 0))

    target_for_global = target_series
    if storage is not None and train_mask is not None and train_mask.any():
        target_for_global = target_series[train_mask]

    global_mean_value: Optional[float] = None
    if target_for_global.notna().sum() > 0:
        global_mean_value = float(target_for_global.mean(skipna=True))

    if global_mean_value is None or math.isnan(global_mean_value):
        fallback_mean = float(target_series.mean(skipna=True))
        if math.isnan(fallback_mean):
            signal.skipped_columns.append("unable to compute global mean")
            return frame, f"Target encoding: unable to compute global mean for '{config.target_column}'", signal
        global_mean_value = fallback_mean

    signal.global_mean = global_mean_value

    def _sanitize_mapping_value(raw: Any) -> Optional[float]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        return numeric

    skipped_configured = set(config.skipped_columns)
    candidate_columns: List[str] = []
    seen: set[str] = set()
    skipped_notes: List[str] = []

    for column in config.columns:
        normalized = str(column or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in skipped_configured:
            skipped_notes.append(f"{normalized} (skipped)")
            continue
        candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured:
                skipped_notes.append(f"{column} (skipped)")
                continue
            candidate_columns.append(column)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Target encoding: no categorical columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    encoded_details: List[str] = []
    skipped_details: List[str] = []
    if skipped_notes:
        skipped_details.extend(skipped_notes)
        signal.skipped_columns.extend(skipped_notes)

    existing_columns = set(working_frame.columns)

    for column in candidate_columns:
        if column == config.target_column:
            skipped_details.append(f"{column} (target column)")
            signal.skipped_columns.append(f"{column} (target column)")
            continue

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

        string_series = series.astype("string")
        original_missing_mask = string_series.isna()

        unique_total = string_series.nunique(dropna=True)
        if unique_total <= 1:
            skipped_details.append(f"{column} (single category)")
            signal.skipped_columns.append(f"{column} (single category)")
            continue

        if config.auto_detect and config.max_categories > 0 and unique_total > config.max_categories:
            skipped_details.append(f"{column} ({unique_total} categories > {config.max_categories})")
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
                transformer_name="target_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="target_encoder",
                column_name=column,
            )

        fit_mode = "fit"
        if storage is not None and train_row_count <= 0:
            if isinstance(stored_payload, dict):
                fit_mode = "reuse"
            else:
                skipped_details.append(f"{column} (no training data)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

        effective_drop_original = config.drop_original
        effective_encode_missing = config.encode_missing
        effective_handle_unknown = config.handle_unknown
        effective_output_suffix = config.output_suffix
        effective_smoothing = config.smoothing
        effective_global_mean = global_mean_value
        placeholder_token = "__MISSING__"
        stored_encoded_column: Optional[str] = None

        mapping_values: Dict[str, Optional[float]]
        unknown_rows: int
        category_count: int
        fragments: List[str]
        preview_mapping: List[TargetEncodingCategoryPreview]

        if fit_mode == "reuse":
            if not isinstance(stored_payload, dict):
                skipped_details.append(f"{column} (stored transformer missing)")
                signal.skipped_columns.append(f"{column} (stored transformer missing)")
                continue

            stored_mapping_payload = stored_payload.get("mapping")
            if not isinstance(stored_mapping_payload, dict) or not stored_mapping_payload:
                skipped_details.append(f"{column} (stored mapping empty)")
                signal.skipped_columns.append(f"{column} (stored mapping empty)")
                continue

            if isinstance(stored_metadata, dict):
                raw_drop_original = stored_metadata.get("drop_original")
                if raw_drop_original is not None:
                    effective_drop_original = bool(raw_drop_original)
                    signal.drop_original = effective_drop_original
                raw_encode_missing = stored_metadata.get("encode_missing")
                if raw_encode_missing is not None:
                    effective_encode_missing = bool(raw_encode_missing)
                    signal.encode_missing = effective_encode_missing
                raw_handle_unknown = stored_metadata.get("handle_unknown")
                if isinstance(raw_handle_unknown, str) and raw_handle_unknown.strip():
                    effective_handle_unknown = raw_handle_unknown.strip()
                    signal.handle_unknown = effective_handle_unknown
                raw_output_suffix = stored_metadata.get("output_suffix")
                if isinstance(raw_output_suffix, str) and raw_output_suffix.strip():
                    effective_output_suffix = raw_output_suffix.strip()
                    signal.output_suffix = effective_output_suffix
                raw_smoothing = stored_metadata.get("smoothing")
                if raw_smoothing is not None:
                    try:
                        effective_smoothing = float(raw_smoothing)
                        signal.smoothing = effective_smoothing
                    except (TypeError, ValueError):
                        pass
                raw_placeholder = stored_metadata.get("placeholder")
                if isinstance(raw_placeholder, str) and raw_placeholder:
                    placeholder_token = raw_placeholder
                raw_global_mean = stored_metadata.get("global_mean")
                if raw_global_mean is not None:
                    try:
                        effective_global_mean = float(raw_global_mean)
                        signal.global_mean = effective_global_mean
                    except (TypeError, ValueError):
                        pass
                encoded_column_meta = stored_metadata.get("encoded_column")
                if isinstance(encoded_column_meta, str) and encoded_column_meta.strip():
                    stored_encoded_column = encoded_column_meta.strip()

            mapping_values = {
                str(key): _sanitize_mapping_value(value)
                for key, value in stored_mapping_payload.items()
            }

            if effective_encode_missing and placeholder_token not in mapping_values:
                mapping_values[placeholder_token] = effective_global_mean

            mapping_series = string_series.fillna(placeholder_token) if effective_encode_missing else string_series
            mapped_series = mapping_series.map(mapping_values)
            if mapped_series is None:
                skipped_details.append(f"{column} (failed to map using stored transformer)")
                signal.skipped_columns.append(f"{column} (failed to map using stored transformer)")
                continue

            unknown_mask = (~original_missing_mask) & mapped_series.isna()
            unknown_rows = int(unknown_mask.sum())

            if effective_encode_missing:
                mapped_series = mapped_series.where(~original_missing_mask, effective_global_mean)

            if effective_handle_unknown == "global_mean":
                mapped_series = mapped_series.where(~unknown_mask, effective_global_mean)

            encoded_series = pd.to_numeric(mapped_series, errors="coerce")
            encoded_series = encoded_series.astype("float64")
            if not effective_encode_missing:
                encoded_series = encoded_series.where(~original_missing_mask, math.nan)

            if isinstance(stored_metadata, dict) and stored_metadata.get("category_count") is not None:
                try:
                    category_count = int(stored_metadata.get("category_count", 0))
                except (TypeError, ValueError):
                    category_count = len([key for key in mapping_values if key != placeholder_token])
            else:
                category_count = len([key for key in mapping_values if key != placeholder_token])

            fragments = [
                f"{category_count} level{'s' if category_count != 1 else ''}",
                f"global_mean={effective_global_mean:.4f}",
            ]
            if effective_smoothing > 0.0:
                fragments.append(f"smoothing={effective_smoothing:g}")
            if effective_encode_missing:
                fragments.append("missing→global_mean")
            if effective_handle_unknown == "global_mean":
                fragments.append("unknown→global_mean")
            elif unknown_rows > 0:
                fragments.append(f"{unknown_rows} unknown row{'s' if unknown_rows != 1 else ''}")

            preview_mapping = []
            for raw_category, encoded_value in list(mapping_values.items())[:3]:
                label = "<MISSING>" if raw_category == placeholder_token else format_category_label(raw_category)
                preview_mapping.append(
                    TargetEncodingCategoryPreview(
                        category=label,
                        encoded_value=_sanitize_mapping_value(encoded_value),
                    )
                )

        else:
            mapping_series = string_series.fillna(placeholder_token) if effective_encode_missing else string_series
            numeric_target = target_series

            fit_mask = numeric_target.notna()
            if not effective_encode_missing:
                fit_mask &= mapping_series.notna()
            if storage is not None and train_mask is not None and train_mask.any():
                fit_mask &= train_mask

            if not fit_mask.any():
                skipped_details.append(f"{column} (no rows with numeric target)")
                signal.skipped_columns.append(f"{column} (no rows with numeric target)")
                continue

            fit_categories = mapping_series[fit_mask]
            fit_targets = numeric_target[fit_mask]

            fit_frame = (
                pd.DataFrame({"category": fit_categories, "target": fit_targets})
                .dropna(subset=["category", "target"])
            )

            if fit_frame.empty:
                skipped_details.append(f"{column} (no rows with numeric target)")
                signal.skipped_columns.append(f"{column} (no rows with numeric target)")
                continue

            grouped = fit_frame.groupby("category")["target"].agg(["mean", "count"])
            if grouped.empty or grouped.shape[0] <= 1:
                skipped_details.append(f"{column} (insufficient category diversity)")
                signal.skipped_columns.append(f"{column} (insufficient category diversity)")
                continue

            means = grouped["mean"]
            counts = grouped["count"]

            if effective_smoothing > 0.0:
                smoothed = (means * counts + effective_smoothing * effective_global_mean) / (
                    counts + effective_smoothing
                )
            else:
                smoothed = means

            mapping_values = {
                str(category): _sanitize_mapping_value(value)
                for category, value in smoothed.to_dict().items()
            }

            if effective_encode_missing and placeholder_token not in mapping_values:
                mapping_values[placeholder_token] = effective_global_mean

            mapped_series = mapping_series.map(mapping_values)
            if mapped_series is None:
                skipped_details.append(f"{column} (failed to map categories)")
                signal.skipped_columns.append(f"{column} (failed to map categories)")
                continue

            unknown_mask = (~original_missing_mask) & mapped_series.isna()
            unknown_rows = int(unknown_mask.sum())

            if effective_encode_missing:
                mapped_series = mapped_series.where(~original_missing_mask, effective_global_mean)

            if effective_handle_unknown == "global_mean":
                mapped_series = mapped_series.where(~unknown_mask, effective_global_mean)

            encoded_series = pd.to_numeric(mapped_series, errors="coerce")
            encoded_series = encoded_series.astype("float64")
            if not effective_encode_missing:
                encoded_series = encoded_series.where(~original_missing_mask, math.nan)

            category_count = grouped.shape[0]
            fragments = [
                f"{category_count} level{'s' if category_count != 1 else ''}",
                f"global_mean={effective_global_mean:.4f}",
            ]
            if effective_smoothing > 0.0:
                fragments.append(f"smoothing={effective_smoothing:g}")
            if effective_encode_missing:
                fragments.append("missing→global_mean")
            if effective_handle_unknown == "global_mean":
                fragments.append("unknown→global_mean")
            elif unknown_rows > 0:
                fragments.append(f"{unknown_rows} unknown row{'s' if unknown_rows != 1 else ''}")

            preview_mapping = []
            for category_value, encoded_value in list(mapping_values.items())[:3]:
                label = "<MISSING>" if category_value == placeholder_token else format_category_label(category_value)
                preview_mapping.append(
                    TargetEncodingCategoryPreview(
                        category=label,
                        encoded_value=_sanitize_mapping_value(encoded_value),
                    )
                )

        if effective_drop_original:
            working_frame[column] = encoded_series
            existing_columns.add(column)
            encoded_column_name = column
            detail = f"{column} (replaced; {'; '.join(fragments)})"
            replaced_original = True
        else:
            encoded_column_name = stored_encoded_column or _resolve_encoded_column_name(
                column,
                effective_output_suffix,
                existing_columns,
            )
            insert_at = working_frame.columns.get_loc(column) + 1
            if encoded_column_name in working_frame.columns:
                working_frame[encoded_column_name] = encoded_series
            else:
                working_frame.insert(insert_at, encoded_column_name, encoded_series)
            existing_columns.add(encoded_column_name)
            detail = f"{column}→{encoded_column_name} ({'; '.join(fragments)})"
            replaced_original = False

        encoded_signal = TargetEncodingAppliedColumnSignal(
            source_column=column,
            encoded_column=encoded_column_name,
            replaced_original=replaced_original,
            category_count=int(category_count),
            global_mean=float(effective_global_mean),
            smoothing=float(effective_smoothing),
            encode_missing=bool(effective_encode_missing),
            handle_unknown=effective_handle_unknown,
            unknown_rows=int(unknown_rows),
            preview=preview_mapping,
        )
        encoded_details.append(detail)
        signal.encoded_columns.append(encoded_signal)

        if storage is not None:
            if fit_mode == "fit":
                transformer_payload = {
                    "mapping": {
                        key: (_sanitize_mapping_value(value) if value is not None else None)
                        for key, value in mapping_values.items()
                    },
                    "placeholder": placeholder_token if effective_encode_missing else None,
                    "global_mean": float(effective_global_mean),
                }
                # Build human-readable method label for transformer audit UI
                method_parts = ["Target Encoding (smoothed mean"]
                if effective_smoothing > 0.0:
                    method_parts.append(f", smoothing={effective_smoothing:g}")
                method_parts.append(")")
                method_label = "".join(method_parts)

                metadata: Dict[str, Any] = {
                    "encoded_column": encoded_column_name,
                    "replaced_original": replaced_original,
                    "category_count": int(category_count),
                    "global_mean": float(effective_global_mean),
                    "smoothing": float(effective_smoothing),
                    "encode_missing": bool(effective_encode_missing),
                    "handle_unknown": effective_handle_unknown,
                    "drop_original": bool(effective_drop_original),
                    "output_suffix": effective_output_suffix,
                    "placeholder": placeholder_token if effective_encode_missing else None,
                    "target_column": config.target_column,
                    "train_rows": train_row_count,
                    "method_label": method_label,
                }

                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="target_encoder",
                    transformer=transformer_payload,
                    column_name=column,
                    metadata=metadata,
                )

                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="target_encoder",
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
                        transformer_name="target_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )
            else:
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="target_encoder",
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
                        transformer_name="target_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )

    if not encoded_details:
        summary = "Target encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Target encoding: encoded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "TARGET_ENCODING_DEFAULT_SUFFIX",
    "TARGET_ENCODING_DEFAULT_MAX_CATEGORIES",
    "TARGET_ENCODING_MAX_CARDINALITY_LIMIT",
    "TARGET_ENCODING_DEFAULT_SMOOTHING",
    "TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN",
    "apply_target_encoding",
]
