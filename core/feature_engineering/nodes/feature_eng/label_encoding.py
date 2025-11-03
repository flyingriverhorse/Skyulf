"""Label encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
from pandas.api import types as pd_types

from core.feature_engineering.recommendations import (
    LabelEncodingSuggestion,
    build_label_encoding_suggestions,
)
from core.feature_engineering.recommendations.categorical import format_category_label
from core.feature_engineering.schemas import (
    LabelEncodingAppliedColumnSignal,
    LabelEncodingCategoryPreview,
    LabelEncodingNodeSignal,
)
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

LABEL_ENCODING_DEFAULT_SUFFIX = "_encoded"
LABEL_ENCODING_DEFAULT_MAX_UNIQUE = 50
LABEL_ENCODING_MAX_CARDINALITY_LIMIT = 1000
LABEL_ENCODING_MISSING_STRATEGIES = {"keep_na", "encode"}


@dataclass
class NormalizedLabelEncodingConfig:
    columns: List[str]
    auto_detect: bool
    drop_original: bool
    output_suffix: str
    max_unique_values: int
    missing_strategy: str
    missing_code: int
    skipped_columns: List[str]


def _normalize_label_encoding_config(config: Any) -> NormalizedLabelEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)

    raw_suffix = str(config.get("output_suffix") or LABEL_ENCODING_DEFAULT_SUFFIX).strip()
    output_suffix = raw_suffix or LABEL_ENCODING_DEFAULT_SUFFIX

    raw_max_unique = config.get("max_unique_values")
    max_unique: int
    try:
        numeric = float(raw_max_unique)
        if math.isnan(numeric):
            raise ValueError
        max_unique = int(round(numeric))
    except (TypeError, ValueError):
        max_unique = LABEL_ENCODING_DEFAULT_MAX_UNIQUE

    if max_unique < 0:
        max_unique = 0
    if max_unique > LABEL_ENCODING_MAX_CARDINALITY_LIMIT:
        max_unique = LABEL_ENCODING_MAX_CARDINALITY_LIMIT

    raw_missing_strategy = str(config.get("missing_strategy") or "keep_na").strip().lower()
    missing_strategy = (
        raw_missing_strategy if raw_missing_strategy in LABEL_ENCODING_MISSING_STRATEGIES else "keep_na"
    )

    missing_code = -1
    if missing_strategy == "encode":
        try:
            numeric = float(config.get("missing_code"))
            if math.isnan(numeric):
                raise ValueError
            missing_code = int(round(numeric))
        except (TypeError, ValueError):
            missing_code = -1

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedLabelEncodingConfig(
        columns=columns,
        auto_detect=auto_detect,
        drop_original=drop_original,
        output_suffix=output_suffix,
        max_unique_values=max_unique,
        missing_strategy=missing_strategy,
        missing_code=missing_code,
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


def _build_category_preview(categories: List[Any], limit: int = 3) -> str:
    if not categories:
        return ""
    entries: List[str] = []
    for index, value in enumerate(categories[:limit]):
        entries.append(f"{format_category_label(value)}={index}")
    if len(categories) > limit:
        entries.append("…")
    return ", ".join(entries)


def apply_label_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, LabelEncodingNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = LabelEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_label_encoding_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.output_suffix = config.output_suffix
    signal.max_unique_values = config.max_unique_values
    signal.missing_strategy = config.missing_strategy
    signal.missing_code = config.missing_code if config.missing_strategy == "encode" else None

    if frame.empty:
        return frame, "Label encoding: no data available", signal

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
        return frame, "Label encoding: no categorical columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    encoded_details: List[str] = []
    skipped_details: List[str] = []
    if skipped_notes:
        skipped_details.extend(skipped_notes)
        signal.skipped_columns.extend(skipped_notes)

    existing_columns = set(working_frame.columns)
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

        string_series = series.astype("string")
        stored_payload: Optional[Dict[str, Any]] = None
        stored_metadata: Optional[Dict[str, Any]] = None
        if storage is not None:
            stored_payload = storage.get_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="label_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="label_encoder",
                column_name=column,
            )

        fit_mode = "fit"
        if storage is not None and train_mask is not None:
            if train_row_count <= 0:
                if isinstance(stored_payload, dict):
                    fit_mode = "reuse"
                else:
                    skipped_details.append(f"{column} (no training data)")
                    signal.skipped_columns.append(f"{column} (no training data)")
                    continue

        categories: List[Any] = []
        category_labels: List[str] = []
        category_mapping: Dict[str, int] = {}
        missing_strategy = config.missing_strategy
        missing_code: Optional[int] = config.missing_code if config.missing_strategy == "encode" else None
        metadata_drop_original: Optional[bool] = None

        if fit_mode == "fit":
            analysis_series = string_series
            if storage is not None and train_mask is not None:
                analysis_series = string_series[train_mask]

            unique_total = analysis_series.nunique(dropna=False)

            if unique_total <= 1:
                skipped_details.append(f"{column} (single category)")
                signal.skipped_columns.append(f"{column} (single category)")
                continue

            if config.max_unique_values > 0 and config.auto_detect and unique_total > config.max_unique_values:
                skipped_details.append(
                    f"{column} ({unique_total} categories > {config.max_unique_values})"
                )
                signal.skipped_columns.append(
                    f"{column} ({unique_total} categories > {config.max_unique_values})"
                )
                continue

            categorical = pd.Categorical(analysis_series, ordered=False)
            categories = list(categorical.categories)
            if not categories:
                skipped_details.append(f"{column} (no categories)")
                signal.skipped_columns.append(f"{column} (no categories)")
                continue

            category_labels = [str(value) for value in categories]
            category_mapping = {label: index for index, label in enumerate(category_labels)}
        else:
            if not isinstance(stored_payload, dict):
                skipped_details.append(f"{column} (stored transformer unavailable)")
                signal.skipped_columns.append(f"{column} (stored transformer unavailable)")
                continue

            stored_mapping = stored_payload.get("mapping") if isinstance(stored_payload, dict) else None
            stored_categories = stored_payload.get("categories") if isinstance(stored_payload, dict) else None
            if not isinstance(stored_mapping, dict) or stored_categories is None:
                skipped_details.append(f"{column} (stored transformer invalid)")
                signal.skipped_columns.append(f"{column} (stored transformer invalid)")
                continue

            for raw_key, raw_value in stored_mapping.items():
                try:
                    category_mapping[str(raw_key)] = int(raw_value)
                except (TypeError, ValueError):
                    continue

            if not category_mapping:
                skipped_details.append(f"{column} (stored mapping empty)")
                signal.skipped_columns.append(f"{column} (stored mapping empty)")
                continue

            categories = list(stored_categories)
            category_labels = [str(value) for value in categories]

            stored_missing_strategy = stored_payload.get("missing_strategy")
            if isinstance(stored_missing_strategy, str) and stored_missing_strategy in LABEL_ENCODING_MISSING_STRATEGIES:
                missing_strategy = stored_missing_strategy
            else:
                missing_strategy = "keep_na"

            stored_missing_code = stored_payload.get("missing_code")
            if missing_strategy == "encode":
                try:
                    missing_code = int(stored_missing_code)
                except (TypeError, ValueError):
                    missing_code = config.missing_code
            else:
                missing_code = None

            if isinstance(stored_metadata, dict):
                raw_drop = stored_metadata.get("replaced_original")
                if raw_drop is not None:
                    metadata_drop_original = bool(raw_drop)

            logger.debug(
                "Reusing stored label encoder",
                extra={
                    "pipeline_id": pipeline_id,
                    "node_id": node_id,
                    "column": column,
                    "categories": len(category_labels),
                },
            )

        def _encode_value(value: Any) -> Any:
            if pd.isna(value):
                return missing_code if missing_strategy == "encode" else pd.NA
            lookup_key = str(value)
            code = category_mapping.get(lookup_key)
            if code is None:
                return missing_code if missing_strategy == "encode" else pd.NA
            return code

        encoded_series = string_series.apply(_encode_value).astype("Int64")

        class_count = len(category_labels)
        class_label = "class" if class_count == 1 else "classes"
        preview = _build_category_preview(categories)

        effective_drop_original = config.drop_original
        if metadata_drop_original is not None and metadata_drop_original != config.drop_original:
            effective_drop_original = metadata_drop_original
            signal.drop_original = metadata_drop_original

        if effective_drop_original:
            working_frame[column] = encoded_series
            encoded_column_name = column
            fragments = [f"{class_count} {class_label}"]
            if preview:
                fragments.append(preview)
            detail = f"{column} (replaced; {'; '.join(fragments)})"
            encoded_symbol = LabelEncodingAppliedColumnSignal(
                source_column=column,
                encoded_column=column,
                class_count=class_count,
                replaced_original=True,
                preview=[
                    LabelEncodingCategoryPreview(value=format_category_label(value), code=index)
                    for index, value in enumerate(categories[:3])
                ],
            )
        else:
            stored_encoded_name: Optional[str] = None
            if isinstance(stored_metadata, dict):
                raw_name = stored_metadata.get("encoded_column")
                if isinstance(raw_name, str) and raw_name.strip():
                    stored_encoded_name = raw_name.strip()

            target_column_name = stored_encoded_name or _resolve_encoded_column_name(
                column,
                config.output_suffix,
                existing_columns,
            )

            encoded_column_name = target_column_name
            if encoded_column_name in working_frame.columns:
                working_frame[encoded_column_name] = encoded_series
            else:
                insert_at = working_frame.columns.get_loc(column) + 1
                working_frame.insert(insert_at, encoded_column_name, encoded_series)
                existing_columns.add(encoded_column_name)

            fragments = [f"{class_count} {class_label}"]
            if preview:
                fragments.append(preview)
            detail = f"{column}→{encoded_column_name} ({'; '.join(fragments)})"
            encoded_symbol = LabelEncodingAppliedColumnSignal(
                source_column=column,
                encoded_column=encoded_column_name,
                class_count=class_count,
                replaced_original=False,
                preview=[
                    LabelEncodingCategoryPreview(value=format_category_label(value), code=index)
                    for index, value in enumerate(categories[:3])
                ],
            )

        encoded_details.append(detail)
        signal.encoded_columns.append(encoded_symbol)

        if storage is not None and fit_mode == "fit":
            mapping_records = [
                {
                    "category": category_labels[index],
                    "code": index,
                    "display": format_category_label(value),
                }
                for index, value in enumerate(categories)
            ]

            transformer_payload = {
                "mapping": category_mapping,
                "categories": category_labels,
                "missing_strategy": missing_strategy,
                "missing_code": missing_code,
            }

            metadata: Dict[str, Any] = {
                "encoded_column": encoded_column_name,
                "replaced_original": bool(effective_drop_original),
                "category_count": class_count,
                "missing_strategy": missing_strategy,
                "missing_code": missing_code,
                "unknown_policy": (
                    "encode_missing_code" if missing_strategy == "encode" else "preserve_na"
                ),
                "mapping": mapping_records,
            }
            if train_mask is not None:
                metadata["train_rows"] = train_row_count

            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="label_encoder",
                transformer=transformer_payload,
                column_name=column,
                metadata=metadata,
            )

            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="label_encoder",
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
                    transformer_name="label_encoder",
                    column_name=column,
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )
        elif storage is not None and fit_mode == "reuse":
            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="label_encoder",
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
                    transformer_name="label_encoder",
                    column_name=column,
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )

    if not encoded_details:
        summary = "Label encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Label encoding: encoded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "LABEL_ENCODING_DEFAULT_SUFFIX",
    "LABEL_ENCODING_DEFAULT_MAX_UNIQUE",
    "LabelEncodingSuggestion",
    "build_label_encoding_suggestions",
    "apply_label_encoding",
]
