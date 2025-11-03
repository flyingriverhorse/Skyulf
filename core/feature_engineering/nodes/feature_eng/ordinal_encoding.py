"""Ordinal encoding helper utilities for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
from pandas.api import types as pd_types

try:  # pragma: no cover - optional dependency
    from sklearn.preprocessing import OrdinalEncoder
except Exception:  # pragma: no cover - scikit-learn missing or incompatible
    OrdinalEncoder = None  # type: ignore

from core.feature_engineering.recommendations.categorical import format_category_label
from core.feature_engineering.schemas import (
    OrdinalEncodingAppliedColumnSignal,
    OrdinalEncodingCategoryPreview,
    OrdinalEncodingNodeSignal,
)
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

ORDINAL_ENCODING_DEFAULT_SUFFIX = "_ordinal"
ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES = 50
ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT = 1000
ORDINAL_ENCODING_HANDLE_UNKNOWN_VALUES = {"error", "use_encoded_value"}
ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN = "use_encoded_value"
ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE = -1


@dataclass
class NormalizedOrdinalEncodingConfig:
    columns: List[str]
    auto_detect: bool
    max_categories: int
    output_suffix: str
    drop_original: bool
    encode_missing: bool
    handle_unknown: str
    unknown_value: int
    skipped_columns: List[str]


def _normalize_ordinal_encoding_config(config: Any) -> NormalizedOrdinalEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)
    encode_missing = _coerce_config_boolean(config.get("encode_missing"), default=False)

    raw_suffix = str(config.get("output_suffix") or ORDINAL_ENCODING_DEFAULT_SUFFIX).strip()
    output_suffix = raw_suffix or ORDINAL_ENCODING_DEFAULT_SUFFIX

    raw_max_categories = config.get("max_categories")
    try:
        numeric = float(raw_max_categories)
        if math.isnan(numeric):
            raise ValueError
        max_categories = int(round(numeric))
    except (TypeError, ValueError):
        max_categories = ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES

    if max_categories < 0:
        max_categories = 0
    if max_categories > ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT:
        max_categories = ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT

    raw_handle_unknown = str(config.get("handle_unknown") or ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN).strip().lower()
    handle_unknown = (
        raw_handle_unknown if raw_handle_unknown in ORDINAL_ENCODING_HANDLE_UNKNOWN_VALUES else ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN
    )

    raw_unknown_value = config.get("unknown_value", ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE)
    try:
        numeric = float(raw_unknown_value)
        if math.isnan(numeric):
            raise ValueError
        unknown_value = int(round(numeric))
    except (TypeError, ValueError):
        unknown_value = ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE

    if unknown_value < -1_000_000:
        unknown_value = -1_000_000
    if unknown_value > 1_000_000:
        unknown_value = 1_000_000

    if handle_unknown != "use_encoded_value":
        unknown_value = ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedOrdinalEncodingConfig(
        columns=columns,
        auto_detect=auto_detect,
        max_categories=max_categories,
        output_suffix=output_suffix,
        drop_original=drop_original,
        encode_missing=encode_missing,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value,
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


def _transform_with_mapping(
    series: pd.Series,
    mapping: Dict[str, int],
    *,
    encode_missing: bool,
    handle_unknown: str,
    unknown_value: int,
) -> Tuple[pd.Series, int]:
    """Apply an ordinal mapping to a pandas Series, handling missing and unknown values."""

    missing_mask = series.isna()
    mapped = series.map(mapping)
    encoded = pd.Series(mapped, index=series.index, dtype="Float64")

    if encode_missing:
        encoded = encoded.where(~missing_mask, float(unknown_value))
    else:
        encoded = encoded.where(~missing_mask, pd.NA)

    unknown_mask = (~missing_mask) & encoded.isna()
    unknown_rows = int(unknown_mask.sum())

    if unknown_rows > 0:
        if handle_unknown == "use_encoded_value":
            encoded = encoded.where(~unknown_mask, float(unknown_value))
        else:
            encoded = encoded.where(~unknown_mask, pd.NA)

    encoded = encoded.astype("Int64")
    return encoded, unknown_rows


def apply_ordinal_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, OrdinalEncodingNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = OrdinalEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    if frame.empty:
        return frame, "Ordinal encoding: no data available", signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_ordinal_encoding_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.max_categories = config.max_categories
    signal.output_suffix = config.output_suffix
    signal.drop_original = config.drop_original
    signal.encode_missing = config.encode_missing
    signal.handle_unknown = config.handle_unknown
    signal.unknown_value = config.unknown_value if config.handle_unknown == "use_encoded_value" or config.encode_missing else None

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
            signal.skipped_columns.append(f"{normalized} (skipped)")
            continue
        candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured:
                skipped_notes.append(f"{column} (skipped)")
                signal.skipped_columns.append(f"{column} (skipped)")
                continue
            candidate_columns.append(column)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Ordinal encoding: no categorical columns selected", signal

    working_frame = frame.copy()
    encoded_details: List[str] = []
    skipped_details: List[str] = []
    if skipped_notes:
        skipped_details.extend(skipped_notes)
        signal.skipped_columns.extend(note for note in skipped_notes if note not in signal.skipped_columns)

    existing_columns = set(working_frame.columns)

    signal.evaluated_columns = list(candidate_columns)

    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None

    split_counts: Dict[str, int] = {}
    train_mask: Optional[pd.Series] = None
    train_row_count = 0
    if has_splits:
        split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_mask = working_frame[SPLIT_TYPE_COLUMN] == "train"
        train_row_count = int(split_counts.get("train", 0))

    def _should_use_unknown_value(encode_missing: bool, handle_unknown: str) -> bool:
        return encode_missing or handle_unknown == "use_encoded_value"

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
        unique_total = string_series.nunique(dropna=True)

        if unique_total <= 1:
            skipped_details.append(f"{column} (single category)")
            signal.skipped_columns.append(f"{column} (single category)")
            continue

        if config.auto_detect and config.max_categories > 0 and unique_total > config.max_categories:
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
                transformer_name="ordinal_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="ordinal_encoder",
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
        effective_unknown_value = config.unknown_value
        stored_encoded_column: Optional[str] = None

        mapping_values: Dict[str, int] = {}
        categories: List[str] = []
        unknown_rows = 0

        if fit_mode == "reuse":
            if not isinstance(stored_payload, dict):
                skipped_details.append(f"{column} (stored transformer missing)")
                signal.skipped_columns.append(f"{column} (stored transformer missing)")
                continue

            raw_mapping = stored_payload.get("mapping")
            if not isinstance(raw_mapping, dict) or not raw_mapping:
                skipped_details.append(f"{column} (stored mapping empty)")
                signal.skipped_columns.append(f"{column} (stored mapping empty)")
                continue

            mapping_values = {
                str(key): int(value)
                for key, value in raw_mapping.items()
                if isinstance(key, str) and value is not None
            }

            if not mapping_values:
                skipped_details.append(f"{column} (stored mapping invalid)")
                signal.skipped_columns.append(f"{column} (stored mapping invalid)")
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
                raw_unknown_value = stored_metadata.get("unknown_value")
                if raw_unknown_value is not None:
                    try:
                        effective_unknown_value = int(raw_unknown_value)
                    except (TypeError, ValueError):
                        effective_unknown_value = config.unknown_value
                    if _should_use_unknown_value(effective_encode_missing, effective_handle_unknown):
                        signal.unknown_value = effective_unknown_value
                encoded_column_meta = stored_metadata.get("encoded_column")
                if isinstance(encoded_column_meta, str) and encoded_column_meta.strip():
                    stored_encoded_column = encoded_column_meta.strip()

                raw_categories = stored_metadata.get("categories")
                if isinstance(raw_categories, list):
                    categories = [str(value) for value in raw_categories if str(value).strip()]

            if not categories:
                categories = [key for key in mapping_values.keys() if key is not None]

        else:
            trainer_series = string_series
            if storage is not None and train_mask is not None and train_mask.any():
                trainer_series = string_series[train_mask]

            if not effective_encode_missing:
                trainer_series = trainer_series.dropna()

            trainer_series = trainer_series.dropna()

            if trainer_series.nunique(dropna=True) <= 1:
                skipped_details.append(f"{column} (insufficient training categories)")
                signal.skipped_columns.append(f"{column} (insufficient training categories)")
                continue

            if OrdinalEncoder is not None:
                encoder_kwargs: Dict[str, Any] = {"handle_unknown": "error"}
                encoder = OrdinalEncoder(**encoder_kwargs)
                fit_values = trainer_series.astype("object").to_numpy().reshape(-1, 1)
                try:
                    encoder.fit(fit_values)
                    categories = [str(value) for value in encoder.categories_[0]]
                except Exception:
                    categories = sorted(str(value) for value in trainer_series.unique())
            else:
                categories = sorted(str(value) for value in trainer_series.unique())

            mapping_values = {category: index for index, category in enumerate(categories)}

        encoded_series, unknown_rows = _transform_with_mapping(
            string_series,
            mapping_values,
            encode_missing=effective_encode_missing,
            handle_unknown=effective_handle_unknown,
            unknown_value=effective_unknown_value,
        )

        class_count = len(categories)
        sample_preview = _build_category_preview(categories)

        preview_entries: List[OrdinalEncodingCategoryPreview] = []
        for index, value in enumerate(categories[:3]):
            preview_entries.append(
                OrdinalEncodingCategoryPreview(
                    category=format_category_label(value),
                    code=index,
                )
            )

        fragments = [f"{class_count} level{'s' if class_count != 1 else ''}"]
        if sample_preview:
            fragments.append(sample_preview)
        if effective_encode_missing:
            fragments.append(f"missing→{effective_unknown_value}")
        if effective_handle_unknown == "use_encoded_value":
            fragments.append(f"unknown→{effective_unknown_value}")
        elif unknown_rows > 0:
            fragments.append(f"{unknown_rows} unknown row{'s' if unknown_rows != 1 else ''}")

        if effective_drop_original:
            working_frame[column] = encoded_series
            existing_columns.add(column)
            detail = f"{column} (replaced; {'; '.join(fragments)})"
            encoded_column_name = column
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

        encoded_details.append(detail)

        signal.encoded_columns.append(
            OrdinalEncodingAppliedColumnSignal(
                source_column=column,
                encoded_column=None if replaced_original else encoded_column_name,
                replaced_original=replaced_original,
                category_count=class_count,
                handle_unknown=effective_handle_unknown,
                unknown_value=effective_unknown_value if _should_use_unknown_value(effective_encode_missing, effective_handle_unknown) else None,
                encode_missing=effective_encode_missing,
                preview=preview_entries,
            )
        )

        if storage is not None:
            if fit_mode == "fit":
                transformer_payload = {
                    "mapping": {str(key): int(value) for key, value in mapping_values.items()},
                    "categories": [str(value) for value in categories],
                }
                metadata: Dict[str, Any] = {
                    "encoded_column": encoded_column_name,
                    "replaced_original": replaced_original,
                    "category_count": class_count,
                    "encode_missing": effective_encode_missing,
                    "handle_unknown": effective_handle_unknown,
                    "unknown_value": effective_unknown_value if _should_use_unknown_value(effective_encode_missing, effective_handle_unknown) else None,
                    "drop_original": effective_drop_original,
                    "output_suffix": effective_output_suffix,
                    "categories": [str(value) for value in categories],
                    "train_rows": train_row_count,
                }

                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="ordinal_encoder",
                    transformer=transformer_payload,
                    column_name=column,
                    metadata=metadata,
                )

                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="ordinal_encoder",
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
                        transformer_name="ordinal_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )
            else:
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="ordinal_encoder",
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
                        transformer_name="ordinal_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )

    if not encoded_details:
        summary = "Ordinal encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Ordinal encoding: encoded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "ORDINAL_ENCODING_DEFAULT_SUFFIX",
    "ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES",
    "ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT",
    "ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN",
    "ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE",
    "apply_ordinal_encoding",
]
