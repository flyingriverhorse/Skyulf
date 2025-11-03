"""Hash encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import pandas as pd
from pandas.api import types as pd_types
from sklearn.feature_extraction import FeatureHasher

from core.feature_engineering.schemas import (
    HashEncodingAppliedColumnSignal,
    HashEncodingNodeSignal,
)
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

HASH_ENCODING_DEFAULT_SUFFIX = "_hashed"
HASH_ENCODING_DEFAULT_MAX_CATEGORIES = 200
HASH_ENCODING_MAX_CARDINALITY_LIMIT = 5000
HASH_ENCODING_DEFAULT_BUCKETS = 64
HASH_ENCODING_MIN_BUCKETS = 2
HASH_ENCODING_MAX_BUCKETS = 1_048_576


@dataclass
class NormalizedHashEncodingConfig:
    columns: List[str]
    auto_detect: bool
    max_categories: int
    n_buckets: int
    drop_original: bool
    output_suffix: str
    encode_missing: bool
    skipped_columns: List[str]


def _normalize_hash_encoding_config(config: Any) -> NormalizedHashEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)

    raw_max_categories = config.get("max_categories")
    try:
        numeric = float(raw_max_categories)
        if math.isnan(numeric):
            raise ValueError
        max_categories = int(round(numeric))
    except (TypeError, ValueError):
        max_categories = HASH_ENCODING_DEFAULT_MAX_CATEGORIES
    if max_categories < 0:
        max_categories = 0
    if max_categories > HASH_ENCODING_MAX_CARDINALITY_LIMIT:
        max_categories = HASH_ENCODING_MAX_CARDINALITY_LIMIT

    raw_buckets = config.get("n_buckets") or config.get("num_buckets")
    try:
        numeric = float(raw_buckets)
        if math.isnan(numeric):
            raise ValueError
        n_buckets = int(round(numeric))
    except (TypeError, ValueError):
        n_buckets = HASH_ENCODING_DEFAULT_BUCKETS
    n_buckets = max(HASH_ENCODING_MIN_BUCKETS, min(n_buckets, HASH_ENCODING_MAX_BUCKETS))

    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)
    encode_missing = _coerce_config_boolean(config.get("encode_missing"), default=False)

    raw_suffix = str(config.get("output_suffix") or HASH_ENCODING_DEFAULT_SUFFIX).strip()
    output_suffix = raw_suffix or HASH_ENCODING_DEFAULT_SUFFIX

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedHashEncodingConfig(
        columns=columns,
        auto_detect=auto_detect,
        max_categories=max_categories,
        n_buckets=n_buckets,
        drop_original=drop_original,
        output_suffix=output_suffix,
        encode_missing=encode_missing,
        skipped_columns=skipped_columns,
    )


def _resolve_output_column_name(base: str, suffix: str, existing: set[str]) -> str:
    candidate_base = base
    counter = 1
    while True:
        candidate = f"{candidate_base}{suffix}"
        if candidate not in existing:
            return candidate
        counter += 1
        candidate_base = f"{base}_{counter}"


def _hash_series(series: pd.Series, *, n_buckets: int, encode_missing: bool) -> pd.Series:
    string_series = series.astype("string")
    missing_mask = string_series.isna()

    values = string_series.fillna("")
    hasher = FeatureHasher(n_features=n_buckets, input_type="string", alternate_sign=False)
    hashed_matrix = hasher.transform([[value] for value in values.tolist()])

    indices = hashed_matrix.indices
    indptr = hashed_matrix.indptr

    hashed_values: List[Any] = []
    for row_index in range(len(values)):
        if missing_mask.iat[row_index]:
            hashed_values.append(pd.NA)
            continue

        start, end = indptr[row_index], indptr[row_index + 1]
        if end > start:
            hashed_values.append(int(indices[start]))
        else:
            hashed_values.append(pd.NA)

    hashed_series = pd.Series(hashed_values, index=series.index, dtype="Int64")

    if encode_missing:
        hashed_series = hashed_series.fillna(n_buckets)

    return hashed_series
def apply_hash_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, HashEncodingNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = HashEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    if frame.empty:
        return frame, "Hash encoding: no data available", signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_hash_encoding_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.max_categories = config.max_categories
    signal.n_buckets = config.n_buckets
    signal.drop_original = config.drop_original
    signal.encode_missing = config.encode_missing
    signal.output_suffix = config.output_suffix

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
        return frame, "Hash encoding: no categorical columns selected", signal

    working_frame = frame.copy()
    encoded_details: List[str] = []

    signal.evaluated_columns = list(candidate_columns)

    existing_columns = set(working_frame.columns)

    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None

    split_counts: Dict[str, int] = {}
    train_row_count = 0
    if has_splits:
        split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_row_count = int(split_counts.get("train", 0))

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
        unique_total = string_series.nunique(dropna=False)

        if unique_total <= 1:
            skipped_details.append(f"{column} (single category)")
            signal.skipped_columns.append(f"{column} (single category)")
            continue

        if config.max_categories > 0 and config.auto_detect and unique_total > config.max_categories:
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
                transformer_name="hash_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="hash_encoder",
                column_name=column,
            )

        fit_mode = "stateless"
        if storage is not None:
            if train_row_count > 0:
                fit_mode = "fit"
            elif stored_payload is not None:
                fit_mode = "reuse"

        if fit_mode == "reuse" and not isinstance(stored_metadata, dict):
            skipped_details.append(f"{column} (stored metadata missing)")
            signal.skipped_columns.append(f"{column} (stored metadata missing)")
            continue

        effective_drop_original = config.drop_original
        effective_output_suffix = config.output_suffix
        effective_encode_missing = config.encode_missing
        effective_n_buckets = config.n_buckets
        stored_output_column: Optional[str] = None

        if fit_mode == "reuse" and isinstance(stored_metadata, dict):
            raw_drop_original = stored_metadata.get("drop_original")
            if raw_drop_original is not None:
                effective_drop_original = bool(raw_drop_original)
                signal.drop_original = effective_drop_original
            raw_output_suffix = stored_metadata.get("output_suffix")
            if isinstance(raw_output_suffix, str) and raw_output_suffix.strip():
                effective_output_suffix = raw_output_suffix.strip()
                signal.output_suffix = effective_output_suffix
            raw_encode_missing = stored_metadata.get("encode_missing")
            if raw_encode_missing is not None:
                effective_encode_missing = bool(raw_encode_missing)
                signal.encode_missing = effective_encode_missing
            raw_n_buckets = stored_metadata.get("n_buckets")
            if raw_n_buckets is not None:
                try:
                    effective_n_buckets = int(raw_n_buckets)
                except (TypeError, ValueError):
                    effective_n_buckets = config.n_buckets
                signal.n_buckets = effective_n_buckets
            raw_output_column = stored_metadata.get("output_column")
            if isinstance(raw_output_column, str) and raw_output_column.strip():
                stored_output_column = raw_output_column.strip()

        hashed_series = _hash_series(
            series,
            n_buckets=effective_n_buckets,
            encode_missing=effective_encode_missing,
        )

        detail_suffix = f"{effective_n_buckets} bucket{'s' if effective_n_buckets != 1 else ''}"
        if effective_encode_missing:
            detail_suffix = f"{detail_suffix}; missing→{effective_n_buckets}"

        if effective_drop_original:
            working_frame[column] = hashed_series
            encoded_details.append(f"{column} (replaced; {detail_suffix})")
            output_column = column
            existing_columns.add(column)
        else:
            encoded_column = stored_output_column or _resolve_output_column_name(
                column,
                effective_output_suffix,
                existing_columns,
            )
            insert_at = working_frame.columns.get_loc(column) + 1
            if encoded_column in working_frame.columns:
                working_frame[encoded_column] = hashed_series
            else:
                working_frame.insert(insert_at, encoded_column, hashed_series)
                existing_columns.add(encoded_column)
            encoded_details.append(f"{column}→{encoded_column} ({detail_suffix})")
            output_column = encoded_column

        sample_hashes: List[int] = []
        for value in hashed_series.dropna().unique()[:3]:
            try:
                sample_hashes.append(int(value))
            except (TypeError, ValueError):
                continue

        signal.encoded_columns.append(
            HashEncodingAppliedColumnSignal(
                source_column=column,
                output_column=output_column,
                replaced_original=effective_drop_original,
                bucket_count=effective_n_buckets,
                encoded_missing=effective_encode_missing,
                category_count=int(unique_total),
                sample_hashes=sample_hashes,
            )
        )

        if storage is not None:
            transformer_payload: Dict[str, Any] = {
                "n_buckets": int(effective_n_buckets),
                "encode_missing": bool(effective_encode_missing),
            }
            metadata: Dict[str, Any] = {
                "output_column": output_column,
                "drop_original": bool(effective_drop_original),
                "output_suffix": effective_output_suffix,
                "encode_missing": bool(effective_encode_missing),
                "n_buckets": int(effective_n_buckets),
                "category_count": int(unique_total),
                "train_rows": train_row_count,
            }

            if stored_payload is None or fit_mode == "fit":
                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="hash_encoder",
                    transformer=transformer_payload,
                    column_name=column,
                    metadata=metadata,
                )

            train_action = "fit_transform" if train_row_count > 0 else "not_available"
            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="hash_encoder",
                column_name=column,
                split_name="train",
                action=train_action,
                row_count=train_row_count,
            )

            for split_name in ("test", "validation"):
                rows_processed = int(split_counts.get(split_name, 0))
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="hash_encoder",
                    column_name=column,
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )

    if not encoded_details:
        summary = "Hash encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Hash encoding: encoded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "HASH_ENCODING_DEFAULT_SUFFIX",
    "HASH_ENCODING_DEFAULT_MAX_CATEGORIES",
    "HASH_ENCODING_DEFAULT_BUCKETS",
    "HASH_ENCODING_MAX_CARDINALITY_LIMIT",
    "apply_hash_encoding",
]
