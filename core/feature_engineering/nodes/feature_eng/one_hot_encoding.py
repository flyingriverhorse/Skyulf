"""One-hot encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
import math

import numpy as np

import pandas as pd
from pandas.api import types as pd_types

try:  # pragma: no cover - optional dependency guard
    from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    OneHotEncoder = None  # type: ignore[assignment]

from core.feature_engineering.schemas import (
    OneHotEncodingAppliedColumnSignal,
    OneHotEncodingNodeSignal,
)

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES = 20
ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT = 200
ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR = "_"
ONE_HOT_MISSING_SENTINEL = "__mlops_missing__"


logger = logging.getLogger(__name__)


@dataclass
class NormalizedOneHotEncodingConfig:
    columns: List[str]
    auto_detect: bool
    drop_original: bool
    drop_first: bool
    include_missing: bool
    max_categories: int
    prefix_separator: str
    skipped_columns: List[str]


def _normalize_one_hot_encoding_config(config: Any) -> NormalizedOneHotEncodingConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)
    drop_first = _coerce_config_boolean(config.get("drop_first"), default=False)
    include_missing = _coerce_config_boolean(config.get("include_missing"), default=False)

    raw_max_categories = config.get("max_categories")
    max_categories: int
    try:
        numeric = float(raw_max_categories)
        if math.isnan(numeric):
            raise ValueError
        max_categories = int(round(numeric))
    except (TypeError, ValueError):
        max_categories = ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES

    if max_categories < 0:
        max_categories = 0
    if max_categories > ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT:
        max_categories = ONE_HOT_ENCODING_MAX_CARDINALITY_LIMIT

    prefix_separator = str(config.get("prefix_separator") or ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR)
    if not prefix_separator:
        prefix_separator = ONE_HOT_ENCODING_DEFAULT_PREFIX_SEPARATOR

    skipped_columns = _coerce_string_list(config.get("skipped_columns"))

    return NormalizedOneHotEncodingConfig(
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


def _coerce_numpy_scalar(value: Any) -> Any:
    """Return a Python scalar for numpy scalar inputs."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_drop_indices(raw_value: Any) -> Optional[List[Optional[int]]]:
    """Convert numpy-based drop indices into JSON-serializable lists."""

    if raw_value is None:
        return None

    if hasattr(raw_value, "tolist"):
        raw_value = raw_value.tolist()

    if isinstance(raw_value, (list, tuple)):
        normalized: List[Optional[int]] = []
        for item in raw_value:
            if isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
                # sklearn may return nested structures for multi-column encoders
                nested = _normalize_drop_indices(item)
                if isinstance(nested, list):
                    normalized.extend(nested)
                else:
                    normalized.append(nested)  # type: ignore[arg-type]
                continue
            if item is None:
                normalized.append(None)
                continue
            try:
                normalized.append(int(_coerce_numpy_scalar(item)))
            except (TypeError, ValueError):
                normalized.append(None)
        return normalized

    try:
        return [int(_coerce_numpy_scalar(raw_value))]
    except (TypeError, ValueError):
        return None


def apply_one_hot_encoding(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, OneHotEncodingNodeSignal]:
    """Apply one-hot encoding to categorical columns.
    
    Supports train/test/validation split awareness:
    - On training data: fit encoder and store for reuse
    - On test/val data: retrieve stored encoder and transform
    - Without split: fit and transform (no storage)
    
    Args:
        frame: Input DataFrame
        node: Node configuration dictionary
        pipeline_id: Unique pipeline identifier for transformer storage
        
    Returns:
        Tuple of (transformed_frame, summary_message, signal_metadata)
    """
    from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
    from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
    
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = OneHotEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_one_hot_encoding_config(config_payload)

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
        return frame, "One-hot encoding: no data available", signal

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
        return frame, "One-hot encoding: no categorical columns selected", signal

    # Detect if we have split awareness
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    
    # Debug logging
    logger.debug(
        "One-hot encoding context resolved",
        extra={
            "node_id": node_id,
            "has_splits": has_splits,
            "pipeline_id": pipeline_id,
        },
    )
    
    # Get transformer storage if we have pipeline_id
    storage = None
    if pipeline_id and has_splits:
        storage = get_pipeline_store()
        logger.debug("One-hot encoding using transformer storage", extra={"node_id": node_id})

    working_frame = frame.copy()
    encoded_details: List[str] = []

    existing_columns = set(working_frame.columns)

    signal.evaluated_columns = list(candidate_columns)

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

        has_missing = series.isna().any()

        stored_encoder: Optional[Any] = None
        stored_metadata: Optional[Dict[str, Any]] = None
        if storage is not None:
            stored_encoder = storage.get_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="one_hot_encoder",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="one_hot_encoder",
                column_name=column,
            )

        fit_mode = "fit"
        if storage is not None and pipeline_id and train_row_count <= 0:
            if stored_encoder is not None:
                fit_mode = "reuse"
            else:
                skipped_details.append(f"{column} (no training data)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

        if OneHotEncoder is None and fit_mode != "reuse":
            signal.skipped_columns.append(f"{column} (scikit-learn unavailable)")
            skipped_details.append(f"{column} (scikit-learn unavailable)")
            continue

        encoder: Any
        encoder_metadata: Dict[str, Any] = {}
        effective_drop_first = config.drop_first
        effective_include_missing = config.include_missing
        effective_prefix_separator = config.prefix_separator
        encoder_has_missing = has_missing

        # If we have splits AND pipeline_id (actual execution), process train/test/val with fit/transform
        # If we have splits but NO pipeline_id (recommendations), treat as normal encoding
        if has_splits and storage and pipeline_id and fit_mode == "fit":
            logger.debug(
                "Applying split-aware one-hot encoding",
                extra={"node_id": node_id, "column": column},
            )
            if train_mask is None or not train_mask.any():
                skipped_details.append(f"{column} (no training data to fit encoder)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

            train_series = series[train_mask].astype("string")
            train_prepared = train_series.fillna(ONE_HOT_MISSING_SENTINEL) if has_missing else train_series

            encoder_kwargs: Dict[str, Any] = {
                "drop": "first" if config.drop_first else None,
                "handle_unknown": "ignore",
                "dtype": np.float64,
            }
            if config.max_categories > 0:
                encoder_kwargs["max_categories"] = config.max_categories

            try:
                encoder = OneHotEncoder(sparse_output=False, **encoder_kwargs)  # type: ignore[arg-type]
            except TypeError:
                encoder_kwargs.pop("dtype", None)
                encoder = OneHotEncoder(sparse=False, **encoder_kwargs)  # type: ignore[arg-type]

            try:
                train_encoded = encoder.fit_transform(train_prepared.to_numpy().reshape(-1, 1))
                logger.info(
                    f"✓ One-hot encoder FITTED on training data for column '{column}'",
                    extra={
                        "node_id": node_id,
                        "column": column,
                        "train_rows": int(train_mask.sum()),
                        "categories_found": len(encoder.categories_[0]) if hasattr(encoder, "categories_") else 0,
                        "output_features": train_encoded.shape[1],
                    }
                )
            except Exception as exc:
                skipped_details.append(f"{column} (failed to fit: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (fit failed)")
                continue

            raw_categories = list(encoder.categories_[0]) if getattr(encoder, "categories_", None) is not None else []
            sanitized_categories = [_coerce_numpy_scalar(category) for category in raw_categories]
            drop_idx_attr = getattr(encoder, "drop_idx_", None)
            sanitized_drop_idx = _normalize_drop_indices(drop_idx_attr)

            encoder_metadata = {
                "categories": sanitized_categories,
                "drop_idx": sanitized_drop_idx,
                "config": {
                    "drop_first": config.drop_first,
                    "include_missing": config.include_missing,
                    "prefix_separator": config.prefix_separator,
                    "has_missing": has_missing,
                },
            }

            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="one_hot_encoder",
                transformer=encoder,
                column_name=column,
                metadata=encoder_metadata,
            )

            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="one_hot_encoder",
                column_name=column,
                split_name="train",
                action="fit_transform",
                row_count=train_row_count,
            )

            logger.info(
                f"✓ Encoder stored in transformer storage",
                extra={
                    "node_id": node_id,
                    "column": column,
                    "pipeline_id": pipeline_id,
                    "categories": len(raw_categories),
                },
            )

            encoder_has_missing = has_missing

            full_prepared = series.astype("string")
            if encoder_has_missing:
                full_prepared = full_prepared.fillna(ONE_HOT_MISSING_SENTINEL)

            try:
                encoded_matrix = encoder.transform(full_prepared.to_numpy().reshape(-1, 1))
                logger.info(
                    f"✓ TRANSFORMED all splits using fitted encoder for column '{column}'",
                    extra={
                        "node_id": node_id,
                        "column": column,
                        "total_rows_transformed": len(full_prepared),
                        "train_rows": int(train_mask.sum()),
                        "test_rows": int(split_counts.get("test", 0)),
                        "validation_rows": int(split_counts.get("validation", 0)),
                        "output_features": encoded_matrix.shape[1],
                    },
                )

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="one_hot_encoder",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )
            except Exception as exc:
                skipped_details.append(f"{column} (failed to transform: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (transform failed)")
                continue

        elif has_splits and storage and pipeline_id and fit_mode == "reuse":
            logger.debug(
                "Reusing stored one-hot encoder",
                extra={"node_id": node_id, "column": column},
            )

            if stored_encoder is None or not hasattr(stored_encoder, "transform"):
                skipped_details.append(f"{column} (stored encoder unavailable)")
                signal.skipped_columns.append(f"{column} (stored encoder unavailable)")
                continue

            encoder = stored_encoder
            if isinstance(stored_metadata, dict):
                encoder_metadata = dict(stored_metadata)
            else:
                encoder_metadata = {}

            metadata_config = encoder_metadata.get("config") if isinstance(encoder_metadata, dict) else None
            if isinstance(metadata_config, dict):
                if "drop_first" in metadata_config:
                    effective_drop_first = bool(metadata_config.get("drop_first"))
                    signal.drop_first = effective_drop_first
                if "include_missing" in metadata_config:
                    effective_include_missing = bool(metadata_config.get("include_missing"))
                    signal.include_missing = effective_include_missing
                if metadata_config.get("prefix_separator"):
                    effective_prefix_separator = str(metadata_config.get("prefix_separator"))
                    signal.prefix_separator = effective_prefix_separator
                encoder_has_missing = bool(metadata_config.get("has_missing"))

            prepared_series = series.astype("string")
            if encoder_has_missing or has_missing:
                prepared_series = prepared_series.fillna(ONE_HOT_MISSING_SENTINEL)

            try:
                encoded_matrix = encoder.transform(prepared_series.to_numpy().reshape(-1, 1))
            except Exception as exc:
                skipped_details.append(f"{column} (failed to transform: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (transform failed)")
                continue

            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="one_hot_encoder",
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
                    transformer_name="one_hot_encoder",
                    column_name=column,
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )

        else:
            # NO SPLITS: Standard fit_transform (or splits without pipeline_id for recommendations)
            logger.debug(
                "Applying standard one-hot encoding",
                extra={
                    "node_id": node_id,
                    "column": column,
                    "has_splits": has_splits,
                    "pipeline_id": pipeline_id,
                },
            )
            prepared_series = series.astype("string")
            if has_missing:
                prepared_series = prepared_series.fillna(ONE_HOT_MISSING_SENTINEL)
            
            encoder_kwargs: Dict[str, Any] = {
                "drop": "first" if config.drop_first else None,
                "handle_unknown": "ignore",
                "dtype": np.float64,
            }
            if config.max_categories > 0:
                encoder_kwargs["max_categories"] = config.max_categories

            try:
                encoder = OneHotEncoder(sparse_output=False, **encoder_kwargs)  # type: ignore[arg-type]
            except TypeError:
                encoder_kwargs.pop("dtype", None)
                encoder = OneHotEncoder(sparse=False, **encoder_kwargs)  # type: ignore[arg-type]
            
            try:
                encoded_matrix = encoder.fit_transform(prepared_series.to_numpy().reshape(-1, 1))
                logger.info(
                    f"✓ Standard one-hot encoding (fit_transform) for column '{column}'",
                    extra={
                        "node_id": node_id,
                        "column": column,
                        "rows": len(prepared_series),
                        "has_splits": has_splits,
                        "pipeline_id": pipeline_id,
                        "output_features": encoded_matrix.shape[1],
                    }
                )
            except Exception as e:
                skipped_details.append(f"{column} (failed to encode: {str(e)[:50]})")
                signal.skipped_columns.append(f"{column} (encoding failed)")
                continue
            
            raw_categories = list(encoder.categories_[0]) if getattr(encoder, "categories_", None) is not None else []
            drop_idx_attr = getattr(encoder, "drop_idx_", None)
            encoder_metadata = {
                "categories": raw_categories,
                "drop_idx": drop_idx_attr,
            }

        if fit_mode == "reuse" and not encoder_metadata and isinstance(stored_metadata, dict):
            encoder_metadata = dict(stored_metadata)

        if hasattr(encoded_matrix, "toarray"):
            encoded_matrix = encoded_matrix.toarray()

        encoded_matrix = np.asarray(encoded_matrix)
        if encoded_matrix.size == 0:
            skipped_details.append(f"{column} (no dummy columns generated)")
            signal.skipped_columns.append(f"{column} (no dummy columns generated)")
            continue

        # Get categories and drop indices from encoder or metadata
        if encoder_metadata:
            raw_categories = encoder_metadata.get("categories", [])
            drop_idx_attr = encoder_metadata.get("drop_idx")
        else:
            raw_categories = list(encoder.categories_[0]) if getattr(encoder, "categories_", None) is not None else []
            drop_idx_attr = getattr(encoder, "drop_idx_", None)

        raw_categories = [_coerce_numpy_scalar(category) for category in raw_categories]
        drop_idx_attr = _normalize_drop_indices(drop_idx_attr)

        drop_indices: set[int] = set()
        if drop_idx_attr is not None:
            for candidate in drop_idx_attr:
                if candidate is None:
                    continue
                try:
                    drop_indices.add(int(candidate))
                except (TypeError, ValueError):
                    continue

        output_categories: List[Any] = []
        for idx, category in enumerate(raw_categories):
            if idx in drop_indices:
                continue
            output_categories.append(category)

        column_positions: List[int] = []
        dummy_column_names: List[str] = []
        includes_missing_dummy = False

        for position, category in enumerate(output_categories):
            is_missing_category = category == ONE_HOT_MISSING_SENTINEL
            if is_missing_category and not (
                effective_include_missing and (encoder_has_missing or has_missing)
            ):
                continue

            label = "nan" if is_missing_category else str(category)
            dummy_name = f"{column}{effective_prefix_separator}{label}"

            column_positions.append(position)
            dummy_column_names.append(dummy_name)

            if is_missing_category:
                includes_missing_dummy = True

        if not column_positions:
            skipped_details.append(f"{column} (no dummy columns generated)")
            signal.skipped_columns.append(f"{column} (no dummy columns generated)")
            continue

        subset_matrix = encoded_matrix[:, column_positions]
        dummy_frame = pd.DataFrame(subset_matrix, index=series.index, columns=dummy_column_names)
        dummy_frame = dummy_frame.round().astype("UInt8")

        rename_mapping: Dict[str, str] = {}
        renamed_columns: List[str] = []
        for dummy_column in dummy_frame.columns:
            target_name = _resolve_unique_column_name(dummy_column, existing_columns)
            rename_mapping[dummy_column] = target_name
            existing_columns.add(target_name)
            renamed_columns.append(target_name)

        if rename_mapping:
            dummy_frame = dummy_frame.rename(columns=rename_mapping)

        insert_at = working_frame.columns.get_loc(column) + (0 if config.drop_original else 1)
        for offset, dummy_column in enumerate(renamed_columns):
            working_frame.insert(insert_at + offset, dummy_column, dummy_frame[dummy_column])

        if config.drop_original:
            working_frame = working_frame.drop(columns=[column])
            existing_columns.discard(column)

        preview_values = series_string.dropna().value_counts().head(3).index.tolist()
        preview = ", ".join(str(value) for value in preview_values)
        encoded_details.append(
            f"{column} → {len(renamed_columns)} column{'s' if len(renamed_columns) != 1 else ''}"
            + (f" ({preview})" if preview else "")
        )

        signal.encoded_columns.append(
            OneHotEncodingAppliedColumnSignal(
                source_column=column,
                dummy_columns=renamed_columns,
                replaced_original=config.drop_original,
                category_count=int(unique_total),
                includes_missing_dummy=bool(
                    effective_include_missing
                    and (encoder_has_missing or has_missing)
                    and includes_missing_dummy
                ),
                preview_categories=[str(value) for value in preview_values],
            )
        )

    if not encoded_details:
        summary = "One-hot encoding: no columns encoded"
    else:
        preview = ", ".join(encoded_details[:3])
        if len(encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"One-hot encoding: expanded {len(encoded_details)} column(s) ({preview})"

    if skipped_details:
        preview = ", ".join(skipped_details[:3])
        if len(skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


__all__ = [
    "ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES",
    "apply_one_hot_encoding",
]
