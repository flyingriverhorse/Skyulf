"""One-hot encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class OneHotEncodingContext:
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    has_splits: bool
    train_mask: Optional[pd.Series]
    split_counts: Dict[str, int]
    train_row_count: int


@dataclass
class SplitActivity:
    split_name: str
    action: str
    row_count: int


@dataclass
class OneHotEncodingAccumulator:
    encoded_details: List[str] = field(default_factory=list)
    skipped_details: List[str] = field(default_factory=list)


@dataclass
class OneHotColumnAnalysis:
    series: pd.Series
    series_string: pd.Series
    unique_total: int
    has_missing: bool
    preview_values: List[str]


@dataclass
class EncodingResult:
    encoded_matrix: np.ndarray
    encoder: Optional[Any]
    metadata: Optional[Dict[str, Any]]
    encoder_has_missing: bool
    store_encoder: bool
    split_activities: List[SplitActivity]
    drop_first_override: Optional[bool] = None
    include_missing_override: Optional[bool] = None
    prefix_separator_override: Optional[str] = None


@dataclass
class OneHotEncodingPlan:
    column: str
    dummy_frame: pd.DataFrame
    dummy_columns: List[str]
    drop_original: bool
    detail: str
    signal_entry: OneHotEncodingAppliedColumnSignal
    store_encoder: bool
    encoder: Optional[Any]
    encoder_metadata: Optional[Dict[str, Any]]
    split_activities: List[SplitActivity]
    drop_first_override: Optional[bool]
    include_missing_override: Optional[bool]
    prefix_separator_override: Optional[str]
    fit_mode: str


def _initialize_signal_from_config(
    signal: OneHotEncodingNodeSignal,
    config: NormalizedOneHotEncodingConfig,
) -> None:
    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.drop_first = config.drop_first
    signal.include_missing = config.include_missing
    signal.max_categories = config.max_categories
    signal.prefix_separator = config.prefix_separator


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedOneHotEncodingConfig,
) -> Tuple[List[str], List[str]]:
    skipped_configured = set(config.skipped_columns)
    candidate_columns: List[str] = []
    skipped_notes: List[str] = []
    seen: set[str] = set()

    for column in config.columns:
        normalized = str(column or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in skipped_configured:
            entry = f"{normalized} (skipped)"
            skipped_notes.append(entry)
            continue
        candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured:
                entry = f"{column} (skipped)"
                skipped_notes.append(entry)
                continue
            candidate_columns.append(column)

    return candidate_columns, skipped_notes


def _build_context(
    frame: pd.DataFrame,
    pipeline_id: Optional[str],
    node_id: Optional[str],
) -> OneHotEncodingContext:
    from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
    from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_splits else {}
    train_row_count = int(split_counts.get("train", 0)) if has_splits else 0
    train_mask = (frame[SPLIT_TYPE_COLUMN] == "train") if has_splits else None

    return OneHotEncodingContext(
        storage=storage,
        pipeline_id=pipeline_id,
        node_id=node_id,
        has_splits=has_splits,
        train_mask=train_mask,
        split_counts=split_counts,
        train_row_count=train_row_count,
    )


def _register_skip(
    accumulator: OneHotEncodingAccumulator,
    signal: OneHotEncodingNodeSignal,
    message: str,
) -> None:
    accumulator.skipped_details.append(message)
    signal.skipped_columns.append(message)


def _compose_summary(accumulator: OneHotEncodingAccumulator) -> str:
    if not accumulator.encoded_details:
        summary = "One-hot encoding: no columns encoded"
    else:
        preview = ", ".join(accumulator.encoded_details[:3])
        if len(accumulator.encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"One-hot encoding: expanded {len(accumulator.encoded_details)} column(s) ({preview})"

    if accumulator.skipped_details:
        preview = ", ".join(accumulator.skipped_details[:3])
        if len(accumulator.skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return summary


def _analyze_column(
    frame: pd.DataFrame,
    column: str,
    config: NormalizedOneHotEncodingConfig,
) -> Tuple[Optional[OneHotColumnAnalysis], Optional[str]]:
    if column not in frame.columns:
        return None, f"{column} (missing)"

    series = frame[column]
    if pd_types.is_bool_dtype(series):
        return None, f"{column} (boolean column)"

    if not (
        pd_types.is_object_dtype(series)
        or pd_types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None, f"{column} (non-categorical dtype)"

    series_string = series.astype("string")
    unique_total = series_string.nunique(dropna=True)

    if unique_total <= 1:
        return None, f"{column} (single category)"

    if config.max_categories and unique_total > config.max_categories:
        return None, f"{column} ({unique_total} categories > {config.max_categories})"

    has_missing = series.isna().any()
    preview_values = series_string.dropna().value_counts().head(3).index.tolist()

    return (
        OneHotColumnAnalysis(
            series=series,
            series_string=series_string,
            unique_total=int(unique_total),
            has_missing=bool(has_missing),
            preview_values=[str(value) for value in preview_values],
        ),
        None,
    )


def _fetch_stored_encoder(
    context: OneHotEncodingContext,
    column: str,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not context.storage:
        return None, None
    payload = context.storage.get_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="one_hot_encoder",
        column_name=column,
    )
    metadata = context.storage.get_metadata(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="one_hot_encoder",
        column_name=column,
    )
    return payload, metadata


def _determine_fit_mode(
    context: OneHotEncodingContext,
    stored_encoder: Optional[Any],
) -> str:
    if not context.storage or not context.has_splits or not context.pipeline_id:
        return "standard"
    if context.train_row_count > 0:
        return "fit"
    if stored_encoder is not None:
        return "reuse"
    return "skip_no_training"


def _build_encoder_kwargs(config: NormalizedOneHotEncodingConfig) -> Dict[str, Any]:
    encoder_kwargs: Dict[str, Any] = {
        "drop": "first" if config.drop_first else None,
        "handle_unknown": "ignore",
        "dtype": np.float64,
    }
    if config.max_categories > 0:
        encoder_kwargs["max_categories"] = config.max_categories
    return encoder_kwargs


def _create_encoder(encoder_kwargs: Dict[str, Any]) -> Any:
    if OneHotEncoder is None:
        raise RuntimeError("scikit-learn unavailable")
    try:
        return OneHotEncoder(sparse_output=False, **encoder_kwargs)  # type: ignore[arg-type]
    except TypeError:
        adjusted_kwargs = dict(encoder_kwargs)
        adjusted_kwargs.pop("dtype", None)
        return OneHotEncoder(sparse=False, **adjusted_kwargs)  # type: ignore[arg-type]


def _fit_split_encoder(
    column: str,
    analysis: OneHotColumnAnalysis,
    config: NormalizedOneHotEncodingConfig,
    context: OneHotEncodingContext,
) -> Tuple[Optional[EncodingResult], Optional[str]]:
    if context.train_mask is None or not context.train_mask.any():
        return None, f"{column} (no training data to fit encoder)"

    encoder_kwargs = _build_encoder_kwargs(config)

    try:
        encoder = _create_encoder(encoder_kwargs)
    except RuntimeError:
        return None, f"{column} (scikit-learn unavailable)"

    train_series = analysis.series[context.train_mask].astype("string")
    train_prepared = train_series.fillna(ONE_HOT_MISSING_SENTINEL) if analysis.has_missing else train_series

    try:
        encoder.fit(train_prepared.to_numpy().reshape(-1, 1))
    except Exception as exc:
        return None, f"{column} (failed to fit: {str(exc)[:50]})"

    full_prepared = analysis.series_string
    if analysis.has_missing:
        full_prepared = full_prepared.fillna(ONE_HOT_MISSING_SENTINEL)

    try:
        encoded_matrix = encoder.transform(full_prepared.to_numpy().reshape(-1, 1))
    except Exception as exc:
        return None, f"{column} (failed to transform: {str(exc)[:50]})"

    if hasattr(encoded_matrix, "toarray"):
        encoded_matrix = encoded_matrix.toarray()

    encoded_matrix = np.asarray(encoded_matrix)

    raw_categories = list(encoder.categories_[0]) if getattr(encoder, "categories_", None) is not None else []
    sanitized_categories = [_coerce_numpy_scalar(category) for category in raw_categories]
    drop_idx_attr = getattr(encoder, "drop_idx_", None)
    sanitized_drop_idx = _normalize_drop_indices(drop_idx_attr)

    metadata: Dict[str, Any] = {
        "categories": sanitized_categories,
        "drop_idx": sanitized_drop_idx,
        "config": {
            "drop_first": config.drop_first,
            "include_missing": config.include_missing,
            "prefix_separator": config.prefix_separator,
            "has_missing": analysis.has_missing,
        },
        "method": "one_hot_encoding",
        "method_label": f"One-Hot Encoding ({len(sanitized_categories)} categories)",
    }
    metadata["train_rows"] = context.train_row_count

    split_activities = [
        SplitActivity(split_name="train", action="fit_transform", row_count=context.train_row_count),
    ]
    for split_name in ("test", "validation"):
        rows_processed = int(context.split_counts.get(split_name, 0))
        split_activities.append(
            SplitActivity(
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )
        )

    return (
        EncodingResult(
            encoded_matrix=encoded_matrix,
            encoder=encoder,
            metadata=metadata,
            encoder_has_missing=analysis.has_missing,
            store_encoder=True,
            split_activities=split_activities,
        ),
        None,
    )


def _reuse_stored_encoder(
    column: str,
    analysis: OneHotColumnAnalysis,
    config: NormalizedOneHotEncodingConfig,
    stored_encoder: Any,
    stored_metadata: Optional[Dict[str, Any]],
    context: OneHotEncodingContext,
) -> Tuple[Optional[EncodingResult], Optional[str]]:
    if stored_encoder is None or not hasattr(stored_encoder, "transform"):
        return None, f"{column} (stored encoder unavailable)"

    metadata = dict(stored_metadata) if isinstance(stored_metadata, dict) else {}
    metadata_config = metadata.get("config") if isinstance(metadata, dict) else None

    drop_first_override: Optional[bool] = None
    include_missing_override: Optional[bool] = None
    prefix_separator_override: Optional[str] = None
    encoder_has_missing = analysis.has_missing

    if isinstance(metadata_config, dict):
        if "drop_first" in metadata_config:
            drop_first_override = bool(metadata_config.get("drop_first"))
        if "include_missing" in metadata_config:
            include_missing_override = bool(metadata_config.get("include_missing"))
        if metadata_config.get("prefix_separator"):
            prefix_separator_override = str(metadata_config.get("prefix_separator"))
        encoder_has_missing = bool(metadata_config.get("has_missing", encoder_has_missing))

    prepared_series = analysis.series_string
    if encoder_has_missing or analysis.has_missing:
        prepared_series = prepared_series.fillna(ONE_HOT_MISSING_SENTINEL)

    try:
        encoded_matrix = stored_encoder.transform(prepared_series.to_numpy().reshape(-1, 1))
    except Exception as exc:
        return None, f"{column} (failed to transform: {str(exc)[:50]})"

    if hasattr(encoded_matrix, "toarray"):
        encoded_matrix = encoded_matrix.toarray()

    encoded_matrix = np.asarray(encoded_matrix)

    raw_categories = metadata.get("categories") if isinstance(metadata, dict) else None
    if raw_categories is None and getattr(stored_encoder, "categories_", None) is not None:
        raw_categories = list(stored_encoder.categories_[0])
    if raw_categories is None:
        raw_categories = []
    sanitized_categories = [_coerce_numpy_scalar(category) for category in raw_categories]

    drop_idx_attr = metadata.get("drop_idx") if isinstance(metadata, dict) else None
    if drop_idx_attr is None and hasattr(stored_encoder, "drop_idx_"):
        drop_idx_attr = getattr(stored_encoder, "drop_idx_", None)
    sanitized_drop_idx = _normalize_drop_indices(drop_idx_attr)

    metadata["categories"] = sanitized_categories
    metadata["drop_idx"] = sanitized_drop_idx

    split_activities = [
        SplitActivity(
            split_name="train",
            action="not_available",
            row_count=context.train_row_count,
        ),
    ]
    for split_name in ("test", "validation"):
        rows_processed = int(context.split_counts.get(split_name, 0))
        split_activities.append(
            SplitActivity(
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )
        )

    return (
        EncodingResult(
            encoded_matrix=encoded_matrix,
            encoder=stored_encoder,
            metadata=metadata,
            encoder_has_missing=encoder_has_missing,
            store_encoder=False,
            split_activities=split_activities,
            drop_first_override=drop_first_override,
            include_missing_override=include_missing_override,
            prefix_separator_override=prefix_separator_override,
        ),
        None,
    )


def _standard_fit_transform(
    column: str,
    analysis: OneHotColumnAnalysis,
    config: NormalizedOneHotEncodingConfig,
) -> Tuple[Optional[EncodingResult], Optional[str]]:
    encoder_kwargs = _build_encoder_kwargs(config)

    try:
        encoder = _create_encoder(encoder_kwargs)
    except RuntimeError:
        return None, f"{column} (scikit-learn unavailable)"

    prepared_series = analysis.series_string
    if analysis.has_missing:
        prepared_series = prepared_series.fillna(ONE_HOT_MISSING_SENTINEL)

    try:
        encoded_matrix = encoder.fit_transform(prepared_series.to_numpy().reshape(-1, 1))
    except Exception as exc:
        return None, f"{column} (failed to encode: {str(exc)[:50]})"

    if hasattr(encoded_matrix, "toarray"):
        encoded_matrix = encoded_matrix.toarray()

    encoded_matrix = np.asarray(encoded_matrix)

    raw_categories = list(encoder.categories_[0]) if getattr(encoder, "categories_", None) is not None else []
    sanitized_categories = [_coerce_numpy_scalar(category) for category in raw_categories]
    drop_idx_attr = getattr(encoder, "drop_idx_", None)
    sanitized_drop_idx = _normalize_drop_indices(drop_idx_attr)

    metadata = {
        "categories": sanitized_categories,
        "drop_idx": sanitized_drop_idx,
    }

    return (
        EncodingResult(
            encoded_matrix=encoded_matrix,
            encoder=encoder,
            metadata=metadata,
            encoder_has_missing=analysis.has_missing,
            store_encoder=False,
            split_activities=[],
        ),
        None,
    )


def _construct_dummy_frame(
    column: str,
    analysis: OneHotColumnAnalysis,
    config: NormalizedOneHotEncodingConfig,
    result: EncodingResult,
    existing_columns: set[str],
) -> Tuple[pd.DataFrame, List[str], bool, str]:
    metadata = result.metadata or {}
    raw_categories = metadata.get("categories", [])
    drop_idx_attr = metadata.get("drop_idx")

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
    for index, category in enumerate(raw_categories):
        if index in drop_indices:
            continue
        output_categories.append(category)

    if not output_categories or result.encoded_matrix.size == 0:
        raise ValueError("no dummy columns generated")

    effective_include_missing = (
        result.include_missing_override
        if result.include_missing_override is not None
        else config.include_missing
    )
    effective_prefix_separator = (
        result.prefix_separator_override or config.prefix_separator
    )

    final_columns: List[str] = []
    columns_data: Dict[str, Any] = {}
    includes_missing_dummy = False

    for position, category in enumerate(output_categories):
        is_missing_category = category == ONE_HOT_MISSING_SENTINEL
        if is_missing_category and not (
            effective_include_missing and (result.encoder_has_missing or analysis.has_missing)
        ):
            continue

        label = "nan" if is_missing_category else str(category)
        candidate_name = f"{column}{effective_prefix_separator}{label}"
        final_name = _resolve_unique_column_name(candidate_name, existing_columns)
        final_columns.append(final_name)
        columns_data[final_name] = result.encoded_matrix[:, position]

        if is_missing_category:
            includes_missing_dummy = True

    if not final_columns:
        raise ValueError("no dummy columns generated")

    dummy_frame = pd.DataFrame(columns_data, index=analysis.series.index).round().astype("UInt8")

    preview = ", ".join(analysis.preview_values)
    detail = (
        f"{column} → {len(final_columns)} column"
        f"{'s' if len(final_columns) != 1 else ''}"
        + (f" ({preview})" if preview else "")
    )

    return dummy_frame, final_columns, includes_missing_dummy, detail


def _record_split_activities(
    plan: OneHotEncodingPlan,
    context: OneHotEncodingContext,
) -> None:
    if not context.storage:
        return
    for activity in plan.split_activities:
        context.storage.record_split_activity(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id or "",
            transformer_name="one_hot_encoder",
            column_name=plan.column,
            split_name=activity.split_name,
            action=activity.action,
            row_count=activity.row_count,
        )


def _store_encoder(plan: OneHotEncodingPlan, context: OneHotEncodingContext) -> None:
    if not context.storage or not plan.store_encoder or plan.encoder is None:
        return
    context.storage.store_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="one_hot_encoder",
        transformer=plan.encoder,
        column_name=plan.column,
        metadata=plan.encoder_metadata or {},
    )
    _record_split_activities(plan, context)


def _build_column_plan(
    column: str,
    frame: pd.DataFrame,
    config: NormalizedOneHotEncodingConfig,
    context: OneHotEncodingContext,
    existing_columns: set[str],
) -> Tuple[Optional[OneHotEncodingPlan], Optional[str]]:
    analysis, error = _analyze_column(frame, column, config)
    if error or analysis is None:
        return None, error

    stored_encoder, stored_metadata = _fetch_stored_encoder(context, column)
    fit_mode = _determine_fit_mode(context, stored_encoder)

    if fit_mode == "skip_no_training":
        return None, f"{column} (no training data)"

    if fit_mode == "fit":
        result, error = _fit_split_encoder(column, analysis, config, context)
    elif fit_mode == "reuse":
        result, error = _reuse_stored_encoder(column, analysis, config, stored_encoder, stored_metadata, context)
    else:
        result, error = _standard_fit_transform(column, analysis, config)

    if error or result is None:
        return None, error

    try:
        dummy_frame, dummy_columns, includes_missing_dummy, detail = _construct_dummy_frame(
            column,
            analysis,
            config,
            result,
            existing_columns,
        )
    except ValueError as exc:
        return None, f"{column} ({str(exc)})"

    signal_entry = OneHotEncodingAppliedColumnSignal(
        source_column=column,
        dummy_columns=list(dummy_columns),
        replaced_original=config.drop_original,
        category_count=analysis.unique_total,
        includes_missing_dummy=bool(
            (result.include_missing_override if result.include_missing_override is not None else config.include_missing)
            and (result.encoder_has_missing or analysis.has_missing)
            and includes_missing_dummy
        ),
        preview_categories=list(analysis.preview_values),
    )

    plan = OneHotEncodingPlan(
        column=column,
        dummy_frame=dummy_frame,
        dummy_columns=list(dummy_columns),
        drop_original=config.drop_original,
        detail=detail,
        signal_entry=signal_entry,
        store_encoder=result.store_encoder,
        encoder=result.encoder,
        encoder_metadata=result.metadata,
        split_activities=list(result.split_activities),
        drop_first_override=result.drop_first_override,
        include_missing_override=result.include_missing_override,
        prefix_separator_override=result.prefix_separator_override,
        fit_mode=fit_mode,
    )

    return plan, None


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
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = OneHotEncodingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_one_hot_encoding_config(config_payload)

    _initialize_signal_from_config(signal, config)

    if frame.empty:
        signal.evaluated_columns = []
        return frame, "One-hot encoding: no data available", signal

    candidate_columns, skipped_notes = _collect_candidate_columns(frame, config)

    accumulator = OneHotEncodingAccumulator()
    for message in skipped_notes:
        _register_skip(accumulator, signal, message)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "One-hot encoding: no categorical columns selected", signal

    working_frame = frame.copy()
    existing_columns = set(working_frame.columns)
    context = _build_context(working_frame, pipeline_id, str(node_id) if node_id is not None else None)

    signal.evaluated_columns = list(candidate_columns)

    for column in candidate_columns:
        plan, error = _build_column_plan(column, working_frame, config, context, existing_columns)
        if error or plan is None:
            _register_skip(accumulator, signal, error or f"{column} (unavailable)")
            continue

        insert_base = working_frame.columns.get_loc(column)
        insert_at = insert_base if plan.drop_original else insert_base + 1

        for offset, dummy_column in enumerate(plan.dummy_columns):
            working_frame.insert(
                insert_at + offset,
                dummy_column,
                plan.dummy_frame[dummy_column],
            )

        if plan.drop_original and column in working_frame.columns:
            working_frame = working_frame.drop(columns=[column])
            existing_columns.discard(column)

        existing_columns.update(plan.dummy_columns)

        accumulator.encoded_details.append(plan.detail)
        signal.encoded_columns.append(plan.signal_entry)

        if plan.drop_first_override is not None:
            signal.drop_first = plan.drop_first_override
        if plan.include_missing_override is not None:
            signal.include_missing = plan.include_missing_override
        if plan.prefix_separator_override:
            signal.prefix_separator = plan.prefix_separator_override

        _store_encoder(plan, context)
        if not plan.store_encoder:
            _record_split_activities(plan, context)

    summary = _compose_summary(accumulator)

    return working_frame, summary, signal


__all__ = [
    "ONE_HOT_ENCODING_DEFAULT_MAX_CATEGORIES",
    "apply_one_hot_encoding",
]
