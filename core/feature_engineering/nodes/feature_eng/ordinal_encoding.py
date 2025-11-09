"""Ordinal encoding helper utilities for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    raw_handle_unknown = str(
        config.get("handle_unknown") or ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN
    ).strip().lower()
    handle_unknown = (
        raw_handle_unknown
        if raw_handle_unknown in ORDINAL_ENCODING_HANDLE_UNKNOWN_VALUES
        else ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN
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


def _should_use_unknown_value(encode_missing: bool, handle_unknown: str) -> bool:
    return encode_missing or handle_unknown == "use_encoded_value"


@dataclass
class OrdinalEncodingContext:
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
class OrdinalEncodingAccumulator:
    encoded_details: List[str] = field(default_factory=list)
    skipped_details: List[str] = field(default_factory=list)


@dataclass
class OrdinalColumnAnalysis:
    series: pd.Series
    string_series: pd.Series
    unique_total: int


@dataclass
class OrdinalEncodingSettings:
    drop_original: bool
    encode_missing: bool
    handle_unknown: str
    output_suffix: str
    unknown_value: int
    stored_encoded_column: Optional[str]


@dataclass
class OrdinalEncodingPlan:
    column: str
    encoded_series: pd.Series
    encoded_column_name: str
    replaced_original: bool
    detail: str
    signal_entry: OrdinalEncodingAppliedColumnSignal
    settings: OrdinalEncodingSettings
    store_transformer: bool
    transformer_payload: Optional[Dict[str, Any]]
    transformer_metadata: Optional[Dict[str, Any]]
    split_activities: List[SplitActivity]
    fit_mode: str


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


def _initialize_signal_from_config(
    signal: OrdinalEncodingNodeSignal,
    config: NormalizedOrdinalEncodingConfig,
) -> None:
    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.max_categories = config.max_categories
    signal.output_suffix = config.output_suffix
    signal.drop_original = config.drop_original
    signal.encode_missing = config.encode_missing
    signal.handle_unknown = config.handle_unknown
    signal.unknown_value = (
        config.unknown_value
        if _should_use_unknown_value(config.encode_missing, config.handle_unknown)
        else None
    )


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedOrdinalEncodingConfig,
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

    return candidate_columns, skipped_notes


def _build_context(
    frame: pd.DataFrame,
    pipeline_id: Optional[str],
    node_id: Optional[str],
) -> OrdinalEncodingContext:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_splits else {}
    train_row_count = int(split_counts.get("train", 0)) if has_splits else 0
    train_mask = (frame[SPLIT_TYPE_COLUMN] == "train") if has_splits else None

    return OrdinalEncodingContext(
        storage=storage,
        pipeline_id=pipeline_id,
        node_id=node_id,
        has_splits=has_splits,
        train_mask=train_mask,
        split_counts=split_counts,
        train_row_count=train_row_count,
    )


def _register_skip(
    accumulator: OrdinalEncodingAccumulator,
    signal: OrdinalEncodingNodeSignal,
    message: str,
) -> None:
    accumulator.skipped_details.append(message)
    signal.skipped_columns.append(message)


def _compose_summary(accumulator: OrdinalEncodingAccumulator) -> str:
    if not accumulator.encoded_details:
        summary = "Ordinal encoding: no columns encoded"
    else:
        preview = ", ".join(accumulator.encoded_details[:3])
        if len(accumulator.encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Ordinal encoding: encoded {len(accumulator.encoded_details)} column(s) ({preview})"

    if accumulator.skipped_details:
        preview = ", ".join(accumulator.skipped_details[:3])
        if len(accumulator.skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return summary


def _analyze_column(
    frame: pd.DataFrame,
    column: str,
    config: NormalizedOrdinalEncodingConfig,
) -> Tuple[Optional[OrdinalColumnAnalysis], Optional[str]]:
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

    string_series = series.astype("string")
    unique_total = string_series.nunique(dropna=True)

    if unique_total <= 1:
        return None, f"{column} (single category)"

    if (
        config.auto_detect
        and config.max_categories > 0
        and unique_total > config.max_categories
    ):
        return None, f"{column} ({unique_total} categories > {config.max_categories})"

    return OrdinalColumnAnalysis(series=series, string_series=string_series, unique_total=int(unique_total)), None


def _fetch_stored_transformer(
    context: OrdinalEncodingContext,
    column: str,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not context.storage:
        return None, None
    payload = context.storage.get_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="ordinal_encoder",
        column_name=column,
    )
    metadata = context.storage.get_metadata(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="ordinal_encoder",
        column_name=column,
    )
    return payload, metadata


def _determine_fit_mode(
    context: OrdinalEncodingContext,
    stored_payload: Optional[Any],
) -> str:
    if not context.storage or not context.has_splits or not context.pipeline_id:
        return "standard"
    if context.train_row_count > 0:
        return "fit"
    if stored_payload is not None:
        return "reuse"
    return "skip_no_training"


def _resolve_effective_settings(
    config: NormalizedOrdinalEncodingConfig,
    stored_metadata: Optional[Dict[str, Any]],
) -> OrdinalEncodingSettings:
    drop_original = config.drop_original
    encode_missing = config.encode_missing
    handle_unknown = config.handle_unknown
    output_suffix = config.output_suffix
    unknown_value = config.unknown_value
    stored_encoded_column: Optional[str] = None

    if isinstance(stored_metadata, dict):
        raw_drop_original = stored_metadata.get("drop_original")
        if raw_drop_original is not None:
            drop_original = bool(raw_drop_original)
        raw_encode_missing = stored_metadata.get("encode_missing")
        if raw_encode_missing is not None:
            encode_missing = bool(raw_encode_missing)
        raw_handle_unknown = stored_metadata.get("handle_unknown")
        if isinstance(raw_handle_unknown, str) and raw_handle_unknown.strip():
            handle_unknown = raw_handle_unknown.strip()
        raw_output_suffix = stored_metadata.get("output_suffix")
        if isinstance(raw_output_suffix, str) and raw_output_suffix.strip():
            output_suffix = raw_output_suffix.strip()
        raw_unknown_value = stored_metadata.get("unknown_value")
        if raw_unknown_value is not None:
            try:
                unknown_value = int(raw_unknown_value)
            except (TypeError, ValueError):
                unknown_value = config.unknown_value
        encoded_column_meta = stored_metadata.get("encoded_column")
        if isinstance(encoded_column_meta, str) and encoded_column_meta.strip():
            stored_encoded_column = encoded_column_meta.strip()

    return OrdinalEncodingSettings(
        drop_original=drop_original,
        encode_missing=encode_missing,
        handle_unknown=handle_unknown,
        output_suffix=output_suffix,
        unknown_value=unknown_value,
        stored_encoded_column=stored_encoded_column,
    )


def _prepare_reuse_mapping(
    stored_payload: Any,
) -> Tuple[Optional[Dict[str, int]], List[str], Optional[str]]:
    if not isinstance(stored_payload, dict):
        return None, [], "stored transformer missing"

    raw_mapping = stored_payload.get("mapping")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        return None, [], "stored mapping empty"

    mapping: Dict[str, int] = {}
    for key, value in raw_mapping.items():
        if key is None or value is None:
            continue
        try:
            mapping[str(key)] = int(value)
        except (TypeError, ValueError):
            continue

    if not mapping:
        return None, [], "stored mapping invalid"

    raw_categories = stored_payload.get("categories")
    categories: List[str] = []
    if isinstance(raw_categories, list):
        categories = [str(item) for item in raw_categories if str(item).strip()]

    return mapping, categories, None


def _fit_mapping(
    column: str,
    analysis: OrdinalColumnAnalysis,
    settings: OrdinalEncodingSettings,
    context: OrdinalEncodingContext,
    *,
    use_training_only: bool,
) -> Tuple[Optional[Dict[str, int]], List[str], Optional[str]]:
    trainer_series = analysis.string_series
    if use_training_only and context.train_mask is not None and context.train_mask.any():
        trainer_series = trainer_series[context.train_mask]

    if not settings.encode_missing:
        trainer_series = trainer_series.dropna()

    trainer_series = trainer_series.dropna()

    if trainer_series.nunique(dropna=True) <= 1:
        return None, [], "insufficient training categories"

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
    return mapping_values, categories, None


def _build_transformer_artifacts(
    column: str,
    encoded_column_name: str,
    categories: List[str],
    mapping: Dict[str, int],
    settings: OrdinalEncodingSettings,
    replaced_original: bool,
    context: OrdinalEncodingContext,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = {
        "mapping": {str(key): int(value) for key, value in mapping.items()},
        "categories": [str(value) for value in categories],
    }

    metadata: Dict[str, Any] = {
        "encoded_column": encoded_column_name,
        "replaced_original": replaced_original,
        "category_count": len(categories),
        "encode_missing": settings.encode_missing,
        "handle_unknown": settings.handle_unknown,
        "unknown_value": (
            settings.unknown_value
            if _should_use_unknown_value(settings.encode_missing, settings.handle_unknown)
            else None
        ),
        "drop_original": settings.drop_original,
        "output_suffix": settings.output_suffix,
        "categories": [str(value) for value in categories],
        "train_rows": context.train_row_count,
    }

    level_count = len(categories)
    metadata["method"] = "ordinal_encoding"
    metadata["method_label"] = (
        f"Ordinal Encoding ({level_count} level{'s' if level_count != 1 else ''})"
    )

    return payload, metadata


def _record_split_activities(plan: OrdinalEncodingPlan, context: OrdinalEncodingContext) -> None:
    if not context.storage:
        return
    for activity in plan.split_activities:
        context.storage.record_split_activity(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id or "",
            transformer_name="ordinal_encoder",
            column_name=plan.column,
            split_name=activity.split_name,
            action=activity.action,
            row_count=activity.row_count,
        )


def _store_transformer(plan: OrdinalEncodingPlan, context: OrdinalEncodingContext) -> None:
    if not context.storage or not plan.store_transformer or plan.transformer_payload is None:
        return
    context.storage.store_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="ordinal_encoder",
        transformer=plan.transformer_payload,
        column_name=plan.column,
        metadata=plan.transformer_metadata or {},
    )
    _record_split_activities(plan, context)


def _resolve_reuse_categories(
    categories: List[str],
    stored_metadata: Optional[Dict[str, Any]],
    mapping_values: Dict[str, int],
) -> List[str]:
    resolved = [str(value) for value in categories if str(value).strip()]
    if isinstance(stored_metadata, dict):
        raw_categories = stored_metadata.get("categories")
        if isinstance(raw_categories, list):
            resolved = [str(value) for value in raw_categories if str(value).strip()]
    if not resolved:
        resolved = [key for key in mapping_values.keys() if key is not None]
    return resolved


def _resolve_mapping_values(
    column: str,
    analysis: OrdinalColumnAnalysis,
    settings: OrdinalEncodingSettings,
    context: OrdinalEncodingContext,
    fit_mode: str,
    stored_payload: Optional[Dict[str, Any]],
    stored_metadata: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, int]], List[str], Optional[str]]:
    if fit_mode == "reuse":
        mapping_values, categories, reuse_error = _prepare_reuse_mapping(stored_payload)
        if reuse_error or mapping_values is None:
            return None, [], reuse_error or None
        categories = _resolve_reuse_categories(categories, stored_metadata, mapping_values)
        return mapping_values, categories, None

    use_training_only = fit_mode == "fit"
    mapping_values, categories, fit_error = _fit_mapping(
        column,
        analysis,
        settings,
        context,
        use_training_only=use_training_only,
    )
    if fit_error or mapping_values is None:
        return None, [], fit_error or None
    return mapping_values, categories, None


def _determine_encoded_column_details(
    column: str,
    settings: OrdinalEncodingSettings,
    existing_columns: set[str],
) -> Tuple[str, bool]:
    if settings.drop_original:
        return column, True
    encoded_column_name = (
        settings.stored_encoded_column
        or _resolve_encoded_column_name(column, settings.output_suffix, existing_columns)
    )
    return encoded_column_name, False


def _build_category_preview_entries(categories: List[str]) -> List[OrdinalEncodingCategoryPreview]:
    entries: List[OrdinalEncodingCategoryPreview] = []
    for index, value in enumerate(categories[:3]):
        entries.append(
            OrdinalEncodingCategoryPreview(
                category=format_category_label(value),
                code=index,
            )
        )
    return entries


def _build_detail_fragments(
    class_count: int,
    sample_preview: str,
    settings: OrdinalEncodingSettings,
    unknown_rows: int,
) -> List[str]:
    fragments = [f"{class_count} level{'s' if class_count != 1 else ''}"]
    if sample_preview:
        fragments.append(sample_preview)
    if settings.encode_missing:
        fragments.append(f"missing→{settings.unknown_value}")
    if settings.handle_unknown == "use_encoded_value":
        fragments.append(f"unknown→{settings.unknown_value}")
    elif unknown_rows > 0:
        fragments.append(f"{unknown_rows} unknown row{'s' if unknown_rows != 1 else ''}")
    return fragments


def _build_signal_entry(
    column: str,
    encoded_column_name: str,
    replaced_original: bool,
    class_count: int,
    settings: OrdinalEncodingSettings,
    preview_entries: List[OrdinalEncodingCategoryPreview],
) -> OrdinalEncodingAppliedColumnSignal:
    return OrdinalEncodingAppliedColumnSignal(
        source_column=column,
        encoded_column=None if replaced_original else encoded_column_name,
        replaced_original=replaced_original,
        category_count=class_count,
        handle_unknown=settings.handle_unknown,
        unknown_value=(
            settings.unknown_value
            if _should_use_unknown_value(settings.encode_missing, settings.handle_unknown)
            else None
        ),
        encode_missing=settings.encode_missing,
        preview=preview_entries,
    )


def _build_split_activities_for_plan(
    context: OrdinalEncodingContext,
    fit_mode: str,
) -> List[SplitActivity]:
    if not context.storage or not context.has_splits:
        return []

    activities: List[SplitActivity] = []
    if fit_mode == "fit":
        activities.append(
            SplitActivity(split_name="train", action="fit_transform", row_count=context.train_row_count)
        )
        for split_name in ("test", "validation"):
            rows_processed = int(context.split_counts.get(split_name, 0))
            activities.append(
                SplitActivity(
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )
            )
        return activities

    if fit_mode == "reuse":
        activities.append(
            SplitActivity(
                split_name="train",
                action="not_available",
                row_count=context.train_row_count,
            )
        )
        for split_name in ("test", "validation"):
            rows_processed = int(context.split_counts.get(split_name, 0))
            activities.append(
                SplitActivity(
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )
            )
    return activities


def _build_column_plan(
    column: str,
    frame: pd.DataFrame,
    config: NormalizedOrdinalEncodingConfig,
    context: OrdinalEncodingContext,
    existing_columns: set[str],
) -> Tuple[Optional[OrdinalEncodingPlan], Optional[str]]:
    analysis, error = _analyze_column(frame, column, config)
    if error or analysis is None:
        return None, error

    stored_payload, stored_metadata = _fetch_stored_transformer(context, column)
    fit_mode = _determine_fit_mode(context, stored_payload)

    if fit_mode == "skip_no_training":
        return None, f"{column} (no training data)"

    settings = _resolve_effective_settings(
        config,
        stored_metadata if fit_mode == "reuse" else None,
    )

    categories: List[str]
    mapping_values: Optional[Dict[str, int]]

    mapping_values, categories, mapping_error = _resolve_mapping_values(
        column,
        analysis,
        settings,
        context,
        fit_mode,
        stored_payload,
        stored_metadata,
    )
    if mapping_error or mapping_values is None:
        return None, f"{column} ({mapping_error})" if mapping_error else f"{column} (unavailable)"

    encoded_column_name, replaced_original = _determine_encoded_column_details(
        column,
        settings,
        existing_columns,
    )

    encoded_series, unknown_rows = _transform_with_mapping(
        analysis.string_series,
        mapping_values,
        encode_missing=settings.encode_missing,
        handle_unknown=settings.handle_unknown,
        unknown_value=settings.unknown_value,
    )

    class_count = len(categories)
    sample_preview = _build_category_preview(categories)

    fragments = _build_detail_fragments(class_count, sample_preview, settings, unknown_rows)
    if replaced_original:
        detail = f"{column} (replaced; {'; '.join(fragments)})"
    else:
        detail = f"{column}->{encoded_column_name} ({'; '.join(fragments)})"

    preview_entries = _build_category_preview_entries(categories)

    signal_entry = _build_signal_entry(
        column,
        encoded_column_name,
        replaced_original,
        class_count,
        settings,
        preview_entries,
    )

    store_transformer = context.storage is not None and fit_mode == "fit"
    transformer_payload: Optional[Dict[str, Any]] = None
    transformer_metadata: Optional[Dict[str, Any]] = None

    if store_transformer:
        transformer_payload, transformer_metadata = _build_transformer_artifacts(
            column,
            encoded_column_name,
            categories,
            mapping_values,
            settings,
            replaced_original,
            context,
        )

    split_activities = _build_split_activities_for_plan(context, fit_mode)

    plan = OrdinalEncodingPlan(
        column=column,
        encoded_series=encoded_series,
        encoded_column_name=encoded_column_name,
        replaced_original=replaced_original,
        detail=detail,
        signal_entry=signal_entry,
        settings=settings,
        store_transformer=store_transformer,
        transformer_payload=transformer_payload,
        transformer_metadata=transformer_metadata,
        split_activities=split_activities,
        fit_mode=fit_mode,
    )

    return plan, None


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

    _initialize_signal_from_config(signal, config)

    candidate_columns, skipped_notes = _collect_candidate_columns(frame, config)

    accumulator = OrdinalEncodingAccumulator()
    for note in skipped_notes:
        _register_skip(accumulator, signal, note)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Ordinal encoding: no categorical columns selected", signal

    working_frame = frame.copy()
    existing_columns = set(working_frame.columns)
    context = _build_context(working_frame, pipeline_id, str(node_id) if node_id is not None else None)

    signal.evaluated_columns = list(candidate_columns)

    for column in candidate_columns:
        plan, error = _build_column_plan(column, working_frame, config, context, existing_columns)
        if error or plan is None:
            _register_skip(accumulator, signal, error or f"{column} (unavailable)")
            continue

        if plan.replaced_original:
            working_frame[plan.column] = plan.encoded_series
            existing_columns.add(plan.column)
        else:
            insert_at = working_frame.columns.get_loc(plan.column) + 1
            if plan.encoded_column_name in working_frame.columns:
                working_frame[plan.encoded_column_name] = plan.encoded_series
            else:
                working_frame.insert(insert_at, plan.encoded_column_name, plan.encoded_series)
            existing_columns.add(plan.encoded_column_name)

        accumulator.encoded_details.append(plan.detail)
        signal.encoded_columns.append(plan.signal_entry)

        signal.drop_original = plan.settings.drop_original
        signal.encode_missing = plan.settings.encode_missing
        signal.handle_unknown = plan.settings.handle_unknown
        signal.output_suffix = plan.settings.output_suffix
        signal.unknown_value = (
            plan.settings.unknown_value
            if _should_use_unknown_value(plan.settings.encode_missing, plan.settings.handle_unknown)
            else None
        )

        if plan.store_transformer:
            _store_transformer(plan, context)
        else:
            _record_split_activities(plan, context)

    summary = _compose_summary(accumulator)

    return working_frame, summary, signal


__all__ = [
    "ORDINAL_ENCODING_DEFAULT_SUFFIX",
    "ORDINAL_ENCODING_DEFAULT_MAX_CATEGORIES",
    "ORDINAL_ENCODING_MAX_CARDINALITY_LIMIT",
    "ORDINAL_ENCODING_DEFAULT_HANDLE_UNKNOWN",
    "ORDINAL_ENCODING_DEFAULT_UNKNOWN_VALUE",
    "apply_ordinal_encoding",
]
