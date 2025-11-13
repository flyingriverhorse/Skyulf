"""Target encoding helper utilities for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import math

import pandas as pd
from pandas.api import types as pd_types

from core.feature_engineering.recommendations.categorical import format_category_label
from core.feature_engineering.schemas import (
    TargetEncodingAppliedColumnSignal,
    TargetEncodingCategoryPreview,
    TargetEncodingNodeSignal,
)
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .utils import _auto_detect_text_columns, _coerce_config_boolean, _coerce_string_list

TARGET_ENCODING_DEFAULT_SUFFIX = "_target"
TARGET_ENCODING_DEFAULT_MAX_CATEGORIES = 50
TARGET_ENCODING_MAX_CARDINALITY_LIMIT = 1000
TARGET_ENCODING_DEFAULT_SMOOTHING = 20.0
TARGET_ENCODING_MAX_SMOOTHING = 1_000_000.0
TARGET_ENCODING_HANDLE_UNKNOWN_VALUES = {"global_mean", "error"}
TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN = "global_mean"
TARGET_ENCODING_PLACEHOLDER_TOKEN = "__MISSING__"


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


def _sanitize_mapping_value(raw: Any) -> Optional[float]:
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


@dataclass
class TargetEncodingEnvironment:
    config: NormalizedTargetEncodingConfig
    target_series: pd.Series
    pipeline_id: Optional[str]
    node_id: Optional[str]
    storage: Optional[Any]
    split_counts: Dict[str, int]
    train_mask: Optional[pd.Series]
    train_row_count: int
    global_mean: float


@dataclass
class TargetEncodingColumnContext:
    column: str
    original_series: pd.Series
    string_series: pd.Series
    original_missing_mask: pd.Series
    unique_total: int


@dataclass
class TargetEncodingSettings:
    drop_original: bool
    encode_missing: bool
    handle_unknown: str
    output_suffix: str
    smoothing: float
    global_mean: float
    placeholder_token: str
    encoded_column: Optional[str]
    fit_mode: Literal["fit", "reuse"]
    stored_payload: Optional[Dict[str, Any]]
    stored_metadata: Optional[Dict[str, Any]]


@dataclass
class TargetEncodingMappingResult:
    encoded_series: pd.Series
    mapping_values: Dict[str, Optional[float]]
    category_count: int
    unknown_rows: int
    preview: List[TargetEncodingCategoryPreview]
    fragments: List[str]


@dataclass
class SkipInfo:
    message: str


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

    raw_handle_unknown = str(
        config.get("handle_unknown") or TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN
    ).strip().lower()
    handle_unknown = (
        raw_handle_unknown
        if raw_handle_unknown in TARGET_ENCODING_HANDLE_UNKNOWN_VALUES
        else TARGET_ENCODING_DEFAULT_HANDLE_UNKNOWN
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


def _prepare_target_series(
    frame: pd.DataFrame,
    config: NormalizedTargetEncodingConfig,
    signal: TargetEncodingNodeSignal,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if frame.empty:
        return None, "Target encoding: no data available"

    if not config.target_column:
        return None, "Target encoding: target column not specified"

    if config.target_column not in frame.columns:
        note = f"target column '{config.target_column}' missing"
        signal.skipped_columns.append(note)
        return None, f"Target encoding: {note}"

    target_series = pd.to_numeric(frame[config.target_column], errors="coerce")
    if target_series.notna().sum() == 0:
        signal.skipped_columns.append("target column non-numeric")
        return (
            None,
            f"Target encoding: target column '{config.target_column}' must be numeric",
        )

    return target_series, None


def _build_target_environment(
    frame: pd.DataFrame,
    config: NormalizedTargetEncodingConfig,
    pipeline_id: Optional[str],
    node_id: Optional[str],
    target_series: pd.Series,
) -> Tuple[Optional[TargetEncodingEnvironment], Optional[str]]:
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
            note = f"unable to compute global mean for '{config.target_column}'"
            return None, f"Target encoding: {note}"
        global_mean_value = fallback_mean

    environment = TargetEncodingEnvironment(
        config=config,
        target_series=target_series,
        pipeline_id=pipeline_id,
        node_id=node_id,
        storage=storage,
        split_counts=split_counts,
        train_mask=train_mask,
        train_row_count=train_row_count,
        global_mean=global_mean_value,
    )
    return environment, None


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedTargetEncodingConfig,
) -> Tuple[List[str], List[str]]:
    skipped_notes: List[str] = []
    seen: set[str] = set()
    candidates: List[str] = []
    skipped_configured = set(config.skipped_columns)

    for column in config.columns:
        normalized = str(column or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in skipped_configured:
            skipped_notes.append(f"{normalized} (skipped)")
            continue
        candidates.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_text_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured:
                skipped_notes.append(f"{column} (skipped)")
                continue
            candidates.append(column)

    return candidates, skipped_notes


def _prepare_column_context(
    frame: pd.DataFrame,
    column: str,
    config: NormalizedTargetEncodingConfig,
) -> Tuple[Optional[TargetEncodingColumnContext], Optional[SkipInfo]]:
    if column == config.target_column:
        return None, SkipInfo(f"{column} (target column)")

    if column not in frame.columns:
        return None, SkipInfo(f"{column} (missing)")

    series = frame[column]

    if pd_types.is_bool_dtype(series):
        return None, SkipInfo(f"{column} (boolean column)")

    if not (
        pd_types.is_object_dtype(series)
        or pd_types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None, SkipInfo(f"{column} (non-categorical dtype)")

    string_series = series.astype("string")
    original_missing_mask = string_series.isna()
    unique_total = int(string_series.nunique(dropna=True))

    if unique_total <= 1:
        return None, SkipInfo(f"{column} (single category)")

    if (
        config.auto_detect
        and config.max_categories > 0
        and unique_total > config.max_categories
    ):
        return None, SkipInfo(f"{column} ({unique_total} categories > {config.max_categories})")

    context = TargetEncodingColumnContext(
        column=column,
        original_series=series,
        string_series=string_series,
        original_missing_mask=original_missing_mask,
        unique_total=unique_total,
    )
    return context, None


def _resolve_column_settings(
    env: TargetEncodingEnvironment,
    context: TargetEncodingColumnContext,
    signal: TargetEncodingNodeSignal,
) -> Tuple[Optional[TargetEncodingSettings], Optional[SkipInfo]]:
    config = env.config
    settings = TargetEncodingSettings(
        drop_original=config.drop_original,
        encode_missing=config.encode_missing,
        handle_unknown=config.handle_unknown,
        output_suffix=config.output_suffix,
        smoothing=config.smoothing,
        global_mean=env.global_mean,
        placeholder_token=TARGET_ENCODING_PLACEHOLDER_TOKEN,
        encoded_column=None,
        fit_mode="fit",
        stored_payload=None,
        stored_metadata=None,
    )

    storage = env.storage
    if storage is None:
        return settings, None

    stored_payload = storage.get_transformer(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="target_encoder",
        column_name=context.column,
    )
    stored_metadata = storage.get_metadata(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="target_encoder",
        column_name=context.column,
    )

    settings.stored_payload = stored_payload
    settings.stored_metadata = stored_metadata

    if env.train_row_count <= 0:
        if isinstance(stored_payload, dict):
            settings.fit_mode = "reuse"
        else:
            return None, SkipInfo(f"{context.column} (no training data)")

    if isinstance(stored_metadata, dict):
        raw_drop_original = stored_metadata.get("drop_original")
        if raw_drop_original is not None:
            settings.drop_original = bool(raw_drop_original)
            signal.drop_original = settings.drop_original

        raw_encode_missing = stored_metadata.get("encode_missing")
        if raw_encode_missing is not None:
            settings.encode_missing = bool(raw_encode_missing)
            signal.encode_missing = settings.encode_missing

        raw_handle_unknown = stored_metadata.get("handle_unknown")
        if isinstance(raw_handle_unknown, str) and raw_handle_unknown.strip():
            settings.handle_unknown = raw_handle_unknown.strip()
            signal.handle_unknown = settings.handle_unknown

        raw_output_suffix = stored_metadata.get("output_suffix")
        if isinstance(raw_output_suffix, str) and raw_output_suffix.strip():
            settings.output_suffix = raw_output_suffix.strip()
            signal.output_suffix = settings.output_suffix

        raw_smoothing = stored_metadata.get("smoothing")
        if raw_smoothing is not None:
            try:
                settings.smoothing = float(raw_smoothing)
                signal.smoothing = settings.smoothing
            except (TypeError, ValueError):
                pass

        raw_placeholder = stored_metadata.get("placeholder")
        if isinstance(raw_placeholder, str) and raw_placeholder:
            settings.placeholder_token = raw_placeholder

        raw_global_mean = stored_metadata.get("global_mean")
        if raw_global_mean is not None:
            try:
                settings.global_mean = float(raw_global_mean)
                signal.global_mean = settings.global_mean
            except (TypeError, ValueError):
                pass

        encoded_column_meta = stored_metadata.get("encoded_column")
        if isinstance(encoded_column_meta, str) and encoded_column_meta.strip():
            settings.encoded_column = encoded_column_meta.strip()

    return settings, None


def _map_categories_to_series(
    string_series: pd.Series,
    original_missing_mask: pd.Series,
    mapping_values: Dict[str, Optional[float]],
    settings: TargetEncodingSettings,
) -> Tuple[Optional[pd.Series], Optional[int]]:
    mapping_series = string_series
    if settings.encode_missing:
        mapping_series = mapping_series.fillna(settings.placeholder_token)

    mapped_series = mapping_series.map(mapping_values)
    if mapped_series is None or not isinstance(mapped_series, pd.Series):
        return None, None

    unknown_mask = (~original_missing_mask) & mapped_series.isna()

    if settings.encode_missing:
        mapped_series = mapped_series.where(~original_missing_mask, settings.global_mean)

    if settings.handle_unknown == "global_mean":
        mapped_series = mapped_series.where(~unknown_mask, settings.global_mean)

    encoded_series = pd.to_numeric(mapped_series, errors="coerce").astype("float64")
    if not settings.encode_missing:
        encoded_series = encoded_series.where(~original_missing_mask, math.nan)

    return encoded_series, int(unknown_mask.sum())


def _build_mapping_fragments(
    category_count: int,
    settings: TargetEncodingSettings,
    unknown_rows: int,
) -> List[str]:
    fragments = [
        f"{category_count} level{'s' if category_count != 1 else ''}",
        f"global_mean={settings.global_mean:.4f}",
    ]

    if settings.smoothing > 0.0:
        fragments.append(f"smoothing={settings.smoothing:g}")

    if settings.encode_missing:
        fragments.append("missing→global_mean")

    if settings.handle_unknown == "global_mean":
        fragments.append("unknown→global_mean")
    elif unknown_rows > 0:
        plural = "s" if unknown_rows != 1 else ""
        fragments.append(f"{unknown_rows} unknown row{plural}")

    return fragments


def _build_preview_mapping(
    mapping_values: Dict[str, Optional[float]],
    placeholder_token: str,
) -> List[TargetEncodingCategoryPreview]:
    preview: List[TargetEncodingCategoryPreview] = []
    for raw_category, encoded_value in list(mapping_values.items())[:3]:
        label = "<MISSING>" if raw_category == placeholder_token else format_category_label(raw_category)
        preview.append(
            TargetEncodingCategoryPreview(
                category=label,
                encoded_value=_sanitize_mapping_value(encoded_value),
            )
        )
    return preview


def _apply_stored_mapping(
    context: TargetEncodingColumnContext,
    settings: TargetEncodingSettings,
) -> Tuple[Optional[TargetEncodingMappingResult], Optional[SkipInfo]]:
    stored_payload = settings.stored_payload
    if not isinstance(stored_payload, dict):
        return None, SkipInfo(f"{context.column} (stored transformer missing)")

    mapping_payload = stored_payload.get("mapping")
    if not isinstance(mapping_payload, dict) or not mapping_payload:
        return None, SkipInfo(f"{context.column} (stored mapping empty)")

    mapping_values = {
        str(key): _sanitize_mapping_value(value)
        for key, value in mapping_payload.items()
    }

    if settings.encode_missing and settings.placeholder_token not in mapping_values:
        mapping_values[settings.placeholder_token] = settings.global_mean

    encoded_series, unknown_rows = _map_categories_to_series(
        context.string_series,
        context.original_missing_mask,
        mapping_values,
        settings,
    )

    if encoded_series is None or unknown_rows is None:
        return None, SkipInfo(f"{context.column} (failed to map using stored transformer)")

    metadata = settings.stored_metadata if isinstance(settings.stored_metadata, dict) else {}
    if metadata.get("category_count") is not None:
        try:
            category_count = int(metadata.get("category_count", 0))
        except (TypeError, ValueError):
            category_count = len([
                key for key in mapping_values if key != settings.placeholder_token
            ])
    else:
        category_count = len([key for key in mapping_values if key != settings.placeholder_token])

    fragments = _build_mapping_fragments(category_count, settings, unknown_rows)
    preview = _build_preview_mapping(mapping_values, settings.placeholder_token)

    result = TargetEncodingMappingResult(
        encoded_series=encoded_series,
        mapping_values=mapping_values,
        category_count=category_count,
        unknown_rows=unknown_rows,
        preview=preview,
        fragments=fragments,
    )
    return result, None


def _fit_target_encoding_mapping(
    context: TargetEncodingColumnContext,
    env: TargetEncodingEnvironment,
    settings: TargetEncodingSettings,
) -> Tuple[Optional[TargetEncodingMappingResult], Optional[SkipInfo]]:
    mapping_series = (
        context.string_series.fillna(settings.placeholder_token)
        if settings.encode_missing
        else context.string_series
    )
    numeric_target = env.target_series

    fit_mask = numeric_target.notna()
    if not settings.encode_missing:
        fit_mask &= mapping_series.notna()
    if env.train_mask is not None and env.train_mask.any():
        fit_mask &= env.train_mask

    if not fit_mask.any():
        return None, SkipInfo(f"{context.column} (no rows with numeric target)")

    fit_frame = (
        pd.DataFrame({"category": mapping_series[fit_mask], "target": numeric_target[fit_mask]})
        .dropna(subset=["category", "target"])
    )

    if fit_frame.empty:
        return None, SkipInfo(f"{context.column} (no rows with numeric target)")

    grouped = fit_frame.groupby("category")["target"].agg(["mean", "count"])
    if grouped.empty or grouped.shape[0] <= 1:
        return None, SkipInfo(f"{context.column} (insufficient category diversity)")

    means = grouped["mean"]
    counts = grouped["count"]

    if settings.smoothing > 0.0:
        smoothed = (means * counts + settings.smoothing * settings.global_mean) / (
            counts + settings.smoothing
        )
    else:
        smoothed = means

    mapping_values = {
        str(category): _sanitize_mapping_value(value)
        for category, value in smoothed.to_dict().items()
    }

    if settings.encode_missing and settings.placeholder_token not in mapping_values:
        mapping_values[settings.placeholder_token] = settings.global_mean

    encoded_series, unknown_rows = _map_categories_to_series(
        context.string_series,
        context.original_missing_mask,
        mapping_values,
        settings,
    )

    if encoded_series is None or unknown_rows is None:
        return None, SkipInfo(f"{context.column} (failed to map categories)")

    category_count = grouped.shape[0]
    fragments = _build_mapping_fragments(category_count, settings, unknown_rows)
    preview = _build_preview_mapping(mapping_values, settings.placeholder_token)

    result = TargetEncodingMappingResult(
        encoded_series=encoded_series,
        mapping_values=mapping_values,
        category_count=category_count,
        unknown_rows=unknown_rows,
        preview=preview,
        fragments=fragments,
    )
    return result, None


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


def _apply_encoded_column(
    working_frame: pd.DataFrame,
    existing_columns: set[str],
    context: TargetEncodingColumnContext,
    settings: TargetEncodingSettings,
    mapping_result: TargetEncodingMappingResult,
) -> Tuple[str, bool, str]:
    if settings.drop_original:
        working_frame[context.column] = mapping_result.encoded_series
        existing_columns.add(context.column)
        detail = f"{context.column} (replaced; {'; '.join(mapping_result.fragments)})"
        return context.column, True, detail

    encoded_column_name = (
        settings.encoded_column
        or _resolve_encoded_column_name(context.column, settings.output_suffix, existing_columns)
    )

    insert_at = working_frame.columns.get_loc(context.column) + 1
    if encoded_column_name in working_frame.columns:
        working_frame[encoded_column_name] = mapping_result.encoded_series
    else:
        working_frame.insert(insert_at, encoded_column_name, mapping_result.encoded_series)
    existing_columns.add(encoded_column_name)

    detail = f"{context.column}→{encoded_column_name} ({'; '.join(mapping_result.fragments)})"
    return encoded_column_name, False, detail


def _persist_target_encoder(
    env: TargetEncodingEnvironment,
    context: TargetEncodingColumnContext,
    settings: TargetEncodingSettings,
    mapping_result: TargetEncodingMappingResult,
    encoded_column: str,
    replaced_original: bool,
) -> None:
    storage = env.storage
    if storage is None or not env.pipeline_id or not env.node_id:
        return

    if settings.fit_mode == "fit":
        transformer_payload = {
            "mapping": {
                key: (_sanitize_mapping_value(value) if value is not None else None)
                for key, value in mapping_result.mapping_values.items()
            },
            "placeholder": settings.placeholder_token if settings.encode_missing else None,
            "global_mean": float(settings.global_mean),
        }

        method_parts = ["Target Encoding (smoothed mean"]
        if settings.smoothing > 0.0:
            method_parts.append(f", smoothing={settings.smoothing:g}")
        method_parts.append(")")
        method_label = "".join(method_parts)

        metadata: Dict[str, Any] = {
            "encoded_column": encoded_column,
            "replaced_original": replaced_original,
            "category_count": int(mapping_result.category_count),
            "global_mean": float(settings.global_mean),
            "smoothing": float(settings.smoothing),
            "encode_missing": bool(settings.encode_missing),
            "handle_unknown": settings.handle_unknown,
            "drop_original": bool(settings.drop_original),
            "output_suffix": settings.output_suffix,
            "placeholder": settings.placeholder_token if settings.encode_missing else None,
            "target_column": env.config.target_column,
            "train_rows": env.train_row_count,
            "method_label": method_label,
        }

        storage.store_transformer(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name="target_encoder",
            transformer=transformer_payload,
            column_name=context.column,
            metadata=metadata,
        )

        storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name="target_encoder",
            column_name=context.column,
            split_name="train",
            action="fit_transform",
            row_count=env.train_row_count,
        )

        for split_name in ("test", "validation"):
            rows_processed = int(env.split_counts.get(split_name, 0))
            storage.record_split_activity(
                pipeline_id=env.pipeline_id,
                node_id=env.node_id,
                transformer_name="target_encoder",
                column_name=context.column,
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )
    else:
        storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name="target_encoder",
            column_name=context.column,
            split_name="train",
            action="not_available",
            row_count=env.train_row_count,
        )

        for split_name in ("test", "validation"):
            rows_processed = int(env.split_counts.get(split_name, 0))
            storage.record_split_activity(
                pipeline_id=env.pipeline_id,
                node_id=env.node_id,
                transformer_name="target_encoder",
                column_name=context.column,
                split_name=split_name,
                action="transform" if rows_processed > 0 else "not_available",
                row_count=rows_processed,
            )


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

    target_series, target_error = _prepare_target_series(frame, config, signal)
    if target_error:
        return frame, target_error, signal

    environment, env_error = _build_target_environment(
        frame,
        config,
        pipeline_id,
        str(node_id) if node_id is not None else None,
        target_series,
    )
    if env_error:
        signal.skipped_columns.append("unable to compute global mean")
        return frame, env_error, signal

    if environment is None:
        fallback_message = "Target encoding: environment unavailable"
        signal.skipped_columns.append("environment unavailable")
        return frame, fallback_message, signal

    signal.global_mean = environment.global_mean

    candidate_columns, skipped_notes = _collect_candidate_columns(frame, config)
    if skipped_notes:
        signal.skipped_columns.extend(skipped_notes)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Target encoding: no categorical columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    existing_columns = set(working_frame.columns)
    encoded_details: List[str] = []
    skipped_details: List[str] = list(skipped_notes)

    for column in candidate_columns:
        context, skip_info = _prepare_column_context(working_frame, column, config)
        if skip_info is not None:
            skipped_details.append(skip_info.message)
            signal.skipped_columns.append(skip_info.message)
            continue

        if context is None:
            message = f"{column} (context unavailable)"
            skipped_details.append(message)
            signal.skipped_columns.append(message)
            continue

        settings, settings_skip = _resolve_column_settings(environment, context, signal)
        if settings_skip is not None or settings is None:
            message = settings_skip.message if settings_skip else f"{column} (unable to resolve settings)"
            skipped_details.append(message)
            signal.skipped_columns.append(message)
            continue

        if settings.fit_mode == "reuse":
            mapping_result, mapping_skip = _apply_stored_mapping(context, settings)
        else:
            mapping_result, mapping_skip = _fit_target_encoding_mapping(context, environment, settings)

        if mapping_skip is not None or mapping_result is None:
            message = mapping_skip.message if mapping_skip else f"{column} (failed to encode)"
            skipped_details.append(message)
            signal.skipped_columns.append(message)
            continue

        encoded_column_name, replaced_original, detail = _apply_encoded_column(
            working_frame,
            existing_columns,
            context,
            settings,
            mapping_result,
        )

        encoded_details.append(detail)
        signal.encoded_columns.append(
            TargetEncodingAppliedColumnSignal(
                source_column=context.column,
                encoded_column=encoded_column_name,
                replaced_original=replaced_original,
                category_count=int(mapping_result.category_count),
                global_mean=float(settings.global_mean),
                smoothing=float(settings.smoothing),
                encode_missing=bool(settings.encode_missing),
                handle_unknown=settings.handle_unknown,
                unknown_rows=int(mapping_result.unknown_rows),
                preview=mapping_result.preview,
            )
        )

        _persist_target_encoder(
            environment,
            context,
            settings,
            mapping_result,
            encoded_column_name,
            replaced_original,
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
