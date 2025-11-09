"""Label encoding helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import logging
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

# Module logger
logger = logging.getLogger(__name__)


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


@dataclass
class LabelEncodingContext:
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    has_splits: bool
    train_mask: Optional[pd.Series]
    split_counts: Dict[str, int]
    train_row_count: int


@dataclass
class LabelEncodingAccumulator:
    encoded_details: List[str] = field(default_factory=list)
    skipped_details: List[str] = field(default_factory=list)


@dataclass
class ColumnEncodingPlan:
    column: str
    encoded_series: pd.Series
    encoded_column: str
    replaced_original: bool
    categories: List[Any]
    category_labels: List[str]
    category_mapping: Dict[str, int]
    missing_strategy: str
    missing_code: Optional[int]
    class_count: int
    detail: str
    preview_entries: List[LabelEncodingCategoryPreview]
    fit_mode: str
    effective_drop_original: bool


@dataclass
class _EncodingResources:
    encoded_series: pd.Series
    categories: List[Any]
    category_labels: List[str]
    category_mapping: Dict[str, int]
    missing_strategy: str
    missing_code: Optional[int]
    class_count: int
    preview: str
    fit_mode: str
    metadata_drop_original: Optional[bool]
    stored_metadata: Optional[Dict[str, Any]]


@dataclass
class _PreparedEncoding:
    categories: List[Any]
    category_labels: List[str]
    category_mapping: Dict[str, int]
    missing_strategy: str
    missing_code: Optional[int]
    metadata_drop_original: Optional[bool]


def _initialize_signal_from_config(
    signal: LabelEncodingNodeSignal,
    config: NormalizedLabelEncodingConfig,
) -> None:
    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.drop_original = config.drop_original
    signal.output_suffix = config.output_suffix
    signal.max_unique_values = config.max_unique_values
    signal.missing_strategy = config.missing_strategy
    signal.missing_code = config.missing_code if config.missing_strategy == "encode" else None


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedLabelEncodingConfig,
) -> Tuple[List[str], List[str]]:
    skipped_configured = set(config.skipped_columns)
    candidate_columns: List[str] = []
    seen: Set[str] = set()
    skipped_notes: List[str] = []

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
) -> LabelEncodingContext:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_splits else {}
    train_row_count = int(split_counts.get("train", 0)) if has_splits else 0
    train_mask = (frame[SPLIT_TYPE_COLUMN] == "train") if has_splits else None

    return LabelEncodingContext(
        storage=storage,
        pipeline_id=pipeline_id,
        node_id=node_id,
        has_splits=has_splits,
        train_mask=train_mask,
        split_counts=split_counts,
        train_row_count=train_row_count,
    )


def _register_skip(
    accumulator: LabelEncodingAccumulator,
    signal: LabelEncodingNodeSignal,
    message: str,
) -> None:
    accumulator.skipped_details.append(message)
    signal.skipped_columns.append(message)


def _fetch_stored_encoder(
    context: LabelEncodingContext,
    column: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not context.storage:
        return None, None
    payload = context.storage.get_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="label_encoder",
        column_name=column,
    )
    metadata = context.storage.get_metadata(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="label_encoder",
        column_name=column,
    )
    return payload, metadata


def _determine_fit_mode(
    context: LabelEncodingContext,
    stored_payload: Optional[Dict[str, Any]],
) -> str:
    if not context.storage or not context.has_splits:
        return "fit"
    if context.train_row_count > 0:
        return "fit"
    if isinstance(stored_payload, dict):
        return "reuse"
    return "skip_no_training"


def _safe_string_mapping(raw_mapping: Dict[Any, Any]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for key, value in raw_mapping.items():
        try:
            mapping[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _validate_column_for_encoding(
    column: str,
    frame: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Optional[str]]:
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

    return series.astype("string"), None


def _prepare_fit_encoding(
    column: str,
    string_series: pd.Series,
    config: NormalizedLabelEncodingConfig,
    context: LabelEncodingContext,
) -> Tuple[Optional[_PreparedEncoding], Optional[str]]:
    analysis_series: pd.Series = string_series
    if context.has_splits and context.train_mask is not None:
        analysis_series = string_series[context.train_mask]

    unique_total = analysis_series.nunique(dropna=False)
    if unique_total <= 1:
        return None, f"{column} (single category)"

    if (
        config.max_unique_values > 0
        and config.auto_detect
        and unique_total > config.max_unique_values
    ):
        return None, f"{column} ({unique_total} categories > {config.max_unique_values})"

    categorical = pd.Categorical(analysis_series, ordered=False)
    categories = list(categorical.categories)
    if not categories:
        return None, f"{column} (no categories)"

    category_labels = [str(value) for value in categories]
    category_mapping = {label: index for index, label in enumerate(category_labels)}
    missing_code = config.missing_code if config.missing_strategy == "encode" else None

    prepared = _PreparedEncoding(
        categories=categories,
        category_labels=category_labels,
        category_mapping=category_mapping,
        missing_strategy=config.missing_strategy,
        missing_code=missing_code,
        metadata_drop_original=None,
    )
    return prepared, None


def _coerce_missing_code(candidate: Any) -> Optional[int]:
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, float) and candidate.is_integer():
        return int(candidate)
    if isinstance(candidate, str):
        stripped_code = candidate.strip()
        if stripped_code and stripped_code.lstrip("-").isdigit():
            try:
                return int(stripped_code)
            except ValueError:
                return None
    return None


def _prepare_reuse_encoding(
    column: str,
    stored_payload: Optional[Dict[str, Any]],
    stored_metadata: Any,
    config: NormalizedLabelEncodingConfig,
) -> Tuple[Optional[_PreparedEncoding], Optional[str]]:
    if not isinstance(stored_payload, dict):
        return None, f"{column} (stored transformer unavailable)"

    stored_mapping = stored_payload.get("mapping")
    stored_categories = stored_payload.get("categories")
    if not isinstance(stored_mapping, dict) or stored_categories is None:
        return None, f"{column} (stored transformer invalid)"

    category_mapping = _safe_string_mapping(stored_mapping)
    if not category_mapping:
        return None, f"{column} (stored mapping empty)"

    categories = list(stored_categories)
    category_labels = [str(value) for value in categories]

    stored_missing_strategy = stored_payload.get("missing_strategy")
    if (
        isinstance(stored_missing_strategy, str)
        and stored_missing_strategy in LABEL_ENCODING_MISSING_STRATEGIES
    ):
        missing_strategy = stored_missing_strategy
    else:
        missing_strategy = "keep_na"

    missing_code = None
    if missing_strategy == "encode":
        missing_code_candidate = _coerce_missing_code(stored_payload.get("missing_code"))
        missing_code = (
            missing_code_candidate if missing_code_candidate is not None else config.missing_code
        )

    metadata_drop_original: Optional[bool] = None
    if isinstance(stored_metadata, dict):
        raw_drop = stored_metadata.get("replaced_original")
        if raw_drop is not None:
            metadata_drop_original = bool(raw_drop)

    prepared = _PreparedEncoding(
        categories=categories,
        category_labels=category_labels,
        category_mapping=category_mapping,
        missing_strategy=missing_strategy,
        missing_code=missing_code,
        metadata_drop_original=metadata_drop_original,
    )
    return prepared, None


def _prepare_column_encoding_resources(
    column: str,
    string_series: pd.Series,
    config: NormalizedLabelEncodingConfig,
    context: LabelEncodingContext,
) -> Tuple[Optional[_EncodingResources], Optional[str]]:
    stored_payload, stored_metadata = _fetch_stored_encoder(context, column)
    fit_mode = _determine_fit_mode(context, stored_payload)

    if fit_mode == "skip_no_training":
        return None, f"{column} (no training data)"

    if fit_mode == "fit":
        prepared, error = _prepare_fit_encoding(column, string_series, config, context)
    else:
        prepared, error = _prepare_reuse_encoding(column, stored_payload, stored_metadata, config)

    if error:
        return None, error
    assert prepared is not None

    missing_strategy = prepared.missing_strategy
    missing_code = prepared.missing_code
    category_mapping = prepared.category_mapping

    def _encode_value(value: Any) -> Any:
        if pd.isna(value):
            return missing_code if missing_strategy == "encode" else pd.NA
        lookup_key = str(value)
        code = category_mapping.get(lookup_key)
        if code is None:
            return missing_code if missing_strategy == "encode" else pd.NA
        return code

    encoded_series = string_series.apply(_encode_value).astype("Int64")

    resources = _EncodingResources(
        encoded_series=encoded_series,
        categories=prepared.categories,
        category_labels=prepared.category_labels,
        category_mapping=prepared.category_mapping,
        missing_strategy=prepared.missing_strategy,
        missing_code=prepared.missing_code,
        class_count=len(prepared.category_labels),
        preview=_build_category_preview(prepared.categories),
        fit_mode=fit_mode,
        metadata_drop_original=prepared.metadata_drop_original,
        stored_metadata=stored_metadata if isinstance(stored_metadata, dict) else None,
    )

    return resources, None


def _build_column_plan(
    column: str,
    frame: pd.DataFrame,
    config: NormalizedLabelEncodingConfig,
    context: LabelEncodingContext,
    signal: LabelEncodingNodeSignal,
    existing_columns: Set[str],
) -> Tuple[Optional[ColumnEncodingPlan], Optional[str]]:
    string_series, error = _validate_column_for_encoding(column, frame)
    if error:
        return None, error
    assert string_series is not None

    resources, error = _prepare_column_encoding_resources(column, string_series, config, context)
    if error:
        return None, error
    assert resources is not None

    class_label = "class" if resources.class_count == 1 else "classes"
    effective_drop_original = config.drop_original
    if (
        resources.metadata_drop_original is not None
        and resources.metadata_drop_original != effective_drop_original
    ):
        effective_drop_original = resources.metadata_drop_original
        signal.drop_original = resources.metadata_drop_original

    if effective_drop_original:
        encoded_column_name = column
        fragments = [f"{resources.class_count} {class_label}"]
        if resources.preview:
            fragments.append(resources.preview)
        detail = f"{column} (replaced; {'; '.join(fragments)})"
        replaced_original = True
    else:
        stored_encoded_name: Optional[str] = None
        if isinstance(resources.stored_metadata, dict):
            raw_name = resources.stored_metadata.get("encoded_column")
            if isinstance(raw_name, str) and raw_name.strip():
                stored_encoded_name = raw_name.strip()

        encoded_column_name = stored_encoded_name or _resolve_encoded_column_name(
            column,
            config.output_suffix,
            existing_columns,
        )

        fragments = [f"{resources.class_count} {class_label}"]
        if resources.preview:
            fragments.append(resources.preview)
        detail = f"{column}→{encoded_column_name} ({'; '.join(fragments)})"
        replaced_original = False

    preview_entries = [
        LabelEncodingCategoryPreview(value=format_category_label(value), code=index)
        for index, value in enumerate(resources.categories[:3])
    ]

    plan = ColumnEncodingPlan(
        column=column,
        encoded_series=resources.encoded_series,
        encoded_column=encoded_column_name,
        replaced_original=replaced_original,
        categories=resources.categories,
        category_labels=resources.category_labels,
        category_mapping=resources.category_mapping,
        missing_strategy=resources.missing_strategy,
        missing_code=resources.missing_code,
        class_count=resources.class_count,
        detail=detail,
        preview_entries=preview_entries,
        fit_mode=resources.fit_mode,
        effective_drop_original=effective_drop_original,
    )

    return plan, None


def _store_label_encoder(
    plan: ColumnEncodingPlan,
    context: LabelEncodingContext,
) -> None:
    if not context.storage:
        return

    mapping_records = [
        {
            "category": plan.category_labels[index],
            "code": index,
            "display": format_category_label(value),
        }
        for index, value in enumerate(plan.categories)
    ]

    transformer_payload = {
        "mapping": plan.category_mapping,
        "categories": plan.category_labels,
        "missing_strategy": plan.missing_strategy,
        "missing_code": plan.missing_code,
    }

    metadata: Dict[str, Any] = {
        "encoded_column": plan.encoded_column,
        "replaced_original": bool(plan.effective_drop_original),
        "category_count": plan.class_count,
        "missing_strategy": plan.missing_strategy,
        "missing_code": plan.missing_code,
        "unknown_policy": (
            "encode_missing_code" if plan.missing_strategy == "encode" else "preserve_na"
        ),
        "mapping": mapping_records,
        "method": "label_encoding",
        "method_label": f"Label Encoding ({plan.class_count} categories)",
    }
    if context.train_mask is not None:
        metadata["train_rows"] = context.train_row_count

    context.storage.store_transformer(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="label_encoder",
        transformer=transformer_payload,
        column_name=plan.column,
        metadata=metadata,
    )

    context.storage.record_split_activity(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="label_encoder",
        column_name=plan.column,
        split_name="train",
        action="fit_transform",
        row_count=context.train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(context.split_counts.get(split_name, 0))
        context.storage.record_split_activity(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id or "",
            transformer_name="label_encoder",
            column_name=plan.column,
            split_name=split_name,
            action="transform" if rows_processed > 0 else "not_available",
            row_count=rows_processed,
        )


def _record_reuse_activity(plan: ColumnEncodingPlan, context: LabelEncodingContext) -> None:
    if not context.storage:
        return

    context.storage.record_split_activity(
        pipeline_id=context.pipeline_id,
        node_id=context.node_id or "",
        transformer_name="label_encoder",
        column_name=plan.column,
        split_name="train",
        action="not_available",
        row_count=context.train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(context.split_counts.get(split_name, 0))
        context.storage.record_split_activity(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id or "",
            transformer_name="label_encoder",
            column_name=plan.column,
            split_name=split_name,
            action="transform" if rows_processed > 0 else "not_available",
            row_count=rows_processed,
        )


def _compose_summary(accumulator: LabelEncodingAccumulator) -> str:
    if not accumulator.encoded_details:
        summary = "Label encoding: no columns encoded"
    else:
        preview = ", ".join(accumulator.encoded_details[:3])
        if len(accumulator.encoded_details) > 3:
            preview = f"{preview}, …"
        summary = f"Label encoding: encoded {len(accumulator.encoded_details)} column(s) ({preview})"

    if accumulator.skipped_details:
        preview = ", ".join(accumulator.skipped_details[:3])
        if len(accumulator.skipped_details) > 3:
            preview = f"{preview}, …"
        summary = f"{summary}; skipped {preview}"

    return summary


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

    _initialize_signal_from_config(signal, config)

    if frame.empty:
        return frame, "Label encoding: no data available", signal

    candidate_columns, skipped_notes = _collect_candidate_columns(frame, config)

    if not candidate_columns:
        signal.evaluated_columns = []
        return frame, "Label encoding: no categorical columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    accumulator = LabelEncodingAccumulator()
    if skipped_notes:
        accumulator.skipped_details.extend(skipped_notes)
        signal.skipped_columns.extend(skipped_notes)

    existing_columns = set(working_frame.columns)
    context = _build_context(working_frame, pipeline_id, str(node_id) if node_id is not None else None)

    for column in candidate_columns:
        plan, error = _build_column_plan(column, working_frame, config, context, signal, existing_columns)
        if error:
            _register_skip(accumulator, signal, error)
            continue

        assert plan is not None  # for type checkers

        if plan.replaced_original:
            working_frame[column] = plan.encoded_series
        else:
            if plan.encoded_column in working_frame.columns:
                working_frame[plan.encoded_column] = plan.encoded_series
            else:
                insert_at = working_frame.columns.get_loc(column) + 1
                working_frame.insert(insert_at, plan.encoded_column, plan.encoded_series)
                existing_columns.add(plan.encoded_column)

        accumulator.encoded_details.append(plan.detail)
        signal.encoded_columns.append(
            LabelEncodingAppliedColumnSignal(
                source_column=plan.column,
                encoded_column=plan.encoded_column,
                class_count=plan.class_count,
                replaced_original=plan.replaced_original,
                preview=plan.preview_entries,
            )
        )

        if plan.fit_mode == "fit":
            _store_label_encoder(plan, context)
        elif plan.fit_mode == "reuse":
            _record_reuse_activity(plan, context)

    summary = _compose_summary(accumulator)

    return working_frame, summary, signal


__all__ = [
    "LABEL_ENCODING_DEFAULT_SUFFIX",
    "LABEL_ENCODING_DEFAULT_MAX_UNIQUE",
    "LabelEncodingSuggestion",
    "build_label_encoding_suggestions",
    "apply_label_encoding",
]
