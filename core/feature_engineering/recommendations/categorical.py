"""Utilities for recommending categorical encoding strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import math

import pandas as pd
from pandas.api import types as pd_types

from core.feature_engineering.preprocessing.encoding.hash_encoding import (
    HASH_ENCODING_DEFAULT_BUCKETS,
    HASH_ENCODING_MAX_CARDINALITY_LIMIT,
)

__all__ = [
    "CategoricalColumnProfile",
    "LabelEncodingSuggestion",
    "OneHotEncodingSuggestion",
    "DummyEncodingSuggestion",
    "OrdinalEncodingSuggestion",
    "TargetEncodingSuggestion",
    "HashEncodingSuggestion",
    "build_label_encoding_suggestions",
    "build_one_hot_encoding_suggestions",
    "build_dummy_encoding_suggestions",
    "build_ordinal_encoding_suggestions",
    "build_target_encoding_suggestions",
    "build_hash_encoding_suggestions",
    "format_category_label",
]


@dataclass
class CategoricalColumnProfile:
    """Lightweight summary describing a categorical column."""

    column: str
    dtype_label: str
    text_category: Optional[str]
    avg_text_length: Optional[float]
    unique_non_missing: int
    non_missing_count: int
    unique_percentage: float
    missing_percentage: float
    sample_values: List[str]

    @property
    def unique_ratio(self) -> float:
        if self.non_missing_count <= 0:
            return 0.0
        return min(max(self.unique_non_missing / float(self.non_missing_count), 0.0), 1.0)


@dataclass
class LabelEncodingSuggestion:
    """Recommendation describing whether a column suits label encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    score: float
    selectable: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "score": self.score,
            "selectable": self.selectable,
        }


@dataclass
class OneHotEncodingSuggestion:
    """Recommendation describing suitability for one-hot encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    estimated_dummy_columns: int
    score: float
    selectable: bool
    recommended_drop_first: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "estimated_dummy_columns": self.estimated_dummy_columns,
            "score": self.score,
            "selectable": self.selectable,
            "recommended_drop_first": self.recommended_drop_first,
        }


@dataclass
class DummyEncodingSuggestion:
    """Recommendation describing suitability for dummy encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    estimated_dummy_columns: int
    score: float
    selectable: bool
    recommended_drop_first: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "estimated_dummy_columns": self.estimated_dummy_columns,
            "score": self.score,
            "selectable": self.selectable,
            "recommended_drop_first": self.recommended_drop_first,
        }


@dataclass
class OrdinalEncodingSuggestion:
    """Recommendation describing suitability for ordinal encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    score: float
    selectable: bool
    recommended_handle_unknown: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "score": self.score,
            "selectable": self.selectable,
            "recommended_handle_unknown": self.recommended_handle_unknown,
        }


@dataclass
class TargetEncodingSuggestion:
    """Recommendation describing suitability for target encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    score: float
    selectable: bool
    recommended_use_global_fallback: bool

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "score": self.score,
            "selectable": self.selectable,
            "recommended_use_global_fallback": self.recommended_use_global_fallback,
        }


@dataclass
class HashEncodingSuggestion:
    """Recommendation describing suitability for hash encoding."""

    column: str
    status: str
    reason: str
    dtype: Optional[str]
    unique_count: Optional[int]
    unique_percentage: Optional[float]
    missing_percentage: Optional[float]
    text_category: Optional[str]
    sample_values: List[str]
    score: float
    selectable: bool
    recommended_bucket_count: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "status": self.status,
            "reason": self.reason,
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "unique_percentage": self.unique_percentage,
            "missing_percentage": self.missing_percentage,
            "text_category": self.text_category,
            "sample_values": self.sample_values,
            "score": self.score,
            "selectable": self.selectable,
            "recommended_bucket_count": self.recommended_bucket_count,
        }


def _safe_int(value: Any) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return int(round(numeric))


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return float(numeric)


def _format_category_label(value: Any) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, float) and math.isnan(value):
        return "(missing)"
    text = str(value)
    if not text:
        return "(blank)"
    if len(text) > 24:
        return f"{text[:21]}…"
    return text


def format_category_label(value: Any) -> str:
    """Public helper that mirrors the private label formatter."""

    return _format_category_label(value)


def _profile_categorical_columns(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[CategoricalColumnProfile]:
    if frame.empty:
        return []

    metadata = column_metadata or {}
    sample_size = frame.shape[0]
    profiles: List[CategoricalColumnProfile] = []

    for column in frame.columns:
        column_name = str(column)
        if not column_name:
            continue

        series = frame[column]
        if pd_types.is_bool_dtype(series):
            continue
        if not (
            pd_types.is_object_dtype(series)
            or pd_types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue

        series_string = series.astype("string")
        non_missing_count = int(series_string.notna().sum())
        unique_non_missing = int(series_string.nunique(dropna=True)) if non_missing_count else 0

        missing_percentage = float(series.isna().mean() * 100.0) if sample_size else 0.0
        unique_percentage = (
            float(unique_non_missing / non_missing_count * 100.0)
            if non_missing_count
            else 0.0
        )

        meta = metadata.get(column_name) or {}
        dtype_label = meta.get("dtype") or str(series.dtype)

        text_category = meta.get("text_category")
        if isinstance(text_category, str):
            text_category = text_category.strip().lower() or None

        avg_text_length = _safe_float(meta.get("avg_text_length"))

        unique_from_meta = _safe_int(meta.get("unique_count"))
        if unique_from_meta is not None and unique_from_meta >= 0:
            unique_non_missing = unique_from_meta

        unique_pct_meta = _safe_float(meta.get("unique_percentage"))
        if unique_pct_meta is not None:
            unique_percentage = unique_pct_meta
        elif non_missing_count:
            unique_percentage = float(unique_non_missing / non_missing_count * 100.0)

        missing_pct_meta = _safe_float(meta.get("null_percentage"))
        if missing_pct_meta is not None:
            missing_percentage = missing_pct_meta

        sample_values: List[str] = []
        try:
            value_counts = series_string.value_counts(dropna=True)
            for value in value_counts.head(3).index:
                sample_values.append(_format_category_label(value))
        except Exception:  # pragma: no cover - defensive fallback
            sample_values = []

        profiles.append(
            CategoricalColumnProfile(
                column=column_name,
                dtype_label=dtype_label,
                text_category=text_category,
                avg_text_length=avg_text_length,
                unique_non_missing=unique_non_missing,
                non_missing_count=non_missing_count,
                unique_percentage=round(unique_percentage, 3),
                missing_percentage=round(missing_percentage, 3),
                sample_values=sample_values,
            )
        )

    return profiles


def build_label_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    auto_detect_limit: int = 50,
    absolute_max_limit: int = 1000,
) -> List[LabelEncodingSuggestion]:
    """Analyse a dataframe and surface label-encoding recommendations."""

    profiles = _profile_categorical_columns(frame, column_metadata)
    if not profiles:
        return []

    suggestions: List[LabelEncodingSuggestion] = []

    for profile in profiles:
        unique_non_missing = profile.unique_non_missing
        unique_percentage = profile.unique_percentage
        missing_percentage = profile.missing_percentage
        text_category = profile.text_category
        avg_text_length = profile.avg_text_length

        status = "recommended"
        reason = "Detected categorical-like text." if unique_non_missing else "No categorical signal detected."
        selectable = True
        score = 0.0

        if unique_non_missing <= 1:
            status = "single_category"
            reason = "Only one distinct non-missing value observed."
            selectable = False
            score = 0.0
        else:
            if text_category in {"identifier", "id", "unique_code"} or profile.unique_ratio >= 0.9:
                status = "identifier"
                reason = "Behaves like a high-uniqueness identifier; encoding adds little value."
                selectable = False
                score = 0.1
            elif text_category in {"free_text", "descriptive_text", "mixed_text"} or (
                avg_text_length is not None and avg_text_length > 40
            ):
                status = "free_text"
                reason = "Detected long-form text better suited for NLP pipelines."
                selectable = False
                score = 0.15
            elif unique_non_missing > absolute_max_limit:
                status = "too_many_categories"
                reason = (
                    f"More than {absolute_max_limit} distinct values detected; consider alternative "
                    "encodings."
                )
                selectable = False
                score = 0.1
            elif unique_non_missing > auto_detect_limit and auto_detect_limit > 0:
                status = "high_cardinality"
                reason = (
                    f"High-cardinality text ({unique_non_missing} categories) exceeds the auto-detect limit "
                    f"of {auto_detect_limit}."
                )
                selectable = True
                score = 0.55
            else:
                if text_category in {"categorical", "codes_labels"}:
                    reason = (
                        "This column is classified as categorical text; label encoding is a good fit."
                    )
                else:
                    reason = f"{unique_non_missing} categories observed; suitable for label encoding."
                selectable = True
                score = 0.85

        suggestions.append(
            LabelEncodingSuggestion(
                column=profile.column,
                status=status,
                reason=reason,
                dtype=profile.dtype_label,
                unique_count=unique_non_missing,
                unique_percentage=unique_percentage,
                missing_percentage=missing_percentage,
                text_category=text_category,
                sample_values=profile.sample_values,
                score=score,
                selectable=selectable,
            )
        )

    suggestions.sort(key=lambda item: (-item.score, item.column))
    return suggestions


def build_one_hot_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    max_safe_categories: int = 20,
    caution_threshold: Optional[int] = None,
    absolute_max_categories: int = 80,
) -> List[OneHotEncodingSuggestion]:
    """Analyse a dataframe and surface one-hot encoding recommendations."""

    profiles = _profile_categorical_columns(frame, column_metadata)
    if not profiles:
        return []

    if caution_threshold is None:
        caution_threshold = max(max_safe_categories * 2, max_safe_categories)
        caution_threshold = min(caution_threshold, absolute_max_categories)

    suggestions: List[OneHotEncodingSuggestion] = []

    for profile in profiles:
        unique_non_missing = profile.unique_non_missing
        unique_percentage = profile.unique_percentage
        missing_percentage = profile.missing_percentage
        text_category = profile.text_category
        avg_text_length = profile.avg_text_length

        estimated_dummy_columns = unique_non_missing + (1 if missing_percentage > 0 else 0)
        recommended_drop_first = estimated_dummy_columns > 2

        status = "recommended"
        reason = "Detected categorical-like text." if unique_non_missing else "No categorical signal detected."
        selectable = True
        score = 0.0

        if unique_non_missing <= 1:
            status = "single_category"
            reason = "Only one distinct non-missing value observed."
            selectable = False
            score = 0.0
            recommended_drop_first = False
        else:
            if text_category in {"identifier", "id", "unique_code"} or profile.unique_ratio >= 0.85:
                status = "identifier"
                reason = "Behaves like an identifier; expanded dummy columns add little value."
                selectable = False
                score = 0.1
                recommended_drop_first = False
            elif text_category in {"free_text", "descriptive_text", "mixed_text"} or (
                avg_text_length is not None and avg_text_length > 40
            ):
                status = "free_text"
                reason = "Detected long-form text better suited for embedding or NLP pipelines."
                selectable = False
                score = 0.15
                recommended_drop_first = False
            elif unique_non_missing > absolute_max_categories:
                status = "too_many_categories"
                reason = (
                    f"More than {absolute_max_categories} distinct values detected; one-hot encoding would "
                    "create excessive features."
                )
                selectable = False
                score = 0.05
                recommended_drop_first = False
            elif unique_non_missing > caution_threshold:
                status = "high_cardinality"
                reason = (
                    f"Very high cardinality ({unique_non_missing} categories) risks sparse, wide matrices. "
                    "Consider target or frequency encoding."
                )
                selectable = False
                score = 0.25
                recommended_drop_first = False
            elif unique_non_missing > max_safe_categories:
                status = "high_cardinality"
                reason = (
                    f"High cardinality ({unique_non_missing} categories) may require dimensionality reduction "
                    "even though one-hot remains possible."
                )
                selectable = True
                score = 0.45
            else:
                status = "recommended"
                reason = (
                    f"{unique_non_missing} categories -> approximately {estimated_dummy_columns} dummy columns."
                )
                selectable = True
                score = 0.9 if unique_non_missing <= (max_safe_categories // 2 or 1) else 0.85

        suggestions.append(
            OneHotEncodingSuggestion(
                column=profile.column,
                status=status,
                reason=reason,
                dtype=profile.dtype_label,
                unique_count=unique_non_missing,
                unique_percentage=unique_percentage,
                missing_percentage=missing_percentage,
                text_category=text_category,
                sample_values=profile.sample_values,
                estimated_dummy_columns=estimated_dummy_columns,
                score=score,
                selectable=selectable,
                recommended_drop_first=recommended_drop_first,
            )
        )

    suggestions.sort(key=lambda item: (-item.score, item.column))
    return suggestions


def build_ordinal_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    auto_detect_limit: int = 50,
    absolute_max_limit: int = 1000,
) -> List[OrdinalEncodingSuggestion]:
    """Build ordinal-encoding suggestions derived from label-encoding insights."""

    label_suggestions = build_label_encoding_suggestions(
        frame,
        column_metadata,
        auto_detect_limit=auto_detect_limit,
        absolute_max_limit=absolute_max_limit,
    )

    ordinal_suggestions: List[OrdinalEncodingSuggestion] = []
    for suggestion in label_suggestions:
        recommended_handle_unknown = bool(
            suggestion.missing_percentage is not None and suggestion.missing_percentage > 0.0
        )
        ordinal_suggestions.append(
            OrdinalEncodingSuggestion(
                column=suggestion.column,
                status=suggestion.status,
                reason=suggestion.reason,
                dtype=suggestion.dtype,
                unique_count=suggestion.unique_count,
                unique_percentage=suggestion.unique_percentage,
                missing_percentage=suggestion.missing_percentage,
                text_category=suggestion.text_category,
                sample_values=suggestion.sample_values,
                score=suggestion.score,
                selectable=suggestion.selectable,
                recommended_handle_unknown=recommended_handle_unknown,
            )
        )

    ordinal_suggestions.sort(key=lambda item: (-item.score, item.column))
    return ordinal_suggestions


def build_target_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    auto_detect_limit: int = 50,
    absolute_max_limit: int = 1000,
) -> List[TargetEncodingSuggestion]:
    """Surface target-encoding recommendations derived from label insights."""

    label_suggestions = build_label_encoding_suggestions(
        frame,
        column_metadata,
        auto_detect_limit=auto_detect_limit,
        absolute_max_limit=absolute_max_limit,
    )

    target_suggestions: List[TargetEncodingSuggestion] = []
    for suggestion in label_suggestions:
        recommended_fallback = bool(
            (suggestion.missing_percentage is not None and suggestion.missing_percentage > 0.0)
            or suggestion.status in {"high_cardinality", "too_many_categories"}
        )

        target_suggestions.append(
            TargetEncodingSuggestion(
                column=suggestion.column,
                status=suggestion.status,
                reason=suggestion.reason,
                dtype=suggestion.dtype,
                unique_count=suggestion.unique_count,
                unique_percentage=suggestion.unique_percentage,
                missing_percentage=suggestion.missing_percentage,
                text_category=suggestion.text_category,
                sample_values=suggestion.sample_values,
                score=suggestion.score,
                selectable=suggestion.selectable,
                recommended_use_global_fallback=recommended_fallback,
            )
        )

    target_suggestions.sort(key=lambda item: (-item.score, item.column))
    return target_suggestions


def _estimate_hash_bucket_count(unique_count: Optional[int], default_buckets: int) -> int:
    if unique_count is None or unique_count <= 0:
        return default_buckets

    safe_unique = max(1, min(unique_count, HASH_ENCODING_MAX_CARDINALITY_LIMIT))
    desired = max(default_buckets, safe_unique * 2)

    bucket_count = 1
    while bucket_count < desired and bucket_count < HASH_ENCODING_MAX_CARDINALITY_LIMIT:
        bucket_count <<= 1

    if bucket_count > HASH_ENCODING_MAX_CARDINALITY_LIMIT:
        bucket_count = HASH_ENCODING_MAX_CARDINALITY_LIMIT

    return max(default_buckets, bucket_count)


def build_hash_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    auto_detect_limit: int = HASH_ENCODING_MAX_CARDINALITY_LIMIT,
    absolute_max_limit: int = HASH_ENCODING_MAX_CARDINALITY_LIMIT,
    default_buckets: int = HASH_ENCODING_DEFAULT_BUCKETS,
) -> List[HashEncodingSuggestion]:
    """Surface hash-encoding recommendations derived from label insights."""

    label_suggestions = build_label_encoding_suggestions(
        frame,
        column_metadata,
        auto_detect_limit=auto_detect_limit,
        absolute_max_limit=absolute_max_limit,
    )

    hash_suggestions: List[HashEncodingSuggestion] = []

    for suggestion in label_suggestions:
        status = suggestion.status
        reason = suggestion.reason
        score = suggestion.score or 0.0

        if status in {"high_cardinality", "too_many_categories"}:
            status = "recommended"
            reason = (
                f"{suggestion.reason} Hash encoding compresses high-cardinality categories into a "
                "bounded set of buckets."
            )
            score = max(score, 0.9)
        elif status == "recommended":
            reason = (
                f"{suggestion.reason} Hash encoding offers a numeric representation without manual "
                "category mapping."
            )
            score = max(score, 0.6)
        else:
            reason = f"{suggestion.reason} Hash encoding is optional for this column."
            score = max(score, 0.3)

        bucket_count = _estimate_hash_bucket_count(suggestion.unique_count, default_buckets)

        hash_suggestions.append(
            HashEncodingSuggestion(
                column=suggestion.column,
                status=status,
                reason=reason,
                dtype=suggestion.dtype,
                unique_count=suggestion.unique_count,
                unique_percentage=suggestion.unique_percentage,
                missing_percentage=suggestion.missing_percentage,
                text_category=suggestion.text_category,
                sample_values=suggestion.sample_values,
                score=score,
                selectable=suggestion.selectable,
                recommended_bucket_count=bucket_count,
            )
        )

    hash_suggestions.sort(key=lambda item: (-item.score, item.column))
    return hash_suggestions


def build_dummy_encoding_suggestions(
    frame: pd.DataFrame,
    column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    max_safe_categories: int = 20,
    caution_threshold: Optional[int] = None,
    absolute_max_categories: int = 80,
) -> List[DummyEncodingSuggestion]:
    """Surface dummy-encoding recommendations by reusing one-hot heuristics."""

    base_suggestions = build_one_hot_encoding_suggestions(
        frame,
        column_metadata,
        max_safe_categories=max_safe_categories,
        caution_threshold=caution_threshold,
        absolute_max_categories=absolute_max_categories,
    )

    if not base_suggestions:
        return []

    enriched: List[DummyEncodingSuggestion] = []

    for suggestion in base_suggestions:
        reason = suggestion.reason
        if suggestion.status == "recommended":
            reason = (
                f"{reason} Dummy encoding will drop the reference category to limit multicollinearity."
            )

        enriched.append(
            DummyEncodingSuggestion(
                column=suggestion.column,
                status=suggestion.status,
                reason=reason,
                dtype=suggestion.dtype,
                unique_count=suggestion.unique_count,
                unique_percentage=suggestion.unique_percentage,
                missing_percentage=suggestion.missing_percentage,
                text_category=suggestion.text_category,
                sample_values=suggestion.sample_values,
                estimated_dummy_columns=suggestion.estimated_dummy_columns,
                score=suggestion.score,
                selectable=suggestion.selectable,
                recommended_drop_first=True,
            )
        )

    enriched.sort(key=lambda item: (-item.score, item.column))
    return enriched
