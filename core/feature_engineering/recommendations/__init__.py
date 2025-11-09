"""Shared recommendation builders for feature-engineering nodes."""

from .categorical import (
    CategoricalColumnProfile,
    DummyEncodingSuggestion,
    HashEncodingSuggestion,
    LabelEncodingSuggestion,
    OneHotEncodingSuggestion,
    OrdinalEncodingSuggestion,
    TargetEncodingSuggestion,
    build_dummy_encoding_suggestions,
    build_hash_encoding_suggestions,
    build_label_encoding_suggestions,
    build_one_hot_encoding_suggestions,
    build_ordinal_encoding_suggestions,
    build_target_encoding_suggestions,
    format_category_label,
)

__all__ = [
    "CategoricalColumnProfile",
    "LabelEncodingSuggestion",
    "OneHotEncodingSuggestion",
    "DummyEncodingSuggestion",
    "HashEncodingSuggestion",
    "OrdinalEncodingSuggestion",
    "TargetEncodingSuggestion",
    "build_label_encoding_suggestions",
    "build_one_hot_encoding_suggestions",
    "build_dummy_encoding_suggestions",
    "build_hash_encoding_suggestions",
    "build_ordinal_encoding_suggestions",
    "build_target_encoding_suggestions",
    "format_category_label",
]
