"""Text analysis helpers for Skyulf MLflow EDA utilities."""

from __future__ import annotations

import re
import string
from typing import Any, Dict

import pandas as pd

__all__ = [
    "categorize_text_column",
    "analyze_text_column",
    "detect_text_patterns",
    "get_text_insights",
]


def categorize_text_column(
    unique_ratio: float,
    avg_length: float,
    avg_words: float,
    punctuation_pct: float,
    unique_count: int,
    total_count: int,
) -> str:
    """Classify a text column into coarse-grained categories.

    The thresholds originate from the exploratory utilities that power the
    platform EDA service. They remain intentionally permissive so columns that
    resemble free-form feedback, descriptions, or reviews are marked as
    ``free_text`` while identifier like values fall into ``identifier``.
    """

    if unique_ratio > 0.7 and avg_length > 20 and avg_words > 3:
        return "free_text"
    if unique_ratio > 0.5 and avg_length > 50 and avg_words > 5:
        return "free_text"
    if avg_words > 10 and unique_ratio > 0.4:
        return "free_text"
    if avg_length > 200 and avg_words > 15:
        return "free_text"
    if avg_length > 40 and avg_words > 5 and punctuation_pct > 80 and unique_ratio > 0.15:
        return "free_text"
    if avg_length > 30 and avg_words > 4 and unique_ratio > 0.2:
        return "free_text"

    if unique_ratio > 0.95 and avg_length < 50:
        return "identifier"
    if unique_ratio > 0.8 and avg_length < 30 and punctuation_pct > 30 and avg_words <= 3:
        return "identifier"
    if unique_ratio <= 0.3 and avg_length < 50 and unique_count < 20:
        return "categorical"
    if unique_ratio < 0.3 and avg_length < 20:
        return "categorical"
    if 0.3 <= unique_ratio <= 0.8 and avg_length < 30 and avg_words < 4:
        return "semi_structured"
    if 0.4 <= unique_ratio <= 0.8 and avg_length > 30 and avg_words > 5:
        return "descriptive_text"
    if avg_length < 20 and unique_count < 200 and avg_words < 3:
        return "codes_labels"

    return "mixed_text"


def analyze_text_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Compute descriptive statistics for an object column."""

    text_values = series.dropna().astype(str)
    if text_values.empty:
        return {
            "data_category": "empty_text",
            "text_category": "empty",
            "avg_text_length": 0,
            "min_text_length": 0,
            "max_text_length": 0,
            "text_length_std": 0,
            "total_characters": 0,
            "total_words": 0,
            "avg_words_per_text": 0,
            "contains_punctuation_pct": 0,
            "contains_numbers_pct": 0,
            "all_caps_pct": 0,
            "whitespace_heavy_pct": 0,
        }

    text_lengths = text_values.str.len()
    avg_length = float(text_lengths.mean())
    min_length = int(text_lengths.min())
    max_length = int(text_lengths.max())
    std_length = float(text_lengths.std()) if len(text_lengths) > 1 else 0.0

    word_counts = text_values.apply(lambda value: len(str(value).split()))
    total_words = int(word_counts.sum())
    avg_words = float(word_counts.mean()) if not word_counts.empty else 0.0

    total_chars = int(text_lengths.sum())

    has_punctuation = text_values.apply(lambda value: bool(re.search(r"[^\w\s]", str(value))))
    has_numbers = text_values.apply(lambda value: bool(re.search(r"\d", str(value))))
    is_all_caps = text_values.apply(lambda value: str(value).isupper() and str(value).isalpha())

    def _whitespace_density(text: str) -> float:
        normalized = re.sub(r"[_\-]+", " ", str(text))
        normalized = re.sub(r"\s+", " ", normalized.strip())
        if not normalized:
            return 0.0
        token_count = len(normalized.split())
        return token_count / max(len(normalized), 1)

    is_whitespace_heavy = text_values.apply(lambda value: _whitespace_density(value) < 0.1)

    punctuation_pct = float(has_punctuation.mean() * 100)
    numbers_pct = float(has_numbers.mean() * 100)
    all_caps_pct = float(is_all_caps.mean() * 100)
    whitespace_pct = float(is_whitespace_heavy.mean() * 100)

    unique_ratio = series.nunique(dropna=False) / max(len(series), 1)
    text_category = categorize_text_column(
        unique_ratio,
        avg_length,
        avg_words,
        punctuation_pct,
        series.nunique(dropna=False),
        len(series),
    )

    quality_flags = []
    if avg_length > 500:
        quality_flags.append("very_long_text")
    if avg_length < 3:
        quality_flags.append("very_short_text")
    if whitespace_pct > 20:
        quality_flags.append("whitespace_heavy")
    if all_caps_pct > 50:
        quality_flags.append("caps_heavy")
    if unique_ratio < 0.01:
        quality_flags.append("highly_repetitive")

    special_char_pct = float(
        text_values.apply(
            lambda value: (
                sum(1 for char in str(value) if char in string.punctuation)
                / max(len(str(value)), 1)
                * 100
            )
        ).mean()
    )

    email_pattern_pct = float(
        text_values.str.contains(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            regex=True,
            na=False,
        ).mean()
        * 100
    )
    url_pattern_pct = float(
        text_values.str.contains(
            r"http[s]?://(?:[A-Za-z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+",
            regex=True,
            na=False,
        ).mean()
        * 100
    )
    phone_pattern_pct = float(
        text_values.str.contains(
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            regex=True,
            na=False,
        ).mean()
        * 100
    )

    return {
        "data_category": "text",
        "text_category": text_category,
        "avg_text_length": round(avg_length, 2),
        "min_text_length": min_length,
        "max_text_length": max_length,
        "text_length_std": round(std_length, 2),
        "total_characters": total_chars,
        "total_words": total_words,
        "avg_words_per_text": round(avg_words, 2),
        "contains_punctuation_pct": round(punctuation_pct, 2),
        "contains_numbers_pct": round(numbers_pct, 2),
        "all_caps_pct": round(all_caps_pct, 2),
        "whitespace_heavy_pct": round(whitespace_pct, 2),
        "special_char_pct": round(special_char_pct, 2),
        "email_pattern_pct": round(email_pattern_pct, 2),
        "url_pattern_pct": round(url_pattern_pct, 2),
        "phone_pattern_pct": round(phone_pattern_pct, 2),
        "quality_flags": quality_flags,
        "text_length_distribution": {
            "q25": float(text_lengths.quantile(0.25)),
            "q50": float(text_lengths.quantile(0.50)),
            "q75": float(text_lengths.quantile(0.75)),
            "q90": float(text_lengths.quantile(0.90)),
            "q95": float(text_lengths.quantile(0.95)),
        },
    }


def detect_text_patterns(series: pd.Series) -> Dict[str, Any]:
    """Detect common patterns such as email, url, or phone numbers."""

    text_values = series.dropna().astype(str)
    if text_values.empty:
        return {"patterns": []}

    checks = [
        ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ("url", r"http[s]?://(?:[A-Za-z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+"),
        ("phone", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ("credit_card", r"\b(?:\d[ -]?){12,18}\d\b"),
        ("ip_address", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        ("date", r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    ]

    patterns = []
    for pattern_type, regex_pattern in checks:
        count = text_values.str.contains(regex_pattern, regex=True, na=False).sum()
        if count:
            patterns.append(
                {
                    "type": pattern_type,
                    "count": int(count),
                    "percentage": float(count / len(text_values) * 100),
                }
            )

    return {"patterns": patterns}


def get_text_insights(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Return aggregate text metrics and lightweight recommendations."""

    analysis = analyze_text_column(series, column_name)
    patterns = detect_text_patterns(series)

    recommendations = []
    text_category = analysis.get("text_category", "unknown")
    avg_length = analysis.get("avg_text_length", 0.0)

    if text_category == "free_text":
        recommendations.extend(
            [
                "Consider text preprocessing: tokenization, lowercasing, stop-word removal",
                "Suitable for NLP analysis: sentiment, topics, or classification",
                "Extract features with TF-IDF, word embeddings, or n-grams",
            ]
        )
    elif text_category == "categorical":
        recommendations.extend(
            [
                "Apply one-hot, ordinal, or target encoding for modeling",
                "Review category distribution for typos and inconsistent casing",
            ]
        )
    elif text_category == "identifier":
        recommendations.extend(
            [
                "Drop or mask identifiers unless needed for joins",
                "Confirm uniqueness to catch ingestion issues",
            ]
        )

    if avg_length > 200:
        recommendations.append("Consider summarizing or truncating long text fields")

    quality_flags = analysis.get("quality_flags") or []
    if quality_flags:
        recommendations.append(
            "Review text quality flags: " + ", ".join(sorted(set(quality_flags)))
        )

    return {
        **analysis,
        **patterns,
        "eda_recommendations": recommendations,
        "nlp_suitability": text_category in {"free_text", "descriptive_text"},
        "requires_preprocessing": text_category in {"free_text", "descriptive_text", "mixed_text"},
    }
