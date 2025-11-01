"""Lightweight data quality reporting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .text import get_text_insights

__all__ = ["generate_quality_report", "QualityReport"]

PII_PATTERN_KEYWORDS: Iterable[str] = (
    "email",
    "phone",
    "ssn",
    "social_security",
    "credit",
    "card",
    "person",
    "name",
    "passport",
    "bank",
    "iban",
    "account",
    "ip",
)

RECOMMENDATION_PRIORITY_META: Dict[str, Dict[str, Any]] = {
    "critical": {"score": 100, "label": "Critical"},
    "high": {"score": 85, "label": "High"},
    "medium": {"score": 70, "label": "Medium"},
    "strategic": {"score": 60, "label": "Strategic"},
    "advanced": {"score": 50, "label": "Advanced"},
    "low": {"score": 40, "label": "Low"},
    "general": {"score": 60, "label": "General"},
}

RECOMMENDATION_CATEGORY_META: Dict[str, Dict[str, Any]] = {
    "data_quality": {
        "label": "Data Quality",
        "default_tags": {"data_quality", "missing_data"},
    },
    "privacy": {
        "label": "Privacy & Governance",
        "default_tags": {"privacy"},
    },
    "text_preprocessing": {
        "label": "Text Preprocessing",
        "default_tags": {"text", "nlp", "text_quality"},
    },
    "categorical_encoding": {
        "label": "Categorical Encoding",
        "default_tags": {"categorical", "encoding"},
    },
    "feature_engineering": {
        "label": "Feature Engineering",
        "default_tags": {"feature_engineering"},
    },
    "pattern_detection": {
        "label": "Pattern Detection",
        "default_tags": {"pattern_detection"},
    },
    "text_quality": {
        "label": "Text Quality",
        "default_tags": {"text_quality"},
    },
    "advanced_analysis": {
        "label": "Advanced Analysis",
        "default_tags": {"advanced_analysis"},
    },
    "project_roadmap": {
        "label": "Project Roadmap",
        "default_tags": {"project_roadmap"},
    },
    "general": {
        "label": "General Guidance",
        "default_tags": {"general"},
    },
}

RECOMMENDATION_SIGNAL_TAGS: Dict[str, List[str]] = {
    "missing_data": ["missing_data", "nulls"],
    "empty_column": ["missing_data", "empty_column"],
    "low_variance": ["constant_feature", "low_variance"],
    "pii": ["pii", "privacy"],
    "text_quality": ["text_quality", "nlp"],
    "nlp": ["nlp", "text"],
    "pattern_detection": ["pattern_detection"],
    "high_cardinality": ["high_cardinality", "categorical"],
}


def _normalize_slug(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower().replace(" ", "_")
    normalized = "".join(char for char in normalized if char.isalnum() or char in {"_", "-"})
    return normalized or None


def _pattern_type_is_pii(pattern_type: Optional[str]) -> bool:
    if not pattern_type:
        return False
    lowered = str(pattern_type).lower()
    return any(keyword in lowered for keyword in PII_PATTERN_KEYWORDS)


@dataclass
class QualityReport:
    """Container that mirrors the public response of ``generate_quality_report``."""

    success: bool
    columns: List[Dict[str, Any]]
    summary: Dict[str, Any]
    quality_checks: Dict[str, bool]
    quality_score: float
    recommendations: List[Dict[str, Any]]
    text_patterns: List[Dict[str, Any]]
    text_quality_flags: List[Dict[str, Any]]
    insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "columns": self.columns,
            "summary": self.summary,
            "quality_checks": self.quality_checks,
            "quality_score": self.quality_score,
            "recommendations": self.recommendations,
            "text_patterns": self.text_patterns,
            "text_quality_flags": self.text_quality_flags,
            "insights": self.insights,
        }


def _build_recommendation(
    title: str,
    description: str,
    *,
    priority: str = "medium",
    category: str = "general",
    columns: Optional[Iterable[str]] = None,
    action: Optional[str] = None,
    why_it_matters: Optional[str] = None,
    feature_impact: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    signal_type: Optional[str] = None,
    metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    normalized_priority = _normalize_slug(priority) or "medium"
    priority_meta = RECOMMENDATION_PRIORITY_META.get(
        normalized_priority,
        RECOMMENDATION_PRIORITY_META["medium"],
    )

    normalized_category = _normalize_slug(category) or "general"
    category_meta = RECOMMENDATION_CATEGORY_META.get(
        normalized_category,
        {
            "label": normalized_category.replace("_", " ").title(),
            "default_tags": {normalized_category},
        },
    )

    payload: Dict[str, Any] = {
        "title": title,
        "description": description,
        "priority": normalized_priority,
        "priority_score": priority_meta["score"],
        "priority_label": priority_meta["label"],
        "category": normalized_category,
        "category_label": category_meta["label"],
    }

    if action:
        payload["action"] = action
    if why_it_matters:
        payload["why_it_matters"] = why_it_matters
    if feature_impact:
        payload["feature_impact"] = feature_impact
    if metrics:
        cleaned = []
        for metric in metrics:
            if isinstance(metric, dict) and metric.get("name"):
                cleaned_metric = {"name": str(metric["name"])}
                if "value" in metric:
                    cleaned_metric["value"] = metric["value"]
                if metric.get("unit"):
                    cleaned_metric["unit"] = str(metric["unit"])
                cleaned.append(cleaned_metric)
        if cleaned:
            payload["metrics"] = cleaned

    normalized_columns = sorted({str(col) for col in columns or []})
    if normalized_columns:
        payload["columns"] = normalized_columns

    normalized_signal = _normalize_slug(signal_type)
    if normalized_signal:
        payload["signal_type"] = normalized_signal

    focus = {normalized_category}
    if normalized_signal:
        focus.add(normalized_signal)
    if focus:
        payload["focus_areas"] = sorted(focus)

    aggregated_tags = set(category_meta.get("default_tags", set()))
    if normalized_signal:
        aggregated_tags.update(RECOMMENDATION_SIGNAL_TAGS.get(normalized_signal, []))
    if tags:
        for tag in tags:
            normalized_tag = _normalize_slug(tag)
            if normalized_tag:
                aggregated_tags.add(normalized_tag)
    if aggregated_tags:
        payload["tags"] = sorted(aggregated_tags)

    payload["meta"] = {"generated_by": "quality_report", "version": 1}
    if normalized_signal:
        payload["meta"]["signal_type"] = normalized_signal

    return payload


def _count_outliers_iqr(series: pd.Series) -> int:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def generate_quality_report(
    df: pd.DataFrame,
    *,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute a compact quality report for a pandas ``DataFrame``.

    Parameters
    ----------
    df:
        Input dataset. The frame is not modified in-place.
    sample_size:
        Optional number of rows to sample before analysis. The default analyses
        the full DataFrame.
    random_state:
        Seed used when sampling rows.
    """

    if df is None or df.empty:
        return QualityReport(
            success=False,
            columns=[],
            summary={
                "rows": 0,
                "columns": 0,
                "completeness_pct": 0.0,
            },
            quality_checks={},
            quality_score=0.0,
            recommendations=[],
            text_patterns=[],
            text_quality_flags=[],
            insights=["Dataset is empty"],
        ).to_dict()

    if sample_size and sample_size < len(df):
        working_df = df.sample(sample_size, random_state=random_state)
    else:
        working_df = df.copy()

    total_rows, total_cols = working_df.shape

    dtype_info: Dict[str, List[str]] = {}
    column_quality: List[Dict[str, Any]] = []
    potential_issues: List[Dict[str, Any]] = []
    text_pattern_tracker: Dict[str, Dict[str, Any]] = {}
    text_quality_flag_tracker: Dict[str, set] = {}
    pii_columns = set()

    for column_name in working_df.columns:
        series = working_df[column_name]
        dtype_str = str(series.dtype)
        dtype_info.setdefault(dtype_str, []).append(column_name)

        non_null_count = int(series.notna().sum())
        null_count = total_rows - non_null_count
        unique_count = int(series.nunique(dropna=False))
        null_pct = float(null_count / max(total_rows, 1) * 100.0)
        unique_pct = float(unique_count / max(total_rows, 1) * 100.0)

        column_payload: Dict[str, Any] = {
            "name": column_name,
            "dtype": dtype_str,
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "unique_percentage": unique_pct,
            "memory_usage": int(series.memory_usage(deep=True)),
        }

        if pd.api.types.is_bool_dtype(series):
            true_count = int(series.fillna(False).sum())
            false_count = non_null_count - true_count
            column_payload.update(
                {
                    "data_category": "boolean",
                    "true_count": true_count,
                    "false_count": false_count,
                    "true_percentage": float(true_count / max(total_rows, 1) * 100.0),
                }
            )
        elif pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_numeric = numeric_series.dropna()
            column_payload["data_category"] = "numeric"
            if not non_null_numeric.empty:
                column_payload.update(
                    {
                        "min_value": float(non_null_numeric.min()),
                        "max_value": float(non_null_numeric.max()),
                        "mean_value": float(non_null_numeric.mean()),
                        "std_value": float(non_null_numeric.std(ddof=0)),
                        "outlier_count": _count_outliers_iqr(non_null_numeric),
                    }
                )
        elif pd.api.types.is_datetime64_any_dtype(series):
            column_payload["data_category"] = "datetime"
        elif series.dtype == "object":
            text_insights = get_text_insights(series, column_name)
            column_payload.update(text_insights)

            for pattern in text_insights.get("patterns", []):
                pattern_type = pattern.get("type", "unknown")
                normalized_type = _normalize_slug(pattern_type) or "unknown"
                tracker_entry = text_pattern_tracker.setdefault(
                    normalized_type,
                    {"count": 0, "columns": set(), "label": pattern_type},
                )
                tracker_entry["count"] += int(pattern.get("count", 0))
                tracker_entry["columns"].add(column_name)

                if _pattern_type_is_pii(pattern_type):
                    pii_columns.add(column_name)

            for flag in text_insights.get("quality_flags", []):
                tracker = text_quality_flag_tracker.setdefault(flag, set())
                tracker.add(column_name)

            text_category = text_insights.get("text_category")
            if text_category == "categorical" and text_insights.get("unique_count", 0) > 50:
                potential_issues.append(
                    {
                        "type": "warning",
                        "column": column_name,
                        "message": (
                            f"Column '{column_name}' has many distinct categories "
                            f"({text_insights.get('unique_count', 0)}). Consider grouping or encoding."
                        ),
                    }
                )

        if unique_pct > 95:
            potential_issues.append(
                {
                    "type": "warning",
                    "column": column_name,
                    "message": "Column appears to be identifier-like (high uniqueness)",
                }
            )
        if null_pct > 30:
            potential_issues.append(
                {
                    "type": "warning",
                    "column": column_name,
                    "message": "Column has more than 30% missing values",
                }
            )

        column_quality.append(column_payload)

    column_quality.sort(key=lambda item: item.get("null_percentage", 0), reverse=True)

    detected_patterns_summary = [
        {
            "pattern_type": entry["label"],
            "total_count": details["count"],
            "columns": sorted(details["columns"]),
        }
        for entry_key, details in text_pattern_tracker.items()
        for entry in [details]
        if details["count"] > 0
    ]

    text_quality_flags_summary = [
        {"flag": flag, "columns": sorted(columns)}
        for flag, columns in text_quality_flag_tracker.items()
        if columns
    ]

    completeness = float(
        working_df.notna().sum().sum() / max(total_rows * max(total_cols, 1), 1) * 100.0
    )

    quality_checks = {
        "no_duplicate_rows": bool(working_df.duplicated().sum() == 0),
        "no_empty_columns": bool(not any(working_df[col].isna().all() for col in working_df.columns)),
        "reasonable_missing_data": bool(all(item.get("null_percentage", 0) < 50 for item in column_quality)),
        "consistent_data_types": bool(len(dtype_info) < total_cols * 0.8 if total_cols else True),
        "no_constant_columns": bool(all(item.get("unique_count", 0) > 1 for item in column_quality if item["non_null_count"] > 0)),
    }

    quality_score = float(
        sum(int(value) for value in quality_checks.values()) / max(len(quality_checks), 1) * 100.0
    )

    free_text_columns = [col for col in column_quality if col.get("text_category") in {"free_text", "descriptive_text"}]
    categorical_text_columns = [col for col in column_quality if col.get("text_category") == "categorical"]
    constant_columns = [col["name"] for col in column_quality if col.get("unique_count", 0) <= 1 and col.get("non_null_count", 0) > 0]
    empty_columns = [col["name"] for col in column_quality if col.get("non_null_count", 0) == 0]
    high_missing = [col for col in column_quality if col.get("null_percentage", 0) > 30]

    recommendations: List[Dict[str, Any]] = []

    if high_missing:
        columns_with_missing = [col["name"] for col in high_missing]
        worst_missing = max(col.get("null_percentage", 0) for col in high_missing)
        priority = "critical" if worst_missing >= 90 else "high" if worst_missing >= 70 else "medium"
        recommendations.append(
            _build_recommendation(
                "Handle Missing Data",
                "Columns show significant missingness; consider targeted imputers or feature removal.",
                priority=priority,
                category="data_quality",
                columns=columns_with_missing,
                action="Profile missingness (df[col].isna().mean()) and decide between imputation, filling, or dropping.",
                why_it_matters="High missingness introduces bias and brittle production pipelines.",
                feature_impact="Improves stability of downstream preprocessing and modeling.",
                tags=["missing_data"],
                signal_type="missing_data",
                metrics=[
                    {"name": "columns_affected", "value": len(columns_with_missing)},
                    {"name": "max_missing_pct", "value": round(float(worst_missing), 2), "unit": "%"},
                ],
            )
        )

    if empty_columns:
        recommendations.append(
            _build_recommendation(
                "Drop Empty Columns",
                "Some columns contain no valid observations and add noise to pipelines.",
                priority="high",
                category="data_quality",
                columns=empty_columns,
                action="Remove empty columns or fix upstream ingestion to populate them.",
                why_it_matters="Empty columns waste compute and complicate schema management.",
                tags=["empty_column"],
                signal_type="empty_column",
                metrics=[{"name": "empty_columns", "value": len(empty_columns)}],
            )
        )

    if constant_columns:
        recommendations.append(
            _build_recommendation(
                "Remove Constant Features",
                "Columns with a single value provide no predictive signal.",
                priority="medium",
                category="data_quality",
                columns=constant_columns,
                action="Drop or combine constant columns prior to modeling.",
                tags=["low_variance"],
                signal_type="low_variance",
            )
        )

    if free_text_columns:
        free_text_names = [col["name"] for col in free_text_columns]
        recommendations.append(
            _build_recommendation(
                "NLP Opportunity",
                "Detected free-text columns suited for NLP feature engineering.",
                priority="medium",
                category="text_preprocessing",
                columns=free_text_names,
                action="Add tokenization, normalization, and embedding steps before modeling.",
                tags=["nlp"],
                signal_type="nlp",
                metrics=[{"name": "text_columns", "value": len(free_text_names)}],
            )
        )

    high_cardinality = [
        col["name"]
        for col in categorical_text_columns
        if col.get("unique_count", 0) > 50
    ]
    if high_cardinality:
        recommendations.append(
            _build_recommendation(
                "High Cardinality Categorical Features",
                "Categorical columns have many unique labels; group rare values or use target encoding.",
                priority="medium",
                category="categorical_encoding",
                columns=high_cardinality,
                tags=["high_cardinality"],
                signal_type="high_cardinality",
            )
        )

    if pii_columns:
        pii_sorted = sorted(pii_columns)
        recommendations.append(
            _build_recommendation(
                "Protect Sensitive Columns",
                "Potential PII detected. Mask, hash, or drop before sharing or modeling.",
                priority="critical",
                category="privacy",
                columns=pii_sorted,
                tags=["pii"],
                signal_type="pii",
            )
        )

    for flag_summary in text_quality_flags_summary:
        recommendations.append(
            _build_recommendation(
                f"Improve Text Quality: {flag_summary['flag']}",
                "Text quality flags detected; plan cleaning steps to resolve them.",
                priority="high",
                category="text_quality",
                columns=flag_summary["columns"],
                tags=[flag_summary["flag"], "text_quality"],
                signal_type="text_quality",
                metrics=[{"name": "columns_affected", "value": len(flag_summary["columns"])}],
            )
        )

    for pattern in detected_patterns_summary:
        if _pattern_type_is_pii(pattern["pattern_type"]):
            continue
        recommendations.append(
            _build_recommendation(
                f"Detected {pattern['pattern_type'].title()} Patterns",
                "Structured patterns were found; consider dedicated parsing or validation logic.",
                priority="medium",
                category="pattern_detection",
                columns=pattern["columns"],
                tags=[pattern["pattern_type"], "pattern_detection"],
                signal_type="pattern_detection",
                metrics=[
                    {"name": "columns_affected", "value": len(pattern["columns"])},
                    {"name": "pattern_hits", "value": pattern.get("total_count", 0)},
                ],
            )
        )

    insights: List[str] = []
    duplicate_rows = int(working_df.duplicated().sum())
    if duplicate_rows:
        insights.append(f"Dataset contains {duplicate_rows} duplicate rows")

    many_missing = [col["name"] for col in high_missing if col.get("null_percentage", 0) > 60]
    if many_missing:
        insights.append(
            "Columns with extreme missingness (>60%): " + ", ".join(many_missing[:5])
        )

    outlier_heavy = [
        col["name"]
        for col in column_quality
        if col.get("data_category") == "numeric" and col.get("outlier_count", 0) > int(total_rows * 0.1)
    ]
    if outlier_heavy:
        insights.append(
            "Numeric outliers detected in: " + ", ".join(outlier_heavy[:5])
        )

    summary = {
        "rows": total_rows,
        "columns": total_cols,
        "completeness_pct": round(completeness, 2),
        "duplicate_rows": duplicate_rows,
    }

    report = QualityReport(
        success=True,
        columns=column_quality,
        summary=summary,
        quality_checks=quality_checks,
        quality_score=round(quality_score, 2),
        recommendations=recommendations,
        text_patterns=detected_patterns_summary,
        text_quality_flags=text_quality_flags_summary,
        insights=insights,
    )
    return report.to_dict()
