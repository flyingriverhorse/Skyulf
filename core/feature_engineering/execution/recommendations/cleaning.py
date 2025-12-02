"""Cleaning recommendation logic."""

from dataclasses import dataclass, field
from statistics import median, StatisticsError
from typing import Any, Dict, Iterable, List, Optional, Set

DROP_COLUMN_FILTER_LABELS: Dict[str, Dict[str, Optional[str]]] = {
    "missing_data": {
        "label": "Missing Data",
        "description": "Columns exceeding the configured missingness threshold.",
    },
    "empty_column": {
        "label": "Empty Columns",
        "description": "Columns that are completely empty (100% missing).",
    },
    "single_value": {
        "label": "Single Value",
        "description": "Columns containing only a single unique value (zero variance).",
    },
    "high_cardinality": {
        "label": "High Cardinality",
        "description": "Categorical columns with too many unique values.",
    },
    "id_column": {
        "label": "ID Columns",
        "description": "Columns detected as identifiers (e.g., sequential IDs).",
    },
    "leakage": {
        "label": "Data Leakage",
        "description": "Columns highly correlated with the target variable.",
    },
    "collinear": {
        "label": "Collinear Features",
        "description": "Columns highly correlated with other features.",
    },
    "other": {
        "label": "Other",
        "description": "Columns flagged for other reasons.",
    },
}


@dataclass
class DropColumnRecommendationFilter:
    id: str
    label: str
    count: int
    description: Optional[str] = None


@dataclass
class DropColumnCandidateEntry:
    name: str
    missing_percentage: Optional[float] = None
    priority: Optional[str] = None
    reasons: Set[str] = field(default_factory=set)
    signals: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


def _build_drop_filter_meta(filter_id: str) -> Dict[str, Optional[str]]:
    base = DROP_COLUMN_FILTER_LABELS.get(filter_id, {})
    label = base.get("label") or filter_id.replace("_", " ").title() or "Other"
    description = base.get("description")
    return {"label": label, "description": description}


@dataclass
class DropColumnRecommendationBuilder:
    candidates: Dict[str, DropColumnCandidateEntry] = field(default_factory=dict)
    column_missing_map: Dict[str, float] = field(default_factory=dict)
    missing_values: List[float] = field(default_factory=list)
    all_columns: Set[str] = field(default_factory=set)

    def _normalize_name(self, column: Any) -> str:
        if column is None:
            return ""
        return str(column).strip()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def record_missing_percentage(
        self,
        column: Any,
        value: Any,
        *,
        include_in_stats: bool = False,
    ) -> Optional[float]:
        name = self._normalize_name(column)
        if not name:
            return None

        numeric = self._safe_float(value)
        if numeric is None or numeric < 0:
            return None

        current = self.column_missing_map.get(name, 0.0)
        self.column_missing_map[name] = max(current, numeric)
        if include_in_stats:
            self.missing_values.append(numeric)

        self.all_columns.add(name)
        return numeric

    def register_candidate(
        self,
        column: Any,
        *,
        reason: str,
        priority: Optional[str] = None,
        missing_pct: Optional[float] = None,
        signals: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        name = self._normalize_name(column)
        if not name:
            return

        entry = self.candidates.setdefault(name, DropColumnCandidateEntry(name=name))

        normalized_reason = str(reason).strip()
        if normalized_reason:
            entry.reasons.add(normalized_reason)

        normalized_priority = str(priority).strip() if priority else None
        if normalized_priority and not entry.priority:
            entry.priority = normalized_priority

        if signals:
            entry.signals.update(
                {
                    signal
                    for signal in (str(item).strip() for item in signals)
                    if signal
                }
            )

        if tags:
            entry.tags.update(
                {
                    tag
                    for tag in (str(item).strip() for item in tags)
                    if tag
                }
            )

        if missing_pct is not None:
            numeric_pct = self.record_missing_percentage(name, missing_pct)
            if numeric_pct is not None:
                if entry.missing_percentage is None or numeric_pct > entry.missing_percentage:
                    entry.missing_percentage = numeric_pct

        self.all_columns.add(name)

    def ingest_missing_summary(self, missing_summary: Iterable[Any]) -> None:
        for record in missing_summary:
            column_name = record.get("column") if isinstance(record, dict) else None
            missing_pct = record.get("missing_percentage") if isinstance(record, dict) else None

            numeric_missing = self.record_missing_percentage(column_name, missing_pct, include_in_stats=True)
            if numeric_missing is None:
                continue

            if numeric_missing >= 30.0:
                priority = "critical" if numeric_missing >= 85 else "high" if numeric_missing >= 60 else "medium"
                signal_labels = ["missing_data"]
                if numeric_missing >= 99.5:
                    signal_labels.append("empty_column")
                self.register_candidate(
                    column_name,
                    reason="High missingness",
                    priority=priority,
                    missing_pct=numeric_missing,
                    signals=signal_labels,
                )

    def ingest_eda_recommendations(self, recommendations: Iterable[Any]) -> None:
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue

            columns = recommendation.get("columns")
            if not columns:
                continue

            category = str(recommendation.get("category", "")).strip()
            if category and category not in {"data_quality", "missing_data", "feature_engineering"}:
                continue

            reason = recommendation.get("title") or recommendation.get("description") or "EDA recommendation"
            priority = recommendation.get("priority")
            signal_type = recommendation.get("signal_type")
            tags = recommendation.get("tags")

            for column in columns:
                self.register_candidate(
                    column,
                    reason=str(reason),
                    priority=str(priority) if priority else None,
                    signals=[signal_type] if signal_type else None,
                    tags=tags,
                )

    def collect_column_details(self, column_details: Iterable[Any]) -> None:
        for detail in column_details:
            if not isinstance(detail, dict):
                continue

            column_name = detail.get("name") or detail.get("column")
            normalized_name = self._normalize_name(column_name)
            if not normalized_name:
                continue

            detail_missing = (
                detail.get("missing_percentage")
                or detail.get("missing_percent")
                or detail.get("missing_pct")
                or detail.get("missing")
            )
            self.record_missing_percentage(normalized_name, detail_missing)

    def collect_sample_preview(self, quality_report: Dict[str, Any]) -> None:
        sample_preview = quality_report.get("sample_preview") or {}
        columns = sample_preview.get("columns") or []
        for column in columns:
            normalized = self._normalize_name(column)
            if normalized:
                self.all_columns.add(normalized)

    def build_candidate_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for entry in self.candidates.values():
            payload.append(
                {
                    "name": entry.name,
                    "reason": ", ".join(sorted(entry.reasons)),
                    "missing_percentage": entry.missing_percentage,
                    "priority": entry.priority,
                    "signals": sorted(entry.signals),
                    "tags": sorted(entry.tags),
                }
            )
        return payload

    def filter_candidates(
        self,
        candidate_payload: List[Dict[str, Any]],
        allowed_columns: Optional[Set[str]],
    ) -> List[Dict[str, Any]]:
        if allowed_columns is None:
            return candidate_payload

        normalized_allowed = {
            self._normalize_name(column)
            for column in allowed_columns
            if self._normalize_name(column)
        }

        filtered = [
            candidate
            for candidate in candidate_payload
            if candidate.get("name") in normalized_allowed
        ]

        self.column_missing_map = {
            name: value
            for name, value in self.column_missing_map.items()
            if name in normalized_allowed
        }
        self.all_columns = {column for column in self.all_columns if column in normalized_allowed}
        self.all_columns.update(normalized_allowed)
        return filtered

    def finalize_all_columns(
        self,
        candidate_payload: Iterable[Dict[str, Any]],
        allowed_columns: Optional[Set[str]],
    ) -> List[str]:
        for candidate in candidate_payload:
            name = self._normalize_name(candidate.get("name"))
            if name:
                self.all_columns.add(name)

        if allowed_columns is not None:
            normalized_allowed = {
                self._normalize_name(column)
                for column in allowed_columns
                if self._normalize_name(column)
            }
            self.all_columns = {column for column in self.all_columns if column in normalized_allowed}
            self.all_columns.update(normalized_allowed)

        sorted_columns = sorted(self.all_columns)
        for column in sorted_columns:
            self.column_missing_map.setdefault(column, 0.0)

        return sorted_columns

    def build_column_missing_map(self, columns: Iterable[str]) -> Dict[str, float]:
        return {name: float(self.column_missing_map.get(name, 0.0)) for name in columns}

    def build_filters(self, candidate_payload: Iterable[Dict[str, Any]]) -> List[DropColumnRecommendationFilter]:
        available_filters_map: Dict[str, Dict[str, Any]] = {}
        orphan_candidates = 0

        for candidate in candidate_payload:
            signals = candidate.get("signals") or []
            if signals:
                for signal in signals:
                    if not signal:
                        continue
                    meta = available_filters_map.setdefault(
                        signal,
                        {
                            "id": signal,
                            **_build_drop_filter_meta(signal),
                            "count": 0,
                        },
                    )
                    meta["count"] += 1
            else:
                orphan_candidates += 1

        if orphan_candidates:
            other_meta = available_filters_map.setdefault(
                "other",
                {
                    "id": "other",
                    **_build_drop_filter_meta("other"),
                    "count": 0,
                },
            )
            other_meta["count"] += orphan_candidates

        filters = [
            DropColumnRecommendationFilter(**value)
            for value in available_filters_map.values()
            if value.get("count", 0) > 0
        ]
        filters.sort(key=lambda item: (-item.count, item.label))
        return filters

    def suggested_threshold(self) -> float:
        if not self.missing_values:
            return 40.0

        try:
            return float(max(20.0, min(95.0, median(self.missing_values))))
        except StatisticsError:
            return 40.0

    @staticmethod
    def sort_candidates(candidate_payload: List[Dict[str, Any]]) -> None:
        candidate_payload.sort(
            key=lambda item: (
                item.get("missing_percentage") is None,
                -(item.get("missing_percentage") or 0.0),
                item.get("name", ""),
            )
        )
