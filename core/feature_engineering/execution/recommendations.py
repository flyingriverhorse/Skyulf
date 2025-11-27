"""Recommendation logic and helpers."""

import json
import logging
from dataclasses import dataclass, field
from statistics import median, StatisticsError
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_engineering.eda_fast import FeatureEngineeringEDAService
from core.feature_engineering.execution.data import build_preview_frame, build_eda_service
from core.feature_engineering.execution.engine import apply_recommendation_graph
from core.feature_engineering.execution.graph import (
    ensure_dataset_node,
    execution_order,
    extract_graph_payload,
    normalize_target_node,
    resolve_catalog_type,
)
from core.feature_engineering.schemas import (
    BinnedColumnDistribution,
    BinnedDistributionResponse,
)

logger = logging.getLogger(__name__)

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


def build_recommendation_column_metadata(
    quality_payload: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    column_metadata: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []

    if not quality_payload.get("success"):
        notes.append("Could not retrieve data quality metrics.")
        return column_metadata, notes

    quality_report = quality_payload.get("quality_report") or {}
    quality_metrics = quality_report.get("quality_metrics") or {}
    column_details = quality_metrics.get("column_details") or []
    if isinstance(column_details, Iterable):
        for col in column_details:
            if isinstance(col, dict):
                name = col.get("column")
                if name:
                    column_metadata[name] = col

    text_summary = quality_report.get("text_analysis_summary") or {}
    categorical_columns = text_summary.get("categorical_text_columns") or []
    if isinstance(categorical_columns, Iterable):
        for col_name in categorical_columns:
            if col_name in column_metadata:
                column_metadata[col_name]["is_categorical_text"] = True

    return column_metadata, notes


async def prepare_categorical_recommendation_context(
    *,
    eda_service: FeatureEngineeringEDAService,
    dataset_source_id: str,
    sample_size: int,
    graph: Optional[Dict[str, Any]],
    target_node_id: Optional[str],
    skip_catalog_types: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], List[str]]:
    preview_payload = await eda_service.preview_source(dataset_source_id, sample_size=sample_size)

    if not preview_payload.get("success"):
        return pd.DataFrame(), {}, ["Could not load dataset preview."]

    frame, _ = build_preview_frame(preview_payload)

    graph_node_map, graph_edges = extract_graph_payload(graph)
    normalized_target_node = normalize_target_node(target_node_id)
    frame = apply_recommendation_graph(
        frame,
        graph_node_map,
        graph_edges,
        normalized_target_node,
        skip_catalog_types,
    )

    quality_payload = await eda_service.quality_report(dataset_source_id, sample_size=sample_size)
    column_metadata, notes = build_recommendation_column_metadata(quality_payload)

    return frame, column_metadata, notes


def parse_skewness_transformations(raw_payload: Optional[str]) -> Dict[str, str]:
    if not raw_payload:
        return {}

    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return {}

    parsed: Dict[str, str] = {}

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                col = item.get("column")
                method = item.get("method")
                if col and method:
                    parsed[col] = method
    elif isinstance(payload, dict):
        for col, method in payload.items():
            if isinstance(col, str) and isinstance(method, str):
                parsed[col] = method
    else:
        pass

    return parsed


def collect_skewness_transformations_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, str]:
    selections: Dict[str, str] = {}

    for node_id in execution_order:
        if node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = resolve_catalog_type(node)
        if catalog_type == "skewness_transform":
            config = node.get("data", {}).get("config", {})
            method = config.get("method")
            columns = config.get("columns", [])
            
            # If method is 'auto', we can't know the specific transform without running it.
            # But if it's explicit (log, sqrt, box-cox), we can track it.
            if method and columns:
                for col in columns:
                    selections[col] = method

    return selections


def collect_binned_columns_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    columns: Dict[str, Dict[str, Any]] = {}

    for node_id in execution_order:
        if node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = resolve_catalog_type(node)
        if catalog_type == "binning_discretization":
            config = node.get("data", {}).get("config", {})
            cols = config.get("columns", [])
            method = config.get("method")
            bins = config.get("bins")
            
            for col in cols:
                columns[col] = {
                    "method": method,
                    "bins": bins,
                    "node_id": node_id
                }

    return columns


def build_candidate_binned_columns(
    frame: pd.DataFrame,
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    candidate_columns: Dict[str, Dict[str, Any]] = {}

    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
            
        # Skip low cardinality (likely categorical/ordinal)
        unique_count = frame[column].nunique()
        if unique_count < 10:
            continue
            
        col_meta = metadata.get(column, {})
        candidate_columns[column] = {
            "column": column,
            "type": str(frame[column].dtype),
            "unique_count": unique_count,
            "missing_count": frame[column].isna().sum(),
            "min": float(frame[column].min()) if not frame[column].empty else 0,
            "max": float(frame[column].max()) if not frame[column].empty else 0,
            **col_meta
        }

    return candidate_columns


def build_binned_distributions_list(
    frame: pd.DataFrame,
    candidate_columns: Dict[str, Dict[str, Any]],
) -> List[BinnedColumnDistribution]:
    distributions: List[BinnedColumnDistribution] = []

    for column_name, metadata in candidate_columns.items():
        try:
            # Simple equal-width binning for visualization
            # In a real app, we might want to use the configured method if available
            series = frame[column_name].dropna()
            if series.empty:
                continue
                
            # Create histogram
            hist_values, bin_edges = pd.cut(series, bins=20, retbins=True, duplicates='drop')
            counts = hist_values.value_counts().sort_index()
            
            dist_data = []
            for interval, count in counts.items():
                dist_data.append({
                    "bin_start": float(interval.left),
                    "bin_end": float(interval.right),
                    "count": int(count)
                })
                
            distributions.append(
                BinnedColumnDistribution(
                    column=column_name,
                    distribution=dist_data,
                    top_count=int(counts.max()) if not counts.empty else 0
                )
            )
        except Exception as e:
            logger.warning(f"Failed to build distribution for {column_name}: {e}")

    distributions.sort(
        key=lambda item: (
            -(item.top_count or 0),
            item.column.lower(),
        )
    )

    return distributions


async def generate_binned_distribution_response(
    session: AsyncSession,
    *,
    dataset_source_id: str,
    sample_size: int,
    graph_input: Any,
    target_node_id: Optional[str],
) -> BinnedDistributionResponse:
    
    eda_service = build_eda_service(session, sample_size)
    
    # Load data
    frame, column_metadata, notes = await prepare_categorical_recommendation_context(
        eda_service=eda_service,
        dataset_source_id=dataset_source_id,
        sample_size=sample_size,
        graph=graph_input,
        target_node_id=target_node_id,
    )
    
    if frame.empty:
        return BinnedDistributionResponse(
            dataset_source_id=dataset_source_id,
            distributions=[],
            notes=notes,
        )
        
    # Identify candidates
    candidates = build_candidate_binned_columns(frame, column_metadata)
    
    # Build distributions
    distributions = build_binned_distributions_list(frame, candidates)
    
    return BinnedDistributionResponse(
        dataset_source_id=dataset_source_id,
        distributions=distributions,
        notes=notes,
    )


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
