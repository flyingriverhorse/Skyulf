"""Transformer audit node.

This node surfaces metadata about stored transformers, highlighting how they
interacted with train/test/validation splits. It is intended for inspection
purposes and does not mutate the dataframe passing through.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import pandas as pd

from core.feature_engineering.schemas import (
    TransformerAuditEntrySignal,
    TransformerAuditNodeSignal,
    TransformerSplitActivitySignal,
)
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store

logger = logging.getLogger(__name__)

TransformerSplit = Literal["train", "test", "validation", "other", "unknown"]
TransformerAction = Literal["fit_transform", "transform", "not_applied", "not_available"]

_KNOWN_SPLITS: Tuple[TransformerSplit, ...] = ("train", "test", "validation")
_SPLIT_ORDER: Dict[TransformerSplit, int] = {name: index for index, name in enumerate(_KNOWN_SPLITS)}


def _normalize_split_name(raw_name: Any) -> Tuple[TransformerSplit, str]:
    """Return normalized split identifier along with a display label."""

    if isinstance(raw_name, str):
        candidate = raw_name.strip()
    else:
        candidate = str(raw_name or "").strip()

    lowered = candidate.lower()
    if lowered in {"train", "training"}:
        return "train", "Train"
    if lowered in {"test", "testing"}:
        return "test", "Test"
    if lowered in {"validation", "valid", "val"}:
        return "validation", "Validation"
    if not candidate:
        return "unknown", "Unknown"
    return "other", candidate or "Other"


def _normalize_action(action: Any) -> TransformerAction:
    if isinstance(action, str):
        candidate = action.strip().lower().replace("-", "_")
        if candidate in {"fit_transform", "transform", "not_applied", "not_available"}:
            return cast(TransformerAction, candidate)
    return "not_applied"


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _build_label_map(node_map: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
    if not isinstance(node_map, dict):
        return {}

    lookup: Dict[str, str] = {}
    for raw_id, entry in node_map.items():
        if raw_id is None:
            continue
        node_id = str(raw_id)
        if not node_id:
            continue

        label: Optional[str] = None
        if isinstance(entry, dict):
            if isinstance(entry.get("label"), str):
                label = entry["label"].strip() or None
            if not label:
                data = entry.get("data")
                if isinstance(data, dict):
                    data_label = data.get("label")
                    if isinstance(data_label, str):
                        label = data_label.strip() or None
        lookup[node_id] = label or node_id

    return lookup


def _sort_split_entries(entries: Iterable[TransformerSplitActivitySignal]) -> List[TransformerSplitActivitySignal]:
    def sort_key(item: TransformerSplitActivitySignal) -> Tuple[int, str]:
        return (_SPLIT_ORDER.get(item.split, len(_SPLIT_ORDER)), item.label or item.split)

    return sorted(entries, key=sort_key)


def apply_transformer_audit(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    *,
    pipeline_id: Optional[str] = None,
    node_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, str, TransformerAuditNodeSignal]:
    """Inspect transformer storage for the current pipeline."""

    node_id = node.get("id") if isinstance(node, dict) else None
    node_id_str = str(node_id) if node_id is not None else None

    signal = TransformerAuditNodeSignal(
        node_id=node_id_str,
        pipeline_id=pipeline_id,
    )

    if not pipeline_id:
        signal.notes.append("Pipeline context unavailable – unable to inspect transformers.")
        logger.warning(
            "Transformer audit called without pipeline_id",
            extra={"node_id": node_id_str}
        )
        return frame, "Transformer audit: pipeline context unavailable", signal

    pipeline_store = get_pipeline_store()

    pipeline_stats = pipeline_store.get_stats()

    records = pipeline_store.list_pipelines(pipeline_id=pipeline_id)
    if not records:
        signal.notes.append("No stored transformers detected for this pipeline.")
        signal.notes.append(
            f"Pipeline store entries total: {pipeline_stats.get('total_transformers', 0)}"
        )
        if pipeline_stats.get('storage_keys'):
            signal.notes.append(
                "No transformer activity recorded yet. Run a preview after executing split-aware "
                "transformation nodes."
            )
        return frame, "Transformer audit: no transformers recorded", signal

    label_map = _build_label_map(node_map)

    for record in records:
        source_node_id = record.get("node_id")
        source_node_id_str = str(source_node_id) if source_node_id is not None else None
        transformer_name = str(record.get("transformer_name") or "transformer")
        raw_metadata = record.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        column_name = record.get("column_name")
        column_label = str(column_name) if column_name is not None and str(column_name).strip() else None
        if not column_label and isinstance(metadata, dict):
            if isinstance(metadata.get("column_summary"), str) and metadata["column_summary"].strip():
                column_label = metadata["column_summary"].strip()
            elif isinstance(metadata.get("input_columns"), (list, tuple)):
                joined = ", ".join(
                    str(column)
                    for column in metadata["input_columns"]
                    if isinstance(column, str) and column.strip()
                ).strip()
                if joined:
                    column_label = joined
        created_at = _coerce_datetime(record.get("created_at"))
        updated_at = _coerce_datetime(record.get("updated_at")) or created_at

        split_activity_map = record.get("split_activity")
        split_entries: Dict[TransformerSplit, TransformerSplitActivitySignal] = {}
        if isinstance(split_activity_map, dict):
            for raw_split, payload in split_activity_map.items():
                norm_split, label = _normalize_split_name(raw_split)
                action_value: TransformerAction = "not_applied"
                row_count: Optional[int] = None
                updated_value: Any = None
                if isinstance(payload, dict):
                    if payload.get("action"):
                        action_value = _normalize_action(payload.get("action"))
                    row_count = payload.get("row_count")
                    updated_value = payload.get("updated_at")

                try:
                    rows_numeric = int(row_count) if row_count is not None else None
                except (TypeError, ValueError):
                    rows_numeric = None
                split_updated = _coerce_datetime(updated_value)

                split_entries[norm_split] = TransformerSplitActivitySignal(
                    split=norm_split,
                    action=action_value,
                    rows=rows_numeric,
                    updated_at=split_updated,
                    label=label,
                )

        for expected_split in _KNOWN_SPLITS:
            if expected_split not in split_entries:
                display_label = expected_split.capitalize()
                split_entries[expected_split] = TransformerSplitActivitySignal(
                    split=expected_split,
                    action="not_available",
                    rows=None,
                    updated_at=None,
                    label=display_label,
                )

        entry_signal = TransformerAuditEntrySignal(
            source_node_id=source_node_id_str,
            source_node_label=label_map.get(source_node_id_str) if source_node_id_str else None,
            transformer_name=transformer_name,
            column_name=column_label,
            created_at=created_at,
            updated_at=updated_at,
            split_activity=_sort_split_entries(split_entries.values()),
            metadata=metadata,
            storage_key=record.get("storage_key"),
        )

        signal.transformers.append(entry_signal)

    signal.total_transformers = len(signal.transformers)
    signal.transformers.sort(
        key=lambda entry: (
            entry.source_node_label or "",
            entry.transformer_name,
            entry.column_name or "",
        )
    )

    summary = (
        "Transformer audit: "
        f"{signal.total_transformers} transformer{'s' if signal.total_transformers != 1 else ''} tracked"
    )
    return frame, summary, signal


__all__ = ["apply_transformer_audit"]
