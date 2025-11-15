"""Node-level helpers for the evaluation canvas."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.feature_engineering.schemas import ModelEvaluationNodeSignal

_DEFAULT_SPLITS: Tuple[str, ...] = ("train", "validation", "test")


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _normalize_splits(raw_value: Any) -> List[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, str):
        parts = [entry.strip().lower() for entry in raw_value.split(",")]
    elif isinstance(raw_value, Iterable):
        parts = [str(entry).strip().lower() for entry in raw_value]
    else:
        return []
    normalized: List[str] = []
    for entry in parts:
        if not entry:
            continue
        if entry in {"train", "training"}:
            normalized.append("train")
        elif entry in {"validation", "valid", "val"}:
            normalized.append("validation")
        elif entry in {"test", "testing"}:
            normalized.append("test")
    return normalized


def apply_model_evaluation(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    *,
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, ModelEvaluationNodeSignal]:
    """Return unchanged frame alongside light-weight node diagnostics."""

    node_id = node.get("id")
    node_id_str = str(node_id) if node_id is not None else None

    config = (node.get("data") or {}).get("config") or {}
    raw_job_id = config.get("training_job_id")
    training_job_id: Optional[str] = None
    if isinstance(raw_job_id, str):
        stripped = raw_job_id.strip()
        training_job_id = stripped or None
    elif raw_job_id is not None:
        training_job_id = str(raw_job_id)

    raw_splits = config.get("splits")
    splits = _normalize_splits(raw_splits)
    if not splits and training_job_id:
        splits = list(_DEFAULT_SPLITS)

    last_evaluated = _parse_iso_datetime(config.get("last_evaluated_at"))

    signal = ModelEvaluationNodeSignal(
        node_id=node_id_str,
        training_job_id=training_job_id,
        splits=splits,
        has_evaluation=bool(last_evaluated),
        last_evaluated_at=last_evaluated,
        notes=[],
    )

    if not training_job_id:
        signal.notes.append("Select a training job from the sidebar to unlock evaluation diagnostics.")
        summary = "Model evaluation: waiting for training job selection"
        return frame, summary, signal

    if not splits:
        signal.notes.append("No dataset splits selected; configure at least one to run diagnostics.")
        summary = f"Model evaluation for job {training_job_id}: no splits selected"
        return frame, summary, signal

    split_label = ", ".join(splits)
    summary = f"Model evaluation configured for job {training_job_id} on {split_label}"
    return frame, summary, signal


__all__ = ["apply_model_evaluation"]
