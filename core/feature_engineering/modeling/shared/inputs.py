"""Shared helpers for loading modeling inputs and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .common import _resolve_training_inputs
from ..training.registry import get_model_spec


@dataclass(frozen=True)
class ModelingInputs:
    frame: pd.DataFrame
    node_config: Dict[str, Any]
    dataset_meta: Optional[Dict[str, Any]]
    upstream_order: List[Any]


async def _load_modeling_inputs(session, job) -> ModelingInputs:
    frame, node_config, dataset_meta, upstream_order = await _resolve_training_inputs(session, job)
    config_dict = node_config if isinstance(node_config, dict) else {}

    if isinstance(upstream_order, list):
        upstream_list = upstream_order
    elif upstream_order is None:
        upstream_list = []
    else:
        try:
            upstream_list = list(upstream_order)
        except TypeError:  # pragma: no cover - safeguard for unexpected input types
            upstream_list = [upstream_order]

    return ModelingInputs(
        frame=frame,
        node_config=config_dict,
        dataset_meta=dataset_meta,
        upstream_order=upstream_list,
    )


def _extract_target_column(node_config: Dict[str, Any], job) -> str:
    job_metadata = job.job_metadata or {}
    target_column = (
        node_config.get("target_column")
        or node_config.get("targetColumn")
        or job_metadata.get("target_column")
    )
    if not target_column:
        raise ValueError("Configuration missing target column")
    return str(target_column)


def _extract_problem_type_hint(node_config: Dict[str, Any]) -> str:
    raw = None
    if isinstance(node_config, dict):
        raw = node_config.get("problem_type") or node_config.get("problemType")
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            return text
    return "auto"


def _resolve_model_spec_from_job(job) -> Any:
    model_type = getattr(job, "model_type", None)
    if not model_type:
        raise ValueError("Job missing model_type")
    return get_model_spec(model_type)


__all__ = [
    "ModelingInputs",
    "_extract_problem_type_hint",
    "_extract_target_column",
    "_load_modeling_inputs",
    "_resolve_model_spec_from_job",
]
