"""Shared helpers for model evaluation modules."""

from __future__ import annotations

import math
from typing import Any, Iterable, List

import numpy as np


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, (int, np.integer)):
        return True
    return False


def _sanitize_structure(value: Any, *, warnings: List[str], context: str) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_structure(inner, warnings=warnings, context=context) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        sanitized_items = [_sanitize_structure(item, warnings=warnings, context=context) for item in value]
        return type(value)(sanitized_items)
    if isinstance(value, (float, np.floating, int, np.integer)):
        if _is_finite_number(value):
            if isinstance(value, (float, np.floating)):
                return float(value)
            return int(value)
        warnings.append(f"Removed non-finite numeric value from {context}.")
        return None
    return value


def _downsample_indices(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    indices = np.linspace(0, length - 1, num=limit, dtype=int)
    return np.unique(indices)


def _align_thresholds(thresholds: np.ndarray, target_size: int) -> np.ndarray:
    if thresholds.size == target_size:
        return thresholds
    if thresholds.size == 0:
        return np.zeros(target_size, dtype=float)
    if thresholds.size == target_size - 1:
        return np.append(thresholds, thresholds[-1])
    if thresholds.size > target_size:
        return thresholds[:target_size]
    pad_size = target_size - thresholds.size
    return np.append(thresholds, np.full(pad_size, thresholds[-1]))


__all__ = [
    "_align_thresholds",
    "_downsample_indices",
    "_sanitize_structure",
]
