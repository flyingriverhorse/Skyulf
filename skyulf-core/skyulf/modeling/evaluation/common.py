"""Shared helpers for model evaluation modules."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .schemas import CurvePoint


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, (int, np.integer)):
        return True
    return False


def _sanitize_structure(value: Any, *, warnings: List[str], context: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_structure(inner, warnings=warnings, context=context)
            for key, inner in value.items()
        }
    if isinstance(value, (list, tuple)):
        sanitized_items = [
            _sanitize_structure(item, warnings=warnings, context=context)
            for item in value
        ]
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
    return np.unique(indices)  # type: ignore


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


def sanitize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Sanitize metrics dictionary to ensure JSON compliance."""
    warnings: List[str] = []
    sanitized = _sanitize_structure(metrics, warnings=warnings, context="metrics")
    # Filter out None values that resulted from non-finite numbers
    return {k: v for k, v in sanitized.items() if v is not None}


def downsample_curve(
    x: np.ndarray, y: np.ndarray, limit: int = 1000
) -> List[CurvePoint]:
    """Downsample curve points to a reasonable limit."""
    indices = _downsample_indices(len(x), limit)

    x_sampled = x[indices]
    y_sampled = y[indices]

    points = []
    for i in range(len(x_sampled)):
        points.append(CurvePoint(x=float(x_sampled[i]), y=float(y_sampled[i])))

    return points
