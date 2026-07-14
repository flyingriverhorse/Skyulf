"""Shared helpers and constants for feature-generation nodes."""

import math
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

# --- Optional Dependencies ---
fuzz: Any = None
try:
    from rapidfuzz import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
        fuzz as _fuzz,
    )

    fuzz = _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# --- Constants ---
DEFAULT_EPSILON = 1e-9
FEATURE_MATH_ALLOWED_TYPES = {
    "arithmetic",
    "ratio",
    "similarity",
    "datetime_extract",
    "group_agg",
    "polynomial",
}
ALLOWED_DATETIME_FEATURES = {
    "year",
    "quarter",
    "month",
    "month_name",
    "week",
    "day",
    "day_name",
    "weekday",
    "is_weekend",
    "hour",
    "minute",
    "second",
    "season",
    "time_of_day",
}

_FUZZ_METHODS: dict[str, str] = {
    "token_sort_ratio": "token_sort_ratio",
    "token_set_ratio": "token_set_ratio",
}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
        return val if not math.isnan(val) else None
    except (TypeError, ValueError):
        return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    adjusted = denominator.copy()
    near_zero = adjusted.isna() | (adjusted.abs() < epsilon)
    if near_zero.any():
        # Preserve the sign of the original denominator when clamping its
        # magnitude to epsilon, so a small negative denominator doesn't flip
        # the sign of the result. NaN defaults to positive epsilon.
        signed_epsilon = pd.Series(
            np.copysign(epsilon, adjusted.fillna(1.0).to_numpy()), index=adjusted.index
        )
        adjusted = adjusted.where(~near_zero, signed_epsilon)
    return numerator / adjusted


def _compute_similarity_score(a: Any, b: Any, method: str) -> float:
    text_a = "" if pd.isna(a) else str(a)
    text_b = "" if pd.isna(b) else str(b)
    if not text_a and not text_b:
        return 100.0
    if not text_a or not text_b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        attr = _FUZZ_METHODS.get(method, "ratio")
        return float(getattr(fuzz, attr)(text_a, text_b))
    return SequenceMatcher(None, text_a, text_b).ratio() * 100.0


def _vectorised_similarity(s_a: pd.Series, s_b: pd.Series, method: str) -> pd.Series:
    """Vectorised element-wise similarity between two pandas string Series.

    Avoids df.apply(axis=1) overhead by:
    1. Pre-computing null/empty masks with numpy/pandas ops (vectorised).
    2. Only invoking the per-pair Python similarity for rows that need it.
    """
    a_str = s_a.fillna("").astype(str)
    b_str = s_b.fillna("").astype(str)
    a_empty = a_str.eq("")
    b_empty = b_str.eq("")
    both_empty = a_empty & b_empty
    one_empty = (a_empty | b_empty) & ~both_empty
    needs_compute = ~(a_empty | b_empty)

    result = pd.Series(0.0, index=s_a.index, dtype=float)
    result.loc[both_empty] = 100.0
    result.loc[one_empty] = 0.0

    if needs_compute.any():
        idx = needs_compute[needs_compute].index
        for i in idx:
            result.loc[i] = _compute_similarity_score(a_str.at[i], b_str.at[i], method)

    return result


def _resolve_output_col(
    op: dict[str, Any], i: int, existing: list[str], allow_overwrite: bool
) -> str:
    """Pick a non-colliding output column name for op ``i``."""
    output_col = op.get("output_column")
    if not output_col:
        base = f"{op.get('operation_type', 'arithmetic')}_{i}"
        prefix = op.get("output_prefix")
        output_col = f"{prefix}_{base}" if prefix else base
    if output_col in existing and not allow_overwrite:
        j = 1
        while f"{output_col}_{j}" in existing:
            j += 1
        output_col = f"{output_col}_{j}"
    return output_col


def _resolve_similarity_pair(op: dict[str, Any], existing: list[str]) -> tuple[str, str] | None:
    """Return ``(col_a, col_b)`` for similarity ops, or ``None`` if unresolved."""
    inputs = op.get("input_columns", [])
    secondary = op.get("secondary_columns", [])
    col_a = inputs[0] if inputs else None
    col_b = secondary[0] if secondary else (inputs[1] if len(inputs) > 1 else None)
    if not col_a or not col_b or col_a not in existing or col_b not in existing:
        return None
    return col_a, col_b


def _resolve_group_agg_cols(
    op: dict[str, Any], existing: list[str]
) -> tuple[str, str, str] | None:
    """Return ``(group_col, target_col, method)`` if op is well-formed, else None."""
    group_cols = [c for c in op.get("input_columns", []) if c in existing]
    target_cols = [c for c in op.get("secondary_columns", []) if c in existing]
    if not group_cols or not target_cols:
        return None
    return group_cols[0], target_cols[0], op.get("method") or "mean"
