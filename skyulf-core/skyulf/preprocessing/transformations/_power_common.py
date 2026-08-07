"""Shared PowerTransformer reconstruction, used by both ``power.py`` (fitted
subset of columns) and ``general.py`` (fitted single columns via the general
transformation node). Both need to rebuild a fitted sklearn ``PowerTransformer``
(plus its optional internal ``StandardScaler``) from artifact-stored lambdas
and scaler params, at apply time, without a real ``.fit()`` call.
"""

from typing import Any

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler


def _narrow_scaler_values(
    values: Any,
    col_indices: list[int] | None,
    n_total_cols: int | None,
) -> np.ndarray:
    """Narrow stored scaler values to the columns being transformed."""
    values_arr = np.array(values)
    if col_indices is not None and n_total_cols is not None and len(values_arr) == n_total_cols:
        return values_arr[col_indices]
    return values_arr


def _build_pretrained_scaler(
    scaler_params: dict[str, Any],
    col_indices: list[int] | None,
    n_total_cols: int | None,
) -> StandardScaler | None:
    """Reconstruct a fitted StandardScaler from stored mean and scale values."""
    mean = scaler_params.get("mean")
    scale = scaler_params.get("scale")
    if mean is None or scale is None:
        return None

    scaler = StandardScaler()
    scaler.mean_ = _narrow_scaler_values(mean, col_indices, n_total_cols)
    scaler.scale_ = _narrow_scaler_values(scale, col_indices, n_total_cols)
    scaler.var_ = np.square(scaler.scale_)
    return scaler


def build_pretrained_power_transformer(
    method: str,
    standardize: bool,
    lambdas_arr: np.ndarray,
    scaler_params: dict[str, Any] | None,
    col_indices: list[int] | None = None,
    n_total_cols: int | None = None,
) -> PowerTransformer:
    """Reconstruct a fitted PowerTransformer from stored lambdas + scaler params.

    ``col_indices``/``n_total_cols`` narrow a multi-column fit's stored
    mean/scale arrays down to the subset being applied now (used by
    ``power.py``); leave them ``None`` for a single-column fit where the
    stored arrays already match 1:1 (used by ``general.py``).
    """
    pt = PowerTransformer(method=method, standardize=standardize)
    pt.lambdas_ = lambdas_arr
    if not standardize or not scaler_params:
        return pt

    scaler = _build_pretrained_scaler(scaler_params, col_indices, n_total_cols)
    if scaler is not None:
        pt._scaler = scaler
    return pt
