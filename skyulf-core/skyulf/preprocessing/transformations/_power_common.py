"""Shared PowerTransformer reconstruction, used by both ``power.py`` (fitted
subset of columns) and ``general.py`` (fitted single columns via the general
transformation node). Both need to rebuild a fitted sklearn ``PowerTransformer``
(plus its optional internal ``StandardScaler``) from artifact-stored lambdas
and scaler params, at apply time, without a real ``.fit()`` call.
"""

from typing import Any

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler


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

    scaler = StandardScaler()
    mean = scaler_params.get("mean")
    scale = scaler_params.get("scale")
    if mean is None or scale is None:
        return pt

    mean_arr = np.array(mean)
    scale_arr = np.array(scale)
    if col_indices is not None and n_total_cols is not None:
        if len(mean_arr) == n_total_cols:
            mean_arr = mean_arr[col_indices]
        if len(scale_arr) == n_total_cols:
            scale_arr = scale_arr[col_indices]

    scaler.mean_ = mean_arr
    scaler.scale_ = scale_arr
    scaler.var_ = np.square(scale_arr)
    pt._scaler = scaler
    return pt
