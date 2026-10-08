"""Min-max scaler node (scale features into a given range)."""

import math
from typing import Any, cast

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...core.portable_state import _normalize
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import user_picked_no_columns
from .._artifacts import MinMaxScalerArtifact
from .._fitted_validation import _columns, _fields, fitted_columns
from .._helpers import (
    decimal_columns_to_float,
    promote_configured_columns_to_float64,
    resolve_valid_columns,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import _select_subset_pandas, _select_subset_polars, validate_scaling_range


class MinMaxScalerApplier(BaseApplier):
    """Rescale the selected columns into the fitted ``feature_range``."""

    @staticmethod
    def validate_fitted_state(raw: dict) -> dict:
        """Inspect this node's supported saved state without fitting or applying data."""
        return _minmax_state(raw)

    @staticmethod
    def resolve_fitted_config(raw: dict, state: dict) -> dict:
        """Bind inference configuration to this node's inspected fitted artifact."""
        return _minmax_config(fitted_columns(_minmax_values(raw), state), state)

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Apply the fitted affine scale/shift to ``X``; ``y`` passes through."""
        return apply_dual_engine(
            X, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        min_val = params.get("min")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or min_val is None or scale is None:
            return X, _y

        exprs = [
            (pl.col(c) * scale[cols.index(c)] + min_val[cols.index(c)]).alias(c) for c in valid
        ]
        return X.with_columns(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        cols = params.get("columns", [])
        min_val = params.get("min")
        scale = params.get("scale")
        valid = resolve_valid_columns(X, cols)
        if not valid or min_val is None or scale is None:
            return X, _y

        X_out = X.copy()
        col_indices = [cols.index(c) for c in valid]
        subset = decimal_columns_to_float(X_out[valid], valid)
        vals = subset.to_numpy(dtype=np.float64, na_value=np.nan)
        vals = vals * np.array(scale)[col_indices] + np.array(min_val)[col_indices]
        X_out[valid] = vals
        return X_out, _y


@NodeRegistry.register(
    "MinMaxScaler",
    MinMaxScalerApplier,
    execution_capabilities=(
        ExecutionCapability("pandas", "apply", "python_batch", "preserve", "row"),
    ),
)
@node_meta(
    id="MinMaxScaler",
    name="Min-Max Scaler",
    category="Preprocessing",
    description="Transform features by scaling each feature to a given range.",
    params={"feature_range": [0, 1], "columns": []},
    learns_from_data=True,
)
class MinMaxScalerCalculator(BaseCalculator):
    """Fit scale/offset statistics that map the data into ``feature_range``."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return a schema with transformed columns promoted to ``float64``."""
        return promote_configured_columns_to_float64(input_schema, config)

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> MinMaxScalerArtifact:  # pylint: disable=arguments-differ
        """Dispatch the fit to the active engine and return the scaler-statistics artifact."""
        if user_picked_no_columns(config):
            return cast(MinMaxScalerArtifact, {})
        return cast(
            MinMaxScalerArtifact,
            fit_dual_engine(X, config, {"polars": self._fit_polars, "pandas": self._fit_pandas}),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, config: dict[str, Any]) -> dict[str, Any]:
        cols, X_subset = _select_subset_polars(X, config)
        if not cols:
            return {}
        return _fit_minmax(X_subset, cols, config)

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, config: dict[str, Any]) -> dict[str, Any]:
        cols, X_subset = _select_subset_pandas(X, config)
        if not cols:
            return {}
        return _fit_minmax(X_subset, cols, config)


def _fit_minmax(X_subset: Any, cols: list[str], config: dict[str, Any]) -> dict[str, Any]:
    # `feature_range` may arrive as a JSON-loaded list (e.g. ``[0, 1]``) from the
    # frontend or pipeline config; sklearn enforces ``tuple`` via param validation.
    feature_range = validate_scaling_range(config.get("feature_range", (0, 1)), "feature_range")
    scaler = MinMaxScaler(feature_range=feature_range)
    X_np, _ = SklearnBridge.to_sklearn(X_subset)
    scaler.fit(X_np)
    return {
        "type": "minmax_scaler",
        "min": scaler.min_.tolist(),
        "scale": scaler.scale_.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "feature_range": feature_range,
        "columns": cols,
    }


def _minmax_values(raw: dict) -> dict:
    """Normalize only the exact tuple range emitted by the built-in scaler fit."""
    if type(raw) is not dict:
        raise ValueError("MinMax state and configuration require a plain mapping.")
    values = dict(raw)
    bounds = values.get("feature_range")
    if type(bounds) is tuple:
        values["feature_range"] = list(bounds)
    return _normalize(values)


def _minmax_range(bounds: Any) -> None:
    """Keep the fitted affine range explicit, finite and strictly increasing."""
    _finite_vector(bounds, 2)
    try:
        validate_scaling_range(bounds, "feature_range")
    except ValueError as exc:
        raise ValueError("MinMax feature_range must be strictly increasing.") from exc


def _finite_vector(values: Any, size: int) -> None:
    """Require aligned finite numeric coefficients without coercion or custom arrays."""
    if type(values) is not list or len(values) != size:
        raise ValueError("MinMax statistic vectors must align with fitted columns.")
    if any(type(value) not in (int, float) for value in values):
        raise ValueError("MinMax statistics require finite numeric scalars.")
    try:
        finite = all(math.isfinite(value) for value in values)
    except OverflowError as exc:
        raise ValueError("MinMax statistics exceed finite numeric bounds.") from exc
    if not finite:
        raise ValueError("MinMax statistics require finite numeric scalars.")


def _minmax_state(raw: dict) -> dict:
    """Inspect the scalar artifact actually executed instead of admitting a native scaler."""
    state = _minmax_values(raw)
    _fields(state, {"type", "columns", "min", "scale", "data_min", "data_max", "feature_range"})
    columns = _columns(state["columns"])
    if state["type"] != "minmax_scaler" or not columns:
        raise ValueError("MinMax requires nonempty fitted affine state.")
    _minmax_range(state["feature_range"])
    for name in ("min", "scale", "data_min", "data_max"):
        _finite_vector(state[name], len(columns))
    if any(value <= 0 for value in state["scale"]):
        raise ValueError("MinMax fitted scales must be positive.")
    if any(low > high for low, high in zip(state["data_min"], state["data_max"], strict=True)):
        raise ValueError("MinMax fitted extrema are inverted.")
    return state


def _minmax_config(params: dict, state: dict) -> dict:
    """Bind defaults and explicit range options to the saved fitted column contract."""
    resolved = {"feature_range": [0, 1], **params}
    _fields(resolved, {"columns", "feature_range"})
    _minmax_range(resolved["feature_range"])
    if resolved["feature_range"] != state["feature_range"]:
        raise ValueError("Configured MinMax range disagrees with fitted state.")
    return resolved
