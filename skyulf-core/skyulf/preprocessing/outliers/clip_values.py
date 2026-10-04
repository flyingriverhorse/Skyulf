"""Clip-values node: limit columns to fixed lower/upper bounds without removing rows."""

import math
from typing import Any

import pandas as pd
import polars as pl

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import is_decimal_series
from .._artifacts import ClipValuesArtifact
from .._helpers import auto_detect_numeric_columns
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


def _is_finite_number(value: Any) -> bool:
    """Accept real finite numbers; booleans are rejected even though they are ints."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _clean_bound(column: str, bound: Any) -> dict[str, float]:
    """Validate one column's bounds and keep only the given finite numbers."""
    if not isinstance(bound, dict):
        raise ValueError(f"ClipValues bounds for '{column}' must be a dict with lower/upper.")
    cleaned = {key: bound[key] for key in ("lower", "upper") if bound.get(key) is not None}
    if not cleaned:
        raise ValueError(f"ClipValues column '{column}' needs a lower or upper bound.")
    for key, value in cleaned.items():
        if not _is_finite_number(value):
            raise ValueError(f"ClipValues {key} bound for '{column}' must be a finite number.")
    if cleaned.get("lower", -math.inf) > cleaned.get("upper", math.inf):
        raise ValueError(f"ClipValues lower bound for '{column}' must not exceed upper.")
    return {key: float(value) for key, value in cleaned.items()}


def _check_columns(X: Any, bounds: dict[str, Any]) -> None:
    """Require every bounded column to exist and be numeric when data is available."""
    columns = list(X.columns)
    missing = [column for column in bounds if column not in columns]
    if missing:
        raise ValueError(f"ClipValues columns are missing from the data: {missing}.")
    numeric = set(auto_detect_numeric_columns(X))
    invalid = [column for column in bounds if column not in numeric]
    if invalid:
        raise ValueError(f"ClipValues requires numeric columns; got non-numeric {invalid}.")


class ClipValuesApplier(BaseApplier):
    """Set values below ``lower`` to ``lower`` and above ``upper`` to ``upper``.

    Nulls stay null, other columns are untouched and every row is kept, so
    unlike ManualBounds this step also runs on scoring rows.
    """

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Clip the configured columns on the active engine; ``y`` passes through."""
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(
            input_data, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        exprs = [
            pl.col(column)
            .cast(pl.Float64)
            .clip(bound.get("lower"), bound.get("upper"))
            .cast(X.schema[column] if X.schema[column].is_integer() else pl.Float64)
            .alias(column)
            for column, bound in params.get("bounds", {}).items()
            if column in X.columns
        ]
        return X.with_columns(exprs), y

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        out = X.copy()
        for column, bound in params.get("bounds", {}).items():
            if column not in out.columns:
                continue
            series = out[column]
            if is_decimal_series(series):
                series = pd.to_numeric(series).astype("float64")
            out[column] = series.clip(lower=bound.get("lower"), upper=bound.get("upper"))
        return out, y


@NodeRegistry.register("ClipValues", ClipValuesApplier)
@node_meta(
    id="ClipValues",
    name="Clip Values",
    category="Preprocessing",
    description=(
        "Limits numeric columns to fixed lower/upper bounds (e.g. employees at most 500) "
        "without removing rows."
    ),
    params={"bounds": {}},
    learns_from_data=False,
)
class ClipValuesCalculator(BaseCalculator):
    """Validate the configured bounds; nothing is learned from the data.

    ``bounds`` maps a column to ``{"lower": ..., "upper": ...}``; either side
    may be omitted. Use Winsorize for bounds learned from percentiles.
    """

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return the input schema unchanged: clipping keeps columns and their types."""
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> ClipValuesArtifact:  # pylint: disable=arguments-differ
        """Return the validated bounds as the artifact."""
        raw = config.get("bounds") or {}
        bounds = {column: _clean_bound(column, bound) for column, bound in raw.items()}
        if bounds and X is not None:
            _check_columns(X.to_native() if hasattr(X, "to_native") else X, bounds)
        return {"type": "clip_values", "bounds": bounds}
