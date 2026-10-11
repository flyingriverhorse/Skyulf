"""Winsorize node (clip extreme values to percentile bounds)."""

from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import (
    detect_numeric_columns,
    is_decimal_series,
    resolve_columns,
    user_picked_no_columns,
)
from .._artifacts import WinsorizeArtifact
from .._fitted_validation import local_state_fields
from .._helpers import (
    auto_detect_numeric_columns,
    promote_configured_columns_to_float64,
)
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import validate_fitted_bounds


def _integer_bounds(bound: dict, column: str) -> dict[str, float]:
    """Reject limits that cannot retain integer values before any rows are matched."""
    if any(not isfinite(value) or value != int(value) for value in bound.values()):
        raise ValueError(
            f"Winsorize column '{column}' needs finite integral bounds for integer input; "
            "add an explicit Casting step for fractional or nonfinite output."
        )
    return {name: int(value) for name, value in bound.items()}


def _pandas_integer_bounds(series: pd.Series, bound: dict, column: str) -> dict[str, float]:
    """Prevent native integer clipping from narrowing or wrapping out-of-range limits."""
    bound = _integer_bounds(bound, column)
    info = np.iinfo(getattr(series.dtype, "numpy_dtype", series.dtype))
    if any(not info.min <= value <= info.max for value in bound.values()):
        raise ValueError(f"Winsorize bounds for '{column}' require an explicit Casting step.")
    return bound


def _clip_pandas_column(series: pd.Series, bound: dict, column: str) -> pd.Series:
    """Retain integer storage while preserving the existing float and Decimal paths."""
    if pd.api.types.is_integer_dtype(series.dtype):
        bound = _pandas_integer_bounds(series, bound, column)
        if isinstance(series.dtype, pd.ArrowDtype):
            dtype = getattr(series.dtype, "numpy_dtype", series.dtype)
            bound = {name: dtype.type(value) for name, value in bound.items()}
    elif isinstance(series.dtype, pd.api.extensions.ExtensionDtype) or is_decimal_series(series):
        series = pd.to_numeric(series).astype("float64")
    return series.clip(lower=bound["lower"], upper=bound["upper"])


def _clip_polars_column(X: Any, column: str, bound: dict) -> pl.Expr:
    """Use native typed bounds without truncating fractions or widening integer columns."""
    expression = pl.col(column)
    if X.schema[column].is_integer():
        bound = _integer_bounds(bound, column)
        try:
            pl.Series(list(bound.values()), dtype=X.schema[column], strict=True)
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                f"Winsorize bounds for '{column}' require an explicit Casting step."
            ) from exc
    else:
        expression = expression.cast(pl.Float64)
    return expression.clip(bound["lower"], bound["upper"]).alias(column)


def _fit_series(X: Any, column: str) -> tuple[pd.Series, bool]:
    """Convert selected Polars integers without routing nulls through float64."""
    series = X[column]
    if isinstance(series, pl.Series):
        if series.dtype.is_integer():
            return pd.Series(series.to_list(), dtype=str(series.dtype)).dropna(), True
        series = series.to_pandas()
    integer = pd.api.types.is_integer_dtype(series.dtype)
    return pd.to_numeric(series, errors="coerce").dropna(), integer


def _fit_bound(series: pd.Series, percentile: float, column: str, *, integer: bool) -> Any:
    """Keep integer endpoints exact; guarded interpolation retains native pandas rounding."""
    if integer:
        if percentile == 0:
            return int(series.min())
        if percentile == 100:
            return int(series.max())
        if series.min() < -(2**53) or series.max() > 2**53:
            raise ValueError(
                f"Winsorize interpolation for '{column}' requires an explicit Casting step "
                "for integers outside the exact float64 range."
            )
    return series.quantile(percentile / 100.0)


class WinsorizeApplier(BaseApplier):
    """Clip values to the fitted percentile bounds; rows are never removed."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect learned bounds and metadata without recomputing request quantiles."""
        if not local_state_fields(
            raw,
            "winsorize",
            {"type", "bounds", "lower_percentile", "upper_percentile", "warnings"},
            allow_empty=True,
        ):
            return raw
        validate_fitted_bounds(raw["bounds"], partial=False)
        for field in ("lower_percentile", "upper_percentile"):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 100:
                raise ValueError("Fitted percentiles must be numbers between zero and 100.")
        if type(raw["warnings"]) is not list or any(
            type(item) is not str for item in raw["warnings"]
        ):
            raise ValueError("Fitted warnings must be a list of strings.")
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe local row clipping with saved limits without admitting Spark workers."""
        if engine not in ("pandas", "polars"):
            return None
        WinsorizeApplier.validate_inference_state(state)
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Clip ``X`` to the fitted per-column bounds; ``y`` passes through untouched."""
        # apply_method already unpacked (X, y); re-wrap so apply_dual_engine's
        # own unpack_pipeline_input doesn't silently drop y. Winsorize never
        # filters rows, but the wrap keeps behavior consistent with the other
        # outlier nodes and avoids losing y from the returned tuple.
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(
            input_data, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        numeric_cols = set(auto_detect_numeric_columns(X))
        exprs = []
        for col, bound in bounds.items():
            if col not in X.columns or col not in numeric_cols:
                continue
            exprs.append(_clip_polars_column(X, col, bound))
        return X.with_columns(exprs), y

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        df_out = X.copy()
        for col, bound in bounds.items():
            if col not in df_out.columns:
                continue
            if pd.api.types.is_numeric_dtype(df_out[col]) or is_decimal_series(df_out[col]):
                df_out[col] = _clip_pandas_column(df_out[col], bound, col)
        return df_out, y


@NodeRegistry.register("Winsorize", WinsorizeApplier)
@node_meta(
    id="Winsorize",
    name="Winsorization",
    category="Preprocessing",
    description="Limit extreme values in the data.",
    params={"lower_percentile": 5.0, "upper_percentile": 95.0, "columns": []},
    learns_from_data=True,
)
class WinsorizeCalculator(BaseCalculator):
    """Fit per-column clip bounds at the configured lower/upper percentiles."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Preserve integer columns and retain existing promotion for other selected types."""
        if not isinstance(input_schema, SkyulfSchema):
            return input_schema
        output = promote_configured_columns_to_float64(input_schema, config)
        for column, dtype in input_schema.dtypes.items():
            if dtype.lower().startswith(("int", "uint")):
                output = output.with_dtype(column, dtype)
        return output

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> WinsorizeArtifact:  # pylint: disable=arguments-differ
        """Compute each column's lower/upper quantiles; warn on empty or non-numeric ones."""
        if user_picked_no_columns(config):
            return {}

        lower_p = config.get("lower_percentile", 5.0)
        upper_p = config.get("upper_percentile", 95.0)
        X = X.to_native() if hasattr(X, "to_native") else X
        cols = resolve_columns(X, config, detect_numeric_columns)
        if not cols:
            return {}

        bounds: dict[str, dict[str, float]] = {}
        warnings = []
        for col in cols:
            series, integer = _fit_series(X, col)
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            bounds[col] = {
                "lower": _fit_bound(series, lower_p, col, integer=integer),
                "upper": _fit_bound(series, upper_p, col, integer=integer),
            }
            if integer:
                bounds[col] = _pandas_integer_bounds(series, bounds[col], col)

        return {
            "type": "winsorize",
            "bounds": bounds,
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "warnings": warnings,
        }
