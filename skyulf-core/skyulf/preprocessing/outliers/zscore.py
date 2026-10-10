"""Z-Score outlier-removal node."""

from decimal import Decimal
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import ZScoreArtifact
from .._fitted_validation import local_scalar, local_state_fields
from .._helpers import resolve_columns_then_to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _apply_pandas_mask, _filter_y_polars, validate_detector_warnings


def _validate_zscore_statistics(stats: Any) -> None:
    """Keep each learned mean and nonnegative population deviation bound to its column."""
    if type(stats) is not dict:
        raise ValueError("Fitted z-score statistics must be a dictionary.")
    for values in stats.values():
        if type(values) is not dict or set(values) != {"mean", "std"}:
            raise ValueError("Fitted z-score entries must contain mean and std.")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values.values()):
            raise ValueError("Fitted z-score statistics must be real numbers.")
        if values["std"] < 0:
            raise ValueError("Fitted z-score deviations must be nonnegative.")


class ZScoreApplier(BaseApplier):
    """Drop rows whose |z| against the fitted mean/std exceeds ``threshold`` in any column."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved numeric statistics and native scalar threshold semantics."""
        if local_state_fields(
            raw, "zscore", {"type", "stats", "threshold", "warnings"}, allow_empty=True
        ):
            _validate_zscore_statistics(raw["stats"])
            if not isinstance(raw["threshold"], (Real, Decimal)):
                local_scalar(raw["threshold"], "ZScore threshold")
            validate_detector_warnings(raw["warnings"])
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved row filtering without bypassing the prediction row-count guard."""
        if engine not in ("pandas", "polars"):
            return None
        ZScoreApplier.validate_inference_state(state)
        effect = "filter" if state.get("stats") else "preserve"
        return ExecutionCapability(engine, "apply", "local", effect, "row")

    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Filter rows through the fitted z-score threshold on the active engine."""
        # apply_method already unpacked (X, y); re-wrap so apply_dual_engine's
        # own unpack_pipeline_input doesn't silently drop y (leaving it
        # unfiltered when X rows are removed). Omit the wrap when y is None
        # to avoid apply_dual_engine's tuple-with-no-y warning log.
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(
            input_data, params, {"polars": self._apply_polars, "pandas": self._apply_pandas}
        )

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)
        if not stats:
            return X, y

        mask = pl.repeat(True, pl.len())
        for col, stat in stats.items():
            if col not in X.columns or stat["std"] == 0:
                continue
            values = pl.col(col).cast(pl.Float64, strict=False)
            z = (values - stat["mean"]) / stat["std"]
            col_mask = z.abs() <= threshold
            missing = values.is_null() | values.is_nan()
            mask = mask & (col_mask | missing)

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)
        if not stats:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, stat in stats.items():
            if col not in X.columns or stat["std"] == 0:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            z = (series - stat["mean"]) / stat["std"]
            col_mask = z.abs() <= threshold
            mask = mask & (col_mask | series.isna())

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("ZScore", ZScoreApplier)
@node_meta(
    id="ZScore",
    name="Z-Score Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Z-Score.",
    params={"threshold": 3.0, "columns": []},
    learns_from_data=True,
)
class ZScoreCalculator(BaseCalculator):
    """Fit per-column mean and population std (``ddof=0``) for z-score filtering."""

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        """Return the input schema unchanged: row filtering preserves the column set."""
        # Z-score removes outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> ZScoreArtifact:  # pylint: disable=arguments-differ
        """Record each column's mean/std; warn on empty or zero-variance columns."""
        if user_picked_no_columns(config):
            return {}

        threshold = config.get("threshold", 3.0)
        # TODO(pandas-removal): mean/std alone would be numpy-safe (Polars std
        # ddof=0 already matches Pandas — verified), but this loop relies on
        # per-column pd.to_numeric(errors="coerce") to gracefully skip
        # non-numeric/mixed-dtype columns and .dropna() per column before
        # computing stats. Revisit once there's a native Polars equivalent
        # (pl.col(col).cast(pl.Float64, strict=False).drop_nulls()) wired in
        # to replace the coercion step without changing which columns get a
        # "non-numeric" warning.
        X_pd, cols = resolve_columns_then_to_pandas(X, config, detect_numeric_columns)
        if not cols:
            return {}

        stats: dict[str, dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            series = series[np.isfinite(series)]
            if series.empty:
                warnings.append(f"Column '{col}': Empty, non-numeric or non-finite")
                continue
            std = series.std(ddof=0)
            if std == 0:
                warnings.append(f"Column '{col}': Zero variance (std=0)")
                continue
            stats[col] = {"mean": series.mean(), "std": std}

        return {
            "type": "zscore",
            "stats": stats,
            "threshold": threshold,
            "warnings": warnings,
        }
