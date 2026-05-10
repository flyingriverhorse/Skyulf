"""Outlier-handling nodes (IQR / ZScore / Winsorize / ManualBounds / EllipticEnvelope).

Per-engine apply paths are split into ``_apply_polars`` / ``_apply_pandas``
static helpers and dispatched via ``apply_dual_engine`` (see
``dispatcher.py``). Fit paths are pandas-only here (we always coerce to
Pandas before computing quantiles / fitting sklearn models), so there is
no fit-time dispatch needed.
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.covariance import EllipticEnvelope

from ..utils import (
    detect_numeric_columns,
    resolve_columns,
    user_picked_no_columns,
)
from .base import BaseApplier, BaseCalculator, apply_method, fit_method
from .dispatcher import apply_dual_engine
from ._artifacts import (
    EllipticEnvelopeArtifact,
    IQRArtifact,
    ManualBoundsArtifact,
    WinsorizeArtifact,
    ZScoreArtifact,
)
from ._helpers import to_pandas
from ._schema import SkyulfSchema
from ..core.meta.decorators import node_meta
from ..registry import NodeRegistry

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Shared filter helpers
# -----------------------------------------------------------------------------


def _filter_y_polars(y: Any, mask_series: Any) -> Any:
    """Apply a Polars boolean mask to ``y`` if it is a Polars Series/DataFrame."""
    import polars as pl

    if y is None:
        return None
    if isinstance(y, (pl.Series, pl.DataFrame)):
        return y.filter(mask_series)
    return y


def _apply_pandas_mask(X_pd: Any, y: Any, mask: pd.Series) -> Tuple[Any, Any]:
    """Apply a Pandas boolean mask to X (and y if non-null)."""
    X_filtered = X_pd[mask]
    if y is None:
        return X_filtered, y
    y_pd: Any = y
    return X_filtered, y_pd[mask]


# -----------------------------------------------------------------------------
# IQR Filter
# -----------------------------------------------------------------------------


class IQRApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pl.lit(True)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            col_mask = (pl.col(col) >= bound["lower"]) & (pl.col(col) <= bound["upper"])
            mask = mask & (col_mask | pl.col(col).is_null())

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            col_mask = (series >= bound["lower"]) & (series <= bound["upper"])
            mask = mask & (col_mask | series.isna())

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("IQR", IQRApplier)
@node_meta(
    id="IQR",
    name="IQR Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Interquartile Range.",
    params={"factor": 1.5, "columns": []},
)
class IQRCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # IQR removes outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> IQRArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        multiplier = config.get("multiplier", 1.5)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        bounds: Dict[str, Dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            bounds[col] = {"lower": q1 - (multiplier * iqr), "upper": q3 + (multiplier * iqr)}

        return {
            "type": "iqr",
            "bounds": bounds,
            "multiplier": multiplier,
            "warnings": warnings,
        }


# -----------------------------------------------------------------------------
# Z-Score Filter
# -----------------------------------------------------------------------------


class ZScoreApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)
        if not stats:
            return X, y

        mask = pl.lit(True)
        for col, stat in stats.items():
            if col not in X.columns or stat["std"] == 0:
                continue
            z = (pl.col(col) - stat["mean"]) / stat["std"]
            col_mask = z.abs() <= threshold
            mask = mask & (col_mask | pl.col(col).is_null())

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
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
)
class ZScoreCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Z-score removes outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> ZScoreArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        threshold = config.get("threshold", 3.0)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        stats: Dict[str, Dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
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


# -----------------------------------------------------------------------------
# Winsorize
# -----------------------------------------------------------------------------


class WinsorizeApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        exprs = []
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            # Cast to float to avoid integer truncation when bounds are float.
            exprs.append(
                pl.col(col).cast(pl.Float64).clip(bound["lower"], bound["upper"]).alias(col)
            )
        return X.with_columns(exprs), y

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        df_out = X.copy()
        for col, bound in bounds.items():
            if col not in df_out.columns:
                continue
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].clip(lower=bound["lower"], upper=bound["upper"])
        return df_out, y


@NodeRegistry.register("Winsorize", WinsorizeApplier)
@node_meta(
    id="Winsorize",
    name="Winsorization",
    category="Preprocessing",
    description="Limit extreme values in the data.",
    params={"limits": [0.05, 0.05], "columns": []},
)
class WinsorizeCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Winsorize clips values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> WinsorizeArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        lower_p = config.get("lower_percentile", 5.0)
        upper_p = config.get("upper_percentile", 95.0)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        bounds: Dict[str, Dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            bounds[col] = {
                "lower": series.quantile(lower_p / 100.0),
                "upper": series.quantile(upper_p / 100.0),
            }

        return {
            "type": "winsorize",
            "bounds": bounds,
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "warnings": warnings,
        }


# -----------------------------------------------------------------------------
# Manual Bounds
# -----------------------------------------------------------------------------


def _manual_bounds_col_mask_polars(col: str, bound: Dict[str, Any]) -> Any:
    """Build a Polars per-column inlier mask from optional lower/upper bounds."""
    import polars as pl

    lower = bound.get("lower")
    upper = bound.get("upper")
    col_mask = pl.lit(True)
    if lower is not None:
        col_mask = col_mask & (pl.col(col) >= lower)
    if upper is not None:
        col_mask = col_mask & (pl.col(col) <= upper)
    return col_mask | pl.col(col).is_null()


def _manual_bounds_col_mask_pandas(series: pd.Series, bound: Dict[str, Any]) -> pd.Series:
    """Build a Pandas per-column inlier mask from optional lower/upper bounds."""
    lower = bound.get("lower")
    upper = bound.get("upper")
    col_mask = pd.Series(True, index=series.index)
    if lower is not None:
        col_mask = col_mask & (series >= lower)
    if upper is not None:
        col_mask = col_mask & (series <= upper)
    return col_mask | series.isna()


class ManualBoundsApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pl.lit(True)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            mask = mask & _manual_bounds_col_mask_polars(col, bound)

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            mask = mask & _manual_bounds_col_mask_pandas(series, bound)

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("ManualBounds", ManualBoundsApplier)
@node_meta(
    id="ManualBounds",
    name="Manual Bounds",
    category="Preprocessing",
    description="Filter outliers by manually specifying lower and upper bounds for columns.",
    params={"bounds": {}},
)
class ManualBoundsCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Manual bounds filter rows; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: Dict[str, Any]) -> ManualBoundsArtifact:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        return {"type": "manual_bounds", "bounds": config.get("bounds", {})}


# -----------------------------------------------------------------------------
# Elliptic Envelope
# -----------------------------------------------------------------------------


def _elliptic_filter_pandas(X_pd: Any, models: Dict[str, Any]) -> pd.Series:
    """Build a row-keep mask by applying every fitted EllipticEnvelope model."""
    mask = pd.Series(True, index=X_pd.index)
    for col, model in models.items():
        if col not in X_pd.columns:
            continue
        series = pd.to_numeric(X_pd[col], errors="coerce")
        valid_idx = series.dropna().index
        if valid_idx.empty:
            continue
        try:
            preds = model.predict(series.loc[valid_idx].to_numpy().reshape(-1, 1))
            col_mask = pd.Series(False, index=X_pd.index)
            col_mask[series.isna()] = True  # keep NaNs; later steps decide
            col_mask.loc[valid_idx] = preds == 1  # 1 == inlier
            mask = mask & col_mask
        except Exception as e:
            logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
    return mask


class EllipticEnvelopeApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        # sklearn models require numpy/pandas input — convert, filter, convert back.
        import polars as pl

        models = params.get("models", {})
        if not models:
            return X, y

        X_pd = X.to_pandas()
        y_pd = y.to_pandas() if (y is not None and hasattr(y, "to_pandas")) else y

        mask = _elliptic_filter_pandas(X_pd, models)
        X_filtered = X_pd[mask]
        X_out = pl.from_pandas(X_filtered)

        if y_pd is None:
            return X_out, y
        y_filtered = y_pd[mask]
        y_out = pl.from_pandas(y_filtered) if y_filtered is not None else None
        return X_out, y_out

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        models = params.get("models", {})
        if not models:
            return X, y
        mask = _elliptic_filter_pandas(X, models)
        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("EllipticEnvelope", EllipticEnvelopeApplier)
@node_meta(
    id="EllipticEnvelope",
    name="Elliptic Envelope",
    category="Preprocessing",
    description="Detect outliers in a Gaussian distributed dataset.",
    params={"contamination": 0.01, "columns": []},
)
class EllipticEnvelopeCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Elliptic envelope filters outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> EllipticEnvelopeArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        contamination = config.get("contamination", 0.01)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        models: Dict[str, Any] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.shape[0] < 5:
                warnings.append(f"Column '{col}': Too few samples ({series.shape[0]})")
                continue
            try:
                model = EllipticEnvelope(contamination=contamination)
                model.fit(series.to_numpy().reshape(-1, 1))
                models[col] = model
            except Exception as e:
                logger.warning(f"EllipticEnvelope fit failed for column {col}: {e}")
                warnings.append(f"Column '{col}': {str(e)}")

        return {
            "type": "elliptic_envelope",
            "models": models,
            "contamination": contamination,
            "warnings": warnings,
        }
