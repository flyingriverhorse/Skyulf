"""Elliptic Envelope outlier node (Gaussian covariance estimation)."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.covariance import EllipticEnvelope

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...types import DEFAULT_RANDOM_STATE
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import EllipticEnvelopeArtifact
from .._helpers import resolve_columns_then_to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _apply_pandas_mask

logger = logging.getLogger(__name__)


def _elliptic_filter_pandas(X_pd: Any, models: dict[str, Any]) -> pd.Series:
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
        except Exception as e:  # noqa: BLE001 - per-column predict failure is logged; column contributes no filtering
            logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
    return mask


def _coerce_column_to_float(X: Any, col: str) -> Any:
    """Cast a polars column to Float64; ``None`` when the column is not coercible."""
    try:
        return X.get_column(col).cast(pl.Float64, strict=False)
    except Exception:  # noqa: BLE001 - not coercible to numbers: contributes no filtering
        return None


def _predict_inliers(model: Any, values: Any, col: str) -> Any:
    """Run ``model.predict`` on valid values; log and return ``None`` on failure (fail open)."""
    try:
        return model.predict(values.reshape(-1, 1))
    except Exception as e:  # noqa: BLE001 - per-column predict failure is logged; column contributes no filtering
        logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
        return None


def _elliptic_mask_numpy(X: Any, models: dict[str, Any]) -> Any:
    """Build a row-keep boolean numpy mask by applying every fitted model.

    Mirrors :func:`_elliptic_filter_pandas` semantics: missing values are kept
    (later steps decide), columns absent from *X* or without valid values are
    skipped, and a failing ``predict`` logs and fails open.
    """
    mask = np.ones(X.height, dtype=bool)
    for col, model in models.items():
        if col not in X.columns:
            continue
        series = _coerce_column_to_float(X, col)
        if series is None:
            continue
        arr = series.to_numpy()
        valid = ~np.isnan(arr)
        if not valid.any():
            continue
        preds = _predict_inliers(model, arr[valid], col)
        if preds is None:
            continue
        col_mask = np.ones(X.height, dtype=bool)
        col_mask[valid] = preds == 1  # 1 == inlier
        mask &= col_mask
    return mask


class EllipticEnvelopeApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
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
        models = params.get("models", {})
        if not models:
            return X, y

        mask = pl.Series(_elliptic_mask_numpy(X, models))
        X_out = X.filter(mask)
        if y is None:
            return X_out, y
        y_out = y.filter(mask) if hasattr(y, "filter") else y
        return X_out, y_out

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    params={"contamination": 0.01, "columns": [], "random_state": DEFAULT_RANDOM_STATE},
    learns_from_data=True,
)
class EllipticEnvelopeCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Elliptic envelope filters outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> EllipticEnvelopeArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return {}

        contamination = config.get("contamination", 0.01)
        random_state = config.get("random_state", DEFAULT_RANDOM_STATE)
        # TODO(pandas-removal): same per-column coercion/dropna caveat as
        # zscore.py — EllipticEnvelope.fit itself takes numpy fine, but the
        # per-column pd.to_numeric(errors="coerce").dropna() skip-logic for
        # non-numeric/short columns needs a native-Polars equivalent first.
        X_pd, cols = resolve_columns_then_to_pandas(X, config, detect_numeric_columns)
        if not cols:
            return {}

        models: dict[str, Any] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.shape[0] < 5:
                warnings.append(f"Column '{col}': Too few samples ({series.shape[0]})")
                continue
            try:
                model = EllipticEnvelope(contamination=contamination, random_state=random_state)
                model.fit(series.to_numpy().reshape(-1, 1))
                models[col] = model
            except Exception as e:  # noqa: BLE001 - per-column fit failure is logged and recorded in warnings
                logger.warning(f"EllipticEnvelope fit failed for column {col}: {e}")
                warnings.append(f"Column '{col}': {str(e)}")

        return {
            "type": "elliptic_envelope",
            "models": models,
            "contamination": contamination,
            "warnings": warnings,
        }
