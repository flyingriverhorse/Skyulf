"""Elliptic Envelope outlier node (Gaussian covariance estimation)."""

import logging
from typing import Any

import pandas as pd
from sklearn.covariance import EllipticEnvelope

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import EllipticEnvelopeArtifact
from .._helpers import to_pandas
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
        except Exception as e:
            logger.warning(f"EllipticEnvelope predict failed for column {col}: {e}")
    return mask


class EllipticEnvelopeApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        # apply_method already unpacked (X, y); re-wrap so apply_dual_engine's
        # own unpack_pipeline_input doesn't silently drop y (leaving it
        # unfiltered when X rows are removed). Omit the wrap when y is None
        # to avoid apply_dual_engine's tuple-with-no-y warning log.
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(input_data, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
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
    params={"contamination": 0.01, "columns": []},
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

        X_pd = to_pandas(X)
        contamination = config.get("contamination", 0.01)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
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
