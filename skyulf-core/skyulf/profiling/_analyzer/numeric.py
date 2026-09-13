"""Numeric column profiling + multicollinearity (VIF)."""

import logging

import numpy as np
import polars as pl

from ..schemas import Alert, NumericStats
from ._utils import _AnalyzerState

logger = logging.getLogger(__name__)

# Beyond this condition number, a float64 inverse may lose ten precision digits.
_VIF_MAX_CONDITION = 1e10
_VIF_UNRESOLVED = 999.0


class NumericMixin(_AnalyzerState):
    """Numeric helpers for :class:`EDAAnalyzer`."""

    def _analyze_numeric(self, col: str, row: dict) -> NumericStats:
        return NumericStats(
            mean=row.get(f"{col}__mean"),
            median=row.get(f"{col}__median"),
            std=row.get(f"{col}__std"),
            variance=row.get(f"{col}__var"),
            min=row.get(f"{col}__min"),
            max=row.get(f"{col}__max"),
            q25=row.get(f"{col}__q25"),
            q75=row.get(f"{col}__q75"),
            skewness=row.get(f"{col}__skew"),
            kurtosis=row.get(f"{col}__kurt"),
            zeros_count=row.get(f"{col}__zeros", 0),
            negatives_count=row.get(f"{col}__negatives", 0),
        )

    @staticmethod
    def _exclude_constant_vif_columns(frame: pl.DataFrame, alerts: list[Alert]) -> pl.DataFrame:
        """Remove constant or unobserved features and explain each VIF omission."""
        variable_cols = []
        for col in frame.columns:
            if frame[col].drop_nulls().n_unique() > 1:
                variable_cols.append(col)
            else:
                alerts.append(
                    Alert(
                        column=col,
                        type="VIF Unavailable",
                        message=(
                            f"Column '{col}' was excluded from VIF because it has "
                            "no variation among the available observations."
                        ),
                        severity="info",
                    )
                )
        return frame.select(variable_cols)

    def _prepare_vif_frame(self, numeric_cols: list[str], alerts: list[Alert]) -> pl.DataFrame:
        """Drop constants before complete-case filtering, then recheck retained observations."""
        frame = self._exclude_constant_vif_columns(self.df.select(numeric_cols), alerts)
        frame = frame.drop_nulls()
        if frame.height:
            frame = self._exclude_constant_vif_columns(frame, alerts)
        return frame

    def _calculate_vif(
        self, numeric_cols: list[str], alerts: list[Alert] | None = None
    ) -> dict[str, float] | None:
        """Calculate VIF, using regression residuals when correlation inversion is unstable.

        Equivalent to ``1 / (1 - R_i^2)`` where ``R_i^2`` is the R² of regressing
        feature *i* against all others. ``VIF > 5`` flags multicollinearity.
        A residual variance ratio at or below float64 precision returns the
        finite high-collinearity marker 999 for that feature only.
        Constants and all-null features are excluded before complete-case
        filtering; excluded or unavailable calculations append explanations
        to ``alerts`` when supplied.
        """
        if len(numeric_cols) < 2:
            return None

        if alerts is None:
            alerts = []
        try:
            df_clean = self._prepare_vif_frame(numeric_cols, alerts)
            if df_clean.width < 2:
                reason = "fewer than two variable numeric features remain"
            elif df_clean.height < df_clean.width + 5:
                reason = "too few complete observations remain for the selected features"
            else:
                values = self._vif_from_data(df_clean.to_numpy(), df_clean.columns)
                if values is not None:
                    return values
                reason = "the selected features do not have finite correlations"
        except Exception as e:  # noqa: BLE001 - VIF is optional; logged, returns None
            logger.warning(f"Error calculating VIF: {e}")
            reason = "the calculation failed for the selected features"
        alerts.append(
            Alert(type="VIF Unavailable", message=f"VIF is unavailable: {reason}.", severity="info")
        )
        return None

    def _vif_from_data(self, data: np.ndarray, numeric_cols: list[str]) -> dict[str, float] | None:
        """Compute stable VIF values for a complete, variable numeric design."""
        corr_matrix = np.corrcoef(data, rowvar=False)
        if not np.isfinite(corr_matrix).all():
            return None
        if np.linalg.cond(corr_matrix) > _VIF_MAX_CONDITION:
            return self._vif_from_residuals(data, numeric_cols)
        try:
            inv_corr = np.linalg.inv(corr_matrix)
        except np.linalg.LinAlgError:
            return self._vif_from_residuals(data, numeric_cols)
        diagonal = np.diag(inv_corr)
        if not np.isfinite(diagonal).all() or (diagonal < 1.0 - 1e-10).any():
            return self._vif_from_residuals(data, numeric_cols)
        return {
            col: max(1.0, float(value)) for col, value in zip(numeric_cols, diagonal, strict=True)
        }

    @staticmethod
    def _vif_from_residuals(data: np.ndarray, numeric_cols: list[str]) -> dict[str, float]:
        """Use centered, normalized least squares to keep VIF specific to each feature."""
        standardized = np.asarray(data, dtype=np.float64)
        standardized = standardized - standardized.mean(axis=0)
        standardized /= np.linalg.norm(standardized, axis=0)

        result = {}
        for i, col in enumerate(numeric_cols):
            target = standardized[:, i]
            predictors = np.delete(standardized, i, axis=1)
            coefficients = np.linalg.lstsq(predictors, target, rcond=None)[0]
            residual = target - predictors @ coefficients
            # Computing 1 - R² would erase tiny residuals through cancellation.
            residual_ratio = float((residual @ residual) / (target @ target))
            result[col] = (
                _VIF_UNRESOLVED
                if residual_ratio <= np.finfo(np.float64).eps
                else max(1.0, 1.0 / residual_ratio)
            )
        return result
