"""Numeric column profiling + multicollinearity (VIF)."""

import logging

import numpy as np

from ..schemas import NumericStats
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

    def _calculate_vif(self, numeric_cols: list[str]) -> dict[str, float] | None:
        """Calculate VIF, using regression residuals when correlation inversion is unstable.

        Equivalent to ``1 / (1 - R_i^2)`` where ``R_i^2`` is the R² of regressing
        feature *i* against all others. ``VIF > 5`` flags multicollinearity.
        A residual variance ratio at or below float64 precision returns the
        finite high-collinearity marker 999 for that feature only.
        """
        if len(numeric_cols) < 2:
            return None

        try:
            df_clean = self.df.select(numeric_cols).drop_nulls()  # type: ignore[attr-defined]

            # Need more rows than features for a stable correlation estimate.
            if df_clean.height < len(numeric_cols) + 5:
                return None

            data = df_clean.to_numpy()
            corr_matrix = np.corrcoef(data, rowvar=False)

            # Constant column → undefined correlation → bail out.
            if np.isnan(corr_matrix).any():
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
                col: max(1.0, float(value))
                for col, value in zip(numeric_cols, diagonal, strict=True)
            }
        except Exception as e:  # noqa: BLE001 - VIF is optional; logged, returns None
            logger.warning(f"Error calculating VIF: {e}")
            return None

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
