"""Numeric column profiling + multicollinearity (VIF)."""

from typing import Dict, List, Optional

import numpy as np

from ..schemas import NumericStats
from ._utils import _AnalyzerState


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

    def _calculate_vif(self, numeric_cols: List[str]) -> Optional[Dict[str, float]]:
        """Variance Inflation Factor via diagonal of the inverse correlation matrix.

        Equivalent to ``1 / (1 - R_i^2)`` where ``R_i^2`` is the R² of regressing
        feature *i* against all others. ``VIF > 5`` flags multicollinearity.
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

            try:
                inv_corr = np.linalg.inv(corr_matrix)
            except np.linalg.LinAlgError:
                # Singular matrix = perfect multicollinearity; flag everything.
                return {col: 999.0 for col in numeric_cols}

            return {
                col: max(1.0, float(inv_corr[i, i]))
                for i, col in enumerate(numeric_cols)
            }
        except Exception as e:
            print(f"Error calculating VIF: {e}")
            return None
