"""Lightweight EDA advisor used by the ML-pipeline API.

Extracted from `api.py` as part of the E9 modularisation. The
profiler/engine pair is intentionally cheap — it is *not* a
replacement for `skyulf.profiling.EDAAnalyzer`. It just produces the
small handful of recommendation rows the canvas surfaces above the
"Run pipeline" button (missing values, duplicates, candidate one-hot
columns, skewed numerics, etc.).

Public surface:
    - `Recommendation`, `AnalysisProfile` Pydantic models
    - `DataProfiler.generate_profile(df) -> AnalysisProfile`
    - `AdvisorEngine().analyze(profile) -> List[Recommendation]`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd
from pydantic import BaseModel


class Recommendation(BaseModel):
    type: str  # "imputation", "cleaning", "encoding", "outlier", "transformation"
    rule_id: Optional[str] = None
    target_columns: List[str]
    action: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = "info"
    suggestion: Optional[str] = None


class AnalysisProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: Dict[str, Any]


class DataProfiler:
    @staticmethod
    def generate_profile(df: pd.DataFrame) -> AnalysisProfile:
        columns = {}
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            columns[col] = {
                "name": col,
                "dtype": str(df[col].dtype),
                "column_type": "numeric" if is_numeric else "categorical",
                "missing_count": int(df[col].isnull().sum()),
                "missing_ratio": float(df[col].isnull().mean()),
                "unique_count": int(df[col].nunique()),
                "min_value": (
                    float(cast(Union[float, int], df[col].min())) if is_numeric else None
                ),
                "max_value": (
                    float(cast(Union[float, int], df[col].max())) if is_numeric else None
                ),
                "mean_value": (
                    float(cast(Union[float, int], df[col].mean())) if is_numeric else None
                ),
                "std_value": (
                    float(cast(Union[float, int], df[col].std())) if is_numeric else None
                ),
                "skewness": (
                    float(cast(Union[float, int], df[col].skew())) if is_numeric else None
                ),
            }
        return AnalysisProfile(
            row_count=len(df),
            column_count=len(df.columns),
            duplicate_row_count=int(df.duplicated().sum()),
            columns=columns,
        )


class AdvisorEngine:
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recs: List[Recommendation] = []

        # 1. Imputation
        missing_cols = [col for col, stats in profile.columns.items() if stats["missing_count"] > 0]
        if missing_cols:
            recs.append(
                Recommendation(
                    type="imputation",
                    rule_id="imputation_mean",
                    target_columns=missing_cols,
                    message=f"Found {len(missing_cols)} columns with missing values.",
                    suggestion="Consider using SimpleImputer or KNNImputer.",
                )
            )

        # 2. Cleaning (Duplicates & High Missing)
        if profile.duplicate_row_count > 0:
            recs.append(
                Recommendation(
                    type="cleaning",
                    rule_id="duplicate_rows_drop",
                    target_columns=[],
                    action="drop_duplicates",
                    message=f"Found {profile.duplicate_row_count} duplicate rows.",
                    suggestion="Add a DropDuplicates node.",
                )
            )

        high_missing_cols = [
            col for col, stats in profile.columns.items() if stats["missing_ratio"] > 0.5
        ]
        if high_missing_cols:
            recs.append(
                Recommendation(
                    type="cleaning",
                    rule_id="high_missing_drop",
                    target_columns=high_missing_cols,
                    action="drop_columns",
                    message=f"Found {len(high_missing_cols)} columns with >50% missing values.",
                    suggestion="Consider dropping these columns.",
                )
            )

        # 3. Encoding (low-cardinality categorical → OHE candidates)
        cat_cols = [
            col
            for col, stats in profile.columns.items()
            if stats["column_type"] == "categorical" and stats["unique_count"] < 20
        ]
        if cat_cols:
            recs.append(
                Recommendation(
                    type="encoding",
                    rule_id="one_hot_encoding",
                    target_columns=cat_cols,
                    message=(
                        f"Found {len(cat_cols)} categorical columns suitable for "
                        "OneHotEncoding."
                    ),
                    suggestion="Consider OneHotEncoder.",
                )
            )

        # 4. Outliers (cheap proxy: any value > mean ± 3σ)
        outlier_cols = []
        for col, stats in profile.columns.items():
            if stats["column_type"] == "numeric" and stats["std_value"] and stats["std_value"] > 0:
                mean = stats["mean_value"]
                std = stats["std_value"]
                if (stats["max_value"] > mean + 3 * std) or (stats["min_value"] < mean - 3 * std):
                    outlier_cols.append(col)

        if outlier_cols:
            recs.append(
                Recommendation(
                    type="outlier",
                    rule_id="outlier_removal_iqr",
                    target_columns=outlier_cols,
                    message=f"Found {len(outlier_cols)} columns with potential outliers.",
                    suggestion="Consider using IsolationForest or Z-score filtering.",
                )
            )

        # 5. Transformation (skewness)
        pos_skewed_cols = []
        neg_skewed_cols = []
        for col, stats in profile.columns.items():
            if (
                stats["column_type"] == "numeric"
                and stats["skewness"]
                and abs(stats["skewness"]) > 1.0
            ):
                if stats["min_value"] > 0:
                    pos_skewed_cols.append(col)
                else:
                    neg_skewed_cols.append(col)

        if pos_skewed_cols:
            recs.append(
                Recommendation(
                    type="transformation",
                    rule_id="power_transform_box_cox",
                    target_columns=pos_skewed_cols,
                    message=(
                        f"Found {len(pos_skewed_cols)} positively skewed columns "
                        "(strictly positive)."
                    ),
                    suggestion="Consider Box-Cox transformation.",
                )
            )

        if neg_skewed_cols:
            recs.append(
                Recommendation(
                    type="transformation",
                    rule_id="power_transform_yeo_johnson",
                    target_columns=neg_skewed_cols,
                    message=(
                        f"Found {len(neg_skewed_cols)} skewed columns (with non-positive "
                        "values)."
                    ),
                    suggestion="Consider Yeo-Johnson transformation.",
                )
            )

        return recs
