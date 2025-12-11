import pandas as pd
import numpy as np
from typing import Dict, Any
from ..recommendations.schemas import AnalysisProfile, ColumnProfile, ColumnType

class DataProfiler:
    @staticmethod
    def generate_profile(df: pd.DataFrame) -> AnalysisProfile:
        """
        Generates a comprehensive statistical profile of the dataframe.
        """
        row_count = len(df)
        column_count = len(df.columns)
        duplicate_row_count = df.duplicated().sum()
        columns_profile = {}

        for col_name in df.columns:
            series = df[col_name]
            dtype = str(series.dtype)
            missing_count = int(series.isna().sum())
            missing_ratio = missing_count / row_count if row_count > 0 else 0.0
            unique_count = series.nunique()
            
            col_type = DataProfiler._detect_column_type(series)
            
            profile = ColumnProfile(
                name=col_name,
                dtype=dtype,
                column_type=col_type,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                unique_count=unique_count
            )
            
            if col_type == ColumnType.NUMERIC:
                # Calculate numeric stats on non-null values
                clean_series = series.dropna()
                if not clean_series.empty:
                    profile.min_value = float(clean_series.min())
                    profile.max_value = float(clean_series.max())
                    profile.mean_value = float(clean_series.mean())
                    profile.std_value = float(clean_series.std())
                    # Skewness can fail on constant values
                    if unique_count > 1:
                        profile.skewness = float(clean_series.skew())
            
            elif col_type == ColumnType.CATEGORICAL or col_type == ColumnType.BOOLEAN or col_type == ColumnType.TEXT:
                # Top 10 values
                value_counts = series.value_counts().head(10).to_dict()
                # Convert keys to string to ensure JSON serializability
                profile.top_values = {str(k): int(v) for k, v in value_counts.items()}
                
                # Calculate avg text length for categorical/text
                if series.dtype == 'object' or str(series.dtype) == 'string':
                     clean_series = series.dropna().astype(str)
                     if not clean_series.empty:
                         profile.avg_text_length = float(clean_series.str.len().mean())
                
            columns_profile[col_name] = profile

        return AnalysisProfile(
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=int(duplicate_row_count),
            columns=columns_profile
        )

    @staticmethod
    def _detect_column_type(series: pd.Series) -> ColumnType:
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN
        elif pd.api.types.is_numeric_dtype(series):
            # Check if it's actually categorical (e.g. 0/1 or few integers)
            # Heuristic: If integer and unique values < 20 and ratio < 5%, might be categorical
            # But for now, let's stick to strict types to avoid confusion in scaling
            return ColumnType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        elif pd.api.types.is_string_dtype(series):
            # Check if text (high cardinality) or categorical (low cardinality)
            if series.nunique() / len(series) > 0.8 and len(series) > 100:
                return ColumnType.TEXT
            return ColumnType.CATEGORICAL
        else:
            return ColumnType.UNKNOWN
