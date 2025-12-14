import polars as pl
from typing import Dict, Any


class DataProfiler:
    """
    Calculates basic statistics for a Polars DataFrame.
    """

    @staticmethod
    def profile(df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate a profile of the dataframe.
        """
        profile: Dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
            "missing_cells": sum(df[col].null_count() for col in df.columns),  # Total missing cells
            "duplicate_rows": df.is_duplicated().sum()
        }

        for col in df.columns:
            col_stats: Dict[str, Any] = {}
            series = df[col]
            dtype = str(series.dtype)

            col_stats["type"] = dtype
            col_stats["null_count"] = series.null_count()
            col_stats["null_percentage"] = (series.null_count() / len(df)) * 100
            col_stats["unique_count"] = series.n_unique()

            if dtype in ["Int64", "Float64", "Int32", "Float32"]:
                col_stats["mean"] = series.mean()
                col_stats["std"] = series.std()
                col_stats["min"] = series.min()
                col_stats["max"] = series.max()
                col_stats["median"] = series.median()
            elif dtype in ["Utf8", "String", "Categorical"]:
                # Top 5 values
                try:
                    value_counts = series.value_counts().sort("count", descending=True).head(5)
                    col_stats["top_values"] = value_counts.to_dicts()
                except BaseException:
                    pass

            profile["columns"][col] = col_stats

        return profile
