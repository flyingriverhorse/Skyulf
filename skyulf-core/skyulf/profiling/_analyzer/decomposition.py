"""Decomposition-tree split: filter → group-by → measure aggregation."""

import contextlib
from typing import Any, cast

import polars as pl

from ._utils import _AnalyzerState


class DecompositionMixin(_AnalyzerState):
    """Decomposition-tree helpers for :class:`EDAAnalyzer`."""

    _NUMERIC_DTYPES = (
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )

    def _coerce_filter_value(self, dtype: pl.DataType, val: Any) -> Any:
        """Coerce a string filter value to numeric when the target column is numeric.

        Tolerates "1.0" strings for int cols; on failure, leaves ``val`` as-is so
        the caller falls through to casting the column to string instead.
        """
        with contextlib.suppress(ValueError):
            val = float(val) if dtype in (pl.Float32, pl.Float64) else int(float(val))
        return val

    def _apply_filter_operator(
        self, filtered_df: pl.DataFrame, col_expr: pl.Expr, op: str, val: Any
    ) -> pl.DataFrame:
        """Apply a single comparison/membership operator to ``filtered_df``."""
        if op == "==":
            return filtered_df.filter(col_expr == val)
        elif op == "!=":
            return filtered_df.filter(col_expr != val)
        elif op == ">":
            return filtered_df.filter(col_expr > val)
        elif op == "<":
            return filtered_df.filter(col_expr < val)
        elif op == ">=":
            return filtered_df.filter(col_expr >= val)
        elif op == "<=":
            return filtered_df.filter(col_expr <= val)
        elif op == "in":
            return filtered_df.filter(col_expr.is_in(cast(Any, val)))
        return filtered_df

    @staticmethod
    def _filter_unknown_numeric(filtered_df: pl.DataFrame, col: str, op: str) -> pl.DataFrame:
        """Filter a numeric column against the FE's "Unknown" (null) sentinel."""
        if op == "==":
            return filtered_df.filter(pl.col(col).is_null())
        elif op == "!=":
            return filtered_df.filter(pl.col(col).is_not_null())
        return filtered_df

    def _apply_single_filter(self, filtered_df: pl.DataFrame, f: dict[str, Any]) -> pl.DataFrame:
        """Apply one filter dict to ``filtered_df``, with numeric-vs-string coercion.

        FE serializes all filter values as strings, so numeric columns need
        coercion, and "Unknown" is the FE's stand-in for nulls.
        """
        col = f["column"]
        op = f["operator"]
        val = f["value"]

        if col not in filtered_df.columns:
            return filtered_df

        dtype = filtered_df.schema[col]
        is_numeric = dtype in self._NUMERIC_DTYPES

        if is_numeric and isinstance(val, str):
            if val == "Unknown":
                return self._filter_unknown_numeric(filtered_df, col, op)
            val = self._coerce_filter_value(dtype, val)

        col_expr = (
            pl.col(col).cast(pl.Utf8) if (is_numeric and isinstance(val, str)) else pl.col(col)
        )
        return self._apply_filter_operator(filtered_df, col_expr, op, val)

    def _apply_decomposition_filters(self, filters: list[dict[str, Any]]) -> pl.DataFrame:
        """Apply all ``filters`` sequentially to the analyzer's dataframe."""
        filtered_df = self.df  # type: ignore[attr-defined]
        for f in filters:
            filtered_df = self._apply_single_filter(filtered_df, f)
        return filtered_df

    def _aggregate_series(self, series: pl.Series, measure_agg: str, default: Any) -> Any:
        """Aggregate a series by name, falling back to ``default`` for unknown aggs."""
        if measure_agg == "sum":
            return series.sum()
        elif measure_agg == "mean":
            return series.mean()
        elif measure_agg == "min":
            return series.min()
        elif measure_agg == "max":
            return series.max()
        return default

    def _compute_global_split(
        self, filtered_df: pl.DataFrame, measure_col: str | None, measure_agg: str
    ) -> list[dict[str, Any]]:
        """Compute the single "Total" row aggregate when no split column is requested.

        Returns an empty list if ``measure_col`` is set but missing from the dataframe.
        """
        if not measure_col:
            val = filtered_df.height
        else:
            if measure_col not in filtered_df.columns:
                return []
            series = filtered_df[measure_col]
            val = self._aggregate_series(series, measure_agg, filtered_df.height)

        if val is None:
            val = 0
        return [{"name": "Total", "value": val, "ratio": 1.0}]

    def _aggregate_grouped(
        self, temp_df: pl.DataFrame, split_col: str, measure_col: str | None, measure_agg: str
    ) -> pl.DataFrame | None:
        """Group ``temp_df`` by ``split_col`` and aggregate ``measure_col`` (or count rows)."""
        if not measure_col:
            return temp_df.group_by(split_col).agg(pl.len().alias("value"))

        if measure_col not in temp_df.columns:
            return None

        agg_exprs = {
            "sum": pl.col(measure_col).sum(),
            "mean": pl.col(measure_col).mean(),
            "min": pl.col(measure_col).min(),
            "max": pl.col(measure_col).max(),
        }
        value_expr = agg_exprs.get(measure_agg, pl.len())
        return temp_df.group_by(split_col).agg(value_expr.alias("value"))

    def _rows_to_split_result(self, agg_df: pl.DataFrame, split_col: str) -> list[dict[str, Any]]:
        """Add a normalized "ratio" column to ``agg_df`` and convert it to result dicts."""
        total_val = agg_df["value"].sum()
        if total_val == 0 or total_val is None:
            result_df = agg_df.with_columns(pl.lit(0.0).alias("ratio"))
        else:
            result_df = agg_df.with_columns((pl.col("value") / total_val).alias("ratio"))

        result_df = result_df.sort("value", descending=True)

        return [
            {
                "name": str(row[split_col]),
                "value": row["value"] if row["value"] is not None else 0,
                "ratio": row["ratio"] if row["ratio"] is not None else 0,
            }
            for row in result_df.iter_rows(named=True)
        ]

    def _compute_grouped_split(
        self,
        filtered_df: pl.DataFrame,
        split_col: str,
        measure_col: str | None,
        measure_agg: str,
    ) -> list[dict[str, Any]]:
        """Compute the per-group aggregate + ratio rows when a split column is requested."""
        if split_col not in filtered_df.columns:
            return []

        # Surface nulls as "Unknown" rather than dropping them.
        temp_df = filtered_df.with_columns(pl.col(split_col).fill_null("Unknown").cast(pl.Utf8))

        agg_df = self._aggregate_grouped(temp_df, split_col, measure_col, measure_agg)
        if agg_df is None:
            return []

        return self._rows_to_split_result(agg_df, split_col)

    def get_decomposition_split(
        self,
        measure_col: str | None,
        measure_agg: str,
        split_col: str | None,
        filters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply ``filters``, then aggregate ``measure_col`` either globally or split by ``split_col``."""
        # 1. Apply filters (with numeric-vs-string coercion since FE serializes everything as strings).
        filtered_df = self._apply_decomposition_filters(filters)

        # 2. Global aggregate when no split column was requested.
        if not split_col:
            return self._compute_global_split(filtered_df, measure_col, measure_agg)

        # 3. Group-by aggregate.
        return self._compute_grouped_split(filtered_df, split_col, measure_col, measure_agg)
