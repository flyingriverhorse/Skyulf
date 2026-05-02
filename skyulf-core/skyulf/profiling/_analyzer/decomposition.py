"""Decomposition-tree split: filter → group-by → measure aggregation."""

from typing import Any, Dict, List, Optional, cast

import polars as pl

from ._utils import _AnalyzerState


class DecompositionMixin(_AnalyzerState):
    """Decomposition-tree helpers for :class:`EDAAnalyzer`."""

    def get_decomposition_split(  # noqa: C901
        self,
        measure_col: Optional[str],
        measure_agg: str,
        split_col: Optional[str],
        filters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply ``filters``, then aggregate ``measure_col`` either globally or split by ``split_col``."""
        # 1. Apply filters (with numeric-vs-string coercion since FE serializes everything as strings).
        filtered_df = self.df  # type: ignore[attr-defined]
        for f in filters:
            col = f["column"]
            op = f["operator"]
            val = f["value"]

            if col not in filtered_df.columns:
                continue

            dtype = filtered_df.schema[col]
            is_numeric = dtype in (
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

            if is_numeric and isinstance(val, str):
                if val == "Unknown":
                    # "Unknown" is the FE's stand-in for nulls.
                    if op == "==":
                        filtered_df = filtered_df.filter(pl.col(col).is_null())
                    elif op == "!=":
                        filtered_df = filtered_df.filter(pl.col(col).is_not_null())
                    continue
                else:
                    try:
                        if dtype in (pl.Float32, pl.Float64):
                            val = float(val)
                        else:
                            val = int(float(val))  # tolerate "1.0" strings for int cols
                    except ValueError:
                        # Fall through; we'll cast the column to string below.
                        pass

            col_expr = (
                pl.col(col).cast(pl.Utf8)
                if (is_numeric and isinstance(val, str))
                else pl.col(col)
            )

            if op == "==":
                filtered_df = filtered_df.filter(col_expr == val)
            elif op == "!=":
                filtered_df = filtered_df.filter(col_expr != val)
            elif op == ">":
                filtered_df = filtered_df.filter(col_expr > val)
            elif op == "<":
                filtered_df = filtered_df.filter(col_expr < val)
            elif op == ">=":
                filtered_df = filtered_df.filter(col_expr >= val)
            elif op == "<=":
                filtered_df = filtered_df.filter(col_expr <= val)
            elif op == "in":
                filtered_df = filtered_df.filter(col_expr.is_in(cast(Any, val)))

        # 2. Global aggregate when no split column was requested.
        if not split_col:
            if not measure_col:
                val = filtered_df.height
            else:
                if measure_col not in filtered_df.columns:
                    return []
                series = filtered_df[measure_col]
                if measure_agg == "sum":
                    val = series.sum()
                elif measure_agg == "mean":
                    val = series.mean()
                elif measure_agg == "min":
                    val = series.min()
                elif measure_agg == "max":
                    val = series.max()
                else:
                    val = filtered_df.height

            if val is None:
                val = 0
            return [{"name": "Total", "value": val, "ratio": 1.0}]

        # 3. Group-by aggregate.
        if split_col not in filtered_df.columns:
            return []

        # Surface nulls as "Unknown" rather than dropping them.
        temp_df = filtered_df.with_columns(
            pl.col(split_col).fill_null("Unknown").cast(pl.Utf8)
        )

        if not measure_col:
            agg_df = temp_df.group_by(split_col).agg(pl.len().alias("value"))
        else:
            if measure_col not in temp_df.columns:
                return []
            if measure_agg == "sum":
                agg_df = temp_df.group_by(split_col).agg(
                    pl.col(measure_col).sum().alias("value")
                )
            elif measure_agg == "mean":
                agg_df = temp_df.group_by(split_col).agg(
                    pl.col(measure_col).mean().alias("value")
                )
            elif measure_agg == "min":
                agg_df = temp_df.group_by(split_col).agg(
                    pl.col(measure_col).min().alias("value")
                )
            elif measure_agg == "max":
                agg_df = temp_df.group_by(split_col).agg(
                    pl.col(measure_col).max().alias("value")
                )
            else:
                agg_df = temp_df.group_by(split_col).agg(pl.len().alias("value"))

        total_val = agg_df["value"].sum()
        if total_val == 0 or total_val is None:
            result_df = agg_df.with_columns(pl.lit(0.0).alias("ratio"))
        else:
            result_df = agg_df.with_columns(
                (pl.col("value") / total_val).alias("ratio")
            )

        result_df = result_df.sort("value", descending=True)

        return [
            {
                "name": str(row[split_col]),
                "value": row["value"] if row["value"] is not None else 0,
                "ratio": row["ratio"] if row["ratio"] is not None else 0,
            }
            for row in result_df.iter_rows(named=True)
        ]
