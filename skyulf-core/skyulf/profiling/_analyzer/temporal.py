"""Time series analysis: trend, seasonality, autocorrelation, stationarity."""

from typing import Any, List, Optional, cast

import numpy as np
import polars as pl

from ..schemas import SeasonalityStats, TimeSeriesAnalysis, TimeSeriesPoint
from ._utils import STATSMODELS_AVAILABLE, _AnalyzerState, _collect


class TemporalMixin(_AnalyzerState):
    """Time-series helpers for :class:`EDAAnalyzer`."""

    def _analyze_timeseries(  # noqa: C901
        self,
        numeric_cols: List[str],
        target_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> Optional[TimeSeriesAnalysis]:
        """Trend (with adaptive resampling), DoW/MoY seasonality, ACF, ADF."""
        try:
            if not date_col:
                # Pick the date col with the highest cardinality — avoids constant
                # metadata fields like "ingested_at".
                date_cols = [
                    col
                    for col in self.columns  # type: ignore[attr-defined]
                    if self._get_semantic_type(self.df[col]) == "DateTime"  # type: ignore[attr-defined]
                ]
                if not date_cols:
                    return None

                best_date_col = None
                max_unique = -1
                for col in date_cols:
                    n_unique = self.df[col].n_unique()  # type: ignore[attr-defined]
                    if n_unique > max_unique:
                        max_unique = n_unique
                        best_date_col = col
                date_col = best_date_col

            if not date_col or date_col not in self.columns:  # type: ignore[attr-defined]
                return None

            ts_df = self.lazy_df.sort(date_col)  # type: ignore[attr-defined]

            min_date = cast(Any, self.df[date_col].min())  # type: ignore[attr-defined]
            max_date = cast(Any, self.df[date_col].max())  # type: ignore[attr-defined]

            cols_to_track = numeric_cols[:3]
            if (
                target_col
                and target_col in numeric_cols
                and target_col not in cols_to_track
            ):
                cols_to_track.append(target_col)

            # Small datasets: skip resampling, plot raw points.
            if self.row_count < 1000:  # type: ignore[attr-defined]
                trend_df = _collect(
                    ts_df.select(
                        [pl.col(date_col).alias("date"), *cols_to_track]
                    ).drop_nulls()
                )

                trend_points = []
                for row in trend_df.iter_rows(named=True):
                    if row["date"] is None:
                        continue
                    vals = {
                        k: v for k, v in row.items() if k != "date" and v is not None
                    }
                    if not vals:
                        continue
                    trend_points.append(
                        TimeSeriesPoint(date=row["date"].isoformat(), values=vals)
                    )
            else:
                # Aim for ~100 points across the date range.
                interval = "1d"
                if min_date and max_date:
                    duration = (max_date - min_date).total_seconds()
                    ideal_seconds = duration / 100
                    if ideal_seconds < 60:
                        interval = "1s"
                    elif ideal_seconds < 3600:
                        interval = "1m"
                    elif ideal_seconds < 86400:
                        interval = "1h"
                    elif ideal_seconds < 604800:
                        interval = "1d"
                    else:
                        interval = "1w"

                if not cols_to_track:
                    trend_df = _collect(
                        ts_df.group_by_dynamic(date_col, every=interval)
                        .agg(pl.count().alias("count"))
                        .sort(date_col)
                    )
                else:
                    aggs = [pl.col(c).mean().alias(c) for c in cols_to_track]
                    trend_df = _collect(
                        ts_df.group_by_dynamic(date_col, every=interval)
                        .agg(aggs)
                        .sort(date_col)
                    )

                trend_points = []
                for row in trend_df.iter_rows(named=True):
                    if row[date_col] is None:
                        continue
                    vals = {
                        k: v for k, v in row.items() if k != date_col and v is not None
                    }
                    if not vals:
                        continue
                    trend_points.append(
                        TimeSeriesPoint(date=row[date_col].isoformat(), values=vals)
                    )

            # Seasonality: aggregate by day-of-week. Keep alias "count" for FE compat
            # even when we're really computing a mean of the primary numeric metric.
            agg_expr = pl.len().alias("count")
            if cols_to_track:
                target_metric = cols_to_track[0]
                agg_expr = pl.col(target_metric).mean().alias("count")

            dow_df = _collect(
                self.lazy_df.with_columns(  # type: ignore[attr-defined]
                    pl.col(date_col).dt.weekday().alias("dow_idx"),
                    pl.col(date_col).dt.strftime("%a").alias("dow_name"),
                )
                .group_by(["dow_idx", "dow_name"])
                .agg(agg_expr)
                .sort("dow_idx")
            )
            dow_stats = [
                {"day": row["dow_name"], "count": row["count"]}
                for row in dow_df.iter_rows(named=True)
            ]

            moy_df = _collect(
                self.lazy_df.with_columns(  # type: ignore[attr-defined]
                    pl.col(date_col).dt.month().alias("month_idx"),
                    pl.col(date_col).dt.strftime("%b").alias("month_name"),
                )
                .group_by(["month_idx", "month_name"])
                .agg(agg_expr)
                .sort("month_idx")
            )
            moy_stats = [
                {"month": row["month_name"], "count": row["count"]}
                for row in moy_df.iter_rows(named=True)
            ]

            # ACF (lags 1..30) on the resampled trend.
            acf_stats: List[dict] = []
            if cols_to_track:
                target_metric = cols_to_track[0]
                series = trend_df[target_metric].to_numpy()

                mask = np.isnan(series)
                if mask.any():
                    series[mask] = np.nanmean(series)

                if len(series) > 10:
                    n = len(series)
                    mean = np.mean(series)
                    var = np.var(series)
                    for lag in range(1, min(31, n // 2)):
                        y1 = series[lag:]
                        y2 = series[:-lag]
                        corr = (
                            0
                            if var == 0
                            else np.sum((y1 - mean) * (y2 - mean)) / n / var
                        )
                        acf_stats.append({"lag": lag, "corr": float(corr)})

            stationarity_test = None
            if STATSMODELS_AVAILABLE and cols_to_track:
                try:
                    from statsmodels.tsa.stattools import adfuller

                    target_metric = cols_to_track[0]
                    series = trend_df[target_metric].to_numpy()
                    mask = np.isnan(series)
                    if mask.any():
                        series[mask] = np.nanmean(series)

                    if len(series) > 20:
                        result = adfuller(series)
                        stationarity_test = {
                            "test_statistic": float(result[0]),
                            "p_value": float(result[1]),
                            "is_stationary": float(result[1]) < 0.05,
                            "metric": target_metric,
                        }
                except Exception as e:
                    print(f"ADF test failed: {e}")

            return TimeSeriesAnalysis(
                date_col=date_col,
                trend=trend_points,
                seasonality=SeasonalityStats(
                    day_of_week=dow_stats, month_of_year=moy_stats
                ),
                autocorrelation=acf_stats,
                stationarity_test=stationarity_test,
            )
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            return None
