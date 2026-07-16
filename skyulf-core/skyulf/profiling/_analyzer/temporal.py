"""Time series analysis: trend, seasonality, autocorrelation, stationarity."""

import logging
from typing import Any, cast

import numpy as np
import polars as pl

from ..schemas import SeasonalityStats, TimeSeriesAnalysis, TimeSeriesPoint
from ._utils import STATSMODELS_AVAILABLE, _AnalyzerState, _collect

logger = logging.getLogger(__name__)


class TemporalMixin(_AnalyzerState):
    """Time-series helpers for :class:`EDAAnalyzer`."""

    def _resolve_date_col(self, date_col: str | None) -> str | None:
        """Pick the date col with the highest cardinality when none is given.

        Avoids constant metadata fields like "ingested_at".
        """
        if date_col:
            return date_col if date_col in self.columns else None  # type: ignore[attr-defined]

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
        return best_date_col

    def _resample_interval(self, min_date: Any, max_date: Any) -> str:
        """Pick a `group_by_dynamic` interval that yields ~100 points across the date range."""
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
        return interval

    def _build_raw_trend(
        self, ts_df: pl.LazyFrame, date_col: str, cols_to_track: list[str]
    ) -> tuple[pl.DataFrame, list[TimeSeriesPoint]]:
        """Build trend points from raw (unresampled) rows for small datasets."""
        trend_df = _collect(
            ts_df.select([pl.col(date_col).alias("date"), *cols_to_track]).drop_nulls()
        )

        trend_points = []
        for row in trend_df.iter_rows(named=True):
            if row["date"] is None:
                continue
            vals = {k: v for k, v in row.items() if k != "date" and v is not None}
            if not vals:
                continue
            trend_points.append(TimeSeriesPoint(date=row["date"].isoformat(), values=vals))
        return trend_df, trend_points

    @staticmethod
    def _resample_trend_df(
        ts_df: pl.LazyFrame, date_col: str, cols_to_track: list[str], interval: str
    ) -> pl.DataFrame:
        """Group-by-dynamic resample of counts (no tracked cols) or per-column means."""
        if not cols_to_track:
            return _collect(
                ts_df.group_by_dynamic(date_col, every=interval)
                .agg(pl.len().alias("count"))
                .sort(date_col)
            )
        aggs = [pl.col(c).mean().alias(c) for c in cols_to_track]
        return _collect(ts_df.group_by_dynamic(date_col, every=interval).agg(aggs).sort(date_col))

    @staticmethod
    def _trend_points_from_df(trend_df: pl.DataFrame, date_col: str) -> list[TimeSeriesPoint]:
        """Convert resampled trend rows into `TimeSeriesPoint` models, skipping empty/null rows."""
        trend_points = []
        for row in trend_df.iter_rows(named=True):
            if row[date_col] is None:
                continue
            vals = {k: v for k, v in row.items() if k != date_col and v is not None}
            if not vals:
                continue
            trend_points.append(TimeSeriesPoint(date=row[date_col].isoformat(), values=vals))
        return trend_points

    def _build_resampled_trend(
        self,
        ts_df: pl.LazyFrame,
        date_col: str,
        cols_to_track: list[str],
        min_date: Any,
        max_date: Any,
    ) -> tuple[pl.DataFrame, list[TimeSeriesPoint]]:
        """Group-by-dynamic resample the trend to ~100 points across the date range."""
        interval = self._resample_interval(min_date, max_date)
        trend_df = self._resample_trend_df(ts_df, date_col, cols_to_track, interval)
        trend_points = self._trend_points_from_df(trend_df, date_col)
        return trend_df, trend_points

    def _compute_seasonality(self, date_col: str, cols_to_track: list[str]) -> SeasonalityStats:
        """Aggregate by day-of-week and month-of-year.

        Keeps alias "count" for FE compat even when we're really computing a
        mean of the primary numeric metric.
        """
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
            {"day": row["dow_name"], "count": row["count"]} for row in dow_df.iter_rows(named=True)
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

        return SeasonalityStats(day_of_week=dow_stats, month_of_year=moy_stats)

    def _compute_acf(self, trend_df: pl.DataFrame, cols_to_track: list[str]) -> list[dict]:
        """Compute autocorrelation (lags 1..30) on the resampled trend."""
        acf_stats: list[dict] = []
        if not cols_to_track:
            return acf_stats

        target_metric = cols_to_track[0]
        # .copy() avoids "assignment destination is read-only" — polars
        # returns a zero-copy read-only view when the column has no nulls.
        series = trend_df[target_metric].to_numpy().copy()

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
                corr = 0 if var == 0 else np.sum((y1 - mean) * (y2 - mean)) / n / var
                acf_stats.append({"lag": lag, "corr": float(corr)})
        return acf_stats

    def _compute_stationarity_test(
        self, trend_df: pl.DataFrame, cols_to_track: list[str]
    ) -> dict | None:
        """Run the Augmented Dickey-Fuller stationarity test on the primary tracked metric."""
        if not (STATSMODELS_AVAILABLE and cols_to_track):
            return None
        try:
            from statsmodels.tsa.stattools import adfuller

            target_metric = cols_to_track[0]
            series = trend_df[target_metric].to_numpy().copy()
            mask = np.isnan(series)
            if mask.any():
                series[mask] = np.nanmean(series)

            if len(series) > 20:
                result = adfuller(series)
                return {
                    "test_statistic": float(result[0]),
                    "p_value": float(result[1]),
                    "is_stationary": float(result[1]) < 0.05,
                    "metric": target_metric,
                }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
        return None

    def _analyze_timeseries(
        self,
        numeric_cols: list[str],
        target_col: str | None = None,
        date_col: str | None = None,
    ) -> TimeSeriesAnalysis | None:
        """Trend (with adaptive resampling), DoW/MoY seasonality, ACF, ADF."""
        try:
            date_col = self._resolve_date_col(date_col)
            if not date_col:
                return None

            ts_df = self.lazy_df.sort(date_col)  # type: ignore[attr-defined]

            min_date = cast(Any, self.df[date_col].min())  # type: ignore[attr-defined]
            max_date = cast(Any, self.df[date_col].max())  # type: ignore[attr-defined]

            cols_to_track = numeric_cols[:3]
            if target_col and target_col in numeric_cols and target_col not in cols_to_track:
                cols_to_track.append(target_col)

            # Small datasets: skip resampling, plot raw points.
            if self.row_count < 1000:  # type: ignore[attr-defined]
                trend_df, trend_points = self._build_raw_trend(ts_df, date_col, cols_to_track)
            else:
                trend_df, trend_points = self._build_resampled_trend(
                    ts_df, date_col, cols_to_track, min_date, max_date
                )

            seasonality = self._compute_seasonality(date_col, cols_to_track)
            acf_stats = self._compute_acf(trend_df, cols_to_track)
            stationarity_test = self._compute_stationarity_test(trend_df, cols_to_track)

            return TimeSeriesAnalysis(
                date_col=date_col,
                trend=trend_points,
                seasonality=seasonality,
                autocorrelation=acf_stats,
                stationarity_test=stationarity_test,
            )
        except Exception as e:
            logger.warning(f"Error in time series analysis: {e}")
            return None
