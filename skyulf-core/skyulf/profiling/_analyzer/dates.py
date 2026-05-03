"""Date detection (string→datetime cast) and date stat profiling."""

import polars as pl

from ..schemas import DateStats
from ._utils import _AnalyzerState


class DatesMixin(_AnalyzerState):
    """Date helpers for :class:`EDAAnalyzer`."""

    def _cast_date_columns(self):  # noqa: C901
        """Auto-cast string columns that look date-like into ``Date``/``Datetime``.

        Uses the column name as a cheap pre-filter (avoids parsing every utf8
        column), then tries generic ISO parsing and a handful of common formats,
        keeping whichever yields the most distinct months (a proxy for "this
        actually parsed something meaningful").
        """
        for col in self.df.columns:  # type: ignore[attr-defined]
            if self.df[col].dtype not in [pl.Utf8, pl.String]:  # type: ignore[attr-defined]
                continue

            col_lower = col.lower()
            date_keywords = [
                "date",
                "time",
                "year",
                "month",
                "day",
                "ts",
                "created",
                "updated",
                "at",
            ]
            if not any(k in col_lower for k in date_keywords):
                continue

            sample = self.df[col].drop_nulls().head(50)  # type: ignore[attr-defined]
            if len(sample) == 0:
                continue

            best_parsed = None
            max_months = -1
            best_method_name = ""

            try:
                parsed = sample.str.to_datetime(strict=False)
                if parsed.null_count() == 0:
                    n_months = parsed.dt.month().n_unique()
                    if n_months > max_months:
                        max_months = n_months
                        best_parsed = (None, "datetime_generic")
                        best_method_name = "Generic Datetime"
            except Exception:
                pass

            try:
                parsed = sample.str.to_date(strict=False)
                if parsed.null_count() == 0:
                    n_months = parsed.dt.month().n_unique()
                    if n_months > max_months:
                        max_months = n_months
                        best_parsed = (None, "date_generic")
                        best_method_name = "Generic Date"
            except Exception:
                pass

            common_formats = [
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%m-%d-%Y",
                "%d-%m-%Y",
                "%Y/%m/%d",
                "%Y-%m-%d",
            ]
            for fmt in common_formats:
                try:
                    parsed = sample.str.to_datetime(format=fmt, strict=False)
                    if parsed.null_count() == 0:
                        n_months = parsed.dt.month().n_unique()
                        # Maximize unique months → disambiguates D/M/Y vs M/D/Y.
                        if n_months > max_months:
                            max_months = n_months
                            best_parsed = (fmt, "datetime_format")
                            best_method_name = f"Format {fmt}"
                except Exception:
                    continue

            if best_parsed:
                fmt, method = best_parsed
                try:
                    if method == "datetime_generic":
                        self.df = self.df.with_columns(  # type: ignore[attr-defined]
                            pl.col(col).str.to_datetime(strict=False).alias(col)
                        )
                    elif method == "date_generic":
                        self.df = self.df.with_columns(  # type: ignore[attr-defined]
                            pl.col(col).str.to_date(strict=False).alias(col)
                        )
                    elif method == "datetime_format":
                        self.df = self.df.with_columns(  # type: ignore[attr-defined]
                            pl.col(col).str.to_datetime(format=fmt, strict=False).alias(col)
                        )
                except Exception as e:
                    print(f"Failed to cast column {col} using {best_method_name}: {e}")

    def _analyze_date(self, col: str, row: dict) -> DateStats:
        min_date = row.get(f"{col}__min")
        max_date = row.get(f"{col}__max")
        duration = None
        if min_date and max_date:
            delta = max_date - min_date
            duration = delta.days if hasattr(delta, "days") else None
        return DateStats(min_date=str(min_date), max_date=str(max_date), duration_days=duration)
