"""Date detection (string→datetime cast) and date stat profiling."""

import logging

import polars as pl

from ..schemas import DateStats
from ._utils import _AnalyzerState

logger = logging.getLogger(__name__)


class DatesMixin(_AnalyzerState):
    """Date helpers for :class:`EDAAnalyzer`."""

    def _cast_date_columns(self):
        """Auto-cast string columns that look date-like into ``Date``/``Datetime``.

        Uses the column name as a cheap pre-filter (avoids parsing every utf8
        column), then tries generic ISO parsing and a handful of common formats,
        keeping whichever yields the most distinct months (a proxy for "this
        actually parsed something meaningful").
        """
        for col in self.df.columns:  # type: ignore[attr-defined]
            if self.df[col].dtype not in [pl.Utf8, pl.String]:  # type: ignore[attr-defined]
                continue

            if not self._looks_like_date_column(col):
                continue

            sample = self.df[col].drop_nulls().head(50)  # type: ignore[attr-defined]
            if len(sample) == 0:
                continue

            best_parsed, best_method_name, tied_candidates = self._find_best_date_parse(sample)

            if len(tied_candidates) > 1:
                logger.warning(
                    f"Column '{col}': multiple date formats {tied_candidates} parsed "
                    f"equally well (tied on distinct months); picked '{best_method_name}' "
                    "arbitrarily. Values may be misinterpreted if the true format differs "
                    "(e.g. day/month swap)."
                )

            if best_parsed:
                self._apply_date_cast(col, best_parsed, best_method_name)

    def _looks_like_date_column(self, col: str) -> bool:
        """Cheap name-based pre-filter for candidate date columns."""
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
        return any(k in col_lower for k in date_keywords)

    def _find_best_date_parse(
        self, sample: pl.Series
    ) -> tuple[tuple[str | None, str] | None, str, list[str]]:
        """Try generic + common-format date parses, keeping the one with the most distinct months.

        Returns the winning (format, method) pair (or ``None``), its display
        name, and the list of formats that tied for best (used to warn about
        ambiguous D/M/Y vs M/D/Y parses).
        """
        best_parsed, max_months, best_method_name = self._try_generic_date_parses(sample)
        return self._try_common_format_parses(sample, best_parsed, max_months, best_method_name)

    def _try_generic_date_parses(
        self, sample: pl.Series
    ) -> tuple[tuple[str | None, str] | None, int, str]:
        """Attempt generic datetime/date parsing, returning the best candidate so far."""
        best_parsed: tuple[str | None, str] | None = None
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
            pass  # nosec B110 - datetime-generic parse is a best-effort candidate, others still tried

        try:
            parsed = sample.str.to_date(strict=False)
            if parsed.null_count() == 0:
                n_months = parsed.dt.month().n_unique()
                if n_months > max_months:
                    max_months = n_months
                    best_parsed = (None, "date_generic")
                    best_method_name = "Generic Date"
        except Exception:
            pass  # nosec B110 - date-generic parse is a best-effort candidate, others still tried

        return best_parsed, max_months, best_method_name

    def _try_common_format_parses(
        self,
        sample: pl.Series,
        best_parsed: tuple[str | None, str] | None,
        max_months: int,
        best_method_name: str,
    ) -> tuple[tuple[str | None, str] | None, str, list[str]]:
        """Try a handful of common explicit date formats against the generic-parse baseline."""
        tied_candidates: list[str] = []
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
                        tied_candidates = [fmt]
                    elif n_months == max_months and best_parsed is not None:
                        tied_candidates.append(fmt)
            except Exception:
                continue  # nosec B112 - format candidate didn't match; next format is tried

        return best_parsed, best_method_name, tied_candidates

    def _apply_date_cast(
        self, col: str, best_parsed: tuple[str | None, str], best_method_name: str
    ) -> None:
        """Cast ``col`` in ``self.df`` in place using the winning parse method."""
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
            logger.warning(f"Failed to cast column {col} using {best_method_name}: {e}")

    def _analyze_date(self, col: str, row: dict) -> DateStats:
        min_date = row.get(f"{col}__min")
        max_date = row.get(f"{col}__max")
        duration = None
        if min_date and max_date:
            delta = max_date - min_date
            duration = delta.days if hasattr(delta, "days") else None
        return DateStats(min_date=str(min_date), max_date=str(max_date), duration_days=duration)
