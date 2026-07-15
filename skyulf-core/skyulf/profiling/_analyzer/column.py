"""Per-column dispatch: semantic typing + the big ``_analyze_column`` orchestrator."""

import logging

import numpy as np
import polars as pl

from ..distributions import calculate_histogram
from ..schemas import (
    Alert,
    CategoricalStats,
    ColumnProfile,
    HistogramBin,
    NormalityTestResult,
    NumericStats,
)
from ._utils import SCIPY_AVAILABLE, _AnalyzerState

logger = logging.getLogger(__name__)


class ColumnMixin(_AnalyzerState):
    """Single-column analysis dispatch."""

    def _get_semantic_type(self, series: pl.Series) -> str:
        """Map polars dtype + cardinality heuristics to a semantic bucket."""
        dtype = series.dtype

        if dtype in [pl.Float32, pl.Float64]:
            return "Numeric"
        if dtype in [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]:
            # Low-cardinality ints behave more like categories (better for plotting).
            n_unique = series.n_unique()
            count = len(series)
            ratio = n_unique / count if count > 0 else 0
            if ratio < 0.05 and n_unique < 20:
                return "Categorical"
            return "Numeric"
        if dtype == pl.Boolean:
            return "Boolean"
        if dtype in [pl.Date, pl.Datetime, pl.Duration]:
            return "DateTime"
        if dtype == pl.Utf8 or dtype == pl.String:
            n_unique = series.n_unique()
            count = len(series)
            ratio = n_unique / count if count > 0 else 0
            if ratio < 0.05:
                return "Categorical"
            return "Text"
        if dtype == pl.Categorical:
            return "Categorical"

        return "Text"

    def _add_high_null_alert(self, col: str, null_pct: float, alerts: list[Alert]) -> None:
        """Flag columns with more than 5% missing values."""
        if null_pct > 5:
            alerts.append(
                Alert(
                    column=col,
                    type="High Null",
                    message=f"Column '{col}' has {null_pct:.1f}% missing values.",
                    severity="warning",
                )
            )

    def _add_normality_test(self, col: str, profile: ColumnProfile) -> None:
        """Run Shapiro-Wilk (small samples) or Kolmogorov-Smirnov (large samples) normality test."""
        if not (
            SCIPY_AVAILABLE
            and profile.numeric_stats
            and profile.numeric_stats.std
            and profile.numeric_stats.std > 0
        ):
            return
        try:
            from scipy.stats import kstest, shapiro

            sample_data = self.df[col].drop_nulls().head(5000).to_numpy()  # type: ignore[attr-defined]

            if len(sample_data) > 20 and np.std(sample_data) > 1e-10:
                if len(sample_data) < 5000:
                    stat, p_value = shapiro(sample_data)
                    test_name = "Shapiro-Wilk"
                else:
                    # Fit normal to data first; otherwise KS would test against N(0,1).
                    mean, std = np.mean(sample_data), np.std(sample_data)
                    stat, p_value = kstest(sample_data, "norm", args=(mean, std))
                    test_name = "Kolmogorov-Smirnov"

                profile.normality_test = NormalityTestResult(
                    test_name=test_name,
                    statistic=float(stat),
                    p_value=float(p_value),
                    is_normal=float(p_value) > 0.05,
                )
        except Exception as e:
            logger.warning(f"Normality test failed for {col}: {e}")

    def _add_outlier_alert(self, col: str, profile: ColumnProfile, alerts: list[Alert]) -> None:
        """IQR-based outlier hint (cheap; just looks at min/max vs whiskers)."""
        if not (
            profile.numeric_stats
            and profile.numeric_stats.q25 is not None
            and profile.numeric_stats.q75 is not None
        ):
            return
        iqr = profile.numeric_stats.q75 - profile.numeric_stats.q25
        if iqr <= 0:
            return
        lower_bound = profile.numeric_stats.q25 - 1.5 * iqr
        upper_bound = profile.numeric_stats.q75 + 1.5 * iqr
        if (
            profile.numeric_stats.min is not None
            and profile.numeric_stats.max is not None
            and (profile.numeric_stats.min < lower_bound or profile.numeric_stats.max > upper_bound)
        ):
            alerts.append(
                Alert(
                    column=col,
                    type="Outlier",
                    message=f"Column '{col}' contains significant outliers.",
                    severity="info",
                )
            )

    def _add_constant_alert(
        self, col: str, profile: ColumnProfile, alerts: list[Alert], numeric_stats: NumericStats
    ) -> None:
        """Flag columns whose numeric std is zero as constant."""
        if numeric_stats.std == 0:
            profile.is_constant = True
            alerts.append(
                Alert(
                    column=col,
                    type="Constant",
                    message=f"Column '{col}' is constant.",
                    severity="warning",
                )
            )

    def _process_numeric_column(
        self, col: str, profile: ColumnProfile, alerts: list[Alert], advanced_stats: dict
    ) -> None:
        """Compute numeric stats, histogram, normality test, outlier and constant alerts."""
        numeric_stats = self._analyze_numeric(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
            col, advanced_stats
        )
        profile.numeric_stats = numeric_stats
        profile.histogram = calculate_histogram(self.lazy_df, col)  # type: ignore[attr-defined]

        self._add_normality_test(col, profile)
        self._add_outlier_alert(col, profile, alerts)
        self._add_constant_alert(col, profile, alerts, numeric_stats)

    def _add_cardinality_alerts(
        self,
        col: str,
        profile: ColumnProfile,
        alerts: list[Alert],
        semantic_type: str,
        categorical_stats: CategoricalStats,
    ) -> None:
        """Distinguish "looks like an ID" from plain high-cardinality Categorical columns."""
        if not (categorical_stats.unique_count > 50 and semantic_type == "Categorical"):
            return
        if categorical_stats.unique_count == self.row_count:  # type: ignore[attr-defined]
            profile.is_unique = True
            alerts.append(
                Alert(
                    column=col,
                    type="Possible ID",
                    message=f"Column '{col}' appears to be an ID.",
                    severity="info",
                )
            )
        elif categorical_stats.unique_count > 1000:
            alerts.append(
                Alert(
                    column=col,
                    type="High Cardinality",
                    message=(
                        f"Column '{col}' has high cardinality ({categorical_stats.unique_count})."
                    ),
                    severity="warning",
                )
            )

    def _add_pii_alert(self, col: str, alerts: list[Alert]) -> None:
        """Flag a column as possibly containing PII (Email/Phone)."""
        alerts.append(
            Alert(
                column=col,
                type="PII",
                message=f"Column '{col}' may contain PII (Email/Phone).",
                severity="error",
            )
        )

    def _process_categorical_column(
        self,
        col: str,
        profile: ColumnProfile,
        alerts: list[Alert],
        advanced_stats: dict,
        basic_stats: dict,
        semantic_type: str,
    ) -> None:
        """Compute categorical stats and cardinality/PII alerts for Categorical/Boolean columns."""
        categorical_stats = self._analyze_categorical(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
            col, advanced_stats, basic_stats
        )
        profile.categorical_stats = categorical_stats

        self._add_cardinality_alerts(col, profile, alerts, semantic_type, categorical_stats)

        if semantic_type == "Categorical" and self._check_pii(col):  # type: ignore[attr-defined]
            self._add_pii_alert(col, alerts)

    def _numpy_histogram_bins(self, values: np.ndarray) -> list[HistogramBin]:
        """Build 10-bin `HistogramBin` list from raw numeric values via numpy."""
        hist, bin_edges = np.histogram(values, bins=10)
        return [
            HistogramBin(
                start=float(bin_edges[i]),
                end=float(bin_edges[i + 1]),
                count=int(hist[i]),
            )
            for i in range(len(hist))
        ]

    def _process_datetime_column(
        self, col: str, profile: ColumnProfile, advanced_stats: dict
    ) -> None:
        """Compute date stats and a timestamp-based histogram for DateTime columns."""
        profile.date_stats = self._analyze_date(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
            col, advanced_stats
        )

        try:
            ts = self.df[col].dt.timestamp("ms").drop_nulls().to_numpy()  # type: ignore[attr-defined]
            if len(ts) > 0:
                profile.histogram = self._numpy_histogram_bins(ts)
        except Exception as e:
            logger.warning(f"Failed to calculate date histogram for {col}: {e}")

    def _process_text_column(
        self, col: str, profile: ColumnProfile, alerts: list[Alert], advanced_stats: dict
    ) -> None:
        """Compute text stats, sentiment, length histogram and PII alert for Text columns."""
        profile.text_stats = self._analyze_text(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
            col, advanced_stats
        )
        if profile.text_stats:
            profile.text_stats.sentiment_distribution = self._analyze_sentiment(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
                self.df[col]  # type: ignore[attr-defined]
            )

        try:
            lengths = self.df[col].str.len_bytes().drop_nulls().to_numpy()  # type: ignore[attr-defined]
            if len(lengths) > 0:
                profile.histogram = self._numpy_histogram_bins(lengths)
        except Exception as e:
            logger.warning(f"Failed to calculate text histogram for {col}: {e}")

        if self._check_pii(col):  # type: ignore[attr-defined]
            self._add_pii_alert(col, alerts)

    def _add_generic_unique_alert(
        self,
        col: str,
        profile: ColumnProfile,
        alerts: list[Alert],
        basic_stats: dict,
        semantic_type: str,
    ) -> None:
        """Flag Text/Numeric columns as "possible ID" when every value is distinct.

        `is_unique` (used by the "possible ID column" recommendation) is
        only ever set above for Categorical columns via categorical_stats.
        Text and Numeric columns never got a chance to be flagged even
        when every value is distinct (e.g. a numeric or free-text ID
        column), so the recommendation was silently dead for those dtypes.
        Compute it generically here from the batched `__unique` aggregate,
        using the same >50-unique-values threshold as the Categorical
        branch above to avoid over-flagging small tables.
        """
        if profile.is_unique or semantic_type not in ("Text", "Numeric"):
            return
        n_unique = basic_stats.get(f"{col}__unique", 0)
        if (
            n_unique > 50
            and self.row_count > 0  # type: ignore[attr-defined]
            and n_unique == self.row_count  # type: ignore[attr-defined]
        ):
            profile.is_unique = True
            alerts.append(
                Alert(
                    column=col,
                    type="Possible ID",
                    message=f"Column '{col}' appears to be an ID.",
                    severity="info",
                )
            )

    def _analyze_column(
        self,
        col: str,
        basic_stats: dict,
        advanced_stats: dict,
        semantic_types: dict,
    ) -> tuple[ColumnProfile, list[Alert]]:
        """Build the per-column ``ColumnProfile`` from already-batched aggregations.

        Reads ``basic_stats`` / ``advanced_stats`` rows produced by
        :meth:`EDAAnalyzer.analyze` and dispatches to the per-type analyzers.
        """
        alerts: list[Alert] = []

        semantic_type = semantic_types[col]
        null_count = basic_stats.get(f"{col}__null", 0)
        null_pct = (null_count / self.row_count) * 100 if self.row_count > 0 else 0  # type: ignore[attr-defined]

        self._add_high_null_alert(col, null_pct, alerts)

        profile = ColumnProfile(
            name=col,
            dtype=semantic_type,
            missing_count=null_count,
            missing_percentage=null_pct,
        )

        if semantic_type == "Numeric":
            self._process_numeric_column(col, profile, alerts, advanced_stats)
        elif semantic_type in ("Categorical", "Boolean"):
            self._process_categorical_column(
                col, profile, alerts, advanced_stats, basic_stats, semantic_type
            )
        elif semantic_type == "DateTime":
            self._process_datetime_column(col, profile, advanced_stats)
        elif semantic_type == "Text":
            self._process_text_column(col, profile, alerts, advanced_stats)

        self._add_generic_unique_alert(col, profile, alerts, basic_stats, semantic_type)

        return profile, alerts
