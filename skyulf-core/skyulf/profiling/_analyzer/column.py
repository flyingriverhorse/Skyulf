"""Per-column dispatch: semantic typing + the big ``_analyze_column`` orchestrator."""

from typing import List, Tuple

import numpy as np
import polars as pl

from ..distributions import calculate_histogram
from ..schemas import Alert, ColumnProfile, HistogramBin, NormalityTestResult
from ._utils import SCIPY_AVAILABLE, _AnalyzerState


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

    def _analyze_column(  # noqa: C901
        self,
        col: str,
        basic_stats: dict,
        advanced_stats: dict,
        semantic_types: dict,
    ) -> Tuple[ColumnProfile, List[Alert]]:
        """Build the per-column ``ColumnProfile`` from already-batched aggregations.

        Reads ``basic_stats`` / ``advanced_stats`` rows produced by
        :meth:`EDAAnalyzer.analyze` and dispatches to the per-type analyzers.
        """
        alerts: List[Alert] = []

        semantic_type = semantic_types[col]
        null_count = basic_stats.get(f"{col}__null", 0)
        null_pct = (null_count / self.row_count) * 100 if self.row_count > 0 else 0  # type: ignore[attr-defined]

        if null_pct > 5:
            alerts.append(
                Alert(
                    column=col,
                    type="High Null",
                    message=f"Column '{col}' has {null_pct:.1f}% missing values.",
                    severity="warning",
                )
            )

        profile = ColumnProfile(
            name=col,
            dtype=semantic_type,
            missing_count=null_count,
            missing_percentage=null_pct,
        )

        if semantic_type == "Numeric":
            profile.numeric_stats = self._analyze_numeric(col, advanced_stats)  # type: ignore[attr-defined]
            profile.histogram = calculate_histogram(self.lazy_df, col)  # type: ignore[attr-defined]

            # Normality test — Shapiro for small samples (more powerful), KS for large.
            if (
                SCIPY_AVAILABLE
                and profile.numeric_stats
                and profile.numeric_stats.std
                and profile.numeric_stats.std > 0
            ):
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
                            stat, p_value = kstest(
                                sample_data, "norm", args=(mean, std)
                            )
                            test_name = "Kolmogorov-Smirnov"

                        profile.normality_test = NormalityTestResult(
                            test_name=test_name,
                            statistic=float(stat),
                            p_value=float(p_value),
                            is_normal=float(p_value) > 0.05,
                        )
                except Exception as e:
                    print(f"Normality test failed for {col}: {e}")

            # IQR-based outlier hint (cheap; just looks at min/max vs whiskers).
            if (
                profile.numeric_stats
                and profile.numeric_stats.q25 is not None
                and profile.numeric_stats.q75 is not None
            ):
                iqr = profile.numeric_stats.q75 - profile.numeric_stats.q25
                if iqr > 0:
                    lower_bound = profile.numeric_stats.q25 - 1.5 * iqr
                    upper_bound = profile.numeric_stats.q75 + 1.5 * iqr
                    if (
                        profile.numeric_stats.min is not None
                        and profile.numeric_stats.max is not None
                        and (
                            profile.numeric_stats.min < lower_bound
                            or profile.numeric_stats.max > upper_bound
                        )
                    ):
                        alerts.append(
                            Alert(
                                column=col,
                                type="Outlier",
                                message=f"Column '{col}' contains significant outliers.",
                                severity="info",
                            )
                        )

            if profile.numeric_stats.std == 0:
                profile.is_constant = True
                alerts.append(
                    Alert(
                        column=col,
                        type="Constant",
                        message=f"Column '{col}' is constant.",
                        severity="warning",
                    )
                )

        elif semantic_type in ("Categorical", "Boolean"):
            profile.categorical_stats = self._analyze_categorical(  # type: ignore[attr-defined]
                col, advanced_stats, basic_stats
            )

            if (
                profile.categorical_stats.unique_count > 50
                and semantic_type == "Categorical"
            ):
                # Distinguish "looks like an ID" from plain high-cardinality.
                if profile.categorical_stats.unique_count == self.row_count:  # type: ignore[attr-defined]
                    profile.is_unique = True
                    alerts.append(
                        Alert(
                            column=col,
                            type="Possible ID",
                            message=f"Column '{col}' appears to be an ID.",
                            severity="info",
                        )
                    )
                elif profile.categorical_stats.unique_count > 1000:
                    alerts.append(
                        Alert(
                            column=col,
                            type="High Cardinality",
                            message=(
                                f"Column '{col}' has high cardinality "
                                f"({profile.categorical_stats.unique_count})."
                            ),
                            severity="warning",
                        )
                    )

            if semantic_type == "Categorical" and self._check_pii(col):  # type: ignore[attr-defined]
                alerts.append(
                    Alert(
                        column=col,
                        type="PII",
                        message=f"Column '{col}' may contain PII (Email/Phone).",
                        severity="error",
                    )
                )

        elif semantic_type == "DateTime":
            profile.date_stats = self._analyze_date(col, advanced_stats)  # type: ignore[attr-defined]

            try:
                ts = self.df[col].dt.timestamp("ms").drop_nulls().to_numpy()  # type: ignore[attr-defined]
                if len(ts) > 0:
                    hist, bin_edges = np.histogram(ts, bins=10)
                    profile.histogram = [
                        HistogramBin(
                            start=float(bin_edges[i]),
                            end=float(bin_edges[i + 1]),
                            count=int(hist[i]),
                        )
                        for i in range(len(hist))
                    ]
            except Exception as e:
                print(f"Failed to calculate date histogram for {col}: {e}")

        elif semantic_type == "Text":
            profile.text_stats = self._analyze_text(col, advanced_stats)  # type: ignore[attr-defined]
            if profile.text_stats:
                profile.text_stats.sentiment_distribution = self._analyze_sentiment(  # type: ignore[attr-defined]
                    self.df[col]  # type: ignore[attr-defined]
                )

            try:
                lengths = self.df[col].str.len_bytes().drop_nulls().to_numpy()  # type: ignore[attr-defined]
                if len(lengths) > 0:
                    hist, bin_edges = np.histogram(lengths, bins=10)
                    profile.histogram = [
                        HistogramBin(
                            start=float(bin_edges[i]),
                            end=float(bin_edges[i + 1]),
                            count=int(hist[i]),
                        )
                        for i in range(len(hist))
                    ]
            except Exception as e:
                print(f"Failed to calculate text histogram for {col}: {e}")

            if self._check_pii(col):  # type: ignore[attr-defined]
                alerts.append(
                    Alert(
                        column=col,
                        type="PII",
                        message=f"Column '{col}' may contain PII (Email/Phone).",
                        severity="error",
                    )
                )

        return profile, alerts
