import numpy as np
import polars as pl
from pydantic import BaseModel

try:
    from scipy.stats import entropy, ks_2samp, wasserstein_distance

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DriftMetric(BaseModel):
    metric: str
    value: float
    has_drift: bool
    threshold: float


class DriftBin(BaseModel):
    bin_start: float
    bin_end: float
    reference_count: int
    current_count: int


class DriftDistribution(BaseModel):
    bins: list[DriftBin]


class ColumnDrift(BaseModel):
    column: str
    metrics: list[DriftMetric]
    drift_detected: bool
    suggestions: list[str] = []
    distribution: DriftDistribution | None = None


class DriftReport(BaseModel):
    reference_rows: int
    current_rows: int
    drifted_columns_count: int
    column_drifts: dict[str, ColumnDrift]
    missing_columns: list[str] = []
    new_columns: list[str] = []


class DriftCalculator:
    """
    Calculates data drift between a reference dataset (training) and current dataset (production).
    Uses Polars for efficient data processing.
    """

    # Categorical columns with more distinct reference values than this are
    # treated as free-text/ID-like (not a meaningful categorical distribution)
    # and skipped, to avoid PSI blowing up on effectively-unique values.
    _MAX_CATEGORICAL_CARDINALITY = 50

    def __init__(self, reference_df: pl.DataFrame, current_df: pl.DataFrame):
        self.reference_df = reference_df
        self.current_df = current_df
        self.common_columns = [col for col in reference_df.columns if col in current_df.columns]

    def calculate_drift(self, thresholds: dict[str, float] | None = None) -> DriftReport:
        """
        Calculates drift for all common columns.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for drift calculation")

        default_thresholds = {
            "psi": 0.2,
            "ks": 0.05,  # p-value < 0.05 means distributions are different
            "wasserstein": 0.1,  # Heuristic, depends on scale
            "kl_divergence": 0.1,
        }
        thresholds = {**default_thresholds, **(thresholds or {})}

        column_drifts = {}
        drifted_count = 0

        # Schema Drift Detection
        ref_cols = set(self.reference_df.columns)
        curr_cols = set(self.current_df.columns)
        missing_columns = list(ref_cols - curr_cols)
        new_columns = list(curr_cols - ref_cols)

        for col in self.common_columns:
            dtype = self.reference_df[col].dtype
            is_numeric = dtype.is_numeric()
            is_categorical = dtype in [pl.Utf8, pl.String, pl.Categorical, pl.Boolean]

            if is_categorical:
                col_drift = self._calculate_categorical_drift(col, thresholds)
                if col_drift is not None:
                    column_drifts[col] = col_drift
                    if col_drift.drift_detected:
                        drifted_count += 1
                continue

            if not is_numeric:
                # Unsupported dtype (e.g. nested/struct columns) — skip.
                continue

            # Ensure current data is also numeric or castable to the reference type
            curr_series = self.current_df[col]
            if curr_series.dtype != dtype:
                try:
                    # Try to cast current to match reference (e.g. Int to Float, or String to Float)
                    curr_series = curr_series.cast(dtype, strict=False)
                except Exception:
                    # If casting fails completely (unlikely with strict=False), skip
                    continue  # nosec B112

            ref_data = self.reference_df[col].drop_nulls().to_numpy()
            curr_data = curr_series.drop_nulls().to_numpy()

            if len(ref_data) == 0 or len(curr_data) == 0:
                continue

            metrics = []
            is_drifted = False

            # 1. Wasserstein Distance
            wd = wasserstein_distance(ref_data, curr_data)
            # Normalize WD? It's scale dependent.
            # Simple normalization: divide by std of reference
            std_ref = np.std(ref_data)
            norm_wd = wd / std_ref if std_ref > 0 else wd

            wd_drift = norm_wd > thresholds["wasserstein"]
            metrics.append(
                DriftMetric(
                    metric="wasserstein_distance",
                    value=float(wd),
                    has_drift=wd_drift,
                    threshold=thresholds["wasserstein"],
                )
            )
            if wd_drift:
                is_drifted = True

            # 2. KS Test
            ks_stat, ks_p = ks_2samp(ref_data, curr_data)
            ks_drift = ks_p < thresholds["ks"]
            metrics.append(
                DriftMetric(
                    metric="ks_test_p_value",
                    value=float(ks_p),
                    has_drift=ks_drift,
                    threshold=thresholds["ks"],
                )
            )
            if ks_drift:
                is_drifted = True

            # 3. PSI (Population Stability Index)
            psi_val = self._calculate_psi(ref_data, curr_data)
            psi_drift = psi_val > thresholds["psi"]
            metrics.append(
                DriftMetric(
                    metric="psi",
                    value=float(psi_val),
                    has_drift=psi_drift,
                    threshold=thresholds["psi"],
                )
            )
            if psi_drift:
                is_drifted = True

            # 4. KL Divergence
            kl_val = self._calculate_kl(ref_data, curr_data)
            kl_drift = kl_val > thresholds["kl_divergence"]
            metrics.append(
                DriftMetric(
                    metric="kl_divergence",
                    value=float(kl_val),
                    has_drift=kl_drift,
                    threshold=thresholds["kl_divergence"],
                )
            )
            if kl_drift:
                is_drifted = True

            # Generate Suggestions
            suggestions = []
            if is_drifted:
                if psi_val > 0.25:
                    suggestions.append(
                        "Critical population shift detected (PSI > 0.25). Immediate model retraining is recommended."
                    )
                elif psi_val > 0.1:
                    suggestions.append(
                        "Moderate population shift detected. Monitor model performance closely."
                    )

                if ks_drift and not psi_drift:
                    suggestions.append(
                        "Statistical distribution has changed, but population stability "
                        "is acceptable. Check for outliers."
                    )

                if wd_drift:
                    suggestions.append(
                        "Significant change in feature scale or shape detected. Verify data preprocessing steps."
                    )

            # Calculate Distribution (Histogram)
            distribution = self._calculate_distribution(ref_data, curr_data)

            column_drifts[col] = ColumnDrift(
                column=col,
                metrics=metrics,
                drift_detected=is_drifted,
                suggestions=suggestions,
                distribution=distribution,
            )

            if is_drifted:
                drifted_count += 1

        return DriftReport(
            reference_rows=len(self.reference_df),
            current_rows=len(self.current_df),
            drifted_columns_count=drifted_count,
            column_drifts=column_drifts,
            missing_columns=missing_columns,
            new_columns=new_columns,
        )

    def _calculate_distribution(
        self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 20
    ) -> DriftDistribution:
        """
        Calculates histogram bins for reference and current data using the same range.
        """
        try:
            # Determine global min/max
            min_val = min(np.min(ref_data), np.min(curr_data))
            max_val = max(np.max(ref_data), np.max(curr_data))

            # Handle constant case
            if min_val == max_val:
                min_val -= 0.5
                max_val += 0.5

            # Compute histogram for both using the same range
            ref_hist, bin_edges = np.histogram(ref_data, bins=bins, range=(min_val, max_val))
            curr_hist, _ = np.histogram(curr_data, bins=bins, range=(min_val, max_val))

            drift_bins = [
                DriftBin(
                    bin_start=float(bin_edges[i]),
                    bin_end=float(bin_edges[i + 1]),
                    reference_count=int(ref_hist[i]),
                    current_count=int(curr_hist[i]),
                )
                for i in range(len(ref_hist))
            ]

            return DriftDistribution(bins=drift_bins)
        except Exception:
            return DriftDistribution(bins=[])

    def _calculate_categorical_drift(
        self, col: str, thresholds: dict[str, float]
    ) -> "ColumnDrift | None":
        """
        Calculates PSI-based drift for a categorical/text/boolean column using
        the category frequency distribution (union of categories seen in
        either dataset). Returns ``None`` if the column looks like free-text
        or a high-cardinality identifier (not a meaningful categorical
        distribution), or if either side has no non-null values.
        """
        ref_data = self.reference_df[col].cast(pl.Utf8, strict=False).drop_nulls()
        curr_data = self.current_df[col].cast(pl.Utf8, strict=False).drop_nulls()

        if len(ref_data) == 0 or len(curr_data) == 0:
            return None

        ref_n_unique = ref_data.n_unique()
        if ref_n_unique > self._MAX_CATEGORICAL_CARDINALITY:
            return None

        ref_counts = ref_data.value_counts()
        curr_counts = curr_data.value_counts()

        categories = sorted(set(ref_counts[col].to_list()) | set(curr_counts[col].to_list()))
        if len(categories) < 2:
            return None

        ref_map = dict(zip(ref_counts[col].to_list(), ref_counts["count"].to_list(), strict=True))
        curr_map = dict(
            zip(curr_counts[col].to_list(), curr_counts["count"].to_list(), strict=True)
        )

        n_ref = len(ref_data)
        n_curr = len(curr_data)
        expected_percents = np.array([ref_map.get(c, 0) / n_ref for c in categories])
        actual_percents = np.array([curr_map.get(c, 0) / n_curr for c in categories])

        # Floor zero-proportion bins with a sample-size-scaled epsilon
        # (rather than a fixed constant) so the log-ratio doesn't blow up on
        # categories that are simply rare/absent in one dataset, while still
        # scaling sensibly for both small and large samples.
        eps_ref = 0.5 / n_ref
        eps_curr = 0.5 / n_curr
        expected_percents = np.where(expected_percents == 0, eps_ref, expected_percents)
        actual_percents = np.where(actual_percents == 0, eps_curr, actual_percents)

        psi_val = float(
            np.sum(
                (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
            )
        )
        psi_drift = psi_val > thresholds["psi"]

        metrics = [
            DriftMetric(
                metric="psi_categorical",
                value=psi_val,
                has_drift=psi_drift,
                threshold=thresholds["psi"],
            )
        ]

        suggestions: list[str] = []
        if psi_drift:
            if psi_val > 0.25:
                suggestions.append(
                    "Critical category-distribution shift detected (PSI > 0.25). "
                    "Immediate model retraining is recommended."
                )
            else:
                suggestions.append(
                    "Moderate category-distribution shift detected. Monitor model performance closely."
                )

        return ColumnDrift(
            column=col,
            metrics=metrics,
            drift_detected=psi_drift,
            suggestions=suggestions,
            distribution=None,
        )

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        """

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Use percentiles from expected (reference) to define bins
        try:
            # Handle constant arrays
            if np.min(expected) == np.max(expected):
                return 0.0

            breakpoints = np.percentile(expected, breakpoints)

            # Ensure unique breakpoints
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 2:
                return 0.0

            # Calculate frequencies
            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

            # Avoid division by zero
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi_value = np.sum(
                (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
            )
            return float(psi_value)

        except Exception:
            return 0.0

    def _calculate_kl(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """
        Calculates KL Divergence (Current || Reference).
        """
        try:
            if len(reference) == 0 or len(current) == 0:
                return 0.0
            if np.min(reference) == np.max(reference):
                return 0.0

            # Use reference percentiles for binning (same as PSI)
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            breakpoints = np.percentile(reference, breakpoints)
            breakpoints = np.unique(breakpoints)

            if len(breakpoints) < 2:
                return 0.0

            ref_percents = np.histogram(reference, breakpoints)[0] / len(reference)
            curr_percents = np.histogram(current, breakpoints)[0] / len(current)

            # Smooth to avoid infinity
            ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
            curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

            return float(entropy(curr_percents, ref_percents))
        except Exception:
            return 0.0
