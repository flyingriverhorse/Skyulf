import polars as pl
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel

try:
    from scipy.stats import ks_2samp, wasserstein_distance, entropy
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
    bins: List[DriftBin]

class ColumnDrift(BaseModel):
    column: str
    metrics: List[DriftMetric]
    drift_detected: bool
    suggestions: List[str] = []
    distribution: Optional[DriftDistribution] = None

class DriftReport(BaseModel):
    reference_rows: int
    current_rows: int
    drifted_columns_count: int
    column_drifts: Dict[str, ColumnDrift]
    missing_columns: List[str] = []
    new_columns: List[str] = []

class DriftCalculator:
    """
    Calculates data drift between a reference dataset (training) and current dataset (production).
    Uses Polars for efficient data processing.
    """
    
    def __init__(self, reference_df: pl.DataFrame, current_df: pl.DataFrame):
        self.reference_df = reference_df
        self.current_df = current_df
        self.common_columns = [
            col for col in reference_df.columns 
            if col in current_df.columns
        ]

    def calculate_drift(self, thresholds: Optional[Dict[str, float]] = None) -> DriftReport:
        """
        Calculates drift for all common columns.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for drift calculation")

        default_thresholds = {
            "psi": 0.2,
            "ks": 0.05, # p-value < 0.05 means distributions are different
            "wasserstein": 0.1, # Heuristic, depends on scale
            "kl_divergence": 0.1
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
            # Skip non-numeric for now (except PSI which can handle categorical with binning)
            # For simplicity, let's focus on numeric columns first
            dtype = self.reference_df[col].dtype
            if dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                continue
                
            ref_data = self.reference_df[col].drop_nulls().to_numpy()
            curr_data = self.current_df[col].drop_nulls().to_numpy()
            
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
            metrics.append(DriftMetric(
                metric="wasserstein_distance",
                value=float(wd),
                has_drift=wd_drift,
                threshold=thresholds["wasserstein"]
            ))
            if wd_drift: is_drifted = True
            
            # 2. KS Test
            ks_stat, ks_p = ks_2samp(ref_data, curr_data)
            ks_drift = ks_p < thresholds["ks"]
            metrics.append(DriftMetric(
                metric="ks_test_p_value",
                value=float(ks_p),
                has_drift=ks_drift,
                threshold=thresholds["ks"]
            ))
            if ks_drift: is_drifted = True
            
            # 3. PSI (Population Stability Index)
            psi_val = self._calculate_psi(ref_data, curr_data)
            psi_drift = psi_val > thresholds["psi"]
            metrics.append(DriftMetric(
                metric="psi",
                value=float(psi_val),
                has_drift=psi_drift,
                threshold=thresholds["psi"]
            ))
            if psi_drift: is_drifted = True

            # 4. KL Divergence
            kl_val = self._calculate_kl(ref_data, curr_data)
            kl_drift = kl_val > thresholds["kl_divergence"]
            metrics.append(DriftMetric(
                metric="kl_divergence",
                value=float(kl_val),
                has_drift=kl_drift,
                threshold=thresholds["kl_divergence"]
            ))
            if kl_drift: is_drifted = True
            
            # Generate Suggestions
            suggestions = []
            if is_drifted:
                if psi_val > 0.25:
                    suggestions.append("Critical population shift detected (PSI > 0.25). Immediate model retraining is recommended.")
                elif psi_val > 0.1:
                    suggestions.append("Moderate population shift detected. Monitor model performance closely.")
                
                if ks_drift and not psi_drift:
                    suggestions.append("Statistical distribution has changed, but population stability is acceptable. Check for outliers.")
                
                if wd_drift:
                    suggestions.append("Significant change in feature scale or shape detected. Verify data preprocessing steps.")

            # Calculate Distribution (Histogram)
            distribution = self._calculate_distribution(ref_data, curr_data)

            column_drifts[col] = ColumnDrift(
                column=col,
                metrics=metrics,
                drift_detected=is_drifted,
                suggestions=suggestions,
                distribution=distribution
            )
            
            if is_drifted:
                drifted_count += 1
                
        return DriftReport(
            reference_rows=len(self.reference_df),
            current_rows=len(self.current_df),
            drifted_columns_count=drifted_count,
            column_drifts=column_drifts,
            missing_columns=missing_columns,
            new_columns=new_columns
        )

    def _calculate_distribution(self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 20) -> DriftDistribution:
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
            
            drift_bins = []
            for i in range(len(ref_hist)):
                drift_bins.append(DriftBin(
                    bin_start=float(bin_edges[i]),
                    bin_end=float(bin_edges[i+1]),
                    reference_count=int(ref_hist[i]),
                    current_count=int(curr_hist[i])
                ))
                
            return DriftDistribution(bins=drift_bins)
        except Exception:
            return DriftDistribution(bins=[])

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

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
            
            psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
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
