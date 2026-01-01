import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .schemas import (
    DatasetProfile, ColumnProfile, NumericStats, CategoricalStats, 
    DateStats, TextStats, Alert, HistogramBin
)
from .distributions import calculate_histogram
from .correlations import calculate_correlations

class EDAAnalyzer:
    def __init__(self, df: pl.DataFrame):
        # We accept eager DataFrame but convert to Lazy for processing
        self.df = df
        self.lazy_df = df.lazy()
        self.row_count = df.height
        self.columns = df.columns
        
    def analyze(self, target_col: Optional[str] = None) -> DatasetProfile:
        """
        Main entry point to generate the full profile.
        """
        # 1. Basic Info
        # null_count() returns a 1-row DataFrame. We want the sum of that row.
        missing_cells = self.df.null_count().sum_horizontal()[0]
        total_cells = self.row_count * len(self.columns)
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
        
        duplicate_rows = self.df.is_duplicated().sum()
        memory_usage = self.df.estimated_size("mb")
        
        # 2. Column Analysis
        col_profiles = {}
        alerts = []
        
        numeric_cols = []
        
        for col in self.columns:
            profile, col_alerts = self._analyze_column(col)
            col_profiles[col] = profile
            alerts.extend(col_alerts)
            
            if profile.dtype == "Numeric":
                numeric_cols.append(col)
                
        # 3. Correlations
        correlations = calculate_correlations(self.lazy_df, numeric_cols)
        
        # 3b. Target Analysis
        target_correlations = {}
        if target_col and target_col in self.columns:
            target_semantic_type = self._get_semantic_type(self.df[target_col])
            
            if target_semantic_type == "Numeric":
                # Only support numeric target for now for correlation
                if target_col in numeric_cols:
                    target_correlations = self._calculate_target_correlations(target_col, numeric_cols)
                    
                    # Check for leakage
                    for col, corr in target_correlations.items():
                        if abs(corr) > 0.95 and col != target_col:
                            alerts.append(Alert(
                                column=col,
                                type="Leakage",
                                message=f"Column '{col}' is highly correlated ({corr:.2f}) with target '{target_col}'. Possible leakage.",
                                severity="warning"
                            ))
            elif target_semantic_type == "Categorical":
                # For categorical target, calculate association with numeric features using ANOVA F-value or similar
                # Or just use correlation ratio (eta)
                # For now, let's use a simple heuristic: GroupBy Mean Variance?
                # Let's implement a simple _calculate_categorical_target_associations
                target_correlations = self._calculate_categorical_target_associations(target_col, numeric_cols)
        
        # 4. Global Alerts (e.g. High Null % overall)
        if missing_pct > 50:
            alerts.append(Alert(
                type="High Null",
                message=f"Dataset is {missing_pct:.1f}% empty.",
                severity="warning"
            ))
            
        # 5. Sample Data (First 100 rows)
        # We need to collect this from the lazy frame or original df
        # Convert to list of dicts
        sample_rows = self.df.head(100).to_dicts()

        return DatasetProfile(
            row_count=self.row_count,
            column_count=len(self.columns),
            duplicate_rows=duplicate_rows,
            missing_cells_percentage=missing_pct,
            memory_usage_mb=memory_usage,
            columns=col_profiles,
            correlations=correlations,
            alerts=alerts,
            sample_data=sample_rows,
            target_col=target_col,
            target_correlations=target_correlations
        )

    def _calculate_categorical_target_associations(self, target_col: str, numeric_cols: List[str]) -> Dict[str, float]:
        """
        Calculate association between categorical target and numeric features.
        Uses Correlation Ratio (eta squared) or similar.
        """
        try:
            associations = {}
            # Filter out target itself
            features = [c for c in numeric_cols if c != target_col]
            
            # We need to collect to use numpy/scikit-learn or do it in Polars
            # Doing it in Polars:
            # Eta^2 = SS_between / SS_total
            # SS_total = sum((x - mean)^2)
            # SS_between = sum(n_group * (mean_group - mean)^2)
            
            for col in features:
                # Calculate global mean
                global_mean = self.df[col].mean()
                
                # Calculate SS_total
                # ss_total = ((self.df[col] - global_mean) ** 2).sum()
                # In Polars:
                ss_total = self.df.select(((pl.col(col) - global_mean) ** 2).sum()).item()
                
                if ss_total == 0:
                    associations[col] = 0.0
                    continue
                
                # Calculate SS_between
                # Group by target, get count and mean
                groups = self.df.group_by(target_col).agg([
                    pl.count().alias("n"),
                    pl.col(col).mean().alias("mean")
                ])
                
                ss_between = 0.0
                for row in groups.iter_rows(named=True):
                    n = row["n"]
                    mean_group = row["mean"]
                    if mean_group is not None:
                        ss_between += n * ((mean_group - global_mean) ** 2)
                    
                eta_squared = ss_between / ss_total
                associations[col] = float(np.sqrt(eta_squared)) # Return eta (0-1)
                
            return dict(sorted(associations.items(), key=lambda item: item[1], reverse=True))
            
        except Exception as e:
            print(f"Error calculating categorical target associations: {e}")
            return {}

    def _calculate_target_correlations(self, target_col: str, numeric_cols: List[str]) -> Dict[str, float]:
        try:
            # Filter out target itself from features to check
            features = [c for c in numeric_cols if c != target_col]
            if not features:
                return {}
                
            # Use Polars to calculate correlation
            # df.select([pl.corr(col, target_col) for col in features])
            
            exprs = [pl.corr(col, target_col).alias(col) for col in features]
            result = self.lazy_df.select(exprs).collect()
            
            corrs = {}
            for col in features:
                val = result[col][0]
                if val is not None and not np.isnan(val):
                    corrs[col] = float(val)
            
            # Sort by absolute correlation
            return dict(sorted(corrs.items(), key=lambda item: abs(item[1]), reverse=True))
            
        except Exception as e:
            print(f"Error calculating target correlations: {e}")
            return {}
        
    def _analyze_column(self, col: str) -> (ColumnProfile, List[Alert]):
        dtype = str(self.df[col].dtype)
        alerts = []
        
        # Determine semantic type
        semantic_type = self._get_semantic_type(self.df[col])
        
        # Basic stats
        null_count = self.df[col].null_count()
        null_pct = (null_count / self.row_count) * 100
        
        if null_pct > 5:
            alerts.append(Alert(
                column=col,
                type="High Null",
                message=f"Column '{col}' has {null_pct:.1f}% missing values.",
                severity="warning"
            ))
            
        # Initialize profile
        profile = ColumnProfile(
            name=col,
            dtype=semantic_type,
            missing_count=null_count,
            missing_percentage=null_pct
        )
        
        # Type-specific analysis
        if semantic_type == "Numeric":
            profile.numeric_stats = self._analyze_numeric(col)
            profile.histogram = calculate_histogram(self.lazy_df, col)
            
            # Outlier detection (IQR)
            if profile.numeric_stats and profile.numeric_stats.q25 is not None and profile.numeric_stats.q75 is not None:
                iqr = profile.numeric_stats.q75 - profile.numeric_stats.q25
                if iqr > 0:
                    # Simple check: are min/max far outside?
                    lower_bound = profile.numeric_stats.q25 - 1.5 * iqr
                    upper_bound = profile.numeric_stats.q75 + 1.5 * iqr
                    if profile.numeric_stats.min < lower_bound or profile.numeric_stats.max > upper_bound:
                        alerts.append(Alert(
                            column=col,
                            type="Outlier",
                            message=f"Column '{col}' contains significant outliers.",
                            severity="info"
                        ))
            
            # Constant check
            if profile.numeric_stats.std == 0:
                profile.is_constant = True
                alerts.append(Alert(column=col, type="Constant", message=f"Column '{col}' is constant.", severity="warning"))

        elif semantic_type == "Categorical" or semantic_type == "Boolean":
            profile.categorical_stats = self._analyze_categorical(col)
            
            # High Cardinality check
            if profile.categorical_stats.unique_count > 50 and semantic_type == "Categorical":
                 # Heuristic: if unique count is high relative to rows, maybe ID?
                 if profile.categorical_stats.unique_count == self.row_count:
                     profile.is_unique = True
                     alerts.append(Alert(column=col, type="Possible ID", message=f"Column '{col}' appears to be an ID.", severity="info"))
                 elif profile.categorical_stats.unique_count > 1000:
                     alerts.append(Alert(column=col, type="High Cardinality", message=f"Column '{col}' has high cardinality ({profile.categorical_stats.unique_count}).", severity="warning"))
            
            # PII Check for Categorical (if it was inferred as Categorical but contains PII)
            # Only check if underlying type is string-like (Categorical in Polars is string-like)
            if semantic_type == "Categorical" and self._check_pii(col):
                 alerts.append(Alert(column=col, type="PII", message=f"Column '{col}' may contain PII (Email/Phone).", severity="error"))

        elif semantic_type == "DateTime":
            profile.date_stats = self._analyze_date(col)
            
        elif semantic_type == "Text":
            profile.text_stats = self._analyze_text(col)
            # PII Check
            if self._check_pii(col):
                alerts.append(Alert(column=col, type="PII", message=f"Column '{col}' may contain PII (Email/Phone).", severity="error"))
                
        return profile, alerts

    def _get_semantic_type(self, series: pl.Series) -> str:
        dtype = series.dtype
        
        if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            return "Numeric"
        elif dtype == pl.Boolean:
            return "Boolean"
        elif dtype in [pl.Date, pl.Datetime, pl.Duration]:
            return "DateTime"
        elif dtype == pl.Utf8 or dtype == pl.String:
            # Check if it looks like a date?
            # For now, assume Text or Categorical based on cardinality?
            # Let's stick to "Text" for raw strings, "Categorical" if we cast it or if low cardinality?
            # The plan says "Categorical" and "Text".
            
            n_unique = series.n_unique()
            count = len(series)
            ratio = n_unique / count if count > 0 else 0
            
            # If low cardinality ratio (e.g. < 5%) -> Categorical
            if ratio < 0.05:
                 return "Categorical"
            
            # If absolute low cardinality (e.g. < 20) AND dataset is not tiny (e.g. > 50 rows)
            # For tiny datasets, we can't be sure, default to Text to be safe, or Categorical?
            # Let's default to Text for high ratio.
            
            return "Text"
        elif dtype == pl.Categorical:
            return "Categorical"
        
        return "Text" # Fallback

    def _analyze_numeric(self, col: str) -> NumericStats:
        # Use lazy stats
        stats = self.lazy_df.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).median().alias("median"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).quantile(0.25).alias("q25"),
            pl.col(col).quantile(0.75).alias("q75"),
            pl.col(col).skew().alias("skew"),
            pl.col(col).kurtosis().alias("kurt"),
            (pl.col(col) == 0).sum().alias("zeros"),
            (pl.col(col) < 0).sum().alias("negatives")
        ]).collect()
        
        row = stats.row(0)
        return NumericStats(
            mean=row[0], median=row[1], std=row[2], min=row[3], max=row[4],
            q25=row[5], q75=row[6], skewness=row[7], kurtosis=row[8],
            zeros_count=row[9], negatives_count=row[10]
        )

    def _analyze_categorical(self, col: str) -> CategoricalStats:
        unique_count = self.df[col].n_unique()
        
        # Top K
        top_k_df = self.df[col].value_counts(sort=True).head(10)
        top_k = []
        for row in top_k_df.iter_rows():
            top_k.append({"value": str(row[0]), "count": row[1]})
            
        # Rare labels (count < 5 or < 1%?)
        # Let's say appearing less than 5 times
        # This is expensive to compute exactly if high cardinality.
        # Approximation: unique_count - count of values with freq > 5?
        # Let's skip complex rare label count for now to save time, or use simple heuristic.
        rare_count = 0 
        
        return CategoricalStats(
            unique_count=unique_count,
            top_k=top_k,
            rare_labels_count=rare_count
        )

    def _analyze_date(self, col: str) -> DateStats:
        stats = self.lazy_df.select([
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max")
        ]).collect()
        
        min_date = stats["min"][0]
        max_date = stats["max"][0]
        
        duration = None
        if min_date and max_date:
            delta = max_date - min_date
            duration = delta.days if hasattr(delta, 'days') else None
            
        return DateStats(
            min_date=str(min_date),
            max_date=str(max_date),
            duration_days=duration
        )

    def _analyze_text(self, col: str) -> TextStats:
        # Length stats
        stats = self.lazy_df.select([
            pl.col(col).str.len_bytes().mean().alias("avg_len"),
            pl.col(col).str.len_bytes().min().alias("min_len"),
            pl.col(col).str.len_bytes().max().alias("max_len")
        ]).collect()
        
        return TextStats(
            avg_length=stats["avg_len"][0],
            min_length=stats["min_len"][0],
            max_length=stats["max_len"][0]
        )

    def _check_pii(self, col: str) -> bool:
        # Simple heuristic check on a sample
        sample = self.df[col].drop_nulls().head(20).to_list()
        email_pattern = r"[^@]+@[^@]+\.[^@]+"
        phone_pattern = r"^\+?1?\d{9,15}$" # Very basic
        
        for val in sample:
            val_str = str(val)
            if re.match(email_pattern, val_str):
                return True
            # Phone check is tricky, maybe skip for now or refine
            
        return False
