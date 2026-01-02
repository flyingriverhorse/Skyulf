import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .schemas import (
    DatasetProfile, ColumnProfile, NumericStats, CategoricalStats, 
    DateStats, TextStats, Alert, HistogramBin, Recommendation, PCAPoint,
    GeospatialStats, GeoPoint, TimeSeriesAnalysis, TimeSeriesPoint, SeasonalityStats,
    TargetInteraction, CategoryBoxPlot, BoxPlotStats, OutlierAnalysis, OutlierPoint,
    NormalityTestResult
)
from .distributions import calculate_histogram
from .correlations import calculate_correlations

# Optional imports for PCA (sklearn is a dependency of skyulf-core)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import f_oneway, shapiro, kstest, norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class EDAAnalyzer:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Auto-detect and cast date columns
        self._cast_date_columns()
        
        # We accept eager DataFrame but convert to Lazy for processing
        self.lazy_df = self.df.lazy()
        self.row_count = self.df.height
        self.columns = self.df.columns

    def _cast_date_columns(self):
        """
        Attempts to cast string columns to DateTime if they look like dates.
        """
        for col in self.df.columns:
            if self.df[col].dtype in [pl.Utf8, pl.String]:
                # Check sample (non-null values)
                sample = self.df[col].drop_nulls().head(20)
                if len(sample) == 0: continue
                
                # 1. Try ISO Datetime (YYYY-MM-DD HH:MM:SS)
                try:
                    # strict=False returns null on failure. 
                    # If sample has no nulls after conversion, it's a match.
                    parsed = sample.str.to_datetime(strict=False)
                    if parsed.null_count() == 0:
                        # Apply to full column
                        self.df = self.df.with_columns(pl.col(col).str.to_datetime(strict=False).alias(col))
                        continue
                except Exception:
                    pass

                # 2. Try ISO Date (YYYY-MM-DD)
                try:
                    parsed = sample.str.to_date(strict=False)
                    if parsed.null_count() == 0:
                        self.df = self.df.with_columns(pl.col(col).str.to_date(strict=False).alias(col))
                        continue
                except Exception:
                    pass
                    
                # 3. Try common formats (MM/DD/YYYY, DD/MM/YYYY)
                # Polars requires specific format strings for these
                common_formats = ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"]
                for fmt in common_formats:
                    try:
                        parsed = sample.str.to_datetime(format=fmt, strict=False)
                        if parsed.null_count() == 0:
                            self.df = self.df.with_columns(pl.col(col).str.to_datetime(format=fmt, strict=False).alias(col))
                            break
                        
                        parsed_date = sample.str.to_date(format=fmt, strict=False)
                        if parsed_date.null_count() == 0:
                            self.df = self.df.with_columns(pl.col(col).str.to_date(format=fmt, strict=False).alias(col))
                            break
                    except Exception:
                        continue
        
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
        target_interactions = None
        
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
                    
                    # Calculate Interactions (Box Plots for Categorical Features vs Numeric Target)
                    # Find categorical columns
                    cat_cols = [c for c in self.columns if self._get_semantic_type(self.df[c]) == "Categorical" and c != target_col]
                    if cat_cols:
                        target_interactions = self._calculate_target_interactions(target_col, cat_cols, is_target_numeric=True)

            elif target_semantic_type == "Categorical":
                # For categorical target, calculate association with numeric features using ANOVA F-value or similar
                # Or just use correlation ratio (eta)
                # For now, let's use a simple heuristic: GroupBy Mean Variance?
                # Let's implement a simple _calculate_categorical_target_associations
                target_correlations = self._calculate_categorical_target_associations(target_col, numeric_cols)
                
                # Calculate Interactions (Box Plots for Numeric Features vs Categorical Target)
                # Use top associated numeric features (limit handled in _calculate_target_interactions)
                top_features = list(target_correlations.keys())
                if top_features:
                    target_interactions = self._calculate_target_interactions(target_col, top_features, is_target_numeric=False)
        
        # 4. Global Alerts (e.g. High Null % overall)
        if missing_pct > 50:
            alerts.append(Alert(
                type="High Null",
                message=f"Dataset is {missing_pct:.1f}% empty.",
                severity="warning"
            ))
            
        # 5. Sample Data (First 1000 rows for scatter plots)
        # We need to collect this from the lazy frame or original df
        # Convert to list of dicts
        sample_rows = self.df.head(5000).to_dicts()

        # 6. Multivariate Analysis (PCA)
        pca_data = None
        if SKLEARN_AVAILABLE and len(numeric_cols) >= 2:
            pca_data = self._calculate_pca(numeric_cols, target_col)

        # 6b. Outlier Detection
        outliers = None
        if SKLEARN_AVAILABLE and len(numeric_cols) >= 1:
            outliers = self._detect_outliers(numeric_cols)

        # 7. Geospatial Analysis
        geospatial = self._analyze_geospatial(numeric_cols, target_col)

        # 8. Time Series Analysis
        timeseries = self._analyze_timeseries(numeric_cols, target_col)

        # 9. Smart Recommendations
        recommendations = self._generate_recommendations(col_profiles, alerts, target_col)

        return DatasetProfile(
            row_count=self.row_count,
            column_count=len(self.columns),
            duplicate_rows=duplicate_rows,
            missing_cells_percentage=missing_pct,
            memory_usage_mb=memory_usage,
            columns=col_profiles,
            correlations=correlations,
            alerts=alerts,
            recommendations=recommendations,
            sample_data=sample_rows,
            target_col=target_col,
            target_correlations=target_correlations,
            target_interactions=target_interactions,
            pca_data=pca_data,
            outliers=outliers,
            geospatial=geospatial,
            timeseries=timeseries
        )

    def _detect_outliers(self, numeric_cols: List[str]) -> Optional[OutlierAnalysis]:
        """
        Detects outliers using Isolation Forest.
        """
        try:
            # Prepare data
            # Limit rows for performance if dataset is huge (e.g. > 50k)
            limit = 50000
            df_numeric = self.df.select(numeric_cols).head(limit)
            
            # Convert to pandas/numpy
            X = df_numeric.to_pandas().values
            
            # Impute
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # Fit Isolation Forest
            clf = IsolationForest(random_state=42, contamination=0.05, n_jobs=-1)
            clf.fit(X)
            
            # Predict: -1 for outliers, 1 for inliers
            preds = clf.predict(X)
            scores = clf.decision_function(X) # Lower is more anomalous
            
            outlier_indices = np.where(preds == -1)[0]
            total_outliers = len(outlier_indices)
            
            if total_outliers == 0:
                return None
                
            # Get top outliers (lowest scores)
            # Zip indices and scores
            scored_indices = list(zip(range(len(scores)), scores))
            # Sort by score ascending (lowest score = most anomalous)
            scored_indices.sort(key=lambda x: x[1])
            
            top_k = 20
            top_outliers = []
            
            for idx, score in scored_indices[:top_k]:
                # Only include if it was actually predicted as outlier
                if preds[idx] == -1:
                    # Get row values
                    row_values = df_numeric.row(idx, named=True)
                    top_outliers.append(OutlierPoint(
                        index=int(idx), # Cast to int for JSON serialization
                        values=row_values,
                        score=float(score)
                    ))
                
            return OutlierAnalysis(
                method="IsolationForest",
                total_outliers=total_outliers,
                outlier_percentage=(total_outliers / len(X)) * 100,
                top_outliers=top_outliers
            )
            
        except Exception as e:
            print(f"Error in outlier detection: {e}")
            return None

    def _analyze_geospatial(self, numeric_cols: List[str], target_col: Optional[str] = None) -> Optional[GeospatialStats]:
        """
        Detects and analyzes geospatial data (Lat/Lon).
        """
        try:
            lat_col = None
            lon_col = None
            
            # Check ALL columns, not just numeric ones, in case they were inferred as strings
            candidates = {c.lower(): c for c in self.columns}
            
            # Latitude detection
            if 'latitude' in candidates: lat_col = candidates['latitude']
            elif 'lat' in candidates: lat_col = candidates['lat']
            
            # Longitude detection
            if 'longitude' in candidates: lon_col = candidates['longitude']
            elif 'lng' in candidates: lon_col = candidates['lng']
            elif 'lon' in candidates: lon_col = candidates['lon']
            elif 'long' in candidates: lon_col = candidates['long']
            
            if not lat_col or not lon_col:
                return None
                
            # Calculate Stats
            # We must ensure they are numeric. If they were not in numeric_cols, they might be strings.
            # We try to cast them to Float64 for the analysis.
            
            geo_df = self.lazy_df.select([
                pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
                pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon")
            ])
            
            # Check if we have valid data after casting
            # We can't easily check "if valid" in lazy mode without collecting.
            # Let's collect stats.
            
            stats = geo_df.select([
                pl.col("lat").min().alias("min_lat"),
                pl.col("lat").max().alias("max_lat"),
                pl.col("lat").mean().alias("mean_lat"),
                pl.col("lon").min().alias("min_lon"),
                pl.col("lon").max().alias("max_lon"),
                pl.col("lon").mean().alias("mean_lon")
            ]).collect().row(0)
            
            # If stats are null (e.g. all cast failed), return None
            if stats[0] is None or stats[3] is None:
                return None
            
            # Sample Points (max 5000)
            sample_size = min(5000, self.row_count)
            
            # We need to fetch original columns + target, then cast in memory or select casted
            cols_to_fetch = [pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"), pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon")]
            
            if target_col and target_col in self.columns:
                cols_to_fetch.append(pl.col(target_col).alias("target"))
                
            sample_df = self.lazy_df.select(cols_to_fetch).collect().sample(n=sample_size, with_replacement=False, seed=42)
            
            points = []
            rows = sample_df.to_dicts()
            for row in rows:
                # Skip if lat/lon is null
                if row["lat"] is None or row["lon"] is None:
                    continue
                    
                label = str(row["target"]) if "target" in row and row["target"] is not None else None
                points.append(GeoPoint(
                    lat=row["lat"],
                    lon=row["lon"],
                    label=label
                ))
                
            return GeospatialStats(
                lat_col=lat_col,
                lon_col=lon_col,
                min_lat=stats[0],
                max_lat=stats[1],
                centroid_lat=stats[2],
                min_lon=stats[3],
                max_lon=stats[4],
                centroid_lon=stats[5],
                sample_points=points
            )
        except Exception as e:
            print(f"Error in geospatial analysis: {e}")
            return None

    def _analyze_timeseries(self, numeric_cols: List[str], target_col: Optional[str] = None) -> Optional[TimeSeriesAnalysis]:
        """
        Detects DateTime column and performs time series analysis.
        """
        try:
            # Find best date column (highest cardinality)
            date_cols = []
            for col in self.columns:
                if self._get_semantic_type(self.df[col]) == "DateTime":
                    date_cols.append(col)
            
            if not date_cols:
                return None
                
            # Pick the one with most unique values to avoid constant metadata dates
            best_date_col = None
            max_unique = -1
            
            for col in date_cols:
                n_unique = self.df[col].n_unique()
                if n_unique > max_unique:
                    max_unique = n_unique
                    best_date_col = col
            
            date_col = best_date_col
                
            # 1. Trend Analysis (Resample by Day)
            ts_df = self.lazy_df.sort(date_col)
            
            # Select top 3 numeric columns + target to track
            cols_to_track = numeric_cols[:3]
            if target_col and target_col in numeric_cols and target_col not in cols_to_track:
                cols_to_track.append(target_col)
                
            if not cols_to_track:
                # Just track count
                trend_df = ts_df.group_by(pl.col(date_col).dt.date().alias("date")).agg(
                    pl.count().alias("count")
                ).sort("date").collect()
            else:
                aggs = [pl.col(c).mean().alias(c) for c in cols_to_track]
                trend_df = ts_df.group_by(pl.col(date_col).dt.date().alias("date")).agg(
                    aggs
                ).sort("date").collect()
                
            # Convert to TimeSeriesPoint list
            trend_points = []
            for row in trend_df.iter_rows(named=True):
                if row["date"] is None: continue
                date_str = str(row["date"])
                values = {k: v for k, v in row.items() if k != "date" and v is not None}
                trend_points.append(TimeSeriesPoint(date=date_str, values=values))
                
            # 2. Seasonality (Day of Week)
            # If we have a numeric column to track, use mean, else count
            agg_expr = pl.count().alias("count")
            if cols_to_track:
                # Use the first numeric column for seasonality magnitude
                target_metric = cols_to_track[0]
                agg_expr = pl.col(target_metric).mean().alias("count") # Alias as count for frontend compatibility, but it's mean
            
            dow_df = self.lazy_df.with_columns(
                pl.col(date_col).dt.weekday().alias("dow_idx"),
                pl.col(date_col).dt.strftime("%a").alias("dow_name")
            ).group_by(["dow_idx", "dow_name"]).agg(
                agg_expr
            ).sort("dow_idx").collect()
            
            dow_stats = [{"day": row["dow_name"], "count": row["count"]} for row in dow_df.iter_rows(named=True)]
            
            # 3. Seasonality (Month of Year)
            moy_df = self.lazy_df.with_columns(
                pl.col(date_col).dt.month().alias("month_idx"),
                pl.col(date_col).dt.strftime("%b").alias("month_name")
            ).group_by(["month_idx", "month_name"]).agg(
                agg_expr
            ).sort("month_idx").collect()
            
            moy_stats = [{"month": row["month_name"], "count": row["count"]} for row in moy_df.iter_rows(named=True)]
            
            # 4. Autocorrelation (ACF)
            acf_stats = []
            if cols_to_track:
                target_metric = cols_to_track[0]
                # We need a contiguous series for ACF. trend_df is already sorted by date.
                # Extract the series
                series = trend_df[target_metric].to_numpy()
                
                # Handle NaNs if any (fill with mean or drop)
                # Simple fill
                mask = np.isnan(series)
                if mask.any():
                    series[mask] = np.nanmean(series)
                
                if len(series) > 10:
                    # Calculate ACF for lags 1 to 30
                    n = len(series)
                    mean = np.mean(series)
                    var = np.var(series)
                    
                    for lag in range(1, min(31, n // 2)):
                        # Slice arrays
                        y1 = series[lag:]
                        y2 = series[:-lag]
                        
                        # Pearson correlation
                        if var == 0:
                            corr = 0
                        else:
                            corr = np.sum((y1 - mean) * (y2 - mean)) / n / var
                            
                        acf_stats.append({"lag": lag, "corr": float(corr)})

            # 5. Stationarity Test (ADF)
            stationarity_test = None
            if STATSMODELS_AVAILABLE and cols_to_track:
                try:
                    target_metric = cols_to_track[0]
                    series = trend_df[target_metric].to_numpy()
                    # Handle NaNs
                    mask = np.isnan(series)
                    if mask.any():
                        series[mask] = np.nanmean(series)
                    
                    if len(series) > 20: # ADF requires sufficient data
                        result = adfuller(series)
                        stationarity_test = {
                            "test_statistic": float(result[0]),
                            "p_value": float(result[1]),
                            "is_stationary": float(result[1]) < 0.05,
                            "metric": target_metric
                        }
                except Exception as e:
                    print(f"ADF test failed: {e}")

            return TimeSeriesAnalysis(
                date_col=date_col,
                trend=trend_points,
                seasonality=SeasonalityStats(
                    day_of_week=dow_stats,
                    month_of_year=moy_stats
                ),
                autocorrelation=acf_stats,
                stationarity_test=stationarity_test
            )
            
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            return None

    def _calculate_pca(self, numeric_cols: List[str], target_col: Optional[str] = None) -> Optional[List[PCAPoint]]:
        """
        Calculates 2D PCA projection of the dataset.
        """
        try:
            # Increase sample size for better representation (max 5000 rows)
            # We use a sample for visualization performance, but 5000 is enough to see structure
            sample_size = min(5000, self.row_count)
            
            # Determine columns to fetch
            cols_to_fetch = list(numeric_cols)
            if target_col and target_col in self.columns and target_col not in cols_to_fetch:
                cols_to_fetch.append(target_col)
                
            # Sample once
            sample_df = self.df.select(cols_to_fetch).sample(n=sample_size, with_replacement=False, seed=42)
            
            # Extract Numeric Data for PCA
            X = sample_df.select(numeric_cols).to_numpy()
            
            # Handle missing values
            if np.isnan(X).any():
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            # Ensure numpy array (CRITICAL FIX)
            X_pca = np.asarray(X_pca)
            
            # Get labels
            labels = None
            if target_col and target_col in self.columns:
                labels = sample_df[target_col].to_list()
            
            # Create result
            points = []
            for i in range(len(X_pca)):
                label_val = str(labels[i]) if labels else None
                points.append(PCAPoint(
                    x=float(X_pca[i, 0]), 
                    y=float(X_pca[i, 1]), 
                    z=float(X_pca[i, 2]) if X_pca.shape[1] > 2 else None,
                    label=label_val
                ))
                
            return points
            
        except Exception as e:
            print(f"Error calculating PCA: {e}")
            return None

    def _generate_recommendations(self, profiles: Dict[str, ColumnProfile], alerts: List[Alert], target_col: Optional[str]) -> List[Recommendation]:
        recs = []
        
        # 1. High Missing
        for col, profile in profiles.items():
            if profile.missing_percentage > 50:
                recs.append(Recommendation(
                    column=col,
                    action="Drop",
                    reason=f"High missing values ({profile.missing_percentage:.1f}%)",
                    suggestion=f"Drop column '{col}' as it contains mostly nulls."
                ))
            elif profile.missing_percentage > 0:
                method = "Median" if profile.dtype == "Numeric" else "Mode"
                recs.append(Recommendation(
                    column=col,
                    action="Impute",
                    reason=f"Missing values ({profile.missing_percentage:.1f}%)",
                    suggestion=f"Impute '{col}' using {method}."
                ))
                
        # 2. High Skewness (Numeric)
        for col, profile in profiles.items():
            if profile.numeric_stats and profile.numeric_stats.skewness:
                if abs(profile.numeric_stats.skewness) > 1.5:
                    recs.append(Recommendation(
                        column=col,
                        action="Transform",
                        reason=f"High skewness ({profile.numeric_stats.skewness:.2f})",
                        suggestion=f"Apply Log or Box-Cox transformation to '{col}'."
                    ))
                    
        # 3. High Cardinality (Categorical)
        for col, profile in profiles.items():
            if profile.categorical_stats and profile.dtype == "Categorical":
                if profile.categorical_stats.unique_count > 50:
                     recs.append(Recommendation(
                        column=col,
                        action="Encode",
                        reason=f"High cardinality ({profile.categorical_stats.unique_count})",
                        suggestion=f"Use Target Encoding or Hashing for '{col}' instead of One-Hot."
                    ))
                    
        # 4. Constant Columns
        for col, profile in profiles.items():
            if profile.is_constant:
                recs.append(Recommendation(
                    column=col,
                    action="Drop",
                    reason="Constant value",
                    suggestion=f"Drop '{col}' as it has zero variance."
                ))
                
        # 5. ID Columns
        for col, profile in profiles.items():
            if profile.is_unique and profile.dtype in ["Categorical", "Text", "Numeric"]:
                 recs.append(Recommendation(
                    column=col,
                    action="Drop",
                    reason="Likely ID column",
                    suggestion=f"Drop '{col}' as it appears to be a unique identifier."
                ))

        # 6. Positive Reinforcement (if no critical issues)
        critical_issues = [r for r in recs if r.action in ["Drop", "Impute"]]
        if not critical_issues:
            recs.append(Recommendation(
                column=None,
                action="Keep",
                reason="Clean Dataset",
                suggestion="No missing values or constant columns found. Data is ready for modeling!"
            ))
            
        # 7. Target Balance (if target selected)
        if target_col and target_col in profiles:
            target_profile = profiles[target_col]
            if target_profile.dtype == "Categorical" and target_profile.categorical_stats:
                # Check balance
                counts = [item['count'] for item in target_profile.categorical_stats.top_k]
                if counts:
                    min_c = min(counts)
                    max_c = max(counts)
                    ratio = min_c / max_c if max_c > 0 else 0
                    if ratio > 0.8:
                        recs.append(Recommendation(
                            column=target_col,
                            action="Info",
                            reason="Balanced Target",
                            suggestion=f"Target classes are well balanced (Ratio: {ratio:.2f})."
                        ))
                    elif ratio < 0.2:
                        recs.append(Recommendation(
                            column=target_col,
                            action="Resample",
                            reason="Imbalanced Target",
                            suggestion=f"Target is imbalanced (Ratio: {ratio:.2f}). Consider SMOTE or Class Weights."
                        ))

        return recs

    def _calculate_target_interactions(self, target_col: str, features: List[str], is_target_numeric: bool) -> List[TargetInteraction]:
        """
        Calculates Box Plot statistics for interactions between Target and Features.
        """
        interactions = []
        try:
            # Limit to top 20 features to avoid clutter (increased from 5)
            features_to_process = features[:20]
            
            for feature in features_to_process:
                # Determine which is categorical (grouping) and which is numeric (values)
                if is_target_numeric:
                    group_col = feature
                    value_col = target_col
                else:
                    group_col = target_col
                    value_col = feature
                
                # Check cardinality of group_col
                n_unique = self.df[group_col].n_unique()
                if n_unique > 20:
                    # Skip high cardinality grouping for box plots
                    continue
                    
                # Calculate Box Plot Stats per Group
                # Group by group_col -> Calculate quantiles of value_col
                
                # Polars doesn't support multiple quantiles in one agg easily in older versions, 
                # but we can do multiple aggs.
                
                stats_df = self.lazy_df.group_by(group_col).agg([
                    pl.col(value_col).min().alias("min"),
                    pl.col(value_col).quantile(0.25).alias("q1"),
                    pl.col(value_col).median().alias("median"),
                    pl.col(value_col).quantile(0.75).alias("q3"),
                    pl.col(value_col).max().alias("max")
                ]).collect()
                
                category_plots = []
                for row in stats_df.iter_rows(named=True):
                    if row[group_col] is None: continue
                    
                    # Ensure values are not None (e.g. empty group)
                    if row["min"] is None: continue
                    
                    category_plots.append(CategoryBoxPlot(
                        name=str(row[group_col]),
                        stats=BoxPlotStats(
                            min=float(row["min"]),
                            q1=float(row["q1"]),
                            median=float(row["median"]),
                            q3=float(row["q3"]),
                            max=float(row["max"])
                        )
                    ))
                
                # Calculate ANOVA p-value if possible
                p_value = None
                if SCIPY_AVAILABLE and len(category_plots) > 1:
                    try:
                        # Fetch data for ANOVA
                        # We need lists of values for each group
                        anova_data = self.lazy_df.select([
                            pl.col(group_col), 
                            pl.col(value_col)
                        ]).group_by(group_col).agg(
                            pl.col(value_col)
                        ).collect()
                        
                        groups_data = []
                        for row in anova_data.iter_rows(named=True):
                            if row[group_col] is not None and row[value_col] is not None:
                                # Filter out nulls
                                vals = [v for v in row[value_col] if v is not None]
                                if len(vals) > 1:
                                    groups_data.append(vals)
                                    
                        if len(groups_data) > 1:
                            f_stat, p_val = f_oneway(*groups_data)
                            if not np.isnan(p_val):
                                p_value = float(p_val)
                    except Exception as e:
                        print(f"ANOVA failed for {feature}: {e}")
                
                if category_plots:
                    interactions.append(TargetInteraction(
                        feature=feature,
                        plot_type="boxplot",
                        data=category_plots,
                        p_value=p_value
                    ))
                    
            return interactions
            
        except Exception as e:
            print(f"Error calculating target interactions: {e}")
            return []

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
            
            # Normality Test (Shapiro-Wilk / KS)
            if SCIPY_AVAILABLE and profile.numeric_stats and profile.numeric_stats.std and profile.numeric_stats.std > 0:
                try:
                    # Sample data (Shapiro is slow on large data, limit to 5000)
                    sample_data = self.df[col].drop_nulls().head(5000).to_numpy()
                    if len(sample_data) > 20:
                        # Use Shapiro-Wilk for N < 5000, else KS test
                        if len(sample_data) < 5000:
                            stat, p_value = shapiro(sample_data)
                            test_name = "Shapiro-Wilk"
                        else:
                            # KS Test against normal distribution
                            stat, p_value = kstest(sample_data, 'norm')
                            test_name = "Kolmogorov-Smirnov"
                            
                        profile.normality_test = NormalityTestResult(
                            test_name=test_name,
                            statistic=float(stat),
                            p_value=float(p_value),
                            is_normal=float(p_value) > 0.05
                        )
                except Exception as e:
                    print(f"Normality test failed for {col}: {e}")

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
            
            # Calculate Histogram for DateTime (Distribution over time)
            try:
                # Convert to ms timestamp for histogram
                ts = self.df[col].dt.timestamp("ms").drop_nulls().to_numpy()
                if len(ts) > 0:
                    hist, bin_edges = np.histogram(ts, bins=10)
                    profile.histogram = []
                    for i in range(len(hist)):
                        profile.histogram.append(HistogramBin(
                            start=float(bin_edges[i]),
                            end=float(bin_edges[i+1]),
                            count=int(hist[i])
                        ))
            except Exception as e:
                print(f"Failed to calculate date histogram for {col}: {e}")
            
        elif semantic_type == "Text":
            profile.text_stats = self._analyze_text(col)
            
            # Calculate Length Histogram for Text
            try:
                lengths = self.df[col].str.len_bytes().drop_nulls().to_numpy()
                if len(lengths) > 0:
                    hist, bin_edges = np.histogram(lengths, bins=10)
                    profile.histogram = []
                    for i in range(len(hist)):
                        profile.histogram.append(HistogramBin(
                            start=float(bin_edges[i]),
                            end=float(bin_edges[i+1]),
                            count=int(hist[i])
                        ))
            except Exception as e:
                print(f"Failed to calculate text histogram for {col}: {e}")

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
            
        # Rare labels (count < 5)
        # We can use the top_k logic to infer if we have a long tail
        # Or count rows where col is not in top_k? No, that's not right.
        # Correct way: group by col, count, filter count < 5, count rows
        try:
            rare_count = self.lazy_df.group_by(col).agg(pl.count().alias("cnt")).filter(pl.col("cnt") < 5).select(pl.count()).collect().item()
        except Exception:
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
        
        # Most common words (simple tokenization by space)
        common_words = []
        try:
            # Split by space, explode, count
            # Limit to first 1000 rows for performance if dataset is huge
            sample_text = self.df.select(col).head(1000)
            words = sample_text.select(
                pl.col(col).str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").explode().alias("word")
            ).filter(pl.col("word") != "")
            
            word_counts = words.group_by("word").agg(pl.count().alias("count")).sort("count", descending=True).head(10)
            
            for row in word_counts.iter_rows(named=True):
                common_words.append({"word": row["word"], "count": row["count"]})
        except Exception as e:
            print(f"Error calculating common words for {col}: {e}")

        return TextStats(
            avg_length=stats["avg_len"][0],
            min_length=stats["min_len"][0],
            max_length=stats["max_len"][0],
            common_words=common_words
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
