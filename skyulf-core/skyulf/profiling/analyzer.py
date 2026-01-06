import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from .schemas import (
    DatasetProfile, ColumnProfile, NumericStats, CategoricalStats, 
    DateStats, TextStats, Alert, HistogramBin, Recommendation, PCAPoint, PCAComponent,
    GeospatialStats, GeoPoint, TimeSeriesAnalysis, TimeSeriesPoint, SeasonalityStats,
    TargetInteraction, CategoryBoxPlot, BoxPlotStats, OutlierAnalysis, OutlierPoint,
    NormalityTestResult, Filter, CausalGraph, CausalNode, CausalEdge,
    RuleTree, RuleNode, ClusteringAnalysis, ClusterStats, ClusteringPoint
)
from .distributions import calculate_histogram
from .correlations import calculate_correlations

# Optional imports for PCA (sklearn is a dependency of skyulf-core)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
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
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from causallearn.search.ConstraintBased.PC import pc
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False

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
                # Heuristic: Check column name first to avoid expensive parsing on non-date columns
                col_lower = col.lower()
                date_keywords = ["date", "time", "year", "month", "day", "ts", "created", "updated", "at"]
                if not any(k in col_lower for k in date_keywords):
                    continue

                # Check sample (non-null values)
                sample = self.df[col].drop_nulls().head(50)
                if len(sample) == 0: continue
                
                best_parsed = None
                max_months = -1
                best_method_name = ""

                # Try Generic ISO Datetime
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

                # Try Generic ISO Date
                try:
                    parsed = sample.str.to_date(strict=False)
                    if parsed.null_count() == 0:
                        n_months = parsed.dt.month().n_unique()
                        # Prefer Datetime over Date if months are equal
                        if n_months > max_months:
                            max_months = n_months
                            best_parsed = (None, "date_generic")
                            best_method_name = "Generic Date"
                except Exception:
                    pass
                    
                # Try Common Formats
                common_formats = ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"]
                
                for fmt in common_formats:
                    try:
                        parsed = sample.str.to_datetime(format=fmt, strict=False)
                        if parsed.null_count() == 0:
                            n_months = parsed.dt.month().n_unique()
                            # Maximize unique months to disambiguate formats (e.g. D/M/Y vs M/D/Y)
                            if n_months > max_months:
                                max_months = n_months
                                best_parsed = (fmt, "datetime_format")
                                best_method_name = f"Format {fmt}"
                    except Exception:
                        continue
                
                # Apply the winner
                if best_parsed:
                    fmt, method = best_parsed
                    try:
                        if method == "datetime_generic":
                            self.df = self.df.with_columns(pl.col(col).str.to_datetime(strict=False).alias(col))
                        elif method == "date_generic":
                            self.df = self.df.with_columns(pl.col(col).str.to_date(strict=False).alias(col))
                        elif method == "datetime_format":
                            self.df = self.df.with_columns(pl.col(col).str.to_datetime(format=fmt, strict=False).alias(col))
                    except Exception as e:
                        print(f"Failed to cast column {col} using {best_method_name}: {e}")
                    continue
        
    def analyze(self, target_col: Optional[str] = None, exclude_cols: Optional[List[str]] = None, filters: Optional[List[Dict[str, Any]]] = None, date_col: Optional[str] = None, lat_col: Optional[str] = None, lon_col: Optional[str] = None, task_type: Optional[str] = None) -> DatasetProfile:
        """
        Main entry point to generate the full profile.
        """
        # Apply Filters
        active_filters = []
        if filters:
            for f in filters:
                col = f.get("column")
                op = f.get("operator")
                val = f.get("value")
                
                if col in self.columns:
                    if op == "==":
                        self.df = self.df.filter(pl.col(col) == val)
                    elif op == "!=":
                        self.df = self.df.filter(pl.col(col) != val)
                    elif op == ">":
                        self.df = self.df.filter(pl.col(col) > val)
                    elif op == "<":
                        self.df = self.df.filter(pl.col(col) < val)
                    elif op == ">=":
                        self.df = self.df.filter(pl.col(col) >= val)
                    elif op == "<=":
                        self.df = self.df.filter(pl.col(col) <= val)
                    elif op == "in" and isinstance(val, list):
                        self.df = self.df.filter(pl.col(col).is_in(val))
                        
                    active_filters.append(Filter(column=col, operator=op, value=val))
            
            # Re-initialize lazy df and stats after filtering
            self.lazy_df = self.df.lazy()
            self.row_count = self.df.height

        # Check for empty dataframe after filtering
        if self.row_count == 0:
            return DatasetProfile(
                row_count=0,
                column_count=len(self.columns),
                duplicate_rows=0,
                missing_cells_percentage=0.0,
                memory_usage_mb=0.0,
                columns={},
                correlations=None,
                alerts=[Alert(type="Empty Data", message="Filters resulted in 0 rows. Please adjust your filters.", severity="warning")],
                recommendations=[],
                sample_data=[],
                active_filters=active_filters,
                target_col=target_col
            )

        # Apply Exclusions
        excluded_columns = []
        if exclude_cols:
            excluded_columns = [c for c in exclude_cols if c in self.columns]
            self.columns = [c for c in self.columns if c not in excluded_columns]

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
            
            # Add to numeric_cols if it's Numeric OR if it's Categorical but underlying type is numeric
            # This ensures it's included in Causal Discovery, PCA, etc.
            is_numeric_type = self.df[col].dtype in [
                pl.Float32, pl.Float64, 
                pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
            ]
            
            if profile.dtype == "Numeric":
                numeric_cols.append(col)
            elif profile.dtype == "Categorical" and is_numeric_type:
                numeric_cols.append(col)

        # Auto-encode String Target for Analysis (Causal only)
        encoded_target_col = None
        if target_col and target_col in self.columns and target_col not in numeric_cols:
             # If target is categorical/string, encode it to numeric so it appears in graphs
             encoded_target = f"{target_col}_encoded"
             self.df = self.df.with_columns(
                 pl.col(target_col).cast(pl.Categorical).to_physical().alias(encoded_target)
             )
             self.lazy_df = self.df.lazy()
             encoded_target_col = encoded_target
        
        # Define feature columns (numeric columns excluding target)
        # This ensures PCA and Correlation Matrix only show features, not the target itself
        feature_cols = [c for c in numeric_cols if c != target_col]

        # 3. Correlations
        correlations = calculate_correlations(self.lazy_df, feature_cols)
        
        # 3.1 Multicollinearity (VIF)
        vif_data = self._calculate_vif(feature_cols)
        if vif_data:
            # Add alerts for high VIF
            for col, val in vif_data.items():
                if val > 10.0:
                    alerts.append(Alert(type="Multicollinearity", message=f"Column '{col}' has very high VIF ({val:.1f}). Consider removing it.", severity="warning"))
                elif val > 5.0:
                    alerts.append(Alert(type="Multicollinearity", message=f"Column '{col}' has high VIF ({val:.1f}).", severity="info"))

        # 3a. Correlations with Target (Feature Selection)
        correlations_with_target = None
        target_corr_cols = []
        
        if target_col:
            if target_col in numeric_cols:
                target_corr_cols = feature_cols + [target_col]
            elif encoded_target_col:
                target_corr_cols = feature_cols + [encoded_target_col]
                
        if len(target_corr_cols) >= 2:
            correlations_with_target = calculate_correlations(self.lazy_df, target_corr_cols)
        
        # 3b. Target Analysis
        target_correlations = {}
        target_interactions = None
        
        if target_col and target_col in self.columns:
            target_semantic_type = self._get_semantic_type(self.df[target_col])
            
            if target_semantic_type == "Numeric":
                # Only support numeric target for now for correlation
                if target_col in numeric_cols:
                    target_correlations = self._calculate_target_correlations(target_col, feature_cols)
                    
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
                target_correlations = self._calculate_categorical_target_associations(target_col, feature_cols)
                
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
        pca_components = None
        if SKLEARN_AVAILABLE and len(feature_cols) >= 2:
            pca_res = self._calculate_pca(feature_cols, target_col)
            if pca_res:
                pca_data, pca_components = pca_res

        # 6b. Outlier Detection
        outliers = None
        if SKLEARN_AVAILABLE and len(numeric_cols) >= 1:
            outliers = self._detect_outliers(numeric_cols)

        # 6c. Clustering (Post-Hoc Analysis)
        clustering = None
        # Use feature_cols (excluding target) for proper unsupervised segmentation
        if SKLEARN_AVAILABLE and len(feature_cols) >= 2:
            clustering = self._perform_clustering(feature_cols, target_col)

        # 7. Geospatial Analysis
        geospatial = self._analyze_geospatial(numeric_cols, target_col, lat_col, lon_col)

        # 8. Time Series Analysis
        timeseries = self._analyze_timeseries(numeric_cols, target_col, date_col)

        # 9. Causal Discovery
        causal_graph = None
        
        # Include encoded target for Causal Discovery
        causal_cols = numeric_cols.copy()
        if encoded_target_col:
            causal_cols.append(encoded_target_col)
            
        if CAUSAL_LEARN_AVAILABLE and len(causal_cols) >= 2:
            causal_graph = self._discover_causal_graph(causal_cols)

        # 10. Rule Discovery (Decision Tree)
        rule_tree = None
        final_task_type = task_type
        
        if SKLEARN_AVAILABLE and target_col and len(feature_cols) >= 1:
            # Run for both Categorical (Classification) and Numeric (Regression)
            target_type = self._get_semantic_type(self.df[target_col])
            if target_type in ["Categorical", "Boolean", "Numeric"]:
                rule_tree = self._discover_rules(feature_cols, target_col, task_type)
                
                # Infer task type if not provided
                if not final_task_type:
                    if target_type == "Numeric":
                        final_task_type = "Regression"
                    else:
                        final_task_type = "Classification"

        # 11. Smart Recommendations
        recommendations = self._generate_recommendations(col_profiles, alerts, target_col)

        return DatasetProfile(
            row_count=self.row_count,
            column_count=len(self.columns),
            duplicate_rows=duplicate_rows,
            missing_cells_percentage=missing_pct,
            memory_usage_mb=memory_usage,
            columns=col_profiles,
            correlations=correlations,
            correlations_with_target=correlations_with_target,
            alerts=alerts,
            recommendations=recommendations,
            sample_data=sample_rows,
            target_col=target_col,
            task_type=final_task_type,
            target_correlations=target_correlations,
            target_interactions=target_interactions,
            pca_data=pca_data,
            pca_components=pca_components,
            outliers=outliers,
            clustering=clustering,
            causal_graph=causal_graph,
            rule_tree=rule_tree,
            vif=vif_data,
            geospatial=geospatial,
            timeseries=timeseries,
            excluded_columns=excluded_columns,
            active_filters=active_filters
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
            
            # Calculate global medians for explanation
            medians = df_numeric.median().row(0, named=True)
            
            # Vectorized explanation for top outliers
            # We only process the top_k rows to save time
            for idx, score in scored_indices[:top_k]:
                # Only include if it was actually predicted as outlier
                if preds[idx] == -1:
                    # Get row values
                    row_values = df_numeric.row(idx, named=True)
                    
                    # Generate simple explanation (Z-Score like contribution)
                    explanation = []
                    
                    # Use list comprehension for speed
                    explanation = [
                        {
                            "feature": col,
                            "value": val,
                            "median": medians.get(col, 0),
                            "diff_pct": abs((val - medians.get(col, 0)) / medians.get(col, 1)) * 100 if medians.get(col, 0) != 0 else 0
                        }
                        for col, val in row_values.items()
                        if val is not None and medians.get(col) is not None
                    ]
                    
                    # Filter significant deviations (> 50%)
                    explanation = [e for e in explanation if e["diff_pct"] > 50]
                    
                    # Sort explanation by deviation
                    explanation.sort(key=lambda x: x["diff_pct"], reverse=True)
                    
                    top_outliers.append(OutlierPoint(
                        index=int(idx), # Cast to int for JSON serialization
                        values=row_values,
                        score=float(score),
                        explanation=explanation[:3] # Top 3 reasons
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

    def _perform_clustering(self, numeric_cols: List[str], target_col: Optional[str] = None) -> Optional[ClusteringAnalysis]:
        """
        Performs KMeans clustering for Post-Hoc Analysis.
        Scales data, runs KMeans (k=3 for simplicity in EDA), and projects to 2D for visualization.
        """
        try:
            # 1. Prepare Data using Helper (ensures same sample as PCA if seed is consistent)
            X_scaled, sample_df, scaler = self._prepare_matrix_sample(numeric_cols, target_col=target_col, limit=5000)
            
            if X_scaled is None or sample_df is None or scaler is None:
                return None
            
            # 2. KMeans
            # We fix k=3 for generic "Low, Medium, High" discovery logic in EDA
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # 3. Calculate Stats for Clusters
            clusters_stats = []
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_points = len(labels)
            
            # Centers (inverse transform to be readable)
            centers_scaled = kmeans.cluster_centers_
            centers_original = scaler.inverse_transform(centers_scaled)
            
            feature_names = numeric_cols
            
            for i, label in enumerate(unique_labels):
                center_dict = {
                    col: float(val) for col, val in zip(feature_names, centers_original[i])
                }
                
                clusters_stats.append(ClusterStats(
                    cluster_id=int(label),
                    size=int(counts[i]),
                    percentage=float(counts[i] / total_points * 100),
                    center=center_dict
                ))
                
            # 4. Visualization Data (PCA projection)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Ensure numpy array
            X_pca = np.asarray(X_pca)
            
            # Ensure X_pca is 2D and has at least 2 columns
            if len(X_pca.shape) == 1:
                X_pca = X_pca.reshape(-1, 1)
                
            if X_pca.shape[1] < 2:
                # Pad with zeros if PCA yielded only 1 component (e.g. perfect correlation or 1 feature)
                padding = np.zeros((X_pca.shape[0], 2 - X_pca.shape[1]))
                X_pca = np.hstack([X_pca, padding])
            
            # Get original labels if target available
            original_labels = None
            if target_col and target_col in self.columns:
                original_labels = sample_df[target_col].to_list()
            
            points = []
            for i in range(len(X_pca)):
                label_val = str(original_labels[i]) if original_labels else None
                points.append(ClusteringPoint(
                    x=float(X_pca[i, 0]),
                    y=float(X_pca[i, 1]),
                    cluster=int(labels[i]),
                    label=label_val
                ))
                
            return ClusteringAnalysis(
                method="KMeans",
                n_clusters=n_clusters,
                inertia=float(kmeans.inertia_),
                clusters=clusters_stats,
                points=points
            )

        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return None

    def _discover_causal_graph(self, numeric_cols: List[str]) -> Optional[CausalGraph]:
        """
        Discovers causal structure using the PC algorithm from causal-learn.
        """
        try:
            # Limit number of columns to prevent hanging on wide datasets
            # PC algorithm complexity grows exponentially with number of variables
            if len(numeric_cols) > 15:
                # Smart Feature Selection
                # If target is present (encoded or numeric), select Target + Top 14 correlated features
                # This avoids filtering out low-variance confounders that are highly predictive
                
                target_col_name = None
                # Check if any column looks like a target (encoded or original)
                # We don't have explicit target_col passed here, but we can infer from numeric_cols
                # Or better, we should pass target_col to this method. 
                # For now, let's assume the last column might be target if it matches known target names
                # But wait, we can calculate correlation matrix for ALL numeric cols
                
                # Calculate full correlation matrix
                corr_matrix = self.df.select([
                    pl.corr(col, numeric_cols[0]).alias(col) for col in numeric_cols
                ]) # This is just one row, we need full matrix.
                
                # Actually, we can just use variance as fallback if no target info is available easily
                # But wait, we can check if 'target' or 'target_encoded' is in numeric_cols
                
                target_candidates = [c for c in numeric_cols if "target" in c.lower() or "label" in c.lower()]
                primary_target = target_candidates[0] if target_candidates else None
                
                if primary_target:
                    # Calculate correlation with target
                    corrs = []
                    for col in numeric_cols:
                        if col == primary_target: continue
                        c = self.df.select(pl.corr(col, primary_target)).item()
                        if c is not None:
                            corrs.append((col, abs(c)))
                    
                    # Sort by correlation
                    corrs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select top 14 + target
                    selected_cols = [x[0] for x in corrs[:14]]
                    selected_cols.append(primary_target)
                    numeric_cols = selected_cols
                else:
                    # Fallback to Variance
                    variances = []
                    for col in numeric_cols:
                        var = self.df.select(pl.col(col).var()).item()
                        if var is not None:
                            variances.append((col, var))
                    
                    # Sort by variance descending
                    variances.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 15
                    numeric_cols = [x[0] for x in variances[:15]]

            # 1. Prepare Data
            # Limit to 5000 rows for performance
            limit = 5000
            df_numeric = self.df.select(numeric_cols).drop_nulls().head(limit)
            
            if df_numeric.height < 50:
                return None

            # Convert to numpy array
            data = df_numeric.to_numpy()
            
            # 2. Run PC Algorithm
            # alpha=0.05 is standard significance level
            # indep_test='fisherz' is standard for continuous data (partial correlation)
            cg = pc(data, alpha=0.05, indep_test='fisherz', show_progress=False)
            
            # cg.G is the GeneralGraph
            # cg.G.graph is the adjacency matrix
            # Nodes are indexed 0..N-1 corresponding to numeric_cols
            
            nodes = []
            edges = []
            
            # Create Nodes
            for i, col in enumerate(numeric_cols):
                nodes.append(CausalNode(id=col, label=col))
                
            # Create Edges
            # causal-learn graph: -1=Tail, 1=Arrow.
            # graph[i,j] is endpoint at j.
            # i -> j: graph[i,j]=1, graph[j,i]=-1
            
            adj_matrix = cg.G.graph
            num_vars = len(numeric_cols)
            
            for i in range(num_vars):
                for j in range(i + 1, num_vars): # Check each pair once
                    end_j = adj_matrix[i, j] # Endpoint at j
                    end_i = adj_matrix[j, i] # Endpoint at i
                    
                    source = numeric_cols[i]
                    target = numeric_cols[j]
                    
                    if end_j == 0 and end_i == 0:
                        continue # No edge
                        
                    edge_type = "undirected"
                    
                    if end_i == -1 and end_j == 1:
                        # i -> j
                        edges.append(CausalEdge(source=source, target=target, type="directed"))
                    elif end_i == 1 and end_j == -1:
                        # j -> i
                        edges.append(CausalEdge(source=target, target=source, type="directed"))
                    elif end_i == -1 and end_j == -1:
                        # i -- j
                        edges.append(CausalEdge(source=source, target=target, type="undirected"))
                    elif end_i == 1 and end_j == 1:
                        # i <-> j
                        edges.append(CausalEdge(source=source, target=target, type="bidirected"))
                        
            return CausalGraph(nodes=nodes, edges=edges)
            
        except Exception as e:
            print(f"Error in causal discovery: {e}")
            return None

    def _analyze_geospatial(self, numeric_cols: List[str], target_col: Optional[str] = None, lat_col: Optional[str] = None, lon_col: Optional[str] = None) -> Optional[GeospatialStats]:
        """
        Detects and analyzes geospatial data (Lat/Lon).
        """
        try:
            # If not provided, try to detect
            if not lat_col or not lon_col:
                # Check ALL columns, not just numeric ones, in case they were inferred as strings
                candidates = {c.lower(): c for c in self.columns}
                
                # Latitude detection
                if not lat_col:
                    if 'latitude' in candidates: lat_col = candidates['latitude']
                    elif 'lat' in candidates: lat_col = candidates['lat']
                
                # Longitude detection
                if not lon_col:
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

    def _analyze_timeseries(self, numeric_cols: List[str], target_col: Optional[str] = None, date_col: Optional[str] = None) -> Optional[TimeSeriesAnalysis]:
        """
        Detects DateTime column and performs time series analysis.
        """
        try:
            if not date_col:
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
            
            if not date_col or date_col not in self.columns:
                return None
                
            # 1. Trend Analysis (Dynamic Resampling)
            ts_df = self.lazy_df.sort(date_col)
            
            # Determine ideal interval
            min_date = self.df[date_col].min()
            max_date = self.df[date_col].max()
            
            # Select top 3 numeric columns + target to track
            cols_to_track = numeric_cols[:3]
            if target_col and target_col in numeric_cols and target_col not in cols_to_track:
                cols_to_track.append(target_col)

            # For small datasets, don't resample, just use raw data
            if self.row_count < 1000:
                trend_df = ts_df.select([
                    pl.col(date_col).alias("date"),
                    *cols_to_track
                ]).drop_nulls().collect()
                
                # Convert to list of points
                trend_points = []
                for row in trend_df.iter_rows(named=True):
                    if row["date"] is None: continue
                    
                    vals = {k: v for k, v in row.items() if k != "date" and v is not None}
                    if not vals: continue
                    
                    trend_points.append(TimeSeriesPoint(
                        date=row["date"].isoformat(),
                        values=vals
                    ))
            else:
                # Dynamic resampling based on duration
                interval = "1d" # Default
                if min_date and max_date:
                    duration = (max_date - min_date).total_seconds()
                    ideal_seconds = duration / 100 # Target ~100 points
                    
                    if ideal_seconds < 60: interval = "1s"
                    elif ideal_seconds < 3600: interval = "1m"
                    elif ideal_seconds < 86400: interval = "1h"
                    elif ideal_seconds < 604800: interval = "1d"
                    else: interval = "1w"

                if not cols_to_track:
                    # Just track count
                    trend_df = ts_df.group_by_dynamic(date_col, every=interval).agg(
                        pl.count().alias("count")
                    ).sort(date_col).collect()
                else:
                    aggs = [pl.col(c).mean().alias(c) for c in cols_to_track]
                    trend_df = ts_df.group_by_dynamic(date_col, every=interval).agg(
                        aggs
                    ).sort(date_col).collect()
                
                # Convert to list of points
                trend_points = []
                for row in trend_df.iter_rows(named=True):
                    # Skip if date is null
                    if row[date_col] is None: continue
                    
                    vals = {k: v for k, v in row.items() if k != date_col and v is not None}
                    if not vals: continue
                    
                    trend_points.append(TimeSeriesPoint(
                        date=row[date_col].isoformat(),
                        values=vals
                    ))
                
            # 2. Seasonality (Day of Week)
            # If we have a numeric column to track, use mean, else count
            agg_expr = pl.len().alias("count")
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

    def _calculate_pca(self, numeric_cols: List[str], target_col: Optional[str] = None) -> Tuple[Optional[List[PCAPoint]], Optional[List[PCAComponent]]]:
        """
        Calculates 2D PCA projection and extracts component loadings.
        """
        try:
            # 1. Prepare Data using Helper
            X_scaled, sample_df, _ = self._prepare_matrix_sample(numeric_cols, target_col=target_col, limit=5000)
            
            if X_scaled is None or sample_df is None:
                return None, None
            
            # 2. PCA
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_scaled)
            
            # Extract Components Info
            components_list = []
            if hasattr(pca, 'components_'):
                # Numeric cols were used in order
                feature_names = numeric_cols 
                for i, comp in enumerate(pca.components_):
                    # Get top features (absolute weight) for this component
                    weights = {feature_names[j]: float(comp[j]) for j in range(len(feature_names))}
                    # Sort by absolute weight and take top 5
                    top_features = dict(sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
                    
                    components_list.append(PCAComponent(
                        component=f"PC{i+1}",
                        explained_variance_ratio=float(pca.explained_variance_ratio_[i]),
                        top_features=top_features
                    ))

            # Ensure numpy array
            X_pca = np.asarray(X_pca)
            
            # Robust shape handling
            if len(X_pca.shape) == 1:
                X_pca = X_pca.reshape(-1, 1)
                
            # Pad if fewer components than expected (PCA might return fewer if input rank is low)
            if X_pca.shape[1] < 3:
                padding = np.zeros((X_pca.shape[0], 3 - X_pca.shape[1]))
                X_pca = np.hstack([X_pca, padding])
            
            # 3. Get labels
            labels = None
            if target_col and target_col in self.columns:
                labels = sample_df[target_col].to_list()
            
            # 4. Create result
            points = []
            for i in range(len(X_pca)):
                label_val = str(labels[i]) if labels else None
                points.append(PCAPoint(
                    x=float(X_pca[i, 0]), 
                    y=float(X_pca[i, 1]), 
                    z=float(X_pca[i, 2]) if X_pca.shape[1] > 2 else None,
                    label=label_val
                ))
                
            return points, components_list
            
        except Exception as e:
            print(f"Error calculating PCA: {e}")
            return None, None

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
                
                # Ensure value_col is numeric (cast if necessary)
                # Sometimes value_col might be inferred as string if it has mixed types
                
                stats_df = self.lazy_df.group_by(group_col).agg([
                    pl.col(value_col).cast(pl.Float64, strict=False).min().alias("min"),
                    pl.col(value_col).cast(pl.Float64, strict=False).quantile(0.25).alias("q1"),
                    pl.col(value_col).cast(pl.Float64, strict=False).median().alias("median"),
                    pl.col(value_col).cast(pl.Float64, strict=False).quantile(0.75).alias("q3"),
                    pl.col(value_col).cast(pl.Float64, strict=False).max().alias("max")
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
                    pl.len().alias("n"),
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
            
            # Suppress numpy warnings during correlation calculation
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
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
        
    def _analyze_column(self, col: str) -> Tuple[ColumnProfile, List[Alert]]:
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
                    
                    # Ensure sample has variance before testing
                    if len(sample_data) > 20 and np.std(sample_data) > 1e-10:
                        # Use Shapiro-Wilk for N < 5000, else KS test
                        if len(sample_data) < 5000:
                            stat, p_value = shapiro(sample_data)
                            test_name = "Shapiro-Wilk"
                        else:
                            # KS Test against normal distribution
                            # Must provide mean/std to test against fitted normal, not standard normal
                            mean, std = np.mean(sample_data), np.std(sample_data)
                            stat, p_value = kstest(sample_data, 'norm', args=(mean, std))
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
            
            # Sentiment Analysis
            if profile.text_stats:
                profile.text_stats.sentiment_distribution = self._analyze_sentiment(self.df[col])
            
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

    def _discover_rules(self, feature_cols: List[str], target_col: str, task_type: Optional[str] = None) -> Optional[RuleTree]:
        """
        Trains a surrogate Decision Tree to extract rules from the data.
        Supports both Classification (Categorical Target) and Regression (Numeric Target).
        """
        try:
            # Prepare Data
            # Limit rows for performance
            limit = 100000
            df_sample = self.df.select(feature_cols + [target_col]).head(limit)
            
            # Determine Task Type
            is_regression = False
            if task_type:
                if task_type.lower() == "regression":
                    is_regression = True
                elif task_type.lower() == "classification":
                    is_regression = False
            else:
                target_type = self._get_semantic_type(df_sample[target_col])
                is_regression = target_type == "Numeric"

            # Handle Missing Values & Encoding
            # Numeric: Fill with Mean
            # Categorical: Fill with "Missing" -> Ordinal Encode (Factorize)
            
            cat_cols = [c for c in feature_cols if self._get_semantic_type(df_sample[c]) in ["Categorical", "Boolean", "Text"]]
            num_cols = [c for c in feature_cols if c not in cat_cols]
            
            X_data = {}
            feature_names = []
            
            for col in num_cols:
                mean_val = df_sample[col].mean()
                X_data[col] = df_sample[col].fill_null(mean_val).to_numpy()
                feature_names.append(col)
                
            for col in cat_cols:
                s = df_sample[col].cast(pl.Utf8).fill_null("Missing")
                # Use simple factorization (Ordinal) for readability
                codes = s.cast(pl.Categorical).to_physical().to_numpy()
                X_data[col] = codes
                feature_names.append(col)
                
            # Construct X matrix
            # Ensure order matches feature_names
            X_list = [X_data[col] for col in feature_names]
            X = np.column_stack(X_list)
            
            # Prepare Y and Model
            if is_regression:
                # Regression: Fill nulls with mean, keep as numeric
                y_mean = df_sample[target_col].mean()
                y = df_sample[target_col].fill_null(y_mean).to_numpy()
                class_names = [] # Not used for regression
                
                # Train Tree (Regressor)
                clf = DecisionTreeRegressor(max_depth=4, random_state=42)
            else:
                # Classification: Encode as categorical
                y_series = df_sample[target_col].cast(pl.Utf8).fill_null("Missing")
                
                # Handle High Cardinality: Group into Top 10 + Other
                if y_series.n_unique() > 10:
                    top_10 = y_series.value_counts().sort("count", descending=True).head(10)[target_col].to_list()
                    # Use Polars expression on a temp DataFrame for speed
                    temp_df = pl.DataFrame({"y": y_series})
                    y_series = temp_df.select(
                        pl.when(pl.col("y").is_in(top_10))
                        .then(pl.col("y"))
                        .otherwise(pl.lit("Other"))
                    ).to_series()

                y = y_series.cast(pl.Categorical).to_physical().to_numpy()
                class_names = y_series.cast(pl.Categorical).cat.get_categories().to_list()
                
                # Train Tree (Classifier)
                clf = DecisionTreeClassifier(max_depth=4, random_state=42)

            clf.fit(X, y)
            
            # Extract Feature Importance
            importances = clf.feature_importances_
            feature_importance_list = []
            for idx, imp in enumerate(importances):
                if imp > 0:
                    feature_importance_list.append({
                        "feature": feature_names[idx],
                        "importance": float(imp)
                    })
            # Sort by importance
            feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

            # Extract Tree Structure
            tree_ = clf.tree_
            
            nodes = []
            
            def recurse(node_id):
                is_leaf = bool(tree_.children_left[node_id] == _tree.TREE_LEAF)
                
                feature = None
                threshold = None
                if not is_leaf:
                    feature_idx = tree_.feature[node_id]
                    feature = feature_names[feature_idx]
                    threshold = float(tree_.threshold[node_id])
                
                # Handle Value & Class Name
                if is_regression:
                    # Value is the predicted mean (single float)
                    val = float(tree_.value[node_id][0][0])
                    value = [val] # Wrap in list for consistency
                    class_name = f"{val:.2f}"
                else:
                    # Value is class counts
                    value = tree_.value[node_id][0].tolist()
                    class_idx = np.argmax(value)
                    class_name = str(class_names[class_idx]) if class_idx < len(class_names) else "Unknown"
                
                children = []
                if not is_leaf:
                    left_id = tree_.children_left[node_id]
                    right_id = tree_.children_right[node_id]
                    children = [int(left_id), int(right_id)]
                    recurse(left_id)
                    recurse(right_id)
                    
                nodes.append(RuleNode(
                    id=int(node_id),
                    feature=feature,
                    threshold=threshold,
                    impurity=float(tree_.impurity[node_id]),
                    samples=int(tree_.n_node_samples[node_id]),
                    value=value,
                    class_name=class_name,
                    is_leaf=is_leaf,
                    children=children
                ))
                
            recurse(0)
            
            # Sort nodes by ID
            nodes.sort(key=lambda x: x.id)

            # Generate Text Rules
            rules_text = []
            
            def recurse_rules(node_id, current_rule):
                if tree_.children_left[node_id] == _tree.TREE_LEAF:
                    # Leaf node
                    total_samples = int(tree_.n_node_samples[node_id])
                    
                    if is_regression:
                        val = float(tree_.value[node_id][0][0])
                        rule_str = f"IF {current_rule} THEN Value = {val:.2f} (Samples: {total_samples})"
                    else:
                        value = tree_.value[node_id][0]
                        class_idx = np.argmax(value)
                        class_name = str(class_names[class_idx]) if class_idx < len(class_names) else "Unknown"
                        total = np.sum(value)
                        confidence = (value[class_idx] / total) * 100 if total > 0 else 0
                        rule_str = f"IF {current_rule} THEN {class_name} (Confidence: {confidence:.1f}%, Samples: {int(total)})"
                        
                    rules_text.append(rule_str)
                    return

                # Internal node
                feature_idx = tree_.feature[node_id]
                feature_name = feature_names[feature_idx]
                threshold = tree_.threshold[node_id]
                
                # Left child (<= threshold)
                left_rule = f"{current_rule} AND {feature_name} <= {threshold:.2f}" if current_rule else f"{feature_name} <= {threshold:.2f}"
                recurse_rules(tree_.children_left[node_id], left_rule)
                
                # Right child (> threshold)
                right_rule = f"{current_rule} AND {feature_name} > {threshold:.2f}" if current_rule else f"{feature_name} > {threshold:.2f}"
                recurse_rules(tree_.children_right[node_id], right_rule)

            recurse_rules(0, "")
            
            return RuleTree(
                nodes=nodes,
                accuracy=float(clf.score(X, y)),
                rules=rules_text,
                feature_importances=feature_importance_list
            )
            
        except Exception as e:
            print(f"Error in rule discovery: {e}")
            return None

    def _get_semantic_type(self, series: pl.Series) -> str:
        dtype = series.dtype
        
        if dtype in [pl.Float32, pl.Float64]:
            return "Numeric"
        elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            # Check cardinality for Integers
            # If low cardinality -> Categorical (better for plotting/analysis)
            n_unique = series.n_unique()
            count = len(series)
            ratio = n_unique / count if count > 0 else 0
            
            if ratio < 0.05 and n_unique < 20:
                 return "Categorical"
            
            return "Numeric"
        elif dtype == pl.Boolean:
            return "Boolean"
        elif dtype in [pl.Date, pl.Datetime, pl.Duration]:
            return "DateTime"
        elif dtype == pl.Utf8 or dtype == pl.String:
            n_unique = series.n_unique()
            count = len(series)
            ratio = n_unique / count if count > 0 else 0
            
            # Low cardinality ratio (< 5%) -> Categorical
            if ratio < 0.05:
                 return "Categorical"
            
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
            pl.col(col).var().alias("variance"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).quantile(0.25).alias("q25"),
            pl.col(col).quantile(0.75).alias("q75"),
            pl.col(col).skew().alias("skew"),
            pl.col(col).kurtosis().alias("kurt"),
            (pl.col(col) == 0).sum().alias("zeros"),
            (pl.col(col) < 0).sum().alias("negatives")
        ]).collect()
        
        row = stats.row(0, named=True)
        return NumericStats(
            mean=row["mean"], median=row["median"], std=row["std"], variance=row.get("variance"),
            min=row["min"], max=row["max"],
            q25=row["q25"], q75=row["q75"], skewness=row["skew"], kurtosis=row["kurt"],
            zeros_count=row["zeros"], negatives_count=row["negatives"]
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
            rare_count = self.lazy_df.group_by(col).agg(pl.len().alias("cnt")).filter(pl.col("cnt") < 5).select(pl.len()).collect().item()
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

    def get_decomposition_split(
        self, 
        measure_col: Optional[str], 
        measure_agg: str, 
        split_col: Optional[str], 
        filters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculates the breakdown of a measure by a split column.
        Used for Decomposition Trees.
        """
        # 1. Apply Filters
        filtered_df = self.df
        for f in filters:
            col = f['column']
            op = f['operator']
            val = f['value']
            
            if col not in filtered_df.columns:
                continue
            
            # Handle type mismatch for numeric columns vs string values (from frontend)
            dtype = filtered_df.schema[col]
            is_numeric = dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
            
            if is_numeric and isinstance(val, str):
                if val == "Unknown":
                    # "Unknown" represents nulls in our visualization
                    if op == '==':
                        filtered_df = filtered_df.filter(pl.col(col).is_null())
                    elif op == '!=':
                        filtered_df = filtered_df.filter(pl.col(col).is_not_null())
                    continue
                else:
                    try:
                        if dtype in (pl.Float32, pl.Float64):
                            val = float(val)
                        else:
                            # Handle "1.0" string for int columns
                            val = int(float(val))
                    except ValueError:
                        # If casting fails, we can't filter numeric col by this string.
                        # It might be a mismatch. We can try casting the column to string?
                        # Or just ignore?
                        # Let's cast column to string as fallback, though slower.
                        pass

            # Apply filter
            # If we still have a string val and numeric col, we cast col to string to be safe
            if is_numeric and isinstance(val, str):
                 col_expr = pl.col(col).cast(pl.Utf8)
            else:
                 col_expr = pl.col(col)

            if op == '==':
                filtered_df = filtered_df.filter(col_expr == val)
            elif op == '!=':
                filtered_df = filtered_df.filter(col_expr != val)
            elif op == '>':
                filtered_df = filtered_df.filter(col_expr > val)
            elif op == '<':
                filtered_df = filtered_df.filter(col_expr < val)
            elif op == '>=':
                filtered_df = filtered_df.filter(col_expr >= val)
            elif op == '<=':
                filtered_df = filtered_df.filter(col_expr <= val)
            elif op == 'in':
                filtered_df = filtered_df.filter(col_expr.is_in(val))

        # 2. Calculate Measure
        # If split_col is None, return global aggregate
        if not split_col:
            if not measure_col: # Count rows
                val = filtered_df.height
            else:
                if measure_col not in filtered_df.columns:
                    return []
                
                # Handle nulls in measure col
                series = filtered_df[measure_col]
                
                if measure_agg == 'sum':
                    val = series.sum()
                elif measure_agg == 'mean':
                    val = series.mean()
                elif measure_agg == 'min':
                    val = series.min()
                elif measure_agg == 'max':
                    val = series.max()
                else:
                    val = filtered_df.height # Default to count
            
            # Handle None/NaN result
            if val is None:
                val = 0
                
            return [{"name": "Total", "value": val, "ratio": 1.0}]

        # 3. Group By Split Col
        if split_col not in filtered_df.columns:
            return []

        # Handle nulls in split col by filling with "Unknown" or dropping
        # For visualization, it's better to show them
        temp_df = filtered_df.with_columns(
            pl.col(split_col).fill_null("Unknown").cast(pl.Utf8)
        )

        if not measure_col:
            # Count
            agg_df = temp_df.group_by(split_col).agg(pl.len().alias("value"))
        else:
            if measure_col not in temp_df.columns:
                return []
                
            if measure_agg == 'sum':
                agg_df = temp_df.group_by(split_col).agg(pl.col(measure_col).sum().alias("value"))
            elif measure_agg == 'mean':
                agg_df = temp_df.group_by(split_col).agg(pl.col(measure_col).mean().alias("value"))
            elif measure_agg == 'min':
                agg_df = temp_df.group_by(split_col).agg(pl.col(measure_col).min().alias("value"))
            elif measure_agg == 'max':
                agg_df = temp_df.group_by(split_col).agg(pl.col(measure_col).max().alias("value"))
            else:
                agg_df = temp_df.group_by(split_col).agg(pl.len().alias("value"))

        # 4. Calculate Ratios
        total_val = agg_df["value"].sum()
        if total_val == 0 or total_val is None:
            # Avoid division by zero
            result_df = agg_df.with_columns(pl.lit(0.0).alias("ratio"))
        else:
            result_df = agg_df.with_columns((pl.col("value") / total_val).alias("ratio"))

        # Sort by value descending
        result_df = result_df.sort("value", descending=True)

        # Convert to list of dicts
        results = []
        for row in result_df.iter_rows(named=True):
            results.append({
                "name": str(row[split_col]),
                "value": row["value"] if row["value"] is not None else 0,
                "ratio": row["ratio"] if row["ratio"] is not None else 0
            })
            
        return results

    def _calculate_vif(self, numeric_cols: List[str]) -> Optional[Dict[str, float]]:
        """
        Calculates Variance Inflation Factor (VIF) for numeric columns to detect multicollinearity.
        Uses the inverse correlation matrix method: VIF_i = (R^-1)_ii
        VIF > 5 indicates high multicollinearity.
        """
        if len(numeric_cols) < 2:
            return None
            
        try:
            # Calculate Correlation Matrix using Polars
            # We use the correlation matrix of the features
            # VIF_i = 1 / (1 - R_i^2) where R_i^2 is the R^2 of regressing X_i on all other X
            # This is equivalent to the diagonal elements of the inverse correlation matrix.
            
            # 1. Get Correlation Matrix
            # We can use the existing calculate_correlations or do it here
            # Let's do it efficiently here using numpy
            
            # Select columns and drop nulls (VIF requires complete cases)
            df_clean = self.df.select(numeric_cols).drop_nulls()
            
            # If too few rows, skip
            if df_clean.height < len(numeric_cols) + 5:
                return None
                
            # Convert to numpy
            data = df_clean.to_numpy()
            
            # Calculate Correlation Matrix (Pearson)
            # rowvar=False means columns are variables
            corr_matrix = np.corrcoef(data, rowvar=False)
            
            # Check for NaNs in correlation matrix (constant columns)
            if np.isnan(corr_matrix).any():
                return None
                
            # 2. Calculate Inverse
            try:
                inv_corr = np.linalg.inv(corr_matrix)
            except np.linalg.LinAlgError:
                # Matrix is singular (perfect multicollinearity)
                # We can try pseudo-inverse or just return high VIF for all
                return {col: 999.0 for col in numeric_cols}
            
            # 3. Extract Diagonal
            vif_data = {}
            for i, col in enumerate(numeric_cols):
                vif = inv_corr[i, i]
                # VIF should be >= 1
                vif = max(1.0, float(vif))
                vif_data[col] = vif
                    
            return vif_data
        except Exception as e:
            print(f"Error calculating VIF: {e}")
            return None

    def _prepare_matrix_sample(self, numeric_cols: List[str], target_col: Optional[str] = None, limit: int = 5000) -> Tuple[Optional[np.ndarray], Optional[pl.DataFrame], Optional[Any]]:
        """
        Helper that:
        1. Selects numeric cols (+ target if exists)
        2. Samples dataset (consistent seed=42)
        3. SimpleImputer (mean)
        4. StandardScaler
        
        Returns: (X_scaled, sample_df, scaler)
        """
        try:
            # Determine columns to fetch
            cols_to_fetch = list(numeric_cols)
            if target_col and target_col in self.columns and target_col not in cols_to_fetch:
                cols_to_fetch.append(target_col)
                
            # Sample once
            # If dataset is smaller than limit, we take all of it.
            # We use seed=42 to ensure other analysis (PCA vs Clustering) see the same subset if called separately
            if self.row_count > limit:
                sample_df = self.df.select(cols_to_fetch).sample(n=limit, with_replacement=False, seed=42)
            else:
                sample_df = self.df.select(cols_to_fetch)
                
            if sample_df.height < 5:
                # Not enough data for meaningful PCA/Clustering
                return None, None, None
            
            # Extract Numeric Data
            X_df = sample_df.select(numeric_cols)
            
            # Impute & Scale
            try:
                # Polars fill_null preferred for speed
                # We use fill_null(0) as a final fallback for columns that are entirely null (where mean is null)
                X_df = X_df.fill_null(strategy="mean").fill_null(0)
                X = X_df.to_numpy()
                
                # Double check for Infs or NaNs in numpy array (can happen with float32 overflows etc)
                if not np.isfinite(X).all():
                     X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                     
            except Exception:
                # Fallback to sklearn/pandas if polars fails or we prefer sklearn
                X = X_df.to_pandas().values
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                # Remaining NaNs?
                X = np.nan_to_num(X, nan=0.0)
                
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            return X_scaled, sample_df, scaler
            
        except Exception as e:
            print(f"Error preparing matrix sample: {e}")
            return None, None, None

    def _analyze_sentiment(self, text_series: pl.Series) -> Optional[Dict[str, float]]:
        """
        Analyzes sentiment of a text column using VADER.
        Returns distribution ratios: {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
        """
        if not VADER_AVAILABLE:
            return None
            
        try:
            # Sample if too large (max 1000 rows for performance)
            if text_series.len() > 1000:
                sample = text_series.sample(1000, seed=42)
            else:
                sample = text_series
                
            # Drop nulls
            texts = sample.drop_nulls().to_list()
            if not texts:
                return None
                
            analyzer = SentimentIntensityAnalyzer()
            
            counts = {"positive": 0, "neutral": 0, "negative": 0}
            total = 0
            
            for text in texts:
                if not isinstance(text, str):
                    continue
                    
                scores = analyzer.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    counts["positive"] += 1
                elif compound <= -0.05:
                    counts["negative"] += 1
                else:
                    counts["neutral"] += 1
                total += 1
            
            if total == 0:
                return None
                
            # Normalize to ratios
            return {
                "positive": counts["positive"] / total,
                "neutral": counts["neutral"] / total,
                "negative": counts["negative"] / total
            }
        except Exception:
            return None
