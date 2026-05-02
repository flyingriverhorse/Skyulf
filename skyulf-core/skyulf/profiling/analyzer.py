"""Public entry point for EDA profiling.

The heavy lifting lives in :mod:`skyulf.profiling._analyzer` mixin modules,
split by concern (numeric / categorical / text / temporal / geo /
multivariate / causal / rules / recommendations / column / decomposition).
This module keeps only the orchestrator: ``EDAAnalyzer.__init__`` and
``EDAAnalyzer.analyze``.
"""

from typing import Any, Dict, List, Optional

import polars as pl

from ._analyzer import (
    CategoricalMixin,
    CausalMixin,
    ColumnMixin,
    DatesMixin,
    DecompositionMixin,
    GeoMixin,
    MultivariateMixin,
    NumericMixin,
    RecommendationsMixin,
    RulesMixin,
    TargetMixin,
    TemporalMixin,
    TextMixin,
)
from ._analyzer._utils import SKLEARN_AVAILABLE, _collect
from .correlations import calculate_correlations
from .schemas import Alert, DatasetProfile, Filter

__all__ = ["EDAAnalyzer"]


class EDAAnalyzer(
    DatesMixin,
    NumericMixin,
    CategoricalMixin,
    TextMixin,
    TargetMixin,
    GeoMixin,
    TemporalMixin,
    MultivariateMixin,
    CausalMixin,
    RulesMixin,
    RecommendationsMixin,
    ColumnMixin,
    DecompositionMixin,
):
    """Generate a :class:`DatasetProfile` for a polars DataFrame.

    Construction auto-casts string columns that look date-like via
    :meth:`_cast_date_columns` (inherited from :class:`DatesMixin`). All
    per-type analysis is delegated to the mixin classes; this orchestrator
    only owns the top-level pipeline in :meth:`analyze`.
    """

    def __init__(self, df: pl.DataFrame):
        self.df = df
        # Detect date-like string columns up front so downstream type checks see Date/Datetime.
        self._cast_date_columns()
        self.lazy_df = self.df.lazy()
        self.row_count = self.df.height
        self.columns = self.df.columns

    def analyze(  # noqa: C901  # top-level orchestrator: each stage delegates to a mixin
        self,
        target_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        date_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> DatasetProfile:
        """Produce the full profile.

        Pipeline (each step is a mixin call when non-trivial):
        1. Filters → exclusions → basic frame stats.
        2. Two batched polars aggregations (basic + advanced) feed
           per-column analysis (A3 optimization).
        3. Correlations / VIF / target relationships.
        4. PCA / clustering / outliers / geo / time-series / causal / rules.
        5. Heuristic recommendations.
        """

        # 1. Apply user filters (mutates self.df / self.lazy_df / self.row_count).
        active_filters: List[Filter] = []
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

                    active_filters.append(
                        Filter(column=str(col), operator=str(op), value=val)
                    )

            self.lazy_df = self.df.lazy()
            self.row_count = self.df.height

        if self.row_count == 0:
            return DatasetProfile(
                row_count=0,
                column_count=len(self.columns),
                duplicate_rows=0,
                missing_cells_percentage=0.0,
                memory_usage_mb=0.0,
                columns={},
                correlations=None,
                alerts=[
                    Alert(
                        type="Empty Data",
                        message="Filters resulted in 0 rows. Please adjust your filters.",
                        severity="warning",
                    )
                ],
                recommendations=[],
                sample_data=[],
                active_filters=active_filters,
                target_col=target_col,
            )

        excluded_columns: List[str] = []
        if exclude_cols:
            excluded_columns = [c for c in exclude_cols if c in self.columns]
            self.columns = [c for c in self.columns if c not in excluded_columns]

        # 2. Frame-level stats.
        missing_cells = self.df.null_count().sum_horizontal()[0]
        total_cells = self.row_count * len(self.columns)
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
        duplicate_rows = int(self.df.is_duplicated().sum())
        memory_usage = self.df.estimated_size("mb")

        col_profiles = {}
        alerts: List[Alert] = []
        numeric_cols: List[str] = []

        # -- A3: BATCHED PER-COLUMN AGGREGATIONS --
        # Two single-pass polars queries replace ~N python-level "select(col)"
        # calls per column. Wins ~3–10× on wide frames.

        # Query 1: Basic stats (null_count, n_unique) for every column.
        basic_aggs = []
        for col in self.columns:
            basic_aggs.extend(
                [
                    pl.col(col).null_count().alias(f"{col}__null"),
                    pl.col(col).n_unique().alias(f"{col}__unique"),
                ]
            )
        basic_stats_df = _collect(self.lazy_df.select(basic_aggs))
        basic_stats = (
            basic_stats_df.row(0, named=True) if len(basic_stats_df) > 0 else {}
        )

        # Inline semantic-type inference (avoids re-fetching n_unique per column).
        semantic_types: Dict[str, str] = {}
        for col in self.columns:
            dtype = self.df[col].dtype
            n_unique = basic_stats.get(f"{col}__unique", 0)
            ratio = n_unique / self.row_count if self.row_count > 0 else 0

            if dtype in [pl.Float32, pl.Float64]:
                stype = "Numeric"
            elif dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                stype = "Categorical" if (ratio < 0.05 and n_unique < 20) else "Numeric"
            elif dtype == pl.Boolean:
                stype = "Boolean"
            elif dtype in [pl.Date, pl.Datetime, pl.Duration]:
                stype = "DateTime"
            elif dtype in [pl.Utf8, pl.String]:
                stype = "Categorical" if ratio < 0.05 else "Text"
            elif str(dtype) == "Categorical":
                stype = "Categorical"
            else:
                stype = "Text"
            semantic_types[col] = stype

        # Query 2: Type-specific advanced aggregations (mean/quantiles/value_counts/...).
        advanced_aggs = []
        for col in self.columns:
            stype = semantic_types[col]
            if stype == "Numeric":
                advanced_aggs.extend(
                    [
                        pl.col(col).mean().alias(f"{col}__mean"),
                        pl.col(col).median().alias(f"{col}__median"),
                        pl.col(col).std().alias(f"{col}__std"),
                        pl.col(col).var().alias(f"{col}__var"),
                        pl.col(col).min().alias(f"{col}__min"),
                        pl.col(col).max().alias(f"{col}__max"),
                        pl.col(col).quantile(0.25).alias(f"{col}__q25"),
                        pl.col(col).quantile(0.75).alias(f"{col}__q75"),
                        pl.col(col).skew().alias(f"{col}__skew"),
                        pl.col(col).kurtosis().alias(f"{col}__kurt"),
                        (pl.col(col) == 0).sum().alias(f"{col}__zeros"),
                        (pl.col(col) < 0).sum().alias(f"{col}__negatives"),
                    ]
                )
            elif stype == "Categorical":
                advanced_aggs.extend(
                    [
                        pl.col(col)
                        .value_counts(sort=True)
                        .head(10)
                        .implode()
                        .alias(f"{col}__top_k"),
                        (pl.col(col).value_counts().struct.field("count") < 5)
                        .sum()
                        .alias(f"{col}__rare"),
                    ]
                )
            elif stype == "DateTime":
                advanced_aggs.extend(
                    [
                        pl.col(col).min().alias(f"{col}__min"),
                        pl.col(col).max().alias(f"{col}__max"),
                    ]
                )
            elif stype == "Text":
                advanced_aggs.extend(
                    [
                        pl.col(col)
                        .str.len_bytes()
                        .cast(pl.Float64)
                        .mean()
                        .alias(f"{col}__avg_len"),
                        pl.col(col).str.len_bytes().min().alias(f"{col}__min_len"),
                        pl.col(col).str.len_bytes().max().alias(f"{col}__max_len"),
                    ]
                )

        advanced_stats: dict = {}
        if advanced_aggs:
            advanced_stats_df = _collect(self.lazy_df.select(advanced_aggs))
            if len(advanced_stats_df) > 0:
                advanced_stats = advanced_stats_df.row(0, named=True)

        # Build per-column profiles from the batched stats.
        for col in self.columns:
            profile, col_alerts = self._analyze_column(
                col, basic_stats, advanced_stats, semantic_types
            )
            col_profiles[col] = profile
            alerts.extend(col_alerts)

            # Include in numeric_cols if it's Numeric OR a Categorical-by-cardinality
            # whose underlying dtype is numeric (so PCA/causal still see it).
            is_numeric_type = self.df[col].dtype in [
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]
            if profile.dtype == "Numeric":
                numeric_cols.append(col)
            elif profile.dtype == "Categorical" and is_numeric_type:
                numeric_cols.append(col)

        # Encode string targets so they appear in causal graphs.
        encoded_target_col: Optional[str] = None
        if target_col and target_col in self.columns and target_col not in numeric_cols:
            encoded_target = f"{target_col}_encoded"
            self.df = self.df.with_columns(
                pl.col(target_col)
                .cast(pl.Categorical)
                .to_physical()
                .alias(encoded_target)
            )
            self.lazy_df = self.df.lazy()
            encoded_target_col = encoded_target

        # Feature columns = numeric cols minus the target itself.
        feature_cols = [c for c in numeric_cols if c != target_col]

        # 3. Correlations + VIF.
        correlations = calculate_correlations(self.lazy_df, feature_cols)

        vif_data = self._calculate_vif(feature_cols)
        if vif_data:
            for col, val in vif_data.items():
                if val > 10.0:
                    alerts.append(
                        Alert(
                            type="Multicollinearity",
                            message=(
                                f"Column '{col}' has very high VIF ({val:.1f}). "
                                "Consider removing it."
                            ),
                            severity="warning",
                        )
                    )
                elif val > 5.0:
                    alerts.append(
                        Alert(
                            type="Multicollinearity",
                            message=f"Column '{col}' has high VIF ({val:.1f}).",
                            severity="info",
                        )
                    )

        # 3a. Feature-vs-target correlations (separate matrix).
        correlations_with_target = None
        target_corr_cols: List[str] = []
        if target_col:
            if target_col in numeric_cols:
                target_corr_cols = feature_cols + [target_col]
            elif encoded_target_col:
                target_corr_cols = feature_cols + [encoded_target_col]
        if len(target_corr_cols) >= 2:
            correlations_with_target = calculate_correlations(
                self.lazy_df, target_corr_cols
            )

        # 3b. Target-relationship analytics (correlations / interactions / leakage).
        target_correlations: Dict[str, float] = {}
        target_interactions = None
        if target_col and target_col in self.columns:
            target_semantic_type = self._get_semantic_type(self.df[target_col])

            if target_semantic_type == "Numeric":
                if target_col in numeric_cols:
                    target_correlations = self._calculate_target_correlations(
                        target_col, feature_cols
                    )

                    for col, corr in target_correlations.items():
                        if abs(corr) > 0.95 and col != target_col:
                            alerts.append(
                                Alert(
                                    column=col,
                                    type="Leakage",
                                    message=(
                                        f"Column '{col}' is highly correlated ({corr:.2f}) "
                                        f"with target '{target_col}'. Possible leakage."
                                    ),
                                    severity="warning",
                                )
                            )

                    cat_cols = [
                        c
                        for c in self.columns
                        if self._get_semantic_type(self.df[c]) == "Categorical"
                        and c != target_col
                    ]
                    if cat_cols:
                        target_interactions = self._calculate_target_interactions(
                            target_col, cat_cols, is_target_numeric=True
                        )

            elif target_semantic_type == "Categorical":
                target_correlations = self._calculate_categorical_target_associations(
                    target_col, feature_cols
                )
                top_features = list(target_correlations.keys())
                if top_features:
                    target_interactions = self._calculate_target_interactions(
                        target_col, top_features, is_target_numeric=False
                    )

        # 4. Frame-level alerts.
        if missing_pct > 50:
            alerts.append(
                Alert(
                    type="High Null",
                    message=f"Dataset is {missing_pct:.1f}% empty.",
                    severity="warning",
                )
            )

        # 5. Sample (used for FE scatter plots).
        sample_rows = self.df.head(5000).to_dicts()

        # 6. Multivariate.
        pca_data = None
        pca_components = None
        if SKLEARN_AVAILABLE and len(feature_cols) >= 2:
            pca_res = self._calculate_pca(feature_cols, target_col)
            if pca_res:
                pca_data, pca_components = pca_res

        outliers = None
        if SKLEARN_AVAILABLE and len(numeric_cols) >= 1:
            outliers = self._detect_outliers(numeric_cols)

        clustering = None
        if SKLEARN_AVAILABLE and len(feature_cols) >= 2:
            clustering = self._perform_clustering(feature_cols, target_col)

        # 7-8. Geo + time series.
        geospatial = self._analyze_geospatial(
            numeric_cols, target_col, lat_col, lon_col
        )
        timeseries = self._analyze_timeseries(numeric_cols, target_col, date_col)

        # 9. Causal discovery (include encoded target so it shows in the graph).
        causal_graph = None
        causal_cols = numeric_cols.copy()
        if encoded_target_col:
            causal_cols.append(encoded_target_col)
        if len(causal_cols) >= 2:
            causal_graph = self._discover_causal_graph(causal_cols)

        # 10. Rule discovery (decision-tree surrogate).
        rule_tree = None
        final_task_type = task_type
        if SKLEARN_AVAILABLE and target_col and len(feature_cols) >= 1:
            target_type = self._get_semantic_type(self.df[target_col])
            if target_type in ["Categorical", "Boolean", "Numeric"]:
                rule_tree = self._discover_rules(feature_cols, target_col, task_type)
                if not final_task_type:
                    final_task_type = (
                        "Regression" if target_type == "Numeric" else "Classification"
                    )

        # 11. Recommendations.
        recommendations = self._generate_recommendations(
            col_profiles, alerts, target_col
        )

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
            active_filters=active_filters,
        )
