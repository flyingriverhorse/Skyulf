"""Public entry point for EDA profiling.

The heavy lifting lives in :mod:`skyulf.profiling._analyzer` mixin modules,
split by concern (numeric / categorical / text / temporal / geo /
multivariate / causal / rules / recommendations / column / decomposition).
This module keeps only the orchestrator: ``EDAAnalyzer.__init__`` and
``EDAAnalyzer.analyze``.
"""

import logging
from typing import Any

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

logger = logging.getLogger(__name__)


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

    _NUMERIC_DTYPES = (
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
    )

    def _apply_single_analyze_filter(self, f: dict[str, Any]) -> Filter | None:
        """Apply one filter dict to `self.df`; returns the resulting `Filter` record or None if skipped."""
        col = f.get("column")
        op = f.get("operator")
        val = f.get("value")

        if col not in self.columns:
            logger.warning("Skipping filter on unknown column %r; filter was not applied.", col)
            return None

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
        else:
            logger.warning(
                "Skipping unsupported filter operator %r for column %r "
                "(value=%r); filter was not applied.",
                op,
                col,
                val,
            )
            return None

        return Filter(column=str(col), operator=str(op), value=val)

    def _apply_analyze_filters(self, filters: list[dict[str, Any]] | None) -> list[Filter]:
        """Apply all user filters, mutating `self.df` / `self.lazy_df` / `self.row_count`."""
        active_filters: list[Filter] = []
        if filters:
            for f in filters:
                applied = self._apply_single_analyze_filter(f)
                if applied:
                    active_filters.append(applied)

            self.lazy_df = self.df.lazy()
            self.row_count = self.df.height
        return active_filters

    def _empty_profile(
        self, target_col: str | None, active_filters: list[Filter]
    ) -> DatasetProfile:
        """Build the placeholder profile returned when filters left 0 rows."""
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

    def _apply_column_exclusions(self, exclude_cols: list[str] | None) -> list[str]:
        """Drop `exclude_cols` from `self.columns`; returns the columns actually excluded."""
        excluded_columns: list[str] = []
        if exclude_cols:
            excluded_columns = [c for c in exclude_cols if c in self.columns]
            self.columns = [c for c in self.columns if c not in excluded_columns]
        return excluded_columns

    def _compute_frame_stats(self, stats_df: pl.DataFrame) -> tuple[float, int, float]:
        """Compute missing-cell percentage, duplicate row count, and memory usage (MB)."""
        missing_cells = stats_df.null_count().sum_horizontal()[0]
        total_cells = self.row_count * len(self.columns)
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0
        duplicate_rows = int(stats_df.is_duplicated().sum())
        memory_usage = self.df.estimated_size("mb")
        return missing_pct, duplicate_rows, memory_usage

    def _compute_basic_stats(self) -> dict:
        """Batched query 1: null_count + n_unique for every column (A3 optimization)."""
        basic_aggs = []
        for col in self.columns:
            basic_aggs.extend(
                [
                    pl.col(col).null_count().alias(f"{col}__null"),
                    pl.col(col).n_unique().alias(f"{col}__unique"),
                ]
            )
        basic_stats_df = _collect(self.lazy_df.select(basic_aggs))
        return basic_stats_df.row(0, named=True) if len(basic_stats_df) > 0 else {}

    @staticmethod
    def _int_semantic_type_from_ratio(ratio: float, n_unique: int) -> str:
        """Classify an integer column as Categorical (low-cardinality) or Numeric."""
        return "Categorical" if (ratio < 0.05 and n_unique < 20) else "Numeric"

    @staticmethod
    def _string_semantic_type_from_ratio(ratio: float) -> str:
        """Classify a string column as Categorical (low-cardinality) or Text."""
        return "Categorical" if ratio < 0.05 else "Text"

    def _semantic_type_for_column(self, dtype, ratio: float, n_unique: int) -> str:
        """Map a single column's dtype + cardinality ratio to a semantic bucket."""
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
            return self._int_semantic_type_from_ratio(ratio, n_unique)
        if dtype == pl.Boolean:
            return "Boolean"
        if dtype in [pl.Date, pl.Datetime, pl.Duration]:
            return "DateTime"
        if dtype in [pl.Utf8, pl.String]:
            return self._string_semantic_type_from_ratio(ratio)
        if str(dtype) == "Categorical":
            return "Categorical"
        return "Text"

    def _infer_semantic_types(self, basic_stats: dict) -> dict[str, str]:
        """Inline semantic-type inference (avoids re-fetching n_unique per column)."""
        semantic_types: dict[str, str] = {}
        for col in self.columns:
            dtype = self.df[col].dtype
            n_unique = basic_stats.get(f"{col}__unique", 0)
            ratio = n_unique / self.row_count if self.row_count > 0 else 0
            semantic_types[col] = self._semantic_type_for_column(dtype, ratio, n_unique)
        return semantic_types

    def _numeric_advanced_aggs(self, col: str) -> list[pl.Expr]:
        """Advanced aggregation expressions for a Numeric column."""
        return [
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

    def _categorical_advanced_aggs(self, col: str) -> list[pl.Expr]:
        """Advanced aggregation expressions for a Categorical column."""
        return [
            pl.col(col).value_counts(sort=True).head(10).implode().alias(f"{col}__top_k"),
            (pl.col(col).value_counts().struct.field("count") < 5).sum().alias(f"{col}__rare"),
        ]

    def _datetime_advanced_aggs(self, col: str) -> list[pl.Expr]:
        """Advanced aggregation expressions for a DateTime column."""
        return [
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).max().alias(f"{col}__max"),
        ]

    def _text_advanced_aggs(self, col: str) -> list[pl.Expr]:
        """Advanced aggregation expressions for a Text column."""
        return [
            pl.col(col).str.len_bytes().cast(pl.Float64).mean().alias(f"{col}__avg_len"),
            pl.col(col).str.len_bytes().min().alias(f"{col}__min_len"),
            pl.col(col).str.len_bytes().max().alias(f"{col}__max_len"),
        ]

    def _compute_advanced_stats(self, semantic_types: dict[str, str]) -> dict:
        """Batched query 2: type-specific advanced aggregations (mean/quantiles/value_counts/...)."""
        aggs_by_type = {
            "Numeric": self._numeric_advanced_aggs,
            "Categorical": self._categorical_advanced_aggs,
            "DateTime": self._datetime_advanced_aggs,
            "Text": self._text_advanced_aggs,
        }

        advanced_aggs: list[pl.Expr] = []
        for col in self.columns:
            build_aggs = aggs_by_type.get(semantic_types[col])
            if build_aggs:
                advanced_aggs.extend(build_aggs(col))

        advanced_stats: dict = {}
        if advanced_aggs:
            advanced_stats_df = _collect(self.lazy_df.select(advanced_aggs))
            if len(advanced_stats_df) > 0:
                advanced_stats = advanced_stats_df.row(0, named=True)
        return advanced_stats

    def _build_column_profiles(
        self, basic_stats: dict, advanced_stats: dict, semantic_types: dict[str, str]
    ) -> tuple[dict, list[Alert], list[str]]:
        """Build per-column profiles from the batched stats and collect the numeric-like columns.

        A column is included in `numeric_cols` if it's Numeric OR a
        Categorical-by-cardinality column whose underlying dtype is numeric
        (so PCA/causal still see it).
        """
        col_profiles = {}
        alerts: list[Alert] = []
        numeric_cols: list[str] = []

        for col in self.columns:
            profile, col_alerts = self._analyze_column(
                col, basic_stats, advanced_stats, semantic_types
            )
            col_profiles[col] = profile
            alerts.extend(col_alerts)

            is_numeric_type = self.df[col].dtype in self._NUMERIC_DTYPES
            if profile.dtype == "Numeric" or profile.dtype == "Categorical" and is_numeric_type:
                numeric_cols.append(col)

        return col_profiles, alerts, numeric_cols

    def _encode_target_if_needed(
        self, target_col: str | None, numeric_cols: list[str]
    ) -> str | None:
        """Encode string/boolean targets so they appear in causal graphs.

        `cast(pl.Categorical)` only accepts string-like columns directly; a
        Boolean target (a common binary-classification target dtype) must
        be cast to Utf8 first or polars raises InvalidOperationError.
        """
        if not (target_col and target_col in self.columns and target_col not in numeric_cols):
            return None

        encoded_target = f"{target_col}_encoded"
        target_expr = pl.col(target_col)
        if self.df.schema[target_col] == pl.Boolean:  # type: ignore[attr-defined]
            target_expr = target_expr.cast(pl.Utf8)
        self.df = self.df.with_columns(
            target_expr.cast(pl.Categorical).to_physical().alias(encoded_target)
        )
        self.lazy_df = self.df.lazy()
        return encoded_target

    def _add_vif_alerts(self, vif_data: dict | None, alerts: list[Alert]) -> None:
        """Flag features with high/very-high variance inflation factor (multicollinearity)."""
        if not vif_data:
            return
        for col, val in vif_data.items():
            if val > 10.0:
                alerts.append(
                    Alert(
                        type="Multicollinearity",
                        message=(
                            f"Column '{col}' has very high VIF ({val:.1f}). Consider removing it."
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

    def _compute_target_correlation_matrix(
        self,
        target_col: str | None,
        feature_cols: list[str],
        numeric_cols: list[str],
        encoded_target_col: str | None,
    ):
        """Compute the separate feature-vs-target correlation matrix, if applicable."""
        target_corr_cols: list[str] = []
        if target_col:
            if target_col in numeric_cols:
                target_corr_cols = feature_cols + [target_col]
            elif encoded_target_col:
                target_corr_cols = feature_cols + [encoded_target_col]
        if len(target_corr_cols) >= 2:
            return calculate_correlations(self.lazy_df, target_corr_cols)
        return None

    def _add_leakage_alerts(
        self, target_col: str, target_correlations: dict[str, float], alerts: list[Alert]
    ) -> None:
        """Flag features suspiciously highly correlated with a numeric target (possible leakage)."""
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

    def _compute_numeric_target_analytics(
        self,
        target_col: str,
        feature_cols: list[str],
        numeric_cols: list[str],
        alerts: list[Alert],
    ) -> tuple[dict[str, float], Any]:
        """Correlations, leakage alerts, and categorical interactions for a numeric target."""
        target_correlations: dict[str, float] = {}
        target_interactions = None
        if target_col not in numeric_cols:
            return target_correlations, target_interactions

        target_correlations = self._calculate_target_correlations(target_col, feature_cols)
        self._add_leakage_alerts(target_col, target_correlations, alerts)

        cat_cols = [
            c
            for c in self.columns
            if self._get_semantic_type(self.df[c]) == "Categorical" and c != target_col
        ]
        if cat_cols:
            target_interactions = self._calculate_target_interactions(
                target_col, cat_cols, is_target_numeric=True
            )
        return target_correlations, target_interactions

    def _compute_categorical_target_analytics(
        self, target_col: str, feature_cols: list[str]
    ) -> tuple[dict[str, float], Any]:
        """Associations and interactions for a Categorical/Boolean target."""
        target_correlations = self._calculate_categorical_target_associations(
            target_col, feature_cols
        )
        target_interactions = None
        top_features = list(target_correlations.keys())
        if top_features:
            target_interactions = self._calculate_target_interactions(
                target_col, top_features, is_target_numeric=False
            )
        return target_correlations, target_interactions

    def _compute_target_relationship_analytics(
        self,
        target_col: str | None,
        feature_cols: list[str],
        numeric_cols: list[str],
        alerts: list[Alert],
    ) -> tuple[dict[str, float], Any]:
        """Dispatch target-relationship analytics (correlations / interactions / leakage) by dtype."""
        target_correlations: dict[str, float] = {}
        target_interactions = None
        if not (target_col and target_col in self.columns):
            return target_correlations, target_interactions

        target_semantic_type = self._get_semantic_type(  # pylint: disable=assignment-from-no-return
            self.df[target_col]
        )

        if target_semantic_type == "Numeric":
            target_correlations, target_interactions = self._compute_numeric_target_analytics(
                target_col, feature_cols, numeric_cols, alerts
            )
        elif target_semantic_type in ("Categorical", "Boolean"):
            target_correlations, target_interactions = self._compute_categorical_target_analytics(
                target_col, feature_cols
            )

        return target_correlations, target_interactions

    def _add_high_missing_alert(self, missing_pct: float, alerts: list[Alert]) -> None:
        """Flag datasets that are more than 50% empty."""
        if missing_pct > 50:
            alerts.append(
                Alert(
                    type="High Null",
                    message=f"Dataset is {missing_pct:.1f}% empty.",
                    severity="warning",
                )
            )

    def _compute_multivariate(
        self, feature_cols: list[str], numeric_cols: list[str], target_col: str | None
    ) -> tuple[Any, Any, Any, Any]:
        """PCA, outlier detection, and clustering (all gated by scikit-learn availability)."""
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

        return pca_data, pca_components, outliers, clustering

    def _compute_causal_graph(self, numeric_cols: list[str], encoded_target_col: str | None):
        """Run causal discovery, including the encoded target so it shows in the graph."""
        causal_cols = numeric_cols.copy()
        if encoded_target_col:
            causal_cols.append(encoded_target_col)
        if len(causal_cols) >= 2:
            return self._discover_causal_graph(causal_cols)
        return None

    def _compute_rule_tree(
        self, feature_cols: list[str], target_col: str | None, task_type: str | None
    ) -> tuple[Any, str | None]:
        """Decision-tree surrogate rule discovery, plus the inferred task type."""
        rule_tree = None
        final_task_type = task_type
        if SKLEARN_AVAILABLE and target_col and len(feature_cols) >= 1:
            target_type = self._get_semantic_type(  # pylint: disable=assignment-from-no-return
                self.df[target_col]
            )
            if target_type in ["Categorical", "Boolean", "Numeric"]:
                rule_tree = self._discover_rules(feature_cols, target_col, task_type)
                if not final_task_type:
                    final_task_type = "Regression" if target_type == "Numeric" else "Classification"
        return rule_tree, final_task_type

    def analyze(
        self,
        target_col: str | None = None,
        exclude_cols: list[str] | None = None,
        filters: list[dict[str, Any]] | None = None,
        date_col: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
        task_type: str | None = None,
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
        active_filters = self._apply_analyze_filters(filters)

        if self.row_count == 0:
            return self._empty_profile(target_col, active_filters)

        excluded_columns = self._apply_column_exclusions(exclude_cols)

        # Narrowed view used for frame-level stats/sample so excluded columns
        # (e.g. PII) never leak into missing/duplicate counts or sample_data.
        stats_df = self.df.select(self.columns) if excluded_columns else self.df

        # 2. Frame-level stats.
        missing_pct, duplicate_rows, memory_usage = self._compute_frame_stats(stats_df)

        # -- A3: BATCHED PER-COLUMN AGGREGATIONS --
        # Two single-pass polars queries replace ~N python-level "select(col)"
        # calls per column. Wins ~3–10× on wide frames.
        basic_stats = self._compute_basic_stats()
        semantic_types = self._infer_semantic_types(basic_stats)
        advanced_stats = self._compute_advanced_stats(semantic_types)

        # Build per-column profiles from the batched stats.
        col_profiles, alerts, numeric_cols = self._build_column_profiles(
            basic_stats, advanced_stats, semantic_types
        )

        encoded_target_col = self._encode_target_if_needed(target_col, numeric_cols)

        # Feature columns = numeric cols minus the target itself.
        feature_cols = [c for c in numeric_cols if c != target_col]

        # 3. Correlations + VIF.
        correlations = calculate_correlations(self.lazy_df, feature_cols)

        vif_data = self._calculate_vif(feature_cols)
        self._add_vif_alerts(vif_data, alerts)

        # 3a. Feature-vs-target correlations (separate matrix).
        correlations_with_target = self._compute_target_correlation_matrix(
            target_col, feature_cols, numeric_cols, encoded_target_col
        )

        # 3b. Target-relationship analytics (correlations / interactions / leakage).
        target_correlations, target_interactions = self._compute_target_relationship_analytics(
            target_col, feature_cols, numeric_cols, alerts
        )

        # 4. Frame-level alerts.
        self._add_high_missing_alert(missing_pct, alerts)

        # 5. Sample (used for FE scatter plots).
        sample_rows = stats_df.head(5000).to_dicts()

        # 6. Multivariate.
        pca_data, pca_components, outliers, clustering = self._compute_multivariate(
            feature_cols, numeric_cols, target_col
        )

        # 7-8. Geo + time series.
        geospatial = self._analyze_geospatial(numeric_cols, target_col, lat_col, lon_col)
        timeseries = self._analyze_timeseries(numeric_cols, target_col, date_col)

        # 9. Causal discovery (include encoded target so it shows in the graph).
        causal_graph = self._compute_causal_graph(numeric_cols, encoded_target_col)

        # 10. Rule discovery (decision-tree surrogate).
        rule_tree, final_task_type = self._compute_rule_tree(feature_cols, target_col, task_type)

        # 11. Recommendations.
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
            active_filters=active_filters,
        )
