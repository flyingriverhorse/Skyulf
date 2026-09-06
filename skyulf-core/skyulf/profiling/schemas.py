"""Pydantic contract for exploratory-data-analysis (EDA) output.

These models are what the EDA route serves to the frontend and what the backend
persists into the report's JSON column, so every field name is public API:
renaming one is a breaking change, not a refactor. Numeric statistics are typed
:data:`FiniteFloat`, which rewrites NaN/inf to ``None`` before validation so no
downstream JSON consumer ever has to parse a bare ``NaN`` token.
"""

import math
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


def _non_finite_to_none(value: Any) -> Any:
    """Map NaN/inf to ``None`` so a profile never serializes a non-finite float.

    ``orjson`` (the EDA route) coerces NaN to ``null`` on its own, but the
    stdlib-JSON paths do not: ``model_dump(mode="json")`` keeps the Python
    ``nan``, ``json.dumps`` then emits a bare ``NaN`` token — invalid JSON that
    the browser's ``JSON.parse`` rejects — and SQLAlchemy writes that straight
    into the EDA report's JSON column. A non-finite stat means "not
    computable", which ``None`` already expresses throughout these schemas.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


FiniteFloat = Annotated[float | None, BeforeValidator(_non_finite_to_none)]
"""A ``float | None`` that rejects NaN/inf by rewriting it to ``None``."""


class NumericStats(BaseModel):
    """Central-tendency, spread and shape statistics for a numeric column."""

    mean: FiniteFloat = None
    median: FiniteFloat = None
    std: FiniteFloat = None
    variance: FiniteFloat = None
    min: FiniteFloat = None
    max: FiniteFloat = None
    q25: FiniteFloat = None
    q75: FiniteFloat = None
    skewness: FiniteFloat = None
    kurtosis: FiniteFloat = None
    zeros_count: int | None = None
    negatives_count: int | None = None
    normality_test: dict[str, Any] | None = None


class CategoricalStats(BaseModel):
    """Cardinality summary for a categorical column, plus its most frequent labels."""

    unique_count: int
    top_k: list[dict[str, Any]] = Field(default_factory=list)  # [{"value": "A", "count": 10}, ...]
    rare_labels_count: int = 0


class DateStats(BaseModel):
    """Coverage window of a datetime column: earliest, latest and the span in days."""

    min_date: str | None = None
    max_date: str | None = None
    duration_days: float | None = None


class TextStats(BaseModel):
    """Length, vocabulary and sentiment summary for a free-text column."""

    avg_length: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    common_words: list[dict[str, Any]] = Field(default_factory=list)
    sentiment_distribution: dict[str, float] | None = (
        None  # {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
    )


class HistogramBin(BaseModel):
    """One equal-width histogram bucket and the number of values falling inside it."""

    start: float
    end: float
    count: int


class NormalityTestResult(BaseModel):
    """Outcome of a distribution-normality test and the verdict drawn from it."""

    test_name: str
    statistic: float
    p_value: float
    is_normal: bool


class CausalNode(BaseModel):
    """One variable in a discovered causal graph."""

    id: str
    label: str


class CausalEdge(BaseModel):
    """One relationship between two causal nodes; ``type`` records its orientation."""

    source: str
    target: str
    type: str  # "directed", "undirected", "bidirected"


class CausalGraph(BaseModel):
    """Causal structure discovered between the profiled columns."""

    nodes: list[CausalNode]
    edges: list[CausalEdge]


class RuleNode(BaseModel):
    """One node of the surrogate decision tree fitted to explain the data."""

    id: int
    feature: str | None = None
    threshold: float | None = None
    impurity: float
    samples: int
    value: list[float]  # Class distribution
    class_name: str | None = None  # Predicted class
    is_leaf: bool
    children: list[int] = Field(default_factory=list)  # IDs of children


class RuleTree(BaseModel):
    """Human-readable decision rules extracted from a surrogate tree.

    A categorical feature's node ``threshold`` is an ordinal split point over
    ``categories`` (physical code ``i`` == ``categories[feature][i]``), not a
    meaningful magnitude — consumers must render ``"feature in [...]"`` through
    that mapping instead of ``"feature <= <code>"``.
    """

    nodes: list[RuleNode]
    accuracy: float | None = None  # Surrogate model accuracy
    rules: list[str] | None = None  # Human readable rules
    feature_importances: list[dict[str, str | float]] | None = (
        None  # Feature importance from surrogate model
    )
    # For categorical features, maps feature name -> ordered category labels
    # (physical code i == categories[feature][i]). A node's numeric
    # `threshold` on such a feature is an internal ordinal-encoding split
    # point, not a meaningful magnitude — consumers should use this mapping
    # to render "feature in [...]" instead of "feature <= <code>".
    categories: dict[str, list[str]] | None = None


class ColumnProfile(BaseModel):
    """Everything the profiler determined about one column, grouped by its detected type."""

    name: str
    dtype: str  # "Numeric", "Categorical", "Boolean", "DateTime", "Text"
    missing_count: int
    missing_percentage: float

    # Type-specific stats
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    date_stats: DateStats | None = None
    text_stats: TextStats | None = None

    # Distribution
    histogram: list[HistogramBin] | None = None
    normality_test: NormalityTestResult | None = None

    # Quality
    is_constant: bool = False
    is_unique: bool = False  # Possible ID


class CorrelationMatrix(BaseModel):
    """Symmetric Pearson matrix over the numeric columns, indexed by ``columns``.

    ``0.0`` doubles for "not computable": the producer emits it both for NaN
    coefficients and for pairs sharing too few observations to yield a
    non-degenerate value, because this model has no "unknown" cell.
    """

    columns: list[str]
    values: list[list[float]]  # 2D array


class ScatterSample(BaseModel):
    """Point pairs for the two named columns, ready to render as a scatter plot."""

    x: str
    y: str
    data: list[dict[str, Any]]  # [{"x": 1, "y": 2}, ...]


class Alert(BaseModel):
    """A data-quality problem the profiler flagged, with its severity."""

    column: str | None = None
    type: str  # "High Null", "Constant", "High Cardinality", "Leakage", "Outlier"
    message: str
    severity: str = "warning"  # "info", "warning", "error"


class Recommendation(BaseModel):
    """A suggested remediation for one column, with the reasoning behind it."""

    column: str | None = None
    action: str  # "Drop", "Impute", "Transform", "Encode"
    reason: str
    suggestion: str


class PCAComponent(BaseModel):
    """One principal component: its explained variance and heaviest feature loadings."""

    component: str  # "PC1", "PC2", "PC3"
    explained_variance_ratio: float
    top_features: dict[str, float]  # feature_name -> weight/loading


class PCAPoint(BaseModel):
    """A row projected into principal-component space for the scatter view."""

    x: float
    y: float
    z: float | None = None
    label: str | None = None  # For target coloring


class GeoPoint(BaseModel):
    """A latitude/longitude pair, optionally labelled with its target value."""

    lat: float
    lon: float
    label: str | None = None


class GeospatialStats(BaseModel):
    """Bounding box and centroid of a detected latitude/longitude column pair."""

    lat_col: str
    lon_col: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    centroid_lat: float
    centroid_lon: float
    sample_points: list[GeoPoint]


class TimeSeriesPoint(BaseModel):
    """One timestamp and the metric values observed at it."""

    date: str
    values: dict[str, float]


class BoxPlotStats(BaseModel):
    """Five-number summary used to draw one box plot."""

    min: float
    q1: float
    median: float
    q3: float
    max: float


class CategoryBoxPlot(BaseModel):
    """Box plot for a single category level of a feature."""

    name: str
    stats: BoxPlotStats


class TargetInteraction(BaseModel):
    """How a feature's distribution varies across the target, with its ANOVA p-value."""

    feature: str
    plot_type: str  # "boxplot"
    data: list[CategoryBoxPlot]
    p_value: float | None = None  # ANOVA p-value


class SeasonalityStats(BaseModel):
    """Time-series aggregates by weekday and by month, to expose seasonal shape."""

    day_of_week: list[dict[str, Any]]
    month_of_year: list[dict[str, Any]]


class TimeSeriesAnalysis(BaseModel):
    """Trend, seasonality and stationarity summary for a detected date column."""

    date_col: str
    trend: list[TimeSeriesPoint]
    seasonality: SeasonalityStats
    autocorrelation: list[dict[str, Any]] | None = None
    stationarity_test: dict[str, Any] | None = None


class OutlierPoint(BaseModel):
    """One anomalous row: its index, key values, anomaly score and per-feature reasons."""

    index: int
    values: dict[str, Any]  # Key values for context
    score: float  # Anomaly score (lower is more anomalous for IF, or distance for others)
    explanation: list[dict[str, Any]] | None = (
        None  # [{"feature": "Age", "value": 95, "mean": 35, "diff": 60}, ...]
    )


class OutlierAnalysis(BaseModel):
    """Result of an outlier sweep: which method ran, and what it flagged."""

    method: str  # "IsolationForest" or "IQR"
    total_outliers: int
    outlier_percentage: float
    top_outliers: list[OutlierPoint]
    plot_data: list[dict[str, Any]] | None = (
        None  # For visualization (e.g. PCA projection of outliers)
    )


class ClusteringPoint(BaseModel):
    """A row projected to 2D and tagged with its assigned cluster."""

    x: float
    y: float
    cluster: int
    label: str | None = None


class ClusterStats(BaseModel):
    """Size and centroid of one discovered cluster."""

    cluster_id: int
    size: int
    percentage: float
    center: dict[str, float]


class ClusteringAnalysis(BaseModel):
    """Unsupervised segment structure found across the profiled rows."""

    method: str = "KMeans"
    n_clusters: int
    inertia: float
    clusters: list[ClusterStats]
    points: list[ClusteringPoint]


class Filter(BaseModel):
    """One row-level filter applied to the dataset before profiling."""

    column: str
    operator: str  # "==", "!=", ">", "<", ">=", "<=", "in"
    value: Any


class DatasetProfile(BaseModel):
    """Root EDA payload: dataset-level facts, per-column profiles and optional analyses."""

    row_count: int
    column_count: int
    duplicate_rows: int
    missing_cells_percentage: float
    memory_usage_mb: float

    columns: dict[str, ColumnProfile]
    correlations: CorrelationMatrix | None = None
    correlations_with_target: CorrelationMatrix | None = None
    alerts: list[Alert] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    sample_data: list[dict[str, Any]] | None = None

    # Target Analysis
    target_col: str | None = None
    task_type: str | None = None  # "Classification" or "Regression"
    target_correlations: dict[str, float] | None = None
    target_interactions: list[TargetInteraction] | None = None

    # Multivariate
    pca_data: list[PCAPoint] | None = None
    pca_components: list[PCAComponent] | None = None
    outliers: OutlierAnalysis | None = None
    clustering: ClusteringAnalysis | None = None
    causal_graph: CausalGraph | None = None
    rule_tree: RuleTree | None = None
    vif: dict[str, float] | None = None  # Variance Inflation Factor for numeric columns

    # Geospatial
    geospatial: GeospatialStats | None = None

    # Time Series
    timeseries: TimeSeriesAnalysis | None = None

    # Metadata
    excluded_columns: list[str] = Field(default_factory=list)
    active_filters: list[Filter] | None = None

    generated_at: datetime = Field(default_factory=datetime.now)
