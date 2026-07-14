from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NumericStats(BaseModel):
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    variance: float | None = None
    min: float | None = None
    max: float | None = None
    q25: float | None = None
    q75: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros_count: int | None = None
    negatives_count: int | None = None
    normality_test: dict[str, Any] | None = None


class CategoricalStats(BaseModel):
    unique_count: int
    top_k: list[dict[str, Any]] = Field(default_factory=list)  # [{"value": "A", "count": 10}, ...]
    rare_labels_count: int = 0


class DateStats(BaseModel):
    min_date: str | None = None
    max_date: str | None = None
    duration_days: float | None = None


class TextStats(BaseModel):
    avg_length: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    common_words: list[dict[str, Any]] = Field(default_factory=list)
    sentiment_distribution: dict[str, float] | None = (
        None  # {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
    )


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class NormalityTestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool


class CausalNode(BaseModel):
    id: str
    label: str


class CausalEdge(BaseModel):
    source: str
    target: str
    type: str  # "directed", "undirected", "bidirected"


class CausalGraph(BaseModel):
    nodes: list[CausalNode]
    edges: list[CausalEdge]


class RuleNode(BaseModel):
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
    columns: list[str]
    values: list[list[float]]  # 2D array


class ScatterSample(BaseModel):
    x: str
    y: str
    data: list[dict[str, Any]]  # [{"x": 1, "y": 2}, ...]


class Alert(BaseModel):
    column: str | None = None
    type: str  # "High Null", "Constant", "High Cardinality", "Leakage", "Outlier"
    message: str
    severity: str = "warning"  # "info", "warning", "error"


class Recommendation(BaseModel):
    column: str | None = None
    action: str  # "Drop", "Impute", "Transform", "Encode"
    reason: str
    suggestion: str


class PCAComponent(BaseModel):
    component: str  # "PC1", "PC2", "PC3"
    explained_variance_ratio: float
    top_features: dict[str, float]  # feature_name -> weight/loading


class PCAPoint(BaseModel):
    x: float
    y: float
    z: float | None = None
    label: str | None = None  # For target coloring


class GeoPoint(BaseModel):
    lat: float
    lon: float
    label: str | None = None


class GeospatialStats(BaseModel):
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
    date: str
    values: dict[str, float]


class BoxPlotStats(BaseModel):
    min: float
    q1: float
    median: float
    q3: float
    max: float


class CategoryBoxPlot(BaseModel):
    name: str
    stats: BoxPlotStats


class TargetInteraction(BaseModel):
    feature: str
    plot_type: str  # "boxplot"
    data: list[CategoryBoxPlot]
    p_value: float | None = None  # ANOVA p-value


class SeasonalityStats(BaseModel):
    day_of_week: list[dict[str, Any]]
    month_of_year: list[dict[str, Any]]


class TimeSeriesAnalysis(BaseModel):
    date_col: str
    trend: list[TimeSeriesPoint]
    seasonality: SeasonalityStats
    autocorrelation: list[dict[str, Any]] | None = None
    stationarity_test: dict[str, Any] | None = None


class OutlierPoint(BaseModel):
    index: int
    values: dict[str, Any]  # Key values for context
    score: float  # Anomaly score (lower is more anomalous for IF, or distance for others)
    explanation: list[dict[str, Any]] | None = (
        None  # [{"feature": "Age", "value": 95, "mean": 35, "diff": 60}, ...]
    )


class OutlierAnalysis(BaseModel):
    method: str  # "IsolationForest" or "IQR"
    total_outliers: int
    outlier_percentage: float
    top_outliers: list[OutlierPoint]
    plot_data: list[dict[str, Any]] | None = (
        None  # For visualization (e.g. PCA projection of outliers)
    )


class ClusteringPoint(BaseModel):
    x: float
    y: float
    cluster: int
    label: str | None = None


class ClusterStats(BaseModel):
    cluster_id: int
    size: int
    percentage: float
    center: dict[str, float]


class ClusteringAnalysis(BaseModel):
    method: str = "KMeans"
    n_clusters: int
    inertia: float
    clusters: list[ClusterStats]
    points: list[ClusteringPoint]


class Filter(BaseModel):
    column: str
    operator: str  # "==", "!=", ">", "<", ">=", "<=", "in"
    value: Any


class DatasetProfile(BaseModel):
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
