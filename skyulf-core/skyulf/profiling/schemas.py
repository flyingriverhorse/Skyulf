from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class NumericStats(BaseModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zeros_count: Optional[int] = None
    negatives_count: Optional[int] = None
    normality_test: Optional[Dict[str, Any]] = None

class CategoricalStats(BaseModel):
    unique_count: int
    top_k: List[Dict[str, Any]] = Field(default_factory=list)  # [{"value": "A", "count": 10}, ...]
    rare_labels_count: int = 0

class DateStats(BaseModel):
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    duration_days: Optional[float] = None

class TextStats(BaseModel):
    avg_length: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    common_words: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_distribution: Optional[Dict[str, float]] = None # {"positive": 0.6, "neutral": 0.3, "negative": 0.1}

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
    type: str # "directed", "undirected", "bidirected"

class CausalGraph(BaseModel):
    nodes: List[CausalNode]
    edges: List[CausalEdge]

class RuleNode(BaseModel):
    id: int
    feature: Optional[str] = None
    threshold: Optional[float] = None
    impurity: float
    samples: int
    value: List[float] # Class distribution
    class_name: Optional[str] = None # Predicted class
    is_leaf: bool
    children: List[int] = Field(default_factory=list) # IDs of children

class RuleTree(BaseModel):
    nodes: List[RuleNode]
    accuracy: Optional[float] = None # Surrogate model accuracy
    rules: Optional[List[str]] = None # Human readable rules
    feature_importances: Optional[List[Dict[str, Union[str, float]]]] = None # Feature importance from surrogate model

class ColumnProfile(BaseModel):
    name: str
    dtype: str  # "Numeric", "Categorical", "Boolean", "DateTime", "Text"
    missing_count: int
    missing_percentage: float
    
    # Type-specific stats
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    date_stats: Optional[DateStats] = None
    text_stats: Optional[TextStats] = None
    
    # Distribution
    histogram: Optional[List[HistogramBin]] = None
    normality_test: Optional[NormalityTestResult] = None
    
    # Quality
    is_constant: bool = False
    is_unique: bool = False  # Possible ID

class CorrelationMatrix(BaseModel):
    columns: List[str]
    values: List[List[float]]  # 2D array

class ScatterSample(BaseModel):
    x: str
    y: str
    data: List[Dict[str, Any]]  # [{"x": 1, "y": 2}, ...]

class Alert(BaseModel):
    column: Optional[str] = None
    type: str  # "High Null", "Constant", "High Cardinality", "Leakage", "Outlier"
    message: str
    severity: str = "warning"  # "info", "warning", "error"

class Recommendation(BaseModel):
    column: Optional[str] = None
    action: str # "Drop", "Impute", "Transform", "Encode"
    reason: str
    suggestion: str

class PCAComponent(BaseModel):
    component: str # "PC1", "PC2", "PC3"
    explained_variance_ratio: float
    top_features: Dict[str, float] # feature_name -> weight/loading

class PCAPoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    label: Optional[str] = None # For target coloring

class GeoPoint(BaseModel):
    lat: float
    lon: float
    label: Optional[str] = None

class GeospatialStats(BaseModel):
    lat_col: str
    lon_col: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    centroid_lat: float
    centroid_lon: float
    sample_points: List[GeoPoint]

class TimeSeriesPoint(BaseModel):
    date: str
    values: Dict[str, float]

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
    plot_type: str # "boxplot"
    data: List[CategoryBoxPlot]
    p_value: Optional[float] = None # ANOVA p-value

class SeasonalityStats(BaseModel):
    day_of_week: List[Dict[str, Any]]
    month_of_year: List[Dict[str, Any]]

class TimeSeriesAnalysis(BaseModel):
    date_col: str
    trend: List[TimeSeriesPoint]
    seasonality: SeasonalityStats
    autocorrelation: Optional[List[Dict[str, Any]]] = None
    stationarity_test: Optional[Dict[str, Any]] = None

class OutlierPoint(BaseModel):
    index: int
    values: Dict[str, Any] # Key values for context
    score: float # Anomaly score (lower is more anomalous for IF, or distance for others)
    explanation: Optional[List[Dict[str, Any]]] = None # [{"feature": "Age", "value": 95, "mean": 35, "diff": 60}, ...]

class OutlierAnalysis(BaseModel):
    method: str # "IsolationForest" or "IQR"
    total_outliers: int
    outlier_percentage: float
    top_outliers: List[OutlierPoint]
    plot_data: Optional[List[Dict[str, Any]]] = None # For visualization (e.g. PCA projection of outliers)

class ClusteringPoint(BaseModel):
    x: float
    y: float
    cluster: int
    label: Optional[str] = None

class ClusterStats(BaseModel):
    cluster_id: int
    size: int
    percentage: float
    center: Dict[str, float]

class ClusteringAnalysis(BaseModel):
    method: str = "KMeans"
    n_clusters: int
    inertia: float
    clusters: List[ClusterStats]
    points: List[ClusteringPoint]

class Filter(BaseModel):
    column: str
    operator: str # "==", "!=", ">", "<", ">=", "<=", "in"
    value: Any

class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_cells_percentage: float
    memory_usage_mb: float
    
    columns: Dict[str, ColumnProfile]
    correlations: Optional[CorrelationMatrix] = None
    correlations_with_target: Optional[CorrelationMatrix] = None
    alerts: List[Alert] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    sample_data: Optional[List[Dict[str, Any]]] = None
    
    # Target Analysis
    target_col: Optional[str] = None
    task_type: Optional[str] = None # "Classification" or "Regression"
    target_correlations: Optional[Dict[str, float]] = None
    target_interactions: Optional[List[TargetInteraction]] = None
    
    # Multivariate
    pca_data: Optional[List[PCAPoint]] = None
    pca_components: Optional[List[PCAComponent]] = None
    outliers: Optional[OutlierAnalysis] = None
    clustering: Optional[ClusteringAnalysis] = None
    causal_graph: Optional[CausalGraph] = None
    rule_tree: Optional[RuleTree] = None
    vif: Optional[Dict[str, float]] = None # Variance Inflation Factor for numeric columns
    
    # Geospatial
    geospatial: Optional[GeospatialStats] = None
    
    # Time Series
    timeseries: Optional[TimeSeriesAnalysis] = None
    
    # Metadata
    excluded_columns: List[str] = Field(default_factory=list)
    active_filters: Optional[List[Filter]] = None
    
    generated_at: datetime = Field(default_factory=datetime.now)
