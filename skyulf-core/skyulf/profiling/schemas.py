from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class NumericStats(BaseModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zeros_count: Optional[int] = None
    negatives_count: Optional[int] = None

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

class HistogramBin(BaseModel):
    start: float
    end: float
    count: int

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

class PCAPoint(BaseModel):
    x: float
    y: float
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

class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_cells_percentage: float
    memory_usage_mb: float
    
    columns: Dict[str, ColumnProfile]
    correlations: Optional[CorrelationMatrix] = None
    alerts: List[Alert] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    sample_data: Optional[List[Dict[str, Any]]] = None
    
    # Target Analysis
    target_col: Optional[str] = None
    target_correlations: Optional[Dict[str, float]] = None
    
    # Multivariate
    pca_data: Optional[List[PCAPoint]] = None
    
    # Geospatial
    geospatial: Optional[GeospatialStats] = None
    
    generated_at: datetime = Field(default_factory=datetime.now)
