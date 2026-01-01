from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    NUMERIC = "Numeric"
    CATEGORICAL = "Categorical"
    BOOLEAN = "Boolean"
    DATETIME = "DateTime"
    TEXT = "Text"
    UNSUPPORTED = "Unsupported"


class AlertType(str, Enum):
    HIGH_CORRELATION = "HighCorrelation"
    HIGH_CARDINALITY = "HighCardinality"
    CONSTANT = "Constant"
    IMBALANCE = "Imbalance"
    SKEWNESS = "Skewness"
    MISSING_DATA = "MissingData"
    DUPLICATES = "Duplicates"


class Alert(BaseModel):
    type: AlertType
    columns: List[str]
    details: str
    severity: str = "Medium"  # Low, Medium, High


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class ValueFrequency(BaseModel):
    value: str
    count: int
    percentage: float


class NumericStats(BaseModel):
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q05: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    q95: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    zeros: Optional[int] = None
    negatives: Optional[int] = None
    outliers: Optional[int] = None  # Count of outliers (1.5 IQR)
    histogram: List[HistogramBin] = Field(default_factory=list)


class CategoricalStats(BaseModel):
    unique_count: int
    top_values: List[ValueFrequency] = Field(default_factory=list)


class DateTimeStats(BaseModel):
    min: Optional[str] = None
    max: Optional[str] = None
    range_days: Optional[float] = None


class VariableProfile(BaseModel):
    name: str
    type: VariableType
    count: int
    missing_count: int
    missing_percentage: float
    memory_usage: Optional[int] = None  # bytes
    
    # Type specific stats
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    datetime_stats: Optional[DateTimeStats] = None


class CorrelationCell(BaseModel):
    x: str
    y: str
    value: float


class CorrelationMatrix(BaseModel):
    method: str  # pearson, spearman
    cells: List[CorrelationCell]


class ScatterSample(BaseModel):
    x: str
    y: str
    data: List[Dict[str, Any]]  # [{"x": 1, "y": 2}, ...]


class DatasetMeta(BaseModel):
    rows: int
    cols: int
    memory_usage: int  # bytes
    duplicate_rows: int
    duplicate_percentage: float
    variable_types: Dict[str, int]  # {"Numeric": 5, "Categorical": 2}
    quality_score: float = Field(..., ge=0, le=100)  # 0-100 score


class DatasetProfile(BaseModel):
    meta: DatasetMeta
    alerts: List[Alert] = Field(default_factory=list)
    variables: Dict[str, VariableProfile]
    correlations: Dict[str, CorrelationMatrix] = Field(default_factory=dict)
    scatter_samples: List[ScatterSample] = Field(default_factory=list)
