from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from enum import Enum

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    column_type: ColumnType
    missing_count: int
    missing_ratio: float
    unique_count: int
    
    # Numeric specific
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    skewness: Optional[float] = None
    
    # Categorical specific
    top_values: Optional[Dict[str, int]] = None # Top K values and counts
    avg_text_length: Optional[float] = None

class AnalysisProfile(BaseModel):
    row_count: int
    column_count: int
    duplicate_row_count: int = 0
    columns: Dict[str, ColumnProfile]
    
class RecommendationType(str, Enum):
    IMPUTATION = "imputation"
    SCALING = "scaling"
    ENCODING = "encoding"
    OUTLIER_REMOVAL = "outlier_removal"
    FEATURE_SELECTION = "feature_selection"
    TRANSFORMATION = "transformation"
    CLEANING = "cleaning"

class Recommendation(BaseModel):
    rule_id: str
    type: RecommendationType
    target_columns: List[str]
    description: str
    suggested_node_type: str # e.g., "SimpleImputer"
    suggested_params: Dict[str, Any]
    confidence: float # 0.0 to 1.0
    reasoning: str
