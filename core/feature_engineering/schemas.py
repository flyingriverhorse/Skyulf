"""Pydantic schemas for feature engineering pipelines."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, computed_field, field_validator, model_validator

from core.feature_engineering.modeling.hyperparameter_tuning.registry import (
    get_default_strategy_value,
    normalize_strategy_value,
)
from core.utils.datetime import utcnow


FullExecutionJobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class DropColumnCandidate(BaseModel):
    """Column candidate recommended for removal."""

    name: str
    reason: str
    missing_percentage: Optional[float] = None
    priority: Optional[str] = None
    signals: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class DropColumnRecommendationFilter(BaseModel):
    """Filter metadata for recommendation categories."""

    id: str
    label: str
    description: Optional[str] = None
    count: int = 0


class DropColumnRecommendations(BaseModel):
    """Response payload for drop column recommendations."""

    dataset_source_id: str
    suggested_threshold: Optional[float] = None
    candidates: List[DropColumnCandidate] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    available_filters: List[DropColumnRecommendationFilter] = Field(default_factory=list)
    all_columns: List[str] = Field(default_factory=list)
    column_missing_map: Dict[str, float] = Field(default_factory=dict)


class LabelEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for label encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    score: float = 0.0
    selectable: bool = True


class LabelEncodingRecommendationsResponse(BaseModel):
    """Response payload for label encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    auto_detect_default: bool = False
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    columns: List[LabelEncodingColumnSuggestion] = Field(default_factory=list)


class OrdinalEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for ordinal encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    score: float = 0.0
    selectable: bool = True
    recommended_handle_unknown: bool = False


class OrdinalEncodingRecommendationsResponse(BaseModel):
    """Response payload for ordinal encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    auto_detect_default: bool = False
    enable_unknown_default: bool = False
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    columns: List[OrdinalEncodingColumnSuggestion] = Field(default_factory=list)


class TargetEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for target encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    score: float = 0.0
    selectable: bool = True
    recommended_use_global_fallback: bool = False


class TargetEncodingRecommendationsResponse(BaseModel):
    """Response payload for target encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    auto_detect_default: bool = False
    enable_global_fallback_default: bool = False
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    columns: List[TargetEncodingColumnSuggestion] = Field(default_factory=list)


class HashEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for hash encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    score: float = 0.0
    selectable: bool = True
    recommended_bucket_count: int = 0


class HashEncodingRecommendationsResponse(BaseModel):
    """Response payload for hash encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    auto_detect_default: bool = False
    suggested_bucket_default: int = 0
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    columns: List[HashEncodingColumnSuggestion] = Field(default_factory=list)


class OneHotEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for one-hot encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    estimated_dummy_columns: int = 0
    score: float = 0.0
    selectable: bool = True
    recommended_drop_first: bool = False


class OneHotEncodingRecommendationsResponse(BaseModel):
    """Response payload for one-hot encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    cautioned_count: int = 0
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    columns: List[OneHotEncodingColumnSuggestion] = Field(default_factory=list)


class DummyEncodingColumnSuggestion(BaseModel):
    """Recommendation payload for dummy encoding candidates."""

    column: str
    status: Literal[
        "recommended",
        "high_cardinality",
        "identifier",
        "free_text",
        "single_category",
        "too_many_categories",
    ]
    reason: str
    dtype: Optional[str] = None
    unique_count: Optional[int] = None
    unique_percentage: Optional[float] = None
    missing_percentage: Optional[float] = None
    text_category: Optional[str] = None
    sample_values: List[str] = Field(default_factory=list)
    estimated_dummy_columns: int = 0
    score: float = 0.0
    selectable: bool = True
    recommended_drop_first: bool = True


class DummyEncodingRecommendationsResponse(BaseModel):
    """Response payload for dummy encoding suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    total_text_columns: int = 0
    recommended_count: int = 0
    cautioned_count: int = 0
    high_cardinality_columns: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    auto_detect_default: bool = False
    columns: List[DummyEncodingColumnSuggestion] = Field(default_factory=list)


class SkewnessMethodDetail(BaseModel):
    """Metadata describing an available skewness transformation method."""

    key: str
    label: str
    description: Optional[str] = None
    direction_bias: Optional[Literal["left", "right", "either"]] = None
    requires_positive: bool = False
    supports_zero: bool = True
    supports_negative: bool = True


class SkewnessMethodStatus(BaseModel):
    """Compatibility status for a skewness method on a given column."""

    status: Literal["ready", "unsupported"]
    reason: Optional[str] = None


class SkewnessColumnDistribution(BaseModel):
    """Histogram-style distribution snapshot for a numeric column."""

    bin_edges: List[float] = Field(default_factory=list)
    counts: List[int] = Field(default_factory=list)
    sample_size: int = 0
    missing_count: int = 0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None


class SkewnessColumnRecommendation(BaseModel):
    """Recommendation payload for a single skewed column."""

    column: str
    skewness: float
    direction: Literal["left", "right"]
    magnitude: Literal["moderate", "substantial", "extreme"]
    summary: str
    recommended_methods: List[str] = Field(default_factory=list)
    method_status: Dict[str, SkewnessMethodStatus] = Field(default_factory=dict)
    applied_method: Optional[str] = None
    distribution_before: Optional[SkewnessColumnDistribution] = None
    distribution_after: Optional[SkewnessColumnDistribution] = None


class SkewnessRecommendationsResponse(BaseModel):
    """Response payload for skewness-driven transformation suggestions."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    skewness_threshold: float
    methods: List[SkewnessMethodDetail] = Field(default_factory=list)
    columns: List[SkewnessColumnRecommendation] = Field(default_factory=list)


class SkewnessConfiguredTransformation(BaseModel):
    """Declared skewness transformation from node configuration."""

    column: str
    method: str
    method_label: Optional[str] = None


class SkewnessAppliedColumnSignal(BaseModel):
    """Details about a column transformed to address skewness."""

    column: str
    method: str
    method_label: Optional[str] = None
    transformed_rows: int = 0
    total_rows: int = 0
    missing_rows: int = 0
    original_skewness: Optional[float] = None
    transformed_skewness: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class SkewnessSkippedColumnSignal(BaseModel):
    """Reason metadata for columns skipped during skewness transforms."""

    column: str
    reason: str
    method: Optional[str] = None
    method_label: Optional[str] = None


class SkewnessNodeSignal(BaseModel):
    """Aggregated metadata for a skewness transform node execution."""

    node_id: Optional[str] = None
    configured_transformations: List[SkewnessConfiguredTransformation] = Field(default_factory=list)
    applied_columns: List[SkewnessAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[SkewnessSkippedColumnSignal] = Field(default_factory=list)


class BinnedColumnBin(BaseModel):
    """Count-level summary for a single binned category."""

    label: str
    count: int
    percentage: float
    is_missing: bool = False


class BinnedColumnDistribution(BaseModel):
    """Distribution profile for a column generated by binning."""

    column: str
    source_column: Optional[str] = None
    total_rows: int
    non_missing_rows: int
    missing_rows: int
    distinct_bins: int
    top_label: Optional[str] = None
    top_count: Optional[int] = None
    top_percentage: Optional[float] = None
    bins: List[BinnedColumnBin] = Field(default_factory=list)


class BinnedDistributionResponse(BaseModel):
    """Response payload summarising distributions for binned columns."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    columns: List[BinnedColumnDistribution] = Field(default_factory=list)


class BinnedDistributionRequest(BaseModel):
    """Request payload for fetching binned column distributions."""

    dataset_source_id: str
    sample_size: Optional[int] = 500
    graph: Optional[Any] = None
    target_node_id: Optional[str] = None


class BinningAppliedColumnSignal(BaseModel):
    """Details about a single column transformed by binning."""

    source_column: str
    output_column: str
    strategy: Literal["equal_width", "equal_frequency", "custom", "kbins"]
    requested_bins: Optional[int] = None
    actual_bins: int = 0
    reduced_bins: bool = False
    drop_original: bool = False
    include_lowest: bool = False
    precision: int = 0
    duplicates: Optional[str] = None
    label_format: Optional[str] = None
    missing_strategy: Optional[str] = None
    missing_label: Optional[str] = None
    custom_labels_applied: bool = False
    sample_bins: List[BinnedColumnBin] = Field(default_factory=list)


class BinningNodeSignal(BaseModel):
    """Aggregated metadata for a binning node execution."""

    node_id: Optional[str] = None
    strategy: Literal["equal_width", "equal_frequency", "custom", "kbins"] = "equal_width"
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    drop_original: bool = False
    include_lowest: bool = True
    precision: int = 0
    duplicates: Optional[str] = None
    label_format: Optional[str] = None
    missing_strategy: Optional[str] = None
    missing_label: Optional[str] = None
    equal_width_bins: Optional[int] = None
    equal_frequency_bins: Optional[int] = None
    column_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    custom_bins: Optional[Dict[str, List[float]]] = None
    custom_labels: Optional[Dict[str, List[str]]] = None
    skipped_columns: List[str] = Field(default_factory=list)
    applied_columns: List["BinningAppliedColumnSignal"] = Field(default_factory=list)


OutlierMethodName = Literal["zscore", "iqr", "elliptic_envelope", "winsorize", "manual"]


class OutlierMethodDetail(BaseModel):
    """Metadata describing an outlier handling method."""

    key: OutlierMethodName
    label: str
    description: Optional[str] = None
    action: Literal["remove", "cap", "manual"]
    notes: List[str] = Field(default_factory=list)
    default_parameters: Dict[str, float] = Field(default_factory=dict)
    parameter_help: Dict[str, str] = Field(default_factory=dict)


class OutlierMethodSummary(BaseModel):
    """Summary of how a method would affect a column."""

    method: OutlierMethodName
    action: Literal["remove", "cap", "manual"] = "remove"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    affected_rows: int = 0
    affected_ratio: float = 0.0
    notes: List[str] = Field(default_factory=list)


class OutlierColumnStats(BaseModel):
    """Descriptive statistics used for outlier heuristics."""

    valid_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None


class OutlierColumnInsight(BaseModel):
    """Outlier impact summary for a single column."""

    column: str
    dtype: Optional[str] = None
    stats: OutlierColumnStats
    method_summaries: List[OutlierMethodSummary] = Field(default_factory=list)
    recommended_method: Optional[OutlierMethodName] = None
    recommended_reason: Optional[str] = None
    has_missing: bool = False


class OutlierRecommendationsResponse(BaseModel):
    """Response payload for outlier removal insights."""

    dataset_source_id: str
    sample_size: int
    default_method: OutlierMethodName
    methods: List[OutlierMethodDetail] = Field(default_factory=list)
    columns: List[OutlierColumnInsight] = Field(default_factory=list)


class OutlierAppliedColumnSignal(BaseModel):
    """Diagnostics captured when removing or capping outliers."""

    column: str
    method: OutlierMethodName
    action: Literal["remove", "cap"]
    affected_rows: int = 0
    total_rows: int = 0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class OutlierNodeSignal(BaseModel):
    """Aggregated metadata for an outlier removal node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    default_method: OutlierMethodName = "iqr"
    column_methods: Dict[str, OutlierMethodName] = Field(default_factory=dict)
    auto_detect: bool = True
    skipped_columns: List[str] = Field(default_factory=list)
    applied_columns: List[OutlierAppliedColumnSignal] = Field(default_factory=list)
    removed_rows: int = 0
    clipped_columns: List[str] = Field(default_factory=list)
    missing_strategy: Optional[str] = None
    missing_label: Optional[str] = None


class QuickProfileDatasetMetrics(BaseModel):
    """High-level dataset metrics for the lightweight profile."""

    row_count: int = 0
    column_count: int = 0
    missing_cells: int = 0
    missing_percentage: float = 0.0
    duplicate_rows: int = 0
    unique_rows: int = 0


class QuickProfileNumericSummary(BaseModel):
    """Numeric summary statistics for a column."""

    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    percentile_25: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_75: Optional[float] = None


class QuickProfileValueCount(BaseModel):
    """Frequency entry for discrete column values."""

    value: Any
    count: int
    percentage: float


class QuickProfileColumnSummary(BaseModel):
    """Column-level insights for the lightweight profile."""

    name: str
    dtype: Optional[str] = None
    semantic_type: Optional[str] = None
    missing_count: int = 0
    missing_percentage: float = 0.0
    distinct_count: Optional[int] = None
    sample_values: List[Any] = Field(default_factory=list)
    numeric_summary: Optional[QuickProfileNumericSummary] = None
    top_values: List[QuickProfileValueCount] = Field(default_factory=list)


class QuickProfileCorrelation(BaseModel):
    """Pairwise correlation metadata for numeric columns."""

    column_a: str
    column_b: str
    coefficient: float


class QuickProfileResponse(BaseModel):
    """Response payload for the lightweight profile endpoint."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    rows_analyzed: int
    columns_analyzed: int
    metrics: QuickProfileDatasetMetrics
    columns: List[QuickProfileColumnSummary] = Field(default_factory=list)
    correlations: List[QuickProfileCorrelation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


ScalingMethodName = Literal["standard", "minmax", "maxabs", "robust"]


BinningStrategyName = Literal["equal_width", "equal_frequency", "kbins"]


class ScalingColumnStats(BaseModel):
    """Summary statistics used to drive scaling recommendations."""

    valid_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    outlier_ratio: Optional[float] = None


class ScalingMethodDetail(BaseModel):
    """Metadata describing an available scaling method."""

    key: ScalingMethodName
    label: str
    description: Optional[str] = None
    handles_negative: bool = True
    handles_zero: bool = True
    handles_outliers: bool = False
    strengths: List[str] = Field(default_factory=list)
    cautions: List[str] = Field(default_factory=list)


class ScalingColumnRecommendation(BaseModel):
    """Recommended scaling strategy for a single column."""

    column: str
    dtype: Optional[str] = None
    recommended_method: ScalingMethodName
    confidence: Literal["high", "medium", "low"] = "medium"
    reasons: List[str] = Field(default_factory=list)
    fallback_methods: List[ScalingMethodName] = Field(default_factory=list)
    stats: ScalingColumnStats
    has_missing: bool = False


class ScalingRecommendationsResponse(BaseModel):
    """Response payload for scaling method insights."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    methods: List[ScalingMethodDetail] = Field(default_factory=list)
    columns: List[ScalingColumnRecommendation] = Field(default_factory=list)


class BinningColumnStats(BaseModel):
    """Summary statistics used to drive binning recommendations."""

    valid_count: int = 0
    missing_count: int = 0
    distinct_count: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    stddev: Optional[float] = None
    skewness: Optional[float] = None
    has_negative: bool = False
    has_zero: bool = False
    has_positive: bool = False


class BinningColumnRecommendation(BaseModel):
    """Recommended binning strategy for a single column."""

    column: str
    dtype: Optional[str] = None
    recommended_strategy: BinningStrategyName
    recommended_bins: int = Field(default=5, ge=2)
    confidence: Literal["high", "medium", "low"] = "medium"
    reasons: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    stats: BinningColumnStats


class BinningExcludedColumn(BaseModel):
    """Columns considered but excluded from binning recommendations."""

    column: str
    reason: str
    dtype: Optional[str] = None


class BinningRecommendationsResponse(BaseModel):
    """Response payload for binning guidance."""

    dataset_source_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    sample_size: int
    columns: List[BinningColumnRecommendation] = Field(default_factory=list)
    excluded_columns: List[BinningExcludedColumn] = Field(default_factory=list)


class ScalingNodeConfig(BaseModel):
    """Configuration payload for the scale numeric features node."""

    columns: List[str] = Field(default_factory=list)
    default_method: ScalingMethodName = "standard"
    column_methods: Dict[str, ScalingMethodName] = Field(default_factory=dict)
    auto_detect: bool = True
    skipped_columns: List[str] = Field(default_factory=list)


class ScalingAppliedColumnSignal(BaseModel):
    """Per-column diagnostics captured after applying numeric scaling."""

    column: str
    method: ScalingMethodName
    method_label: Optional[str] = None
    total_rows: int = 0
    valid_rows: int = 0
    missing_rows: int = 0
    original_mean: Optional[float] = None
    original_stddev: Optional[float] = None
    original_min: Optional[float] = None
    original_max: Optional[float] = None
    scaled_mean: Optional[float] = None
    scaled_stddev: Optional[float] = None
    scaled_min: Optional[float] = None
    scaled_max: Optional[float] = None


class ScalingNodeSignal(BaseModel):
    """Aggregated metadata for a scaling node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    default_method: ScalingMethodName = "standard"
    column_methods: Dict[str, ScalingMethodName] = Field(default_factory=dict)
    scaled_columns: List[ScalingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class PolynomialGeneratedFeature(BaseModel):
    """Description of an individual polynomial feature that was generated."""

    column: str
    degree: int
    terms: List[str] = Field(default_factory=list)
    expression: str
    raw_feature: Optional[str] = None


class PolynomialFeaturesNodeSignal(BaseModel):
    """Aggregated metadata for a polynomial features node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    applied_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False
    include_input_features: bool = False
    output_prefix: str = "poly"
    generated_columns: List[str] = Field(default_factory=list)
    generated_features: List[PolynomialGeneratedFeature] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    filled_columns: Dict[str, int] = Field(default_factory=dict)
    feature_count: int = 0
    transform_mode: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


ImputationMethodName = Literal["mean", "median", "mode", "knn", "regression", "mice"]


class ImputationConfiguredStrategySignal(BaseModel):
    """Description of a configured imputation strategy before execution."""

    method: ImputationMethodName
    columns: List[str] = Field(default_factory=list)
    auto_detected: bool = False
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ImputationAppliedColumnSignal(BaseModel):
    """Details for a column where missing values were imputed."""

    column: str
    method: ImputationMethodName
    method_label: Optional[str] = None
    filled_cells: int = 0
    original_missing: int = 0
    remaining_missing: int = 0
    dtype: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class ImputationNodeSignal(BaseModel):
    """Aggregated metadata for an imputation node execution."""

    node_id: Optional[str] = None
    configured_strategies: List[ImputationConfiguredStrategySignal] = Field(default_factory=list)
    applied_columns: List[ImputationAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    touched_columns: List[str] = Field(default_factory=list)
    filled_cells: int = 0
    method_usage: Dict[ImputationMethodName, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class DropMissingColumnsNodeSignal(BaseModel):
    node_id: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    removed_columns: List[str] = Field(default_factory=list)
    requested_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    removed_count: int = 0
    total_columns_before: int = 0
    total_columns_after: int = 0
    threshold: Optional[float] = None


class DropMissingRowsNodeSignal(BaseModel):
    node_id: Optional[str] = None
    removed_rows: int = 0
    removed_count: int = 0
    total_rows_before: int = 0
    total_rows_after: int = 0
    drop_if_any_missing: bool = False
    threshold: Optional[float] = None


class RemoveDuplicatesNodeSignal(BaseModel):
    node_id: Optional[str] = None
    removed_rows: int = 0
    total_rows_before: int = 0
    total_rows_after: int = 0
    keep: str = "first"
    subset_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)


class ClassResamplingNodeSignalBase(BaseModel):
    node_id: Optional[str] = None
    method: Optional[str] = None
    method_label: Optional[str] = None
    target_column: Optional[str] = None
    sampling_strategy: Optional[Union[str, float]] = None
    sampling_strategy_label: Optional[str] = None
    random_state: Optional[int] = None
    class_counts_before: Dict[str, int] = Field(default_factory=dict)
    class_counts_after: Dict[str, int] = Field(default_factory=dict)
    total_rows_before: int = 0
    total_rows_after: int = 0
    preserved_missing_rows: int = 0
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ClassUndersamplingNodeSignal(ClassResamplingNodeSignalBase):
    replacement: Optional[bool] = None


class ClassOversamplingNodeSignal(ClassResamplingNodeSignalBase):
    replacement: Optional[bool] = None
    k_neighbors: Optional[int] = None
    effective_k_neighbors: Optional[int] = None
    min_class_size: Optional[int] = None
    integer_cast_columns: List[str] = Field(default_factory=list)
    adjusted_parameters: Dict[str, Any] = Field(default_factory=dict)


class FeatureTargetSplitNodeSignal(BaseModel):
    node_id: Optional[str] = None
    target_column: Optional[str] = None
    configured_feature_columns: List[str] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)
    auto_included_columns: List[str] = Field(default_factory=list)
    missing_feature_columns: List[str] = Field(default_factory=list)
    excluded_columns: List[str] = Field(default_factory=list)
    feature_missing_counts: Dict[str, int] = Field(default_factory=dict)
    target_dtype: Optional[str] = None
    target_missing_count: int = 0
    target_missing_percentage: Optional[float] = None
    preview_row_count: int = 0
    total_rows: int = 0
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TrainTestSplitNodeSignal(BaseModel):
    node_id: Optional[str] = None
    train_size: Optional[int] = None
    validation_size: Optional[int] = None
    test_size: Optional[int] = None
    total_size: Optional[int] = None
    test_ratio: Optional[float] = None
    validation_ratio: Optional[float] = None
    stratified: bool = False
    target_column: Optional[str] = None
    random_state: Optional[int] = None
    shuffle: bool = True
    splits_created: List[str] = Field(default_factory=list)


class MissingIndicatorAppliedColumnSignal(BaseModel):
    source_column: str
    indicator_column: str
    flagged_rows: int = 0
    total_rows: int = 0
    created: bool = True
    overwritten_existing: bool = False


class MissingIndicatorNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    indicator_suffix: str = "_was_missing"
    total_flagged_rows: int = 0
    indicators: List[MissingIndicatorAppliedColumnSignal] = Field(default_factory=list)


class TrimWhitespaceAppliedColumnSignal(BaseModel):
    column: str
    updated_cells: int = 0
    total_rows: int = 0
    dtype: Optional[str] = None


class TrimWhitespaceNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[str] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    mode: str = "both"
    updated_cells: int = 0
    applied_columns: List[TrimWhitespaceAppliedColumnSignal] = Field(default_factory=list)


class NormalizeTextCaseAppliedColumnSignal(BaseModel):
    column: str
    updated_cells: int = 0
    total_rows: int = 0
    dtype: Optional[str] = None


class NormalizeTextCaseNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[str] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    mode: str = "lower"
    updated_cells: int = 0
    applied_columns: List[NormalizeTextCaseAppliedColumnSignal] = Field(default_factory=list)


class CastColumnAttemptSignal(BaseModel):
    column: str
    original_dtype: Optional[str] = None
    requested_dtype: Optional[str] = None
    resolved_dtype: Optional[str] = None
    dtype_family: Optional[str] = None
    changed_dtype: bool = False
    values_coerced_to_missing: int = 0
    error: Optional[str] = None


class CastColumnTypesNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    column_overrides: Dict[str, str] = Field(default_factory=dict)
    candidate_columns: List[str] = Field(default_factory=list)
    attempted_columns: List[CastColumnAttemptSignal] = Field(default_factory=list)
    applied_columns: List[str] = Field(default_factory=list)
    skipped_missing_dtype: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    errors: Dict[str, str] = Field(default_factory=dict)
    coerce_on_error: bool = False
    coerced_values: int = 0


class ReplaceAliasesAppliedColumnSignal(BaseModel):
    column: str
    mode: str
    mode_label: Optional[str] = None
    replacements: int = 0
    total_rows: int = 0
    auto_detected: bool = False
    dtype: Optional[str] = None


class ReplaceAliasesNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[ReplaceAliasesAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    replacements: int = 0
    strategies_executed: int = 0
    modes_used: List[str] = Field(default_factory=list)
    custom_pairs_used: bool = False
    skipped_custom_strategies: int = 0


class StandardizeDatesAppliedColumnSignal(BaseModel):
    column: str
    mode: str
    mode_label: Optional[str] = None
    converted_values: int = 0
    parse_failures: int = 0
    auto_detected: bool = False
    dtype: Optional[str] = None


class StandardizeDatesNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[StandardizeDatesAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    total_converted_values: int = 0
    total_parse_failures: int = 0
    mode_counts: Dict[str, int] = Field(default_factory=dict)


class RemoveSpecialCharactersAppliedColumnSignal(BaseModel):
    column: str
    mode: str
    mode_label: Optional[str] = None
    updated_cells: int = 0
    replacement: Optional[str] = None
    auto_detected: bool = False
    dtype: Optional[str] = None
    total_rows: int = 0


class RemoveSpecialCharactersNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[RemoveSpecialCharactersAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    mode: str = "keep_alphanumeric"
    replacement: Optional[str] = None
    total_updated_cells: int = 0


class ReplaceInvalidValuesAppliedColumnSignal(BaseModel):
    column: str
    mode: str
    mode_label: Optional[str] = None
    replacements: int = 0
    auto_detected: bool = False
    dtype: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ReplaceInvalidValuesNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[ReplaceInvalidValuesAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    mode: str = "negative_to_nan"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    total_replacements: int = 0


class RegexCleanupAppliedColumnSignal(BaseModel):
    column: str
    mode: str
    mode_label: Optional[str] = None
    updated_cells: int = 0
    auto_detected: bool = False
    dtype: Optional[str] = None
    pattern: Optional[str] = None


class RegexCleanupNodeSignal(BaseModel):
    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    auto_detected_columns: List[str] = Field(default_factory=list)
    processed_columns: List[RegexCleanupAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)
    missing_columns: List[str] = Field(default_factory=list)
    mode: str = "normalize_slash_dates"
    pattern: Optional[str] = None
    total_updated_cells: int = 0


class FeatureGraph(BaseModel):
    """Serialized graph definition."""

    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class FeaturePipelineBase(BaseModel):
    name: str = Field(default="Draft pipeline", max_length=150)
    description: Optional[str] = None
    graph: FeatureGraph
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="pipeline_metadata",
        validation_alias=AliasChoices("metadata", "pipeline_metadata"),
        serialization_alias="metadata",
    )

    model_config = {
        "populate_by_name": True,
    }


class FeaturePipelineCreate(FeaturePipelineBase):
    pass


class FeaturePipelineResponse(FeaturePipelineBase):
    id: int
    dataset_source_id: str
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }


class DatasetSourceSummary(BaseModel):
    id: int
    source_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True,
    }


class TrainModelDraftTargetInsight(BaseModel):
    """Target column summary used when validating modeling readiness."""

    name: str
    configured_problem_type: Literal["classification", "regression"] = "classification"
    inferred_problem_type: Literal["classification", "regression"] = "classification"
    pandas_dtype: Optional[str] = None
    distinct_count: int = 0
    missing_count: int = 0


class TrainModelDraftReadinessSnapshot(BaseModel):
    """Structured metadata surfaced by the train model draft node."""

    row_count: int = 0
    feature_count: int = 0
    feature_columns: List[str] = Field(default_factory=list)
    numeric_features: List[str] = Field(default_factory=list)
    non_numeric_features: List[str] = Field(default_factory=list)
    features_with_missing: List[str] = Field(default_factory=list)
    target: Optional[TrainModelDraftTargetInsight] = None
    warnings: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    ready_for_training: bool = False


class LabelEncodingCategoryPreview(BaseModel):
    """Sample mapping between original categories and encoded codes."""

    value: Optional[str] = None
    code: int


class LabelEncodingAppliedColumnSignal(BaseModel):
    """Description of a single column transformed by label encoding."""

    source_column: str
    encoded_column: Optional[str] = None
    class_count: int = 0
    replaced_original: bool = False
    preview: List[LabelEncodingCategoryPreview] = Field(default_factory=list)


class LabelEncodingNodeSignal(BaseModel):
    """Aggregated metadata for a label encoding node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    output_suffix: Optional[str] = None
    max_unique_values: Optional[int] = None
    missing_strategy: Optional[str] = None
    missing_code: Optional[int] = None
    encoded_columns: List[LabelEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class TargetEncodingCategoryPreview(BaseModel):
    """Sample mapping between categories and their encoded target mean."""

    category: str
    encoded_value: Optional[float] = None


class TargetEncodingAppliedColumnSignal(BaseModel):
    """Description of a single column transformed by target encoding."""

    source_column: str
    encoded_column: Optional[str] = None
    replaced_original: bool = False
    category_count: int = 0
    global_mean: Optional[float] = None
    smoothing: Optional[float] = None
    encode_missing: bool = False
    handle_unknown: Optional[str] = None
    unknown_rows: int = 0
    preview: List[TargetEncodingCategoryPreview] = Field(default_factory=list)


class TargetEncodingNodeSignal(BaseModel):
    """Aggregated metadata for a target encoding node execution."""

    node_id: Optional[str] = None
    target_column: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    output_suffix: Optional[str] = None
    max_categories: Optional[int] = None
    smoothing: Optional[float] = None
    encode_missing: bool = False
    handle_unknown: Optional[str] = None
    global_mean: Optional[float] = None
    encoded_columns: List[TargetEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class FeatureSelectionFeatureSummary(BaseModel):
    """Per-column scoring details from feature selection."""

    column: str
    selected: bool = True
    score: Optional[float] = None
    p_value: Optional[float] = None
    rank: Optional[int] = None
    importance: Optional[float] = None
    note: Optional[str] = None


class FeatureSelectionNodeSignal(BaseModel):
    """Structured signal emitted by the feature selection node."""

    node_id: Optional[str] = None
    method: Optional[str] = None
    score_func: Optional[str] = None
    mode: Optional[str] = None
    estimator: Optional[str] = None
    problem_type: Optional[str] = None
    target_column: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    selected_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    feature_summaries: List[FeatureSelectionFeatureSummary] = Field(default_factory=list)
    drop_unselected: bool = False
    auto_detect: bool = False
    k: Optional[int] = None
    percentile: Optional[float] = None
    alpha: Optional[float] = None
    threshold: Optional[float] = None
    transform_mode: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class FeatureMathOperationResult(BaseModel):
    """Execution metadata for a feature math operation."""

    operation_id: str
    operation_type: str
    method: Optional[str] = None
    output_columns: List[str] = Field(default_factory=list)
    status: Literal["applied", "skipped", "failed"]
    message: Optional[str] = None


class FeatureMathNodeSignal(BaseModel):
    """Aggregated metadata for a feature math node execution."""

    node_id: Optional[str] = None
    total_operations: int = 0
    applied_operations: int = 0
    skipped_operations: int = 0
    failed_operations: int = 0
    operations: List[FeatureMathOperationResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)
    encoded_columns: List[TargetEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class OneHotEncodingAppliedColumnSignal(BaseModel):
    """Description of a single column transformed by one-hot encoding."""

    source_column: str
    dummy_columns: List[str] = Field(default_factory=list)
    replaced_original: bool = False
    category_count: int = 0
    includes_missing_dummy: bool = False
    preview_categories: List[str] = Field(default_factory=list)


class OneHotEncodingNodeSignal(BaseModel):
    """Aggregated metadata for a one-hot encoding node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    drop_first: bool = False
    include_missing: bool = False
    max_categories: Optional[int] = None
    prefix_separator: Optional[str] = None
    encoded_columns: List[OneHotEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class DummyEncodingAppliedColumnSignal(BaseModel):
    """Description of a single column transformed by dummy encoding."""

    source_column: str
    dummy_columns: List[str] = Field(default_factory=list)
    replaced_original: bool = False
    category_count: int = 0
    includes_missing_dummy: bool = False
    preview_categories: List[str] = Field(default_factory=list)


class DummyEncodingNodeSignal(BaseModel):
    """Aggregated metadata for a dummy encoding node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    drop_first: bool = False
    include_missing: bool = False
    max_categories: Optional[int] = None
    prefix_separator: Optional[str] = None
    encoded_columns: List[DummyEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class TransformerSplitActivitySignal(BaseModel):
    """Tracks how a transformer interacted with a specific split."""

    split: Literal["train", "test", "validation", "other", "unknown"]
    action: Literal["fit_transform", "transform", "not_applied", "not_available"]
    rows: Optional[int] = None
    updated_at: Optional[datetime] = None
    label: Optional[str] = None


class TransformerAuditEntrySignal(BaseModel):
    """An individual transformer instance captured by the audit node."""

    source_node_id: Optional[str] = None
    source_node_label: Optional[str] = None
    transformer_name: str
    column_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    split_activity: List[TransformerSplitActivitySignal] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    storage_key: Optional[str] = None


class TransformerAuditNodeSignal(BaseModel):
    """Aggregated transformer state for the transformer audit node."""

    node_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    transformers: List[TransformerAuditEntrySignal] = Field(default_factory=list)
    total_transformers: int = 0
    notes: List[str] = Field(default_factory=list)


class HashEncodingAppliedColumnSignal(BaseModel):
    source_column: str
    output_column: str
    replaced_original: bool = False
    bucket_count: int = 0
    encoded_missing: bool = False
    category_count: Optional[int] = None
    sample_hashes: List[int] = Field(default_factory=list)


class HashEncodingNodeSignal(BaseModel):
    """Aggregated metadata for a hash encoding node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    encode_missing: bool = False
    max_categories: Optional[int] = None
    n_buckets: Optional[int] = None
    output_suffix: Optional[str] = None
    encoded_columns: List[HashEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class OrdinalEncodingCategoryPreview(BaseModel):
    category: str
    code: int


class OrdinalEncodingAppliedColumnSignal(BaseModel):
    source_column: str
    encoded_column: Optional[str] = None
    replaced_original: bool = False
    category_count: int = 0
    handle_unknown: Optional[str] = None
    unknown_value: Optional[int] = None
    encode_missing: bool = False
    preview: List[OrdinalEncodingCategoryPreview] = Field(default_factory=list)


class OrdinalEncodingNodeSignal(BaseModel):
    """Aggregated metadata for an ordinal encoding node execution."""

    node_id: Optional[str] = None
    configured_columns: List[str] = Field(default_factory=list)
    evaluated_columns: List[str] = Field(default_factory=list)
    auto_detect: bool = False
    drop_original: bool = False
    encode_missing: bool = False
    max_categories: Optional[int] = None
    output_suffix: Optional[str] = None
    handle_unknown: Optional[str] = None
    unknown_value: Optional[int] = None
    encoded_columns: List[OrdinalEncodingAppliedColumnSignal] = Field(default_factory=list)
    skipped_columns: List[str] = Field(default_factory=list)


class ModelEvaluationConfusionMatrix(BaseModel):
    """Structured confusion matrix payload for classification diagnostics."""

    labels: List[str] = Field(default_factory=list)
    matrix: List[List[int]] = Field(default_factory=list)
    normalized: Optional[List[List[float]]] = None
    totals: List[int] = Field(default_factory=list)
    accuracy: Optional[float] = None


class ModelEvaluationRocCurve(BaseModel):
    """Receiver operating characteristic curve payload."""

    label: str
    fpr: List[float] = Field(default_factory=list)
    tpr: List[float] = Field(default_factory=list)
    thresholds: List[float] = Field(default_factory=list)
    auc: Optional[float] = None


class ModelEvaluationPrecisionRecallCurve(BaseModel):
    """Precision/recall curve payload."""

    label: str
    recall: List[float] = Field(default_factory=list)
    precision: List[float] = Field(default_factory=list)
    thresholds: List[float] = Field(default_factory=list)
    average_precision: Optional[float] = None


class ModelEvaluationResidualHistogram(BaseModel):
    """Histogram summary for residual diagnostics."""

    bin_edges: List[float] = Field(default_factory=list)
    counts: List[int] = Field(default_factory=list)


class ModelEvaluationResidualPoint(BaseModel):
    """Sampled residual point used for scatter plots."""

    actual: float
    predicted: float


class ModelEvaluationResiduals(BaseModel):
    """Residual diagnostics payload for regression analyses."""

    histogram: ModelEvaluationResidualHistogram
    scatter: List[ModelEvaluationResidualPoint] = Field(default_factory=list)
    summary: Dict[str, float] = Field(default_factory=dict)


class ModelEvaluationSplitPayload(BaseModel):
    """Evaluation artefacts for a single dataset split."""

    split: str
    row_count: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: Optional[ModelEvaluationConfusionMatrix] = None
    roc_curves: List[ModelEvaluationRocCurve] = Field(default_factory=list)
    pr_curves: List[ModelEvaluationPrecisionRecallCurve] = Field(default_factory=list)
    residuals: Optional[ModelEvaluationResiduals] = None
    notes: List[str] = Field(default_factory=list)


class ModelEvaluationReport(BaseModel):
    """Aggregated evaluation report returned by the evaluation endpoint."""

    job_id: str
    pipeline_id: Optional[str] = None
    node_id: Optional[str] = None
    generated_at: datetime
    problem_type: Literal["classification", "regression"]
    target_column: Optional[str] = None
    splits: Dict[str, ModelEvaluationSplitPayload] = Field(default_factory=dict)


class ModelEvaluationRequest(BaseModel):
    """Request payload controlling model evaluation diagnostics."""

    splits: Optional[List[str]] = None
    include_confusion: bool = True
    include_curves: bool = True
    include_residuals: bool = True
    max_curve_points: Optional[int] = None
    max_scatter_points: Optional[int] = None


class ModelEvaluationNodeSignal(BaseModel):
    """Lightweight preview signal for the model evaluation node."""

    node_id: Optional[str] = None
    training_job_id: Optional[str] = None
    splits: List[str] = Field(default_factory=list)
    has_evaluation: bool = False
    last_evaluated_at: Optional[datetime] = None
    notes: List[str] = Field(default_factory=list)


class FullExecutionSignal(BaseModel):
    """Status payload describing full-dataset pipeline execution."""

    status: Literal["succeeded", "deferred", "skipped", "failed"]
    reason: Optional[str] = None
    total_rows: Optional[int] = None
    processed_rows: Optional[int] = None
    applied_steps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    job_id: Optional[str] = None
    job_status: Optional[Literal["queued", "running", "succeeded", "failed", "cancelled"]] = None
    dataset_source_id: Optional[str] = None
    last_updated: Optional[datetime] = None
    poll_after_seconds: Optional[int] = None
    eta_seconds: Optional[int] = None


class PipelinePreviewSignals(BaseModel):
    """Container for node-level structured metadata during previews."""

    modeling: Optional[TrainModelDraftReadinessSnapshot] = None
    cast_column_types: List[CastColumnTypesNodeSignal] = Field(default_factory=list)
    feature_math: List[FeatureMathNodeSignal] = Field(default_factory=list)
    label_encoding: List[LabelEncodingNodeSignal] = Field(default_factory=list)
    target_encoding: List[TargetEncodingNodeSignal] = Field(default_factory=list)
    one_hot_encoding: List[OneHotEncodingNodeSignal] = Field(default_factory=list)
    dummy_encoding: List[DummyEncodingNodeSignal] = Field(default_factory=list)
    hash_encoding: List[HashEncodingNodeSignal] = Field(default_factory=list)
    ordinal_encoding: List[OrdinalEncodingNodeSignal] = Field(default_factory=list)
    transformer_audit: List[TransformerAuditNodeSignal] = Field(default_factory=list)
    binning: List[BinningNodeSignal] = Field(default_factory=list)
    skewness_transform: List[SkewnessNodeSignal] = Field(default_factory=list)
    scaling: List[ScalingNodeSignal] = Field(default_factory=list)
    polynomial_features: List[PolynomialFeaturesNodeSignal] = Field(default_factory=list)
    feature_selection: List[FeatureSelectionNodeSignal] = Field(default_factory=list)
    imputation: List[ImputationNodeSignal] = Field(default_factory=list)
    outlier_removal: List[OutlierNodeSignal] = Field(default_factory=list)
    drop_missing_columns: List[DropMissingColumnsNodeSignal] = Field(default_factory=list)
    drop_missing_rows: List[DropMissingRowsNodeSignal] = Field(default_factory=list)
    remove_duplicates: List[RemoveDuplicatesNodeSignal] = Field(default_factory=list)
    missing_value_indicator: List[MissingIndicatorNodeSignal] = Field(default_factory=list)
    trim_whitespace: List[TrimWhitespaceNodeSignal] = Field(default_factory=list)
    normalize_text_case: List[NormalizeTextCaseNodeSignal] = Field(default_factory=list)
    replace_aliases: List[ReplaceAliasesNodeSignal] = Field(default_factory=list)
    standardize_dates: List[StandardizeDatesNodeSignal] = Field(default_factory=list)
    remove_special_characters: List[RemoveSpecialCharactersNodeSignal] = Field(default_factory=list)
    replace_invalid_values: List[ReplaceInvalidValuesNodeSignal] = Field(default_factory=list)
    regex_cleanup: List[RegexCleanupNodeSignal] = Field(default_factory=list)
    feature_target_split: List[FeatureTargetSplitNodeSignal] = Field(default_factory=list)
    train_test_split: List[TrainTestSplitNodeSignal] = Field(default_factory=list)
    class_undersampling: List[ClassUndersamplingNodeSignal] = Field(default_factory=list)
    class_oversampling: List[ClassOversamplingNodeSignal] = Field(default_factory=list)
    model_evaluation: List[ModelEvaluationNodeSignal] = Field(default_factory=list)
    full_execution: Optional[FullExecutionSignal] = None


class TrainingJobStatus(str, Enum):
    """Allowed training job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJobCreate(BaseModel):
    """Payload used when enqueueing one or more model training jobs."""

    dataset_source_id: str
    pipeline_id: str
    node_id: str
    model_types: List[str] = Field(default_factory=list)
    hyperparameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="job_metadata")
    run_training: bool = True
    graph: FeatureGraph
    target_node_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_types(cls, data: Any) -> Any:
        """Allow legacy payloads that submit `model_type` as a scalar."""

        if not isinstance(data, dict):
            return data

        existing = data.get("model_types")
        if existing is not None:
            if isinstance(existing, list):
                return data
            data["model_types"] = [existing]
            return data

        single = data.get("model_type")
        if single is None:
            return data

        if isinstance(single, list):
            data["model_types"] = single
        else:
            data["model_types"] = [single]

        return data

    @model_validator(mode="after")
    def _ensure_model_types(self) -> "TrainingJobCreate":
        normalized: List[str] = []
        seen: set[str] = set()
        for entry in self.model_types:
            if entry is None:
                continue
            candidate = str(entry).strip()
            if candidate and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)

        if not normalized:
            raise ValueError("Training job payload requires at least one model type")

        object.__setattr__(self, "model_types", normalized)
        return self

    @computed_field  # type: ignore[misc]
    @property
    def model_type(self) -> str:
        """Return the primary model type for backward compatibility."""

        return self.model_types[0]


class TrainingJobResponse(BaseModel):
    """Detailed representation of a training job state."""

    job_id: str = Field(alias="id")
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    user_id: Optional[int] = None
    status: TrainingJobStatus
    version: int
    model_type: str
    hyperparameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="job_metadata",
        serialization_alias="metadata",
    )
    metrics: Optional[Dict[str, Any]] = None
    artifact_uri: Optional[str] = None
    error_message: Optional[str] = None
    progress: Optional[int] = 0
    current_step: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    graph: FeatureGraph

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @computed_field  # type: ignore[misc]
    @property
    def target_node_id(self) -> Optional[str]:
        metadata = self.metadata or {}
        return metadata.get("target_node_id")


class TrainingJobSummary(BaseModel):
    """Compact representation used in lists and canvas hydration."""

    job_id: str = Field(alias="id")
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    status: TrainingJobStatus
    version: int
    model_type: str
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="job_metadata",
        serialization_alias="metadata",
    )
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: Optional[int] = 0
    current_step: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @computed_field  # type: ignore[misc]
    @property
    def problem_type(self) -> Optional[str]:
        metadata = self.metadata or {}
        value = metadata.get("resolved_problem_type") or metadata.get("problem_type")
        return value.lower() if isinstance(value, str) else None


class TrainingJobListResponse(BaseModel):
    """Paginated-ish response for training job listings."""

    jobs: List[TrainingJobSummary] = Field(default_factory=list)
    total: int = 0


class TrainingJobBatchResponse(BaseModel):
    """Response payload when multiple training jobs are created at once."""

    jobs: List[TrainingJobResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def total(self) -> int:
        return len(self.jobs)


class HyperparameterTuningJobStatus(str, Enum):
    """Lifecycle states for hyperparameter tuning jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HyperparameterTuningJobCreate(BaseModel):
    """Payload used when enqueueing one or more hyperparameter tuning jobs."""

    dataset_source_id: str
    pipeline_id: str
    node_id: str
    model_types: List[str] = Field(default_factory=list)
    search_strategy: str = Field(default_factory=get_default_strategy_value)
    search_space: Dict[str, Any]
    n_iterations: Optional[int] = Field(default=None, validation_alias=AliasChoices("n_iterations", "n_iter"))
    scoring: Optional[str] = None
    random_state: Optional[int] = None
    baseline_hyperparameters: Optional[Dict[str, Any]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="job_metadata")
    run_tuning: bool = True
    graph: FeatureGraph
    target_node_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_types(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        existing = data.get("model_types")
        if existing is not None:
            if isinstance(existing, list):
                return data
            data["model_types"] = [existing]
            return data

        single = data.get("model_type")
        if single is None:
            return data

        if isinstance(single, list):
            data["model_types"] = single
        else:
            data["model_types"] = [single]

        return data

    @model_validator(mode="after")
    def _ensure_model_types(self) -> "HyperparameterTuningJobCreate":
        normalized: List[str] = []
        seen: set[str] = set()
        for entry in self.model_types:
            if entry is None:
                continue
            candidate = str(entry).strip()
            if candidate and candidate not in seen:
                normalized.append(candidate)
                seen.add(candidate)

        if not normalized:
            raise ValueError("Hyperparameter tuning payload requires at least one model type")

        object.__setattr__(self, "model_types", normalized)
        return self

    @computed_field  # type: ignore[misc]
    @property
    def model_type(self) -> str:
        return self.model_types[0]

    @field_validator("search_strategy", mode="before")
    @classmethod
    def _normalize_search_strategy(cls, value: Any) -> str:
        return normalize_strategy_value(value)


class HyperparameterTuningJobResponse(BaseModel):
    """Detailed representation of a hyperparameter tuning job."""

    job_id: str = Field(alias="id")
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    user_id: Optional[int] = None
    status: HyperparameterTuningJobStatus
    run_number: int
    model_type: str
    search_strategy: str
    search_space: Optional[Dict[str, Any]] = None
    baseline_hyperparameters: Optional[Dict[str, Any]] = None
    n_iterations: Optional[int] = None
    scoring: Optional[str] = None
    random_state: Optional[int] = None
    cross_validation: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="job_metadata",
        serialization_alias="metadata",
    )
    metrics: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    artifact_uri: Optional[str] = None
    error_message: Optional[str] = None
    progress: Optional[int] = 0
    current_step: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    graph: FeatureGraph

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @computed_field  # type: ignore[misc]
    @property
    def target_node_id(self) -> Optional[str]:
        metadata = self.metadata or {}
        return metadata.get("target_node_id")

    @field_validator("search_strategy", mode="before")
    @classmethod
    def _normalize_search_strategy(cls, value: Any) -> str:
        return normalize_strategy_value(value)


class HyperparameterTuningJobSummary(BaseModel):
    """Compact representation of tuning jobs for listings."""

    job_id: str = Field(alias="id")
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    status: HyperparameterTuningJobStatus
    run_number: int
    model_type: str
    search_strategy: str
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="job_metadata",
        serialization_alias="metadata",
    )
    metrics: Optional[Dict[str, Any]] = None
    results: Optional[List[Dict[str, Any]]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    error_message: Optional[str] = None
    progress: Optional[int] = 0
    current_step: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @field_validator("search_strategy", mode="before")
    @classmethod
    def _normalize_search_strategy(cls, value: Any) -> str:
        return normalize_strategy_value(value)


class HyperparameterTuningJobListResponse(BaseModel):
    """Response payload for tuning job listings."""

    jobs: List[HyperparameterTuningJobSummary] = Field(default_factory=list)
    total: int = 0


class HyperparameterTuningJobBatchResponse(BaseModel):
    """Response payload when multiple tuning jobs are created."""

    jobs: List[HyperparameterTuningJobResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def total(self) -> int:
        return len(self.jobs)


class PipelinePreviewColumnStat(BaseModel):
    name: str
    dtype: Optional[str] = None
    missing_count: int = 0
    missing_percentage: float = 0.0
    distinct_count: Optional[int] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None


class PipelinePreviewMetrics(BaseModel):
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_cells: int
    preview_rows: int = 0
    total_rows: int = 0
    requested_sample_size: int = 0


class PipelinePreviewRowMissingStat(BaseModel):
    index: int
    missing_percentage: float


class PipelinePreviewColumnSchema(BaseModel):
    name: str
    pandas_dtype: Optional[str] = None
    logical_family: Literal[
        'numeric',
        'integer',
        'categorical',
        'string',
        'datetime',
        'boolean',
        'unknown',
    ] = 'unknown'
    nullable: bool = True


class PipelinePreviewSchema(BaseModel):
    signature: Optional[str] = None
    columns: List[PipelinePreviewColumnSchema] = Field(default_factory=list)


class PipelinePreviewRequest(BaseModel):
    dataset_source_id: str
    graph: FeatureGraph
    target_node_id: Optional[str] = None
    sample_size: int = Field(default=1000, ge=0, le=5000)  # Increased from 200/1000 to 1000/5000
    include_signals: bool = Field(
        default=True,
        description="When false, pipeline preview omits node-level signal payloads to reduce response size.",
    )
    include_preview_rows: bool = Field(
        default=True,
        description=(
            "When false, the preview response excludes sampled rows or metrics and runs the pipeline on an empty frame."
        ),
    )


class PipelinePreviewResponse(BaseModel):
    node_id: Optional[str]
    columns: List[str] = Field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: PipelinePreviewMetrics
    column_stats: List[PipelinePreviewColumnStat] = Field(default_factory=list)
    applied_steps: List[str] = Field(default_factory=list)
    row_missing_stats: List[PipelinePreviewRowMissingStat] = Field(default_factory=list)
    schema_summary: Optional[PipelinePreviewSchema] = Field(
        default=None,
        validation_alias=AliasChoices("schema", "schema_summary"),
        serialization_alias="schema",
    )
    modeling_signals: Optional[TrainModelDraftReadinessSnapshot] = None
    signals: Optional[PipelinePreviewSignals] = None

    model_config = {
        "populate_by_name": True,
    }


class PipelinePreviewRowsResponse(BaseModel):
    """Windowed preview payload used for incremental row loading."""

    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    offset: int = 0
    limit: int = 0
    returned_rows: int = 0
    total_rows: Optional[int] = None
    next_offset: Optional[int] = None
    has_more: bool = False
    sampling_mode: Optional[str] = None
    sampling_adjustments: List[str] = Field(default_factory=list)
    large_dataset: bool = False

