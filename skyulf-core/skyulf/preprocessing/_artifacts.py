"""Typed artifact shapes returned by Calculator.fit() methods.

Each TypedDict documents the exact keys a downstream Applier can
expect in its ``params`` argument. Using these as return-type
annotations on Calculator.fit() lets ``ty`` / IDEs catch missing
or mis-spelled keys early.

All fields are optional (``total=False``) so that the early-return
``{}`` (no columns selected) is still assignment-compatible at the
call sites that collect params.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


# ── Scalers ───────────────────────────────────────────────────────────────────


class StandardScalerArtifact(TypedDict, total=False):
    type: str
    mean: List[float]
    scale: Optional[List[float]]
    var: List[float]
    with_mean: bool
    with_std: bool
    columns: List[str]


class MinMaxScalerArtifact(TypedDict, total=False):
    type: str
    min: List[float]
    scale: List[float]
    data_min: List[float]
    data_max: List[float]
    feature_range: List[float]
    columns: List[str]


class RobustScalerArtifact(TypedDict, total=False):
    type: str
    center: Optional[List[float]]
    scale: Optional[List[float]]
    quantile_range: Any
    with_centering: bool
    with_scaling: bool
    columns: List[str]


class MaxAbsScalerArtifact(TypedDict, total=False):
    type: str
    scale: Optional[List[float]]
    max_abs: Optional[List[float]]
    columns: List[str]


# ── Imputers ──────────────────────────────────────────────────────────────────


class SimpleImputerArtifact(TypedDict, total=False):
    type: str
    strategy: str
    fill_values: Dict[str, Any]
    columns: List[str]
    missing_counts: Dict[str, int]
    total_missing: int


class KNNImputerArtifact(TypedDict, total=False):
    type: str
    # sklearn object — intentionally Any; not JSON-serialisable
    imputer_object: Any
    columns: List[str]
    n_neighbors: int
    weights: str


class IterativeImputerArtifact(TypedDict, total=False):
    type: str
    imputer_object: Any
    columns: List[str]
    estimator: str


# ── Outlier detectors ─────────────────────────────────────────────────────────


class IQRArtifact(TypedDict, total=False):
    type: str
    bounds: Dict[str, Dict[str, float]]
    multiplier: float
    warnings: List[str]


class ZScoreArtifact(TypedDict, total=False):
    type: str
    stats: Dict[str, Dict[str, float]]
    threshold: float
    warnings: List[str]


class WinsorizeArtifact(TypedDict, total=False):
    type: str
    bounds: Dict[str, Dict[str, float]]
    lower_percentile: float
    upper_percentile: float
    warnings: List[str]


class ManualBoundsArtifact(TypedDict, total=False):
    type: str
    bounds: Dict[str, Any]


class EllipticEnvelopeArtifact(TypedDict, total=False):
    type: str
    # sklearn EllipticEnvelope objects per column — not JSON-serialisable
    models: Dict[str, Any]
    contamination: float
    warnings: List[str]


# ── Transformations ──────────────────────────────────────────────────────────


class PowerTransformerArtifact(TypedDict, total=False):
    type: str
    lambdas: List[float]
    method: str
    standardize: bool
    columns: List[str]
    # Optional fitted scaler params when standardize=True
    scaler_params: Optional[Dict[str, Any]]


class SimpleTransformationArtifact(TypedDict, total=False):
    type: str
    # List of {"column": str, "method": str, ...} dicts copied from config
    transformations: List[Dict[str, Any]]


class GeneralTransformationArtifact(TypedDict, total=False):
    type: str
    # Each item is {"column": str, "method": str, "params": {...}, ...}
    transformations: List[Dict[str, Any]]


# ── Resampling ───────────────────────────────────────────────────────────────


class OversamplingArtifact(TypedDict, total=False):
    type: str
    method: str
    target_column: Optional[str]
    sampling_strategy: Any
    random_state: int
    k_neighbors: int
    m_neighbors: int
    kind: str
    svm_estimator: Any
    out_step: float
    kmeans_estimator: Any
    cluster_balance_threshold: float
    density_exponent: Any
    n_jobs: int


class UndersamplingArtifact(TypedDict, total=False):
    type: str
    method: str
    target_column: Optional[str]
    sampling_strategy: Any
    random_state: int
    replacement: bool
    version: int
    n_neighbors: int
    kind_sel: str
    n_jobs: int


# ── Drop / missing ───────────────────────────────────────────────────────────


class DeduplicateArtifact(TypedDict, total=False):
    type: str
    subset: Optional[List[str]]
    keep: str


class DropMissingColumnsArtifact(TypedDict, total=False):
    type: str
    columns_to_drop: List[str]
    threshold: Optional[float]


class DropMissingRowsArtifact(TypedDict, total=False):
    type: str
    subset: Optional[List[str]]
    how: str
    threshold: Optional[int]


class MissingIndicatorArtifact(TypedDict, total=False):
    type: str
    columns: List[str]


# ── Casting ──────────────────────────────────────────────────────────────────


class CastingArtifact(TypedDict, total=False):
    type: str
    type_map: Dict[str, str]
    coerce_on_error: bool


# ── Bucketing ────────────────────────────────────────────────────────────────


class GeneralBinningArtifact(TypedDict, total=False):
    type: str
    bin_edges: Dict[str, List[float]]
    custom_labels: Dict[str, List[Any]]
    output_suffix: str
    drop_original: bool
    label_format: str
    missing_strategy: str
    missing_label: str
    include_lowest: bool
    precision: int


# ── Feature generation ───────────────────────────────────────────────────────


class PolynomialFeaturesArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    degree: int
    interaction_only: bool
    include_bias: bool
    include_input_features: bool
    output_prefix: str
    feature_names: List[str]


class FeatureGenerationArtifact(TypedDict, total=False):
    type: str
    operations: List[Dict[str, Any]]
    epsilon: float
    allow_overwrite: bool


# ── Feature selection ────────────────────────────────────────────────────────


class VarianceThresholdArtifact(TypedDict, total=False):
    type: str
    selected_columns: List[str]
    candidate_columns: List[str]
    threshold: float
    drop_columns: bool
    variances: Dict[str, float]


class CorrelationThresholdArtifact(TypedDict, total=False):
    type: str
    columns_to_drop: List[str]
    threshold: float
    method: str
    drop_columns: bool


class UnivariateSelectionArtifact(TypedDict, total=False):
    type: str
    selected_columns: List[str]
    candidate_columns: List[str]
    method: str
    drop_columns: bool
    feature_scores: Dict[str, float]
    p_values: Dict[str, float]
    # No-target fallback uses these legacy keys instead
    scores: Dict[str, float]
    pvalues: Dict[str, float]


class ModelBasedSelectionArtifact(TypedDict, total=False):
    type: str
    selected_columns: List[str]
    candidate_columns: List[str]
    method: str
    drop_columns: bool
    feature_importances: Dict[str, float]


# ── Cleaning ─────────────────────────────────────────────────────────────────


class TextCleaningArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    operations: List[Dict[str, Any]]


class InvalidValueReplacementArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    replace_inf: bool
    replace_neg_inf: bool
    rule: Optional[str]
    replacement: Any
    value: Any
    min_value: Any
    max_value: Any


class ValueReplacementArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    mapping: Optional[Dict[Any, Any]]
    to_replace: Any
    value: Any


class AliasReplacementArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    alias_type: Optional[str]
    custom_map: Optional[Dict[str, Any]]


# ── Inspection ───────────────────────────────────────────────────────────────


class DatasetProfileArtifact(TypedDict, total=False):
    type: str
    # Profile is a deeply nested dict produced by EDAAnalyzer; intentionally
    # left as Dict[str, Any] — schema is unstable and evolves with new metrics.
    profile: Dict[str, Any]


class DataSnapshotArtifact(TypedDict, total=False):
    type: str
    # Snapshot is metadata-only; shape depends on the snapshot strategy.
    snapshot: Dict[str, Any]


# ── Encoders ─────────────────────────────────────────────────────────────────


class OneHotArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    # sklearn OneHotEncoder — not JSON-serialisable
    encoder_object: Any
    feature_names: List[str]
    prefix_separator: str
    drop_original: bool
    include_missing: bool


class OrdinalArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    encoder_object: Any
    # Per-column LabelEncoder objects keyed by column name (plus optional "__target__")
    encoders: Dict[str, Any]
    categories_count: List[int]


class LabelEncoderArtifact(TypedDict, total=False):
    type: str
    columns: Optional[List[str]]
    encoders: Dict[str, Any]
    classes_count: Dict[str, int]
    missing_code: int


class TargetEncoderArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    # sklearn TargetEncoder — not JSON-serialisable
    encoder_object: Any


class HashEncoderArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    n_features: int


class DummyEncoderArtifact(TypedDict, total=False):
    type: str
    columns: List[str]
    categories: Dict[str, List[str]]
    drop_first: bool
