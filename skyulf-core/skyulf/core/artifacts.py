"""Typed artifact shapes returned by Calculator.fit() methods.

Each TypedDict documents the exact keys a downstream Applier can
expect in its ``params`` argument. Using these as return-type
annotations on Calculator.fit() lets ``ty`` / IDEs catch missing
or mis-spelled keys early.

All fields are optional (``total=False``) so that the early-return
``{}`` (no columns selected) is still assignment-compatible at the
call sites that collect params.
"""

from typing import Any, TypedDict

# ── Scalers ───────────────────────────────────────────────────────────────────


class StandardScalerArtifact(TypedDict, total=False):
    type: str
    mean: list[float]
    scale: list[float] | None
    var: list[float]
    with_mean: bool
    with_std: bool
    columns: list[str]


class MinMaxScalerArtifact(TypedDict, total=False):
    type: str
    min: list[float]
    scale: list[float]
    data_min: list[float]
    data_max: list[float]
    feature_range: list[float]
    columns: list[str]


class RobustScalerArtifact(TypedDict, total=False):
    type: str
    center: list[float] | None
    scale: list[float] | None
    quantile_range: Any
    with_centering: bool
    with_scaling: bool
    columns: list[str]


class MaxAbsScalerArtifact(TypedDict, total=False):
    type: str
    scale: list[float] | None
    max_abs: list[float] | None
    columns: list[str]


# ── Imputers ──────────────────────────────────────────────────────────────────


class SimpleImputerArtifact(TypedDict, total=False):
    type: str
    strategy: str
    fill_values: dict[str, Any]
    columns: list[str]
    missing_counts: dict[str, int]
    total_missing: int


class KNNImputerArtifact(TypedDict, total=False):
    type: str
    # sklearn object — intentionally Any; not JSON-serialisable
    imputer_object: Any
    columns: list[str]
    n_neighbors: int
    weights: str


class IterativeImputerArtifact(TypedDict, total=False):
    type: str
    imputer_object: Any
    columns: list[str]
    estimator: str


# ── Outlier detectors ─────────────────────────────────────────────────────────


class IQRArtifact(TypedDict, total=False):
    type: str
    bounds: dict[str, dict[str, float]]
    multiplier: float
    warnings: list[str]


class ZScoreArtifact(TypedDict, total=False):
    type: str
    stats: dict[str, dict[str, float]]
    threshold: float
    warnings: list[str]


class WinsorizeArtifact(TypedDict, total=False):
    type: str
    bounds: dict[str, dict[str, float]]
    lower_percentile: float
    upper_percentile: float
    warnings: list[str]


class ManualBoundsArtifact(TypedDict, total=False):
    type: str
    bounds: dict[str, Any]


class EllipticEnvelopeArtifact(TypedDict, total=False):
    type: str
    # sklearn EllipticEnvelope objects per column — not JSON-serialisable
    models: dict[str, Any]
    contamination: float
    warnings: list[str]


# ── Transformations ──────────────────────────────────────────────────────────


class PowerTransformerArtifact(TypedDict, total=False):
    type: str
    lambdas: list[float]
    method: str
    standardize: bool
    columns: list[str]
    # Optional fitted scaler params when standardize=True
    scaler_params: dict[str, Any] | None


class SimpleTransformationArtifact(TypedDict, total=False):
    type: str
    # List of {"column": str, "method": str, ...} dicts copied from config
    transformations: list[dict[str, Any]]


class GeneralTransformationArtifact(TypedDict, total=False):
    type: str
    # Each item is {"column": str, "method": str, "params": {...}, ...}
    transformations: list[dict[str, Any]]


# ── Resampling ───────────────────────────────────────────────────────────────


class OversamplingArtifact(TypedDict, total=False):
    type: str
    method: str
    target_column: str | None
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
    target_column: str | None
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
    subset: list[str] | None
    keep: str


class DropMissingColumnsArtifact(TypedDict, total=False):
    type: str
    columns_to_drop: list[str]
    threshold: float | None


class DropMissingRowsArtifact(TypedDict, total=False):
    type: str
    subset: list[str] | None
    how: str
    threshold: int | None


class MissingIndicatorArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    flag_suffix: str


# ── Casting ──────────────────────────────────────────────────────────────────


class CastingArtifact(TypedDict, total=False):
    type: str
    type_map: dict[str, str]
    coerce_on_error: bool


# ── Bucketing ────────────────────────────────────────────────────────────────


class GeneralBinningArtifact(TypedDict, total=False):
    type: str
    bin_edges: dict[str, list[float]]
    custom_labels: dict[str, list[Any]]
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
    columns: list[str]
    degree: int
    interaction_only: bool
    include_bias: bool
    include_input_features: bool
    output_prefix: str
    feature_names: list[str]


class FeatureGenerationArtifact(TypedDict, total=False):
    type: str
    operations: list[dict[str, Any]]
    epsilon: float
    allow_overwrite: bool


class FeatureInteractionArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    degree: int
    interaction_only: bool
    include_bias: bool
    combinations: list[list[str]]
    feature_names: list[str]


# ── Geo ───────────────────────────────────────────────────────────────────────


class GeoDistanceArtifact(TypedDict, total=False):
    type: str
    lat1_col: str
    lon1_col: str
    lat2_col: str
    lon2_col: str
    method: str
    unit: str
    output_column: str


class H3IndexArtifact(TypedDict, total=False):
    type: str
    lat_col: str
    lon_col: str
    resolution: int
    output_column: str


# ── Feature selection ────────────────────────────────────────────────────────


class VarianceThresholdArtifact(TypedDict, total=False):
    type: str
    selected_columns: list[str]
    candidate_columns: list[str]
    threshold: float
    drop_columns: bool
    variances: dict[str, float]


class CorrelationThresholdArtifact(TypedDict, total=False):
    type: str
    columns_to_drop: list[str]
    threshold: float
    method: str
    drop_columns: bool


class UnivariateSelectionArtifact(TypedDict, total=False):
    type: str
    selected_columns: list[str]
    candidate_columns: list[str]
    method: str
    drop_columns: bool
    feature_scores: dict[str, float]
    p_values: dict[str, float]
    # No-target fallback uses these legacy keys instead
    scores: dict[str, float]
    pvalues: dict[str, float]


class ModelBasedSelectionArtifact(TypedDict, total=False):
    type: str
    selected_columns: list[str]
    candidate_columns: list[str]
    method: str
    drop_columns: bool
    feature_importances: dict[str, float]


# ── Cleaning ─────────────────────────────────────────────────────────────────


class TextCleaningArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    operations: list[dict[str, Any]]


class InvalidValueReplacementArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    replace_inf: bool
    replace_neg_inf: bool
    rule: str | None
    replacement: Any
    value: Any
    min_value: Any
    max_value: Any


class ValueReplacementArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    mapping: dict[Any, Any] | None
    to_replace: Any
    value: Any


# ── Text / Vectorization ──────────────────────────────────────────────────────


class CountVectorizerArtifact(TypedDict, total=False):
    type: str
    columns: list[str]  # source text column(s) fed to the vectorizer
    output_columns: list[str]  # one name per vocabulary term
    vocabulary: dict[str, int]  # token → column-index mapping
    max_features: int | None
    lowercase: bool
    stop_words: str | None  # e.g. "english" or None
    binary: bool  # presence/absence (1/0) instead of counts
    vectorizer_object: Any  # fitted sklearn object (not JSON-serialisable)
    drop_original: bool


class TfidfVectorizerArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    output_columns: list[str]
    vocabulary: dict[str, int]
    idf: list[float]  # one value per vocabulary term
    max_features: int | None
    lowercase: bool
    stop_words: str | None  # e.g. "english" or None
    vectorizer_object: Any  # fitted sklearn object (not JSON-serialisable)
    drop_original: bool


class HashingVectorizerArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    output_columns: list[str]  # indexed: ``{src}__hash__{i}``
    n_features: int
    norm: str | None
    lowercase: bool
    stop_words: str | None  # e.g. "english" or None
    vectorizer_object: Any  # configured (but stateless) sklearn object
    drop_original: bool


class TokenizerArtifact(TypedDict, total=False):
    type: str
    columns: list[str]  # source text column(s)
    analyzer: str  # word | char | char_wb
    lowercase: bool
    stop_words: str | None  # e.g. "english" or None
    ngram_range: list[int]
    output_columns: list[str]  # tokenized-text column name(s)
    add_token_count: bool
    drop_original: bool


class SentenceEmbedderArtifact(TypedDict, total=False):
    type: str
    columns: list[str]  # source text column(s)
    model_name: str  # sentence-transformers model id
    embedding_dim: int
    normalize: bool
    output_columns: list[str]  # indexed: ``{src}__emb__{i}``
    drop_original: bool


class AliasReplacementArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    alias_type: str | None
    custom_map: dict[str, Any] | None


# ── Inspection ───────────────────────────────────────────────────────────────


class DatasetProfileArtifact(TypedDict, total=False):
    type: str
    # Profile is a deeply nested dict produced by EDAAnalyzer; intentionally
    # left as Dict[str, Any] — schema is unstable and evolves with new metrics.
    profile: dict[str, Any]


class DataSnapshotArtifact(TypedDict, total=False):
    type: str
    # Snapshot is metadata-only; shape depends on the snapshot strategy.
    snapshot: dict[str, Any]


# ── Encoders ─────────────────────────────────────────────────────────────────


class OneHotArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    # sklearn OneHotEncoder — not JSON-serialisable
    encoder_object: Any
    feature_names: list[str]
    prefix_separator: str
    drop_original: bool
    include_missing: bool


class OrdinalArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    encoder_object: Any
    # Per-column LabelEncoder objects keyed by column name (plus optional "__target__")
    encoders: dict[str, Any]
    categories_count: list[int]


class LabelEncoderArtifact(TypedDict, total=False):
    type: str
    columns: list[str] | None
    encoders: dict[str, Any]
    classes_count: dict[str, int]
    missing_code: int


class TargetEncoderArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    # sklearn TargetEncoder — not JSON-serialisable
    encoder_object: Any


class HashEncoderArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    n_features: int


class DummyEncoderArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    categories: dict[str, list[str]]
    drop_first: bool


# ── Splitters ────────────────────────────────────────────────────────────────


class SplitArtifact(TypedDict, total=False):
    """Train/test/validation split parameters (passed through from config)."""

    type: str
    test_size: float
    validation_size: float
    random_state: int
    shuffle: bool
    stratify: bool
    target_column: str | None


class FeatureTargetSplitArtifact(TypedDict, total=False):
    """Feature/target split parameters (passed through from config)."""

    type: str
    target_column: str


# ── Time series ──────────────────────────────────────────────────────────────


class LagFeaturesArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    lags: list[int]
    group_by: list[str] | None
    sort_by: str | None
    drop_na: bool


class RollingAggregateArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    window: int
    aggregations: list[str]
    min_periods: int
    group_by: list[str] | None
    sort_by: str | None


class DateFeaturesArtifact(TypedDict, total=False):
    type: str
    columns: list[str]
    features: list[str]
    drop_original: bool
