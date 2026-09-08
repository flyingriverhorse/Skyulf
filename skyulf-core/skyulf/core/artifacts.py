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
    """Standard-scaling parameters (per-column mean, scale and variance)."""

    type: str
    mean: list[float]
    scale: list[float] | None
    var: list[float]
    with_mean: bool
    with_std: bool
    columns: list[str]


class MinMaxScalerArtifact(TypedDict, total=False):
    """Min-max scaling parameters (per-column data range and target feature range)."""

    type: str
    min: list[float]
    scale: list[float]
    data_min: list[float]
    data_max: list[float]
    feature_range: list[float]
    columns: list[str]


class RobustScalerArtifact(TypedDict, total=False):
    """Robust-scaling parameters (per-column centre, scale and quantile range)."""

    type: str
    center: list[float] | None
    scale: list[float] | None
    quantile_range: Any
    with_centering: bool
    with_scaling: bool
    columns: list[str]


class MaxAbsScalerArtifact(TypedDict, total=False):
    """Max-abs scaling parameters (per-column maximum absolute value and scale)."""

    type: str
    scale: list[float] | None
    max_abs: list[float] | None
    columns: list[str]


# ── Imputers ──────────────────────────────────────────────────────────────────


class SimpleImputerArtifact(TypedDict, total=False):
    """Simple-imputation parameters (per-column fill values, strategy and missing counts)."""

    type: str
    strategy: str
    fill_values: dict[str, Any]
    columns: list[str]
    missing_counts: dict[str, int]
    total_missing: int


class KNNImputerArtifact(TypedDict, total=False):
    """KNN-imputation parameters (fitted sklearn imputer object and neighbour settings)."""

    type: str
    # sklearn object — intentionally Any; not JSON-serialisable
    imputer_object: Any
    columns: list[str]
    n_neighbors: int
    weights: str


class IterativeImputerArtifact(TypedDict, total=False):
    """Iterative-imputation parameters (fitted sklearn imputer object and estimator)."""

    type: str
    imputer_object: Any
    columns: list[str]
    estimator: str


# ── Outlier detectors ─────────────────────────────────────────────────────────


class IQRArtifact(TypedDict, total=False):
    """IQR outlier-detection parameters (per-column bounds and the multiplier used)."""

    type: str
    bounds: dict[str, dict[str, float]]
    multiplier: float
    warnings: list[str]


class ZScoreArtifact(TypedDict, total=False):
    """Z-score outlier-detection parameters (per-column statistics and cutoff threshold)."""

    type: str
    stats: dict[str, dict[str, float]]
    threshold: float
    warnings: list[str]


class WinsorizeArtifact(TypedDict, total=False):
    """Winsorization parameters (per-column clipping bounds and the bounding percentiles)."""

    type: str
    bounds: dict[str, dict[str, float]]
    lower_percentile: float
    upper_percentile: float
    warnings: list[str]


class ManualBoundsArtifact(TypedDict, total=False):
    """Manual outlier-bound parameters (user-supplied per-column bounds)."""

    type: str
    bounds: dict[str, Any]


class EllipticEnvelopeArtifact(TypedDict, total=False):
    """Elliptic Envelope parameters (per-column sklearn models and contamination rate)."""

    type: str
    # sklearn EllipticEnvelope objects per column — not JSON-serialisable
    models: dict[str, Any]
    contamination: float
    warnings: list[str]


# ── Transformations ──────────────────────────────────────────────────────────


class PowerTransformerArtifact(TypedDict, total=False):
    """Power-transform parameters (per-column lambdas and optional standardizing scaler)."""

    type: str
    lambdas: list[float]
    method: str
    standardize: bool
    columns: list[str]
    # Optional fitted scaler params when standardize=True
    scaler_params: dict[str, Any] | None


class SimpleTransformationArtifact(TypedDict, total=False):
    """Simple transformation parameters (per-column method specs copied from config)."""

    type: str
    # List of {"column": str, "method": str, ...} dicts copied from config
    transformations: list[dict[str, Any]]


class GeneralTransformationArtifact(TypedDict, total=False):
    """General transformation parameters (per-column method and parameter specs)."""

    type: str
    # Each item is {"column": str, "method": str, "params": {...}, ...}
    transformations: list[dict[str, Any]]


# ── Resampling ───────────────────────────────────────────────────────────────


class OversamplingArtifact(TypedDict, total=False):
    """Oversampling parameters (method plus its estimator, neighbour and sampling settings)."""

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
    """Undersampling parameters (method plus its neighbour, replacement and selection settings)."""

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
    """Deduplication parameters (column subset compared and which duplicate to keep)."""

    type: str
    subset: list[str] | None
    keep: str


class DropMissingColumnsArtifact(TypedDict, total=False):
    """Drop-missing-columns parameters (columns to drop and the missingness threshold)."""

    type: str
    columns_to_drop: list[str]
    threshold: float | None


class DropMissingRowsArtifact(TypedDict, total=False):
    """Drop-missing-rows parameters (subset considered and the drop rule/threshold)."""

    type: str
    subset: list[str] | None
    how: str
    threshold: int | None
    missing_threshold: float | None


class MissingIndicatorArtifact(TypedDict, total=False):
    """Missing-indicator parameters (flagged columns and the indicator-column suffix)."""

    type: str
    columns: list[str]
    flag_suffix: str


# ── Casting ──────────────────────────────────────────────────────────────────


class CastingArtifact(TypedDict, total=False):
    """Type-casting parameters (target dtype per column and coerce-on-error flag)."""

    type: str
    type_map: dict[str, str]
    coerce_on_error: bool
    categories: dict[str, list[Any]]


# ── Bucketing ────────────────────────────────────────────────────────────────


class GeneralBinningArtifact(TypedDict, total=False):
    """Binning parameters (per-column bin edges plus bucket labels and formatting)."""

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
    """Polynomial-feature parameters (source columns, degree and generated feature names)."""

    type: str
    columns: list[str]
    degree: int
    interaction_only: bool
    include_bias: bool
    include_input_features: bool
    output_prefix: str
    feature_names: list[str]


class FeatureGenerationArtifact(TypedDict, total=False):
    """Feature-generation parameters (operations, epsilon and allow-overwrite flag)."""

    type: str
    operations: list[dict[str, Any]]
    epsilon: float
    allow_overwrite: bool


class FeatureInteractionArtifact(TypedDict, total=False):
    """Feature-interaction parameters (expanded column combinations and feature names)."""

    type: str
    columns: list[str]
    degree: int
    interaction_only: bool
    include_bias: bool
    combinations: list[list[str]]
    feature_names: list[str]


# ── Geo ───────────────────────────────────────────────────────────────────────


class GeoDistanceArtifact(TypedDict, total=False):
    """Geo-distance parameters (paired lat/lon columns, method, unit and output column)."""

    type: str
    lat1_col: str
    lon1_col: str
    lat2_col: str
    lon2_col: str
    method: str
    unit: str
    output_column: str


class H3IndexArtifact(TypedDict, total=False):
    """H3-indexing parameters (lat/lon columns, H3 resolution and output column)."""

    type: str
    lat_col: str
    lon_col: str
    resolution: int
    output_column: str


# ── Feature selection ────────────────────────────────────────────────────────


class VarianceThresholdArtifact(TypedDict, total=False):
    """Variance-threshold selection parameters (per-column variances and columns kept)."""

    type: str
    selected_columns: list[str]
    candidate_columns: list[str]
    threshold: float
    drop_columns: bool
    variances: dict[str, float]


class CorrelationThresholdArtifact(TypedDict, total=False):
    """Correlation-threshold selection parameters (correlated columns to drop and method)."""

    type: str
    columns_to_drop: list[str]
    threshold: float
    method: str
    drop_columns: bool


class UnivariateSelectionArtifact(TypedDict, total=False):
    """Univariate-selection params: per-feature scores, p-values and legacy no-target keys."""

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
    """Model-based selection parameters (per-feature importances and retained columns)."""

    type: str
    selected_columns: list[str]
    candidate_columns: list[str]
    method: str
    drop_columns: bool
    feature_importances: dict[str, float]


# ── Cleaning ─────────────────────────────────────────────────────────────────


class TextCleaningArtifact(TypedDict, total=False):
    """Text-cleaning parameters (columns cleaned and the sequence of operations)."""

    type: str
    columns: list[str]
    operations: list[dict[str, Any]]


class InvalidValueReplacementArtifact(TypedDict, total=False):
    """Invalid-value replacement parameters (columns, inf-replacement flags and the rule/value)."""

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
    """Value-replacement parameters (columns affected and the mapping or to-replace/value pair)."""

    type: str
    columns: list[str]
    mapping: dict[Any, Any] | None
    to_replace: Any
    value: Any


# ── Text / Vectorization ──────────────────────────────────────────────────────


class CountVectorizerArtifact(TypedDict, total=False):
    """Count-vectorizer parameters (vocabulary, output columns and fitted sklearn vectorizer)."""

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
    """TF-IDF vectorizer params: vocabulary, per-term IDF weights and the fitted vectorizer."""

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
    """Hashing-vectorizer parameters (feature-bucket count and the stateless sklearn vectorizer)."""

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
    """Tokenizer parameters (analyzer, n-gram range and output token-column names)."""

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
    """Sentence-embedder params: model id, embedding dimension and output column names."""

    type: str
    columns: list[str]  # source text column(s)
    model_name: str  # sentence-transformers model id
    embedding_dim: int
    normalize: bool
    output_columns: list[str]  # indexed: ``{src}__emb__{i}``
    drop_original: bool


class AliasReplacementArtifact(TypedDict, total=False):
    """Alias-replacement parameters (columns normalised and the alias type or custom map)."""

    type: str
    columns: list[str]
    alias_type: str | None
    custom_map: dict[str, Any] | None


# ── Inspection ───────────────────────────────────────────────────────────────


class DatasetProfileArtifact(TypedDict, total=False):
    """Dataset-profiling parameters (the EDAAnalyzer profile payload; schema intentionally open)."""

    type: str
    # Profile is a deeply nested dict produced by EDAAnalyzer; intentionally
    # left as dict[str, Any] — schema is unstable and evolves with new metrics.
    profile: dict[str, Any]


class DataSnapshotArtifact(TypedDict, total=False):
    """Data-snapshot parameters (metadata-only snapshot payload; shape depends on strategy)."""

    type: str
    # Snapshot is metadata-only; shape depends on the snapshot strategy.
    snapshot: dict[str, Any]


# ── Encoders ─────────────────────────────────────────────────────────────────


class OneHotArtifact(TypedDict, total=False):
    """One-hot encoding parameters (fitted sklearn encoder, feature names and prefix separator)."""

    type: str
    columns: list[str]
    # sklearn OneHotEncoder — not JSON-serialisable
    encoder_object: Any
    feature_names: list[str]
    prefix_separator: str
    drop_original: bool
    include_missing: bool


class OrdinalArtifact(TypedDict, total=False):
    """Ordinal encoding parameters (per-column LabelEncoders and category counts)."""

    type: str
    columns: list[str]
    encoder_object: Any
    # Per-column LabelEncoder objects keyed by column name (plus optional "__target__")
    encoders: dict[str, Any]
    categories_count: list[int]


class LabelEncoderArtifact(TypedDict, total=False):
    """Label-encoding parameters (per-column encoders, class counts and the missing-value code)."""

    type: str
    columns: list[str] | None
    encoders: dict[str, Any]
    classes_count: dict[str, int]
    missing_code: int


class TargetEncoderArtifact(TypedDict, total=False):
    """Target-encoding parameters (the fitted sklearn target encoder for the encoded columns)."""

    type: str
    columns: list[str]
    # sklearn TargetEncoder — not JSON-serialisable
    encoder_object: Any


class HashEncoderArtifact(TypedDict, total=False):
    """Hash-encoding parameters (columns hashed and the number of output buckets)."""

    type: str
    columns: list[str]
    n_features: int
    numeric_normalization_version: int


class DummyEncoderArtifact(TypedDict, total=False):
    """Dummy-encoding parameters (per-column categories and the drop-first flag)."""

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
    """Lag-feature parameters (columns lagged, lag offsets and group/sort ordering)."""

    type: str
    columns: list[str]
    lags: list[int]
    group_by: list[str] | None
    sort_by: str | None
    drop_na: bool


class RollingAggregateArtifact(TypedDict, total=False):
    """Rolling-aggregate parameters (window size, aggregation functions and group/sort ordering)."""

    type: str
    columns: list[str]
    window: int
    aggregations: list[str]
    min_periods: int
    group_by: list[str] | None
    sort_by: str | None


class DateFeaturesArtifact(TypedDict, total=False):
    """Date-feature parameters (datetime columns expanded and the parts extracted)."""

    type: str
    columns: list[str]
    features: list[str]
    drop_original: bool
