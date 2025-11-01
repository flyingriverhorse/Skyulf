"""Core type definitions and enums for Skyulf-MLFlow library."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Type aliases for better readability
DataFrame = pd.DataFrame
Series = pd.Series
Array = np.ndarray
ColumnName = str
ColumnList = List[ColumnName]
DataType = Union[str, type, np.dtype]


class SplitType(str, Enum):
    """Enumeration of data split types."""

    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class NodeCategory(str, Enum):
    """Categorization of nodes by their data processing behavior."""

    # Transformers: Need to fit on train, transform on test/validation
    TRANSFORMER = "transformer"

    # Filters: Apply same logic to all splits independently
    FILTER = "filter"

    # Splitters: Create splits (like train_test_split)
    SPLITTER = "splitter"

    # Models: Fit on train, predict on test/validation
    MODEL = "model"

    # Passthrough: No special handling needed
    PASSTHROUGH = "passthrough"


class EncodingStrategy(str, Enum):
    """Encoding strategies for categorical variables."""

    ONE_HOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    HASH = "hash"
    BINARY = "binary"
    FREQUENCY = "frequency"


class ScalingStrategy(str, Enum):
    """Scaling strategies for numeric variables."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NORMALIZER = "normalizer"


class ImputationStrategy(str, Enum):
    """Imputation strategies for missing values."""

    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    KNN = "knn"
    ITERATIVE = "iterative"


class SamplingStrategy(str, Enum):
    """Sampling strategies for imbalanced data."""

    # Over-sampling
    SMOTE = "smote"
    ADASYN = "adasyn"
    RANDOM_OVER = "random_over"

    # Under-sampling
    RANDOM_UNDER = "random_under"
    TOMEK = "tomek"
    EDITED_NEAREST = "edited_nearest"
    CLUSTER_CENTROIDS = "cluster_centroids"

    # Combination
    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"


class BinningStrategy(str, Enum):
    """Binning strategies for numeric variables."""

    EQUAL_WIDTH = "equal_width"
    EQUAL_FREQUENCY = "equal_frequency"
    QUANTILE = "quantile"
    KMEANS = "kmeans"
    CUSTOM = "custom"


class OutlierMethod(str, Enum):
    """Methods for outlier detection and removal."""

    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"


class TransformerState(str, Enum):
    """States of a transformer."""

    NOT_FITTED = "not_fitted"
    FITTED = "fitted"
    PARTIAL_FITTED = "partial_fitted"


# Configuration types
TransformerConfig = Dict[str, Any]
PipelineConfig = Dict[str, Any]
NodeConfig = Dict[str, Any]

# Metadata types
TransformerMetadata = Dict[str, Any]
PipelineMetadata = Dict[str, Any]

# Split column name constant
SPLIT_TYPE_COLUMN = "__split_type__"
