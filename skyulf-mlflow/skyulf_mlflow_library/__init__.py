"""
Skyulf-MLFlow: A comprehensive machine learning library for data ingestion and feature engineering.

This library provides intuitive APIs for building robust ML pipelines, similar to scikit-learn and LangChain.
"""

__version__ = "0.1.1"
__author__ = "Murat"
__license__ = "MIT"

# Core exports
from skyulf_mlflow_library.core.base import (
    BaseTransformer,
    BaseEncoder,
    BaseScaler,
    BaseImputer,
    BaseFilter,
)
from skyulf_mlflow_library.core.types import (
    SplitType,
    NodeCategory,
    EncodingStrategy,
    ScalingStrategy,
    ImputationStrategy,
    SamplingStrategy,
    BinningStrategy,
    OutlierMethod,
    SPLIT_TYPE_COLUMN,
)
from skyulf_mlflow_library.exceptions import (
    SkyulfMLFlowException,
    DataIngestionError,
    PreprocessingError,
    FeatureEngineeringError,
    TransformerError,
    TransformerNotFittedError,
    PipelineError,
    ValidationError,
    ConfigurationError,
    ColumnNotFoundError,
)

__all__ = [
    # Version
    "__version__",
    # Base classes
    "BaseTransformer",
    "BaseEncoder",
    "BaseScaler",
    "BaseImputer",
    "BaseFilter",
    # Types and enums
    "SplitType",
    "NodeCategory",
    "EncodingStrategy",
    "ScalingStrategy",
    "ImputationStrategy",
    "SamplingStrategy",
    "BinningStrategy",
    "OutlierMethod",
    "SPLIT_TYPE_COLUMN",
    # Exceptions
    "SkyulfMLFlowException",
    "DataIngestionError",
    "PreprocessingError",
    "FeatureEngineeringError",
    "TransformerError",
    "TransformerNotFittedError",
    "PipelineError",
    "ValidationError",
    "ConfigurationError",
    "ColumnNotFoundError",
]
