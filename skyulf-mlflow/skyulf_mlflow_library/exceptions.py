"""Core exceptions for Skyulf-MLFlow library."""

from typing import Optional


class SkyulfMLFlowException(Exception):
    """Base exception for all Skyulf-MLFlow errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataIngestionError(SkyulfMLFlowException):
    """Raised when data ingestion fails."""

    pass


class DataLoadError(DataIngestionError):
    """Raised when data loading fails."""

    pass


class DataValidationError(DataIngestionError):
    """Raised when data validation fails."""

    pass


class DataConversionError(DataIngestionError):
    """Raised when data type conversion fails."""

    pass


class PreprocessingError(SkyulfMLFlowException):
    """Raised when preprocessing fails."""

    pass


class MissingValueError(PreprocessingError):
    """Raised when missing value handling fails."""

    pass


class DuplicateError(PreprocessingError):
    """Raised when duplicate handling fails."""

    pass


class OutlierError(PreprocessingError):
    """Raised when outlier handling fails."""

    pass


class FeatureEngineeringError(SkyulfMLFlowException):
    """Raised when feature engineering fails."""

    pass


class TransformerError(FeatureEngineeringError):
    """Raised when transformer operation fails."""

    pass


class TransformerNotFittedError(TransformerError):
    """Raised when transformer is used before being fitted."""

    pass


class EncodingError(FeatureEngineeringError):
    """Raised when encoding fails."""

    pass


class ScalingError(FeatureEngineeringError):
    """Raised when scaling fails."""

    pass


class ImputationError(FeatureEngineeringError):
    """Raised when imputation fails."""

    pass


class SamplingError(FeatureEngineeringError):
    """Raised when sampling fails."""

    pass


class PipelineError(SkyulfMLFlowException):
    """Raised when pipeline operation fails."""

    pass


class PipelineBuildError(PipelineError):
    """Raised when pipeline building fails."""

    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""

    pass


class SplitError(PipelineError):
    """Raised when train/test split handling fails."""

    pass


class ValidationError(SkyulfMLFlowException):
    """Raised when validation fails."""

    pass


class ConfigurationError(SkyulfMLFlowException):
    """Raised when configuration is invalid."""

    pass


class ColumnNotFoundError(SkyulfMLFlowException):
    """Raised when a specified column is not found in the dataframe."""

    def __init__(self, column: str, available_columns: list):
        message = f"Column '{column}' not found in dataframe"
        details = {
            "column": column,
            "available_columns": available_columns,
        }
        super().__init__(message, details)


class IncompatibleDataError(SkyulfMLFlowException):
    """Raised when data is incompatible with the operation."""

    pass


class EmptyDataError(SkyulfMLFlowException):
    """Raised when data is empty."""

    pass
