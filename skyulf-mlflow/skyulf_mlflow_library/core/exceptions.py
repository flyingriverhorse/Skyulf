"""
Core exception classes for Skyulf MLflow library.

This module defines custom exception types used throughout the library.
"""


class SkyulfMLflowError(Exception):
    """Base exception class for all Skyulf MLflow errors."""
    pass


class FeatureEngineeringError(SkyulfMLflowError):
    """Exception raised for errors in feature engineering operations."""
    pass


class DataProcessingError(SkyulfMLflowError):
    """Exception raised for errors in data processing operations."""
    pass


class DataIngestionError(SkyulfMLflowError):
    """Exception raised for errors in data ingestion operations."""
    pass


class ValidationError(SkyulfMLflowError):
    """Exception raised for validation errors."""
    pass


class ConfigurationError(SkyulfMLflowError):
    """Exception raised for configuration errors."""
    pass


class TransformationError(SkyulfMLflowError):
    """Exception raised for transformation errors."""
    pass
