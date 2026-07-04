"""Core utilities for Skyulf."""

from .compute import (
    ComputeBackend,
    LocalComputeBackend,
    get_compute_backend,
    set_compute_backend,
)
from .deprecation import deprecated, warn_deprecated
from .model_registry import InMemoryModelRegistry, ModelRegistry, ModelVersion
from .protocols import ApplierProtocol, CalculatorProtocol, PipelineStep
from .schema import SchemaMismatchError, SkyulfSchema, validate_schema
from .serialization import (
    JoblibModelSerializer,
    ModelSerializer,
    get_model_serializer,
    set_model_serializer,
)
from .warnings import SkyulfWarning, WarningCategory

__all__ = [
    "deprecated",
    "warn_deprecated",
    "SkyulfWarning",
    "WarningCategory",
    "ComputeBackend",
    "LocalComputeBackend",
    "get_compute_backend",
    "set_compute_backend",
    "ModelSerializer",
    "JoblibModelSerializer",
    "get_model_serializer",
    "set_model_serializer",
    "ModelRegistry",
    "InMemoryModelRegistry",
    "ModelVersion",
    "SkyulfSchema",
    "SchemaMismatchError",
    "validate_schema",
    "CalculatorProtocol",
    "ApplierProtocol",
    "PipelineStep",
]
