"""Core utilities for Skyulf."""

from .compute import (
    ComputeBackend,
    LocalComputeBackend,
    compute_backend,
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
    model_serializer,
    set_model_serializer,
)
from .warnings import SkyulfWarning, WarningCategory

__all__ = [
    "ApplierProtocol",
    "CalculatorProtocol",
    "ComputeBackend",
    "InMemoryModelRegistry",
    "JoblibModelSerializer",
    "LocalComputeBackend",
    "ModelRegistry",
    "ModelSerializer",
    "ModelVersion",
    "PipelineStep",
    "SchemaMismatchError",
    "SkyulfSchema",
    "SkyulfWarning",
    "WarningCategory",
    "compute_backend",
    "deprecated",
    "get_compute_backend",
    "get_model_serializer",
    "model_serializer",
    "set_compute_backend",
    "set_model_serializer",
    "validate_schema",
    "warn_deprecated",
]
