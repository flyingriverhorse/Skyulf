"""Local and distributed dataframe protocols, adapters and engine registry."""

from .pandas_engine import PandasEngine
from .polars_engine import (
    POLARS_NUMERIC_BOOL_DTYPES,
    POLARS_NUMERIC_DTYPES,
    PolarsEngine,
    SkyulfPolarsWrapper,
)
from .protocol import DistributedDataFrame, PandasBackedFrame, PolarsBackedFrame, SkyulfDataFrame
from .registry import BaseEngine, EngineName, EngineRegistry, get_engine
from .spark_engine import DistributedMaterializationError, SkyulfSparkWrapper, SparkEngine

__all__ = [
    "POLARS_NUMERIC_BOOL_DTYPES",
    "POLARS_NUMERIC_DTYPES",
    "BaseEngine",
    "DistributedDataFrame",
    "DistributedMaterializationError",
    "EngineName",
    "EngineRegistry",
    "PandasBackedFrame",
    "PandasEngine",
    "PolarsBackedFrame",
    "PolarsEngine",
    "SkyulfDataFrame",
    "SkyulfPolarsWrapper",
    "SkyulfSparkWrapper",
    "SparkEngine",
    "get_engine",
]
