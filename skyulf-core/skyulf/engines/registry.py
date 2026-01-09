"""
Engine Registry for Skyulf.

This module handles the auto-detection of the appropriate compute engine
(Pandas, Polars, etc.) based on the input data type.
"""

from typing import Any, Dict, Optional, Type
import logging

# We import the protocol for type checking, but we don't strictly need it at runtime here
# to avoid circular imports if engines import protocol.
from .protocol import SkyulfDataFrame

logger = logging.getLogger(__name__)

class BaseEngine:
    """Abstract base class for all engines."""
    
    name: str = "base"

    @classmethod
    def is_compatible(cls, data: Any) -> bool:
        """Check if this engine can handle the given data object."""
        raise NotImplementedError

    @classmethod
    def from_pandas(cls, df: Any) -> Any:
        """Convert a pandas DataFrame to this engine's native format."""
        raise NotImplementedError

    @classmethod
    def to_numpy(cls, df: Any) -> Any:
        """Convert to numpy array (for sklearn compatibility)."""
        raise NotImplementedError

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        """Wrap the native dataframe in a SkyulfDataFrame compliant wrapper."""
        raise NotImplementedError

    @classmethod
    def create_dataframe(cls, data: Any) -> Any:
        """Create a native dataframe from a dictionary or list."""
        raise NotImplementedError

class EngineRegistry:
    _engines: Dict[str, Type[BaseEngine]] = {}
    _active_engine: str = "pandas"  # Default

    @classmethod
    def register(cls, name: str, engine_cls: Type[BaseEngine]):
        """Register a new engine."""
        cls._engines[name] = engine_cls
        logger.debug(f"Registered engine: {name}")

    @classmethod
    def get(cls, name: str) -> Type[BaseEngine]:
        """Get an engine by name."""
        if name not in cls._engines:
            raise ValueError(f"Engine '{name}' not found. Available: {list(cls._engines.keys())}")
        return cls._engines[name]

    @classmethod
    def resolve(cls, data: Any = None) -> Type[BaseEngine]:
        """
        Auto-detect engine based on input data type.
        
        Args:
            data: The data object (DataFrame) to inspect.
            
        Returns:
            The compatible Engine class.
        """
        if data is None:
            return cls.get(cls._active_engine)
        
        # Check module path to identify the library
        module = type(data).__module__
        
        if "polars" in module:
            return cls.get("polars")
        if "pandas" in module:
            return cls.get("pandas")
        if "pyspark" in module:
            # Future proofing
            if "spark" in cls._engines:
                return cls.get("spark")
        if "dask" in module:
            # Future proofing
            if "dask" in cls._engines:
                return cls.get("dask")
                
        # Fallback to default if unknown (or let it fail later)
        logger.warning(f"Unknown data type {type(data)}, falling back to default engine: {cls._active_engine}")
        return cls.get(cls._active_engine)

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        """
        Auto-detect engine and wrap the data.
        """
        engine = cls.resolve(data)
        return engine.wrap(data)

# Global Helper
def get_engine(data: Any = None) -> Type[BaseEngine]:
    return EngineRegistry.resolve(data)
