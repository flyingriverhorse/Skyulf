"""
Engine Registry for Skyulf.

This module handles the auto-detection of the appropriate compute engine
(Pandas, Polars, etc.) based on the input data type.
"""

import logging
from enum import StrEnum
from typing import Any

# We import the protocol for type checking, but we don't strictly need it at runtime here
# to avoid circular imports if engines import protocol.
from .protocol import SkyulfDataFrame

logger = logging.getLogger(__name__)


class EngineName(StrEnum):
    PANDAS = "pandas"
    POLARS = "polars"
    BASE = "base"


class BaseEngine:
    """Abstract base class for all engines."""

    name: EngineName = EngineName.BASE

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
    _engines: dict[str, type[BaseEngine]] = {}
    _active_engine: str = "pandas"  # Default

    @classmethod
    def register(cls, name: str, engine_cls: type[BaseEngine]):
        """Register a new engine."""
        cls._engines[name] = engine_cls
        logger.debug(f"Registered engine: {name}")

    @classmethod
    def get(cls, name: str) -> type[BaseEngine]:
        """Get an engine by name."""
        if name not in cls._engines:
            raise ValueError(f"Engine '{name}' not found. Available: {list(cls._engines.keys())}")
        return cls._engines[name]

    @classmethod
    def resolve(cls, data: Any = None) -> type[BaseEngine]:
        """
        Auto-detect engine based on input data type.

        Args:
            data: The data object (DataFrame) to inspect.

        Returns:
            The compatible Engine class.
        """
        if data is None:
            return cls.get(cls._active_engine)

        # Check the top-level package of the module path to identify the
        # library. Using the top-level component (rather than a bare
        # substring check) avoids false positives from unrelated modules
        # that merely contain "pandas"/"polars" in their name, e.g. a
        # third-party "fake_polars_stub" or "my_pandas_wrapper" module.
        module = type(data).__module__
        top_level = module.split(".", 1)[0]

        # Our own engine wrappers (SkyulfPandasWrapper/SkyulfPolarsWrapper,
        # under `skyulf.engines.*`) hold the real dataframe in `._df`; unwrap
        # and re-check *only* in that case so detection is based on the
        # underlying library. We deliberately don't do this unconditionally
        # for any object with a `._df` attribute, since e.g. polars'
        # DataFrame itself has an internal `._df` (its Rust-backed handle)
        # that would otherwise be mistakenly "unwrapped".
        if top_level == "skyulf" and hasattr(data, "_df"):
            module = type(data._df).__module__
            top_level = module.split(".", 1)[0]

        if top_level == "polars":
            return cls.get("polars")
        if top_level == "pandas":
            return cls.get("pandas")
        if top_level == "pyspark" and "spark" in cls._engines:
            # Future proofing
            return cls.get("spark")
        if top_level == "dask" and "dask" in cls._engines:
            # Future proofing
            return cls.get("dask")

        # Fallback to default if unknown (or let it fail later)
        # Plain Python sequences (list/tuple) are a common, expected input
        # shape (e.g. a raw y target list) rather than a genuinely unknown
        # type, so don't warn for those - only for anything else.
        if not isinstance(data, list | tuple):
            logger.warning(
                f"Unknown data type {type(data)}, falling back to default engine: {cls._active_engine}"
            )
        return cls.get(cls._active_engine)

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        """
        Auto-detect engine and wrap the data.
        """
        engine = cls.resolve(data)
        return engine.wrap(data)


# Global Helper
def get_engine(data: Any = None) -> type[BaseEngine]:
    return EngineRegistry.resolve(data)
