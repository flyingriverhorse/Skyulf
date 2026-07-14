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

    # Maps a data object's detected top-level module package to the engine
    # name registered for it. "spark"/"dask" are future-proofing: only used
    # if/when those engines are actually registered in `_engines`.
    _TOP_LEVEL_TO_ENGINE: dict[str, str] = {
        "polars": "polars",
        "pandas": "pandas",
        "pyspark": "spark",
        "dask": "dask",
    }

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
    def set_active_engine(cls, name: str) -> None:
        """Set the default/fallback engine used when no data is given to resolve()."""
        if name not in cls._engines:
            raise ValueError(f"Engine '{name}' not found. Available: {list(cls._engines.keys())}")
        cls._active_engine = name
        logger.debug(f"Active engine set to: {name}")

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

        top_level = cls._detect_top_level_package(data)
        engine_name = cls._TOP_LEVEL_TO_ENGINE.get(top_level)
        if engine_name is not None and engine_name in cls._engines:
            return cls.get(engine_name)

        # Fallback to default if unknown (or let it fail later)
        cls._warn_unknown_data_type(data)
        return cls.get(cls._active_engine)

    @staticmethod
    def _detect_top_level_package(data: Any) -> str:
        """Return the top-level module package name backing `data`'s type.

        Checks the top-level component of the module path (rather than a
        bare substring check) to identify the library, avoiding false
        positives from unrelated modules that merely contain "pandas"/
        "polars" in their name, e.g. a third-party "fake_polars_stub" or
        "my_pandas_wrapper" module.

        Our own engine wrappers (SkyulfPandasWrapper/SkyulfPolarsWrapper,
        under `skyulf.engines.*`) hold the real dataframe in `._df`; unwrap
        and re-check *only* in that case so detection is based on the
        underlying library. We deliberately don't do this unconditionally
        for any object with a `._df` attribute, since e.g. polars' DataFrame
        itself has an internal `._df` (its Rust-backed handle) that would
        otherwise be mistakenly "unwrapped".
        """
        top_level = type(data).__module__.split(".", 1)[0]
        if top_level == "skyulf" and hasattr(data, "_df"):
            top_level = type(data._df).__module__.split(".", 1)[0]
        return top_level

    @classmethod
    def _warn_unknown_data_type(cls, data: Any) -> None:
        """Log a warning for a genuinely unrecognized data type.

        Plain Python sequences (list/tuple) are a common, expected input
        shape (e.g. a raw y target list) rather than a genuinely unknown
        type, so don't warn for those - only for anything else.
        """
        if not isinstance(data, list | tuple):
            logger.warning(
                f"Unknown data type {type(data)}, falling back to default engine: {cls._active_engine}"
            )

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
