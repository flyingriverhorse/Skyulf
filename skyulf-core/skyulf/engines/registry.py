"""Engine Registry for Skyulf.

This module handles the auto-detection of the appropriate compute engine
(Pandas, Polars, etc.) based on the input data type.
"""

import logging
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, overload

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

# protocol.py is a leaf module (no engine imports), so this top-level import is safe.
from .protocol import DistributedDataFrame, SkyulfDataFrame

logger = logging.getLogger(__name__)


class EngineName(StrEnum):
    """Enumeration of local and distributed compute-engine identities."""

    PANDAS = "pandas"
    POLARS = "polars"
    SPARK = "spark"
    BASE = "base"


class BaseEngine:
    """Abstract base class for all engines."""

    name: EngineName = EngineName.BASE

    @classmethod
    def ensure_available(cls) -> None:
        """Validate optional runtime dependencies; local engines need no extra check."""

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
    def wrap(cls, data: Any) -> SkyulfDataFrame | DistributedDataFrame:
        """Wrap the native dataframe using its local or distributed contract."""
        raise NotImplementedError

    @classmethod
    def create_dataframe(cls, data: Any) -> Any:
        """Create a native dataframe from a dictionary or list."""
        raise NotImplementedError


class EngineRegistry:
    """Registry mapping engine names to ``BaseEngine`` classes and resolving the active one."""

    _engines: ClassVar[dict[str, type[BaseEngine]]] = {}
    # Polars is the default fallback; recognized frames select their own engine.
    # Explicit overrides stay local to the caller's thread or async context.
    _active_engine: ClassVar[ContextVar[str]] = ContextVar("skyulf_active_engine", default="polars")

    # Maps a data object's detected top-level module package to the engine
    # name registered for it. Recognized Spark inputs never fall back locally.
    _TOP_LEVEL_TO_ENGINE: ClassVar[dict[str, str]] = {
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
        engine = cls._engines[name]
        engine.ensure_available()
        return engine

    @classmethod
    def set_active_engine(cls, name: str) -> None:
        """Set the fallback engine for the current thread or async context.

        New async tasks inherit their creator's selection; independent threads
        start with Polars unless a context is explicitly copied into them.
        Recognized input types still determine their own engine in ``resolve``.
        """
        cls.get(name)
        cls._active_engine.set(name)
        logger.debug(f"Active engine set to: {name}")

    @classmethod
    def resolve(cls, data: Any = None) -> type[BaseEngine]:
        """Auto-detect engine based on input data type.

        Args:
            data: The data object (DataFrame) to inspect.

        Returns:
            The compatible Engine class.
        """
        if data is None:
            return cls.get(cls._active_engine.get())

        top_level = cls._detect_top_level_package(data)
        engine_name = cls._TOP_LEVEL_TO_ENGINE.get(top_level)
        if engine_name == "spark":
            if "spark" not in cls._engines:
                raise ImportError("Spark engine is not registered; cannot use a local fallback.")
            engine = cls.get("spark")
            if not engine.is_compatible(data):
                raise TypeError("Spark engine requires a pyspark.sql.DataFrame.")
            return engine
        if engine_name is not None and engine_name in cls._engines:
            return cls.get(engine_name)

        # Fallback to default if unknown (or let it fail later)
        cls._warn_unknown_data_type(data)
        return cls.get(cls._active_engine.get())

    @staticmethod
    def _detect_top_level_package(data: Any) -> str:
        """Return the top-level module package name backing `data`'s type.

        Checks the top-level component of the module path (rather than a
        bare substring check) to identify the library, avoiding false
        positives from unrelated modules that merely contain "pandas"/
        "polars" in their name, e.g. a third-party "fake_polars_stub" or
        "my_pandas_wrapper" module.

        Our own engine wrappers (SkyulfPandasWrapper/SkyulfPolarsWrapper,
        under `skyulf.engines.*`) hold the real dataframe behind a public
        `to_native()` accessor; unwrap and re-check *only* in that case so
        detection is based on the underlying library. Only our wrappers
        define `to_native()` (raw polars/pandas frames have no such method),
        so it is a reliable wrapper discriminator and we never touch polars'
        own internal `._df` handle.
        """
        top_level = type(data).__module__.split(".", 1)[0]
        if top_level == "skyulf" and hasattr(data, "to_native"):
            data = data.to_native()
            top_level = type(data).__module__.split(".", 1)[0]
        if any(base.__module__.split(".", 1)[0] == "pyspark" for base in type(data).__mro__):
            return "pyspark"
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
                f"Unknown data type {type(data)}, falling back to default engine: "
                f"{cls._active_engine.get()}"
            )

    @overload
    @classmethod
    def wrap(cls, data: "pd.DataFrame | pl.DataFrame | SkyulfDataFrame") -> SkyulfDataFrame: ...

    @overload
    @classmethod
    def wrap(cls, data: Any) -> SkyulfDataFrame | DistributedDataFrame: ...

    @classmethod
    def wrap(cls, data: Any) -> SkyulfDataFrame | DistributedDataFrame:
        """Auto-detect engine and wrap the data."""
        engine = cls.resolve(data)
        return engine.wrap(data)


# Global Helper
def get_engine(data: Any = None) -> type[BaseEngine]:
    """Return the ``BaseEngine`` that handles ``data`` (shortcut for ``EngineRegistry.resolve``)."""
    return EngineRegistry.resolve(data)
