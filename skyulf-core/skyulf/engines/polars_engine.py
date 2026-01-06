from typing import Any, List, Sequence, Tuple
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from .registry import BaseEngine, EngineRegistry
from .protocol import SkyulfDataFrame

class SkyulfPolarsWrapper:
    """Wrapper for Polars DataFrame to implement SkyulfDataFrame protocol."""
    
    def __init__(self, df: Any):
        # df is pl.DataFrame
        self._df = df
        
    @property
    def columns(self) -> Sequence[str]:
        return self._df.columns
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._df.shape
        
    def select(self, columns: List[str]) -> "SkyulfDataFrame":
        return SkyulfPolarsWrapper(self._df.select(columns))
        
    def drop(self, columns: List[str]) -> "SkyulfDataFrame":
        return SkyulfPolarsWrapper(self._df.drop(columns))
        
    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        # Polars with_columns takes expressions or series
        # We need to ensure values is compatible
        return SkyulfPolarsWrapper(self._df.with_columns(pl.Series(name, values)))
        
    def to_pandas(self) -> Any:
        return self._df.to_pandas()
        
    def to_arrow(self) -> Any:
        return self._df.to_arrow()

    def copy(self) -> "SkyulfDataFrame":
        return SkyulfPolarsWrapper(self._df.clone())

    def __getitem__(self, key):
        return self._df[key]
        
    def __getattr__(self, name):
        return getattr(self._df, name)

class PolarsEngine(BaseEngine):
    name = "polars"

    @classmethod
    def is_compatible(cls, data: Any) -> bool:
        if not HAS_POLARS: return False
        return isinstance(data, pl.DataFrame)

    @classmethod
    def from_pandas(cls, df: Any) -> Any:
        if not HAS_POLARS: raise ImportError("Polars not installed")
        return pl.from_pandas(df)

    @classmethod
    def to_numpy(cls, df: Any) -> Any:
        if isinstance(df, SkyulfPolarsWrapper):
            return df.to_pandas().to_numpy() # Polars -> Arrow -> Pandas -> Numpy (safest path for now)
        if hasattr(df, "to_numpy"):
            return df.to_numpy()
        return np.array(df)

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        if isinstance(data, SkyulfPolarsWrapper):
            return data
        return SkyulfPolarsWrapper(data)

    @classmethod
    def create_dataframe(cls, data: Any) -> Any:
        if not HAS_POLARS: raise ImportError("Polars not installed")
        return pl.DataFrame(data)

# Register automatically
if HAS_POLARS:
    EngineRegistry.register("polars", PolarsEngine)
