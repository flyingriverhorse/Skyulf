from typing import Any, List, Sequence, Tuple
import pandas as pd
import numpy as np
import pyarrow as pa

from .registry import BaseEngine, EngineRegistry
from .protocol import SkyulfDataFrame

class SkyulfPandasWrapper:
    """Wrapper for Pandas DataFrame to implement SkyulfDataFrame protocol."""
    
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    @property
    def columns(self) -> Sequence[str]:
        return self._df.columns.tolist()
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._df.shape
        
    def select(self, columns: List[str]) -> "SkyulfDataFrame":
        return SkyulfPandasWrapper(self._df[columns])
        
    def drop(self, columns: List[str]) -> "SkyulfDataFrame":
        return SkyulfPandasWrapper(self._df.drop(columns=columns))
        
    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        # Handle various input types for values
        return SkyulfPandasWrapper(self._df.assign(**{name: values}))
        
    def to_pandas(self) -> pd.DataFrame:
        return self._df
        
    def to_arrow(self) -> Any:
        return pa.Table.from_pandas(self._df)

    def copy(self) -> "SkyulfDataFrame":
        return SkyulfPandasWrapper(self._df.copy())
        
    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        self._df[key] = value

    # Allow access to underlying dataframe methods for flexibility, 
    # but this breaks the protocol abstraction if used.
    def __getattr__(self, name):
        return getattr(self._df, name)

class PandasEngine(BaseEngine):
    name = "pandas"

    @classmethod
    def is_compatible(cls, data: Any) -> bool:
        return isinstance(data, pd.DataFrame)

    @classmethod
    def from_pandas(cls, df: Any) -> Any:
        return df

    @classmethod
    def to_numpy(cls, df: Any) -> Any:
        if hasattr(df, "to_numpy"):
            return df.to_numpy()
        if isinstance(df, SkyulfPandasWrapper):
            return df.to_pandas().to_numpy()
        return np.array(df)

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        if isinstance(data, SkyulfPandasWrapper):
            return data
        return SkyulfPandasWrapper(data)

    @classmethod
    def create_dataframe(cls, data: Any) -> Any:
        return pd.DataFrame(data)

# Register automatically
EngineRegistry.register("pandas", PandasEngine)
