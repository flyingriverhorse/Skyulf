from typing import Any, List, Protocol, Sequence, Tuple, runtime_checkable
import pandas as pd

@runtime_checkable
class SkyulfDataFrame(Protocol):
    """
    The Universal DataFrame Interface for Skyulf.
    
    This protocol defines the minimum set of operations that any compute engine
    (Pandas, Polars, Spark, Dask) must support to be used within Skyulf nodes.
    """
    
    @property
    def columns(self) -> Sequence[str]:
        """Return the list of column names."""
        ...
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataframe (rows, cols)."""
        ...
    
    # Core Operations
    def select(self, columns: List[str]) -> "SkyulfDataFrame":
        """Select a subset of columns."""
        ...
        
    def drop(self, columns: List[str]) -> "SkyulfDataFrame":
        """Drop specified columns."""
        ...
        
    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        """
        Add or replace a column.
        
        Args:
            name: The name of the column.
            values: The data for the column (list, array, series).
        """
        ...
    
    # Bridges
    def to_pandas(self) -> pd.DataFrame:
        """Convert to a Pandas DataFrame."""
        ...
        
    def to_arrow(self) -> Any:
        """
        Convert to an Arrow Table/RecordBatch.
        Critical for zero-copy data transfer between engines.
        """
        ...

    def copy(self) -> "SkyulfDataFrame":
        """Return a copy of the dataframe."""
        ...
