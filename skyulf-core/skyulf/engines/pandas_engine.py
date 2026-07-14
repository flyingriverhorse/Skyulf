from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .protocol import SkyulfDataFrame
from .registry import BaseEngine, EngineName, EngineRegistry


def _to_positional_values(values: Any, target_index: pd.Index) -> Any:
    """Strip index alignment from Series-like values so assignment behaves
    positionally, matching Polars semantics instead of pandas' index-based
    alignment (which silently produces NaNs on mismatched indices)."""
    if isinstance(values, pd.Series) and not values.index.equals(target_index):
        if len(values) != len(target_index):
            raise ValueError(
                "Length mismatch: cannot assign a Series of length "
                f"{len(values)} to a DataFrame with {len(target_index)} rows."
            )
        values = pd.Series(values.to_numpy(), index=target_index)
    return values


class SkyulfPandasWrapper:
    """Wrapper for Pandas DataFrame to implement SkyulfDataFrame protocol."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def columns(self) -> Sequence[str]:
        return self._df.columns.tolist()

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def select(self, columns: list[str] | str) -> "SkyulfDataFrame":
        # Normalize a bare string to a single-element list so pandas returns
        # a DataFrame (matching Polars' select(), which always returns a
        # DataFrame) instead of a bare Series.
        if isinstance(columns, str):
            columns = [columns]
        return SkyulfPandasWrapper(self._df[columns])

    def drop(self, columns: list[str]) -> "SkyulfDataFrame":
        return SkyulfPandasWrapper(self._df.drop(columns=columns))

    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        # Pandas aligns Series/DataFrame-like values by index, while Polars
        # always assigns positionally. If `values` carries its own index that
        # doesn't match self._df's, a naive assign() would silently produce
        # NaNs instead of raising or matching Polars' positional semantics.
        values = _to_positional_values(values, self._df.index)
        return SkyulfPandasWrapper(self._df.assign(**{name: values}))

    def to_pandas(self) -> pd.DataFrame:
        return self._df

    def to_arrow(self) -> Any:
        import pyarrow as pa  # lazy import — optional dependency

        return pa.Table.from_pandas(self._df)

    def copy(self) -> "SkyulfDataFrame":
        return SkyulfPandasWrapper(self._df.copy())

    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        value = _to_positional_values(value, self._df.index)
        self._df[key] = value

    def __len__(self) -> int:
        return len(self._df)

    # Allow access to underlying dataframe methods for flexibility,
    # but this breaks the protocol abstraction if used.
    def __getattr__(self, name):
        return getattr(self._df, name)


class PandasEngine(BaseEngine):
    name = EngineName.PANDAS

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
