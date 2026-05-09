import os
from typing import Dict, Optional, cast

import polars as pl

from .base import BaseConnector


class LocalFileConnector(BaseConnector):
    """
    Connector for local files (CSV, Excel, Parquet, JSON).

    Schema and small samples are resolved lazily via `polars.scan_*`
    when the format supports it (CSV, Parquet) so that previewing a
    multi-GB file does not pull the whole thing into memory. Excel
    and JSON fall back to eager reads (polars has no `scan_excel` /
    `scan_json` for our supported file shapes).
    """

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
    # Formats where polars exposes a streaming/lazy reader.
    _LAZY_EXTENSIONS = {".csv", ".parquet"}

    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path
        self.kwargs = kwargs
        self._df: Optional[pl.DataFrame] = None
        self._schema: Optional[Dict[str, str]] = None

    async def connect(self) -> bool:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        return True

    def _ext(self) -> str:
        return os.path.splitext(self.file_path)[1].lower()

    def _scan(self) -> Optional[pl.LazyFrame]:
        """Return a lazy frame for formats that support scanning."""
        ext = self._ext()
        try:
            if ext == ".csv":
                return pl.scan_csv(self.file_path, **self.kwargs)
            if ext == ".parquet":
                return pl.scan_parquet(self.file_path, **self.kwargs)
        except Exception:
            return None
        return None

    async def _load_data(self) -> None:
        """Eagerly materialise the full file. Only call when the full frame is needed."""
        if self._df is not None:
            return

        ext = self._ext()

        try:
            if ext == ".csv":
                self._df = pl.read_csv(self.file_path, **self.kwargs)
            elif ext in [".xlsx", ".xls"]:
                self._df = pl.read_excel(self.file_path, **self.kwargs)
            elif ext == ".parquet":
                self._df = pl.read_parquet(self.file_path, **self.kwargs)
            elif ext == ".json":
                self._df = pl.read_json(self.file_path, **self.kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to read file {self.file_path}: {str(e)}")

    async def get_schema(self) -> Dict[str, str]:
        if self._schema is not None:
            return self._schema

        lazy_schema = self._try_lazy_schema()
        if lazy_schema is not None:
            self._schema = lazy_schema
            return self._schema

        await self._load_data()
        if self._df is None:
            raise RuntimeError("Data not loaded")

        self._schema = {col: str(dtype) for col, dtype in self._df.schema.items()}
        return self._schema

    def _try_lazy_schema(self) -> Optional[Dict[str, str]]:
        """Read just the file header / parquet footer when the format supports it."""
        if self._df is not None or self._ext() not in self._LAZY_EXTENSIONS:
            return None
        lf = self._scan()
        if lf is None:
            return None
        try:
            schema = lf.collect_schema()
            return {col: str(dtype) for col, dtype in schema.items()}
        except Exception:
            return None

    async def fetch_data(
        self, query: Optional[str] = None, limit: Optional[int] = None
    ) -> pl.DataFrame:
        lazy_head = self._try_lazy_head(query=query, limit=limit)
        if lazy_head is not None:
            return lazy_head

        await self._load_data()
        if self._df is None:
            raise RuntimeError("Data not loaded")

        df = self._df

        if limit:
            df = df.head(limit)

        return df

    def _try_lazy_head(
        self, *, query: Optional[str], limit: Optional[int]
    ) -> Optional[pl.DataFrame]:
        """Stream a bounded head from CSV/Parquet without materialising the file."""
        if (
            limit is None
            or limit <= 0
            or self._df is not None
            or query is not None
            or self._ext() not in self._LAZY_EXTENSIONS
        ):
            return None
        lf = self._scan()
        if lf is None:
            return None
        try:
            return cast(pl.DataFrame, lf.head(limit).collect())
        except Exception:
            return None

    async def validate(self) -> bool:
        return await self.connect()
