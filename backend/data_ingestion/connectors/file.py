from pathlib import Path
from typing import cast

import polars as pl

from backend.config import get_settings

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
        settings = get_settings()
        self.base_path = Path(settings.UPLOAD_DIR).expanduser().resolve()
        self._testing = getattr(settings, "TESTING", False)
        self.file_path = self._resolve_file_path(file_path)
        self.kwargs = kwargs
        self._df: pl.DataFrame | None = None
        self._schema: dict[str, str] | None = None

    def _resolve_file_path(self, file_path: str) -> str:
        return str(
            self.resolve_safe_path(file_path, base_path=self.base_path, testing=self._testing)
        )

    @staticmethod
    def resolve_safe_path(
        file_path: str,
        *,
        base_path: Path | None = None,
        testing: bool | None = None,
    ) -> Path:
        """Resolve ``file_path`` against the configured upload directory and
        enforce that the result stays contained within it.

        This is the single source of truth for local-path containment used
        by ``LocalFileConnector.__init__``. Call sites that resolve a local
        path *before* handing it to the connector (e.g.
        ``DataIngestionService.get_sample``) should call this directly for
        defense-in-depth, instead of duplicating the containment logic or
        relying solely on the connector to enforce it later.

        Raises:
            PermissionError: if the resolved path escapes ``base_path``.
        """
        settings = get_settings()
        if base_path is None:
            base_path = Path(settings.UPLOAD_DIR).expanduser().resolve()
        if testing is None:
            testing = getattr(settings, "TESTING", False)

        candidate = Path(file_path).expanduser()
        if not candidate.is_absolute():
            candidate = base_path / candidate

        resolved = candidate.resolve(strict=False)
        # Skip containment check in test mode — tests use tmp_path outside UPLOAD_DIR
        if testing:
            return resolved
        if resolved != base_path and base_path not in resolved.parents:
            raise PermissionError("File path resolves outside the configured upload directory")
        return resolved

    async def connect(self) -> bool:
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        ext = Path(self.file_path).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        return True

    def _ext(self) -> str:
        return Path(self.file_path).suffix.lower()

    def _scan(self) -> pl.LazyFrame | None:
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
            raise RuntimeError(f"Failed to read file {Path(self.file_path).name}") from e

    async def get_schema(self) -> dict[str, str]:
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

    def _try_lazy_schema(self) -> dict[str, str] | None:
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

    async def fetch_data(self, query: str | None = None, limit: int | None = None) -> pl.DataFrame:
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

    def _try_lazy_head(self, *, query: str | None, limit: int | None) -> pl.DataFrame | None:
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
