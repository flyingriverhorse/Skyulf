"""Load entire datasets for feature-engineering pipeline execution."""

from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandas.errors import DtypeWarning
from sqlalchemy.ext.asyncio import AsyncSession

from core.data_ingestion.service import DataIngestionService


class FullDatasetCaptureService:
    """Utility wrapper that materialises full datasets without sampling."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.data_service = DataIngestionService(session)
        self._project_root = Path(__file__).resolve().parents[3]
        self._uploads_dir = self._project_root / "uploads" / "data"

    async def capture(self, source_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Return the full dataset frame alongside basic metadata."""
        normalized = (source_id or "").strip()
        if not normalized:
            raise ValueError("dataset_source_id must not be empty")

        file_path = await self._resolve_source_file_path(normalized)
        if file_path is None:
            raise FileNotFoundError(f"Data source '{normalized}' not found")

        frame = await asyncio.to_thread(self._read_full_frame, file_path)
        metadata: Dict[str, Any] = {
            "columns": frame.columns.tolist(),
            "total_rows": int(frame.shape[0]),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "file_path": str(file_path),
        }
        return frame, metadata

    async def _resolve_source_file_path(self, source_id: str) -> Optional[Path]:
        source = await self.data_service.get_data_source_by_source_id(source_id)
        if source is None and source_id.isdigit():
            try:
                source = await self.data_service.get_data_source(int(source_id))
            except Exception:
                source = None

        if source is not None:
            config = getattr(source, "config", {}) or {}
            for key in ("file_path", "filepath", "path"):
                candidate = config.get(key)
                if not candidate:
                    continue
                resolved = Path(candidate)
                if resolved.exists():
                    return resolved

        if self._uploads_dir.exists():
            for file_path in self._uploads_dir.iterdir():
                if file_path.is_file() and source_id in file_path.name:
                    return file_path

        return None

    def _read_full_frame(self, file_path: Path) -> pd.DataFrame:
        extension = file_path.suffix.lower()
        if extension == ".csv":
            return self._safe_read_csv(file_path)
        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        if extension == ".json":
            return self._read_json(file_path)
        if extension == ".parquet":
            return pd.read_parquet(file_path)
        return self._safe_read_csv(file_path)

    def _safe_read_csv(self, handle: Any, **kwargs: Any) -> pd.DataFrame:
        read_kwargs: Dict[str, Any] = {"low_memory": False}
        read_kwargs.update(kwargs)

        if "engine" not in read_kwargs:
            try:
                dtype_backend = getattr(pd.options.mode, "dtype_backend", None)
            except AttributeError:
                dtype_backend = None
            if dtype_backend == "pyarrow":
                read_kwargs["engine"] = "pyarrow"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            return pd.read_csv(handle, **read_kwargs)

    def _read_json(self, file_path: Path) -> pd.DataFrame:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame()


__all__ = ["FullDatasetCaptureService"]
