"""Fast EDA helpers dedicated to feature-engineering workflows.

This module narrows the scope of the broader :mod:`core.eda` package to the
subset of capabilities the feature-engineering API depends on.  The goal is to
serve small previews and light-weight quality metadata without the heavier
text analysis or caching layers that slowed down the original service.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pandas.api import types as pd_types
from pandas.errors import DtypeWarning, ParserError
from sqlalchemy.ext.asyncio import AsyncSession

from core.data_ingestion.serialization import JSONSafeSerializer
from core.data_ingestion.service import DataIngestionService

__all__ = ["FeatureEngineeringEDAService"]


@dataclass
class _PreviewPayload:
    frame: pd.DataFrame
    total_rows: int


DEFAULT_SAMPLE_CAP = 200
LARGE_DATASET_ROW_THRESHOLD = 100_000
LARGE_DATASET_SIZE_THRESHOLD = 50 * 1024 * 1024  # 50MB
LARGE_DATASET_PREVIEW_CAP = 200


class FeatureEngineeringEDAService:
    """Lean EDA surface optimised for feature-engineering routes.

    The implementation focuses on two entrypoints that the canvas relies on:

    * :meth:`preview_source` – return a small sample of the dataset
    * :meth:`quality_report` – expose column-level quality metrics used by
      recommendation endpoints

    Expensive text analytics, recommendation scoring, and persistence logic are
    intentionally omitted here to keep requests fast.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        row_count_ttl: int = 300,
        sample_cap: Optional[int] = None,
        quality_report_ttl: int = 300,
    ) -> None:
        self.session = session
        self.data_service = DataIngestionService(session)
        self._row_count_cache: Dict[str, Tuple[int, float]] = {}
        self._row_count_ttl = max(0, int(row_count_ttl))
        self._quality_report_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._quality_report_ttl = max(0, int(quality_report_ttl))
        resolved_cap = sample_cap if isinstance(sample_cap, int) and sample_cap > 0 else DEFAULT_SAMPLE_CAP
        self._sample_cap = int(resolved_cap)
        self._large_dataset_preview_cap = min(self._sample_cap, LARGE_DATASET_PREVIEW_CAP)
        self._project_root = Path(__file__).resolve().parents[3]
        self._uploads_dir = self._project_root / "uploads" / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def preview_source(
        self,
        source_id: str,
        *,
        sample_size: int = 500,
        mode: str = "head",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Return a small dataset snapshot.

        Args:
            source_id: External source identifier used throughout the canvas.
            sample_size: Number of rows to sample. ``0`` loads the entire file.
            mode: Sampling flavour. ``"head"`` collects the top rows, whereas
                ``"first_last"`` stitches together the first and last chunks.
            force_refresh: Present for API compatibility; ignored because this
                lightweight implementation has no persistent cache.
        """

        file_path = await self._resolve_source_file_path(source_id)
        if not file_path:
            return {
                "success": False,
                "error": f"Data source '{source_id}' not found",
                "message": "Unable to locate file for preview.",
            }

        try:
            effective_sample, sampling_mode, adjustments, is_large = self._resolve_sampling_strategy(
                sample_size,
                mode,
                file_path,
            )

            preview = await asyncio.to_thread(
                self._generate_preview,
                file_path,
                effective_sample,
                sampling_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {
                "success": False,
                "error": f"Preview failed: {exc}",
                "message": "Unable to read sample from dataset.",
            }

        frame = preview.frame
        preview_payload = {
            "columns": frame.columns.tolist(),
            "sample_data": JSONSafeSerializer.clean_for_json(frame.to_dict("records")),
            "sample_size": int(frame.shape[0]),
            "total_rows": preview.total_rows,
            "estimated_total_rows": preview.total_rows,
            "dtypes": {column: self._normalize_dtype(dtype) for column, dtype in frame.dtypes.items()},
            "mode": sampling_mode,
        }

        if adjustments:
            preview_payload["sampling_adjustments"] = adjustments
        if is_large:
            preview_payload["large_dataset"] = True

        return {
            "success": True,
            "preview": preview_payload,
            "source_id": source_id,
            "meta": {"generated_at": time.time(), "from_cache": False},
        }

    async def quality_report(self, source_id: str, *, sample_size: int = 500) -> Dict[str, Any]:
        """Generate reduced quality metadata for the requested dataset."""

        # Check cache first
        cached_report = self._get_cached_quality_report(source_id, sample_size)
        if cached_report is not None:
            return cached_report

        file_path = await self._resolve_source_file_path(source_id)
        if not file_path:
            return {
                "success": False,
                "error": f"Data source '{source_id}' not found",
                "message": "Unable to locate file for quality report.",
            }

        try:
            effective_sample, sampling_mode, adjustments, is_large = self._resolve_sampling_strategy(
                sample_size,
                "head",
                file_path,
            )
            preview = await asyncio.to_thread(
                self._generate_preview,
                file_path,
                effective_sample,
                sampling_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {
                "success": False,
                "error": f"Quality report failed: {exc}",
                "message": "Unable to analyse dataset sample.",
            }

        quality_report = self._build_quality_report(preview.frame)
        payload = {
            "success": True,
            "quality_report": JSONSafeSerializer.clean_for_json(quality_report),
            "source_id": source_id,
            "meta": {
                "sample_size": int(preview.frame.shape[0]),
                "total_rows": preview.total_rows,
                "generated_at": time.time(),
                "sampling_mode": sampling_mode,
                "sampling_adjustments": adjustments,
                "large_dataset": is_large,
                "from_cache": False,
            },
        }

        # Cache the result
        self._set_cached_quality_report(source_id, sample_size, payload)

        return payload

    async def preview_rows_window(
        self,
        source_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        mode: str = "head",
    ) -> Dict[str, Any]:
        """Return a specific window of rows for incremental preview pagination."""

        file_path = await self._resolve_source_file_path(source_id)
        if not file_path:
            return {
                "success": False,
                "error": f"Data source '{source_id}' not found",
                "message": "Unable to locate file for preview window.",
            }

        effective_offset, effective_limit, adjustments, is_large = self._resolve_window_strategy(
            offset,
            limit,
            file_path,
        )

        normalized_mode = (mode or "head").strip().lower() or "head"
        if normalized_mode not in {"head"}:
            adjustments.append(f"window_mode_fallback:{normalized_mode}")

        try:
            frame = await asyncio.to_thread(
                self._read_window,
                file_path,
                effective_offset,
                effective_limit,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {
                "success": False,
                "error": f"Preview window failed: {exc}",
                "message": "Unable to load requested preview rows.",
            }

        if frame.empty:
            columns = []
        else:
            columns = frame.columns.tolist()

        total_rows = self._estimate_total_rows(file_path, frame, effective_limit)
        returned_rows = int(frame.shape[0])
        next_offset = effective_offset + returned_rows
        has_more = total_rows > next_offset if total_rows is not None else False

        preview_payload = {
            "columns": columns,
            "rows": JSONSafeSerializer.clean_for_json(frame.to_dict("records")),
            "offset": effective_offset,
            "limit": effective_limit,
            "returned_rows": returned_rows,
            "total_rows": total_rows,
            "next_offset": next_offset,
            "has_more": has_more,
            "mode": "window",
        }

        if adjustments:
            preview_payload["sampling_adjustments"] = adjustments
        if is_large:
            preview_payload["large_dataset"] = True

        return {
            "success": True,
            "preview": preview_payload,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _resolve_source_file_path(self, source_id: str) -> Optional[Path]:
        normalized = (source_id or "").strip()
        if not normalized:
            return None

        source = await self.data_service.get_data_source_by_source_id(normalized)
        if source is None and normalized.isdigit():
            try:
                source = await self.data_service.get_data_source(int(normalized))
            except Exception:  # pragma: no cover - defensive
                source = None

        if source is not None:
            config = getattr(source, "config", {}) or {}
            for key in ("file_path", "filepath", "path"):
                candidate = config.get(key)
                if candidate:
                    resolved = Path(candidate)
                    if resolved.exists():
                        return resolved

        if self._uploads_dir.exists():
            candidate = next(
                (
                    file_path
                    for file_path in self._uploads_dir.iterdir()
                    if file_path.is_file() and normalized in file_path.name
                ),
                None,
            )
            if candidate:
                return candidate

        return None

    def _generate_preview(self, file_path: Path, sample_size: int, mode: str) -> _PreviewPayload:
        frame = self._read_sample(file_path, sample_size, mode)
        total_rows = self._estimate_total_rows(file_path, frame, sample_size)
        return _PreviewPayload(frame=frame, total_rows=total_rows)

    def _read_sample(self, file_path: Path, sample_size: int, mode: str) -> pd.DataFrame:
        sample_limit = max(0, int(sample_size))
        mode_normalized = (mode or "head").strip().lower() or "head"
        extension = file_path.suffix.lower()

        if extension == ".csv":
            return self._read_csv(file_path, sample_limit, mode_normalized)
        if extension in {".xlsx", ".xls"}:
            return self._read_excel(file_path, sample_limit)
        if extension == ".json":
            return self._read_json(file_path, sample_limit)

        # Fallback: attempt pandas autodetection
        return self._safe_read_csv(file_path, nrows=sample_limit if sample_limit > 0 else None)

    def _read_csv(self, file_path: Path, sample_limit: int, mode: str) -> pd.DataFrame:
        if sample_limit == 0:
            return self._safe_read_csv(file_path)

        if mode == "first_last" and sample_limit > 0:
            total_rows = self._row_count_hint(file_path)
            if total_rows <= 0:
                total_rows = self._compute_row_count(file_path, fallback=sample_limit)
                self._set_cached_row_count(file_path, total_rows)

            if total_rows <= sample_limit:
                return self._safe_read_csv(file_path, nrows=sample_limit)

            head_rows = max(sample_limit // 2, 1)
            tail_rows = max(sample_limit - head_rows, 0)

            head = self._safe_read_csv(file_path, nrows=head_rows)
            if tail_rows:
                tail = self._read_csv_tail(file_path, tail_rows)
                if not tail.empty:
                    return pd.concat([head, tail], ignore_index=True)
            return head

        return self._safe_read_csv(file_path, nrows=sample_limit)

    def _read_csv_tail(self, file_path: Path, rows: int) -> pd.DataFrame:
        if rows <= 0:
            return pd.DataFrame()

        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.readline()
            buffer = deque(handle, maxlen=rows)

        if not header or not buffer:
            return pd.DataFrame()

        tail_data = header + "".join(buffer)
        return self._safe_read_csv(io.StringIO(tail_data))

    def _read_excel(self, file_path: Path, sample_limit: int) -> pd.DataFrame:
        frame = pd.read_excel(file_path)
        if sample_limit and sample_limit > 0:
            return frame.head(sample_limit)
        return frame

    def _read_json(self, file_path: Path, sample_limit: int) -> pd.DataFrame:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            if sample_limit and sample_limit > 0:
                data = data[:sample_limit]
            return pd.DataFrame(data)

        if isinstance(data, dict):
            return pd.DataFrame([data])

        return pd.DataFrame([])

    def _read_window(self, file_path: Path, offset: int, limit: int) -> pd.DataFrame:
        if limit <= 0:
            return pd.DataFrame()

        extension = file_path.suffix.lower()

        if extension == ".csv":
            return self._read_csv_window(file_path, offset, limit)
        if extension in {".xlsx", ".xls"}:
            return self._read_excel_window(file_path, offset, limit)
        if extension == ".json":
            frame = self._read_json(file_path, 0)
            if offset or limit:
                return frame.iloc[offset : offset + limit].reset_index(drop=True)
            return frame

        # Fallback: read using pandas autodetection then slice.
        frame = self._safe_read_csv(
            file_path,
            nrows=offset + limit if offset or limit else None,
        )
        return frame.iloc[offset : offset + limit].reset_index(drop=True)

    def _read_csv_window(self, file_path: Path, offset: int, limit: int) -> pd.DataFrame:
        if offset <= 0:
            return self._safe_read_csv(file_path, nrows=limit)

        skip_rows = range(1, offset + 1)
        return self._safe_read_csv(file_path, skiprows=skip_rows, nrows=limit)

    def _read_excel_window(self, file_path: Path, offset: int, limit: int) -> pd.DataFrame:
        if offset <= 0:
            return pd.read_excel(file_path, nrows=limit)

        skip_rows = range(1, offset + 1)
        return pd.read_excel(file_path, skiprows=skip_rows, nrows=limit)

    def _row_count_hint(self, file_path: Path) -> int:
        cached = self._get_cached_row_count(file_path)
        if cached is not None:
            return cached
        return 0

    def _safe_read_csv(self, handle: Any, **kwargs: Any) -> pd.DataFrame:
        read_kwargs: Dict[str, Any] = {"low_memory": False}
        read_kwargs.update(kwargs)

        # When the global pandas dtype backend prefers pyarrow, reuse that engine.
        if "engine" not in read_kwargs:
            try:
                dtype_backend = getattr(pd.options.mode, "dtype_backend", None)
            except AttributeError:  # pragma: no cover - pandas < 2.0
                dtype_backend = None
            if dtype_backend == "pyarrow":
                read_kwargs["engine"] = "pyarrow"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            try:
                return pd.read_csv(handle, **read_kwargs)
            except ParserError:
                fallback_kwargs = dict(read_kwargs)
                fallback_kwargs.pop("engine", None)
                fallback_kwargs.setdefault("engine", "python")
                fallback_kwargs.setdefault("on_bad_lines", "skip")

                if hasattr(handle, "seek"):
                    try:  # pragma: no cover - defensive reset for IO handles
                        handle.seek(0)
                    except (OSError, ValueError):
                        pass

                warnings.simplefilter("ignore", DtypeWarning)
                return pd.read_csv(handle, **fallback_kwargs)

    def _estimate_total_rows(self, file_path: Path, frame: pd.DataFrame, sample_size: int) -> int:
        # If we already cached an explicit row count, reuse it.
        cached = self._get_cached_row_count(file_path)
        if cached is not None:
            return cached

        # When the caller requested the full dataset, the sample already
        # contains every row.
        if sample_size <= 0:
            total = int(frame.shape[0])
            self._set_cached_row_count(file_path, total)
            return total

        # If the sampled frame is smaller than the requested sample, assume the
        # dataset is small and we already captured every row.
        if frame.shape[0] < sample_size:
            total = int(frame.shape[0])
            self._set_cached_row_count(file_path, total)
            return total

        # Otherwise fall back to a lightweight row count for supported formats.
        total_rows = self._compute_row_count(file_path, fallback=frame.shape[0])
        self._set_cached_row_count(file_path, total_rows)
        return total_rows

    def _compute_row_count(self, file_path: Path, *, fallback: int) -> int:
        extension = file_path.suffix.lower()
        try:
            if extension == ".csv":
                # Use faster buffered reading for large CSV files
                count = 0
                with file_path.open("rb") as handle:
                    # Read in 64KB chunks for faster counting
                    buffer_size = 65536
                    read_f = handle.raw.read if hasattr(handle, "raw") else handle.read
                    buf = read_f(buffer_size)
                    while buf:
                        count += buf.count(b'\n')
                        buf = read_f(buffer_size)
                # Subtract header row
                return max(count - 1, 0)
            if extension in {".xlsx", ".xls"}:
                return int(pd.read_excel(file_path, usecols=[0]).shape[0])
            if extension == ".json":
                with file_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    return len(data)
                return 1
        except Exception:  # pragma: no cover - defensive fallback
            return int(fallback)
        return int(fallback)

    def _get_cached_row_count(self, file_path: Path) -> Optional[int]:
        if self._row_count_ttl == 0:
            return None
        key = str(file_path.resolve())
        cached = self._row_count_cache.get(key)
        if not cached:
            return None
        count, timestamp = cached
        if time.time() - timestamp > self._row_count_ttl:
            self._row_count_cache.pop(key, None)
            return None
        return count

    def _set_cached_row_count(self, file_path: Path, count: int) -> None:
        if self._row_count_ttl == 0:
            return
        key = str(file_path.resolve())
        self._row_count_cache[key] = (int(count), time.time())

    def _get_cached_quality_report(self, source_id: str, sample_size: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached quality report if valid."""
        if self._quality_report_ttl == 0:
            return None

        cache_key = f"{source_id}:{sample_size}"
        cached = self._quality_report_cache.get(cache_key)
        if not cached:
            return None

        report, timestamp = cached
        if time.time() - timestamp > self._quality_report_ttl:
            self._quality_report_cache.pop(cache_key, None)
            return None

        # Update meta to indicate cache hit
        cached_report = dict(report)
        if "meta" in cached_report:
            cached_report["meta"] = dict(cached_report["meta"])
            cached_report["meta"]["from_cache"] = True
            cached_report["meta"]["cache_age_seconds"] = int(time.time() - timestamp)

        return cached_report

    def _set_cached_quality_report(self, source_id: str, sample_size: int, report: Dict[str, Any]) -> None:
        """Cache quality report for future requests."""
        if self._quality_report_ttl == 0:
            return

        cache_key = f"{source_id}:{sample_size}"
        self._quality_report_cache[cache_key] = (report, time.time())

    def _resolve_sampling_strategy(
        self,
        sample_size: int,
        mode: str,
        file_path: Path,
    ) -> Tuple[int, str, List[str], bool]:
        effective = max(0, int(sample_size))
        sample_mode = (mode or "head").strip().lower() or "head"
        adjustments: List[str] = []

        if effective > 0:
            effective = min(effective, self._sample_cap)

        is_large = self._is_large_dataset(file_path)

        if is_large:
            if effective <= 0:
                effective = self._large_dataset_preview_cap
                adjustments.append("limited_full_sample_for_large_dataset")
            elif effective > self._large_dataset_preview_cap:
                effective = self._large_dataset_preview_cap
                adjustments.append("capped_sample_for_large_dataset")

            if sample_mode == "head":
                sample_mode = "first_last"
                adjustments.append("switched_to_first_last_for_large_dataset")

        return effective, sample_mode, adjustments, is_large

    def _is_large_dataset(self, file_path: Path) -> bool:
        row_hint = self._row_count_hint(file_path)
        if row_hint and row_hint >= LARGE_DATASET_ROW_THRESHOLD:
            return True

        try:
            size_bytes = file_path.stat().st_size
        except OSError:  # pragma: no cover - filesystem edge case
            size_bytes = 0

        return size_bytes >= LARGE_DATASET_SIZE_THRESHOLD

    def _resolve_window_strategy(
        self,
        offset: int,
        limit: int,
        file_path: Path,
    ) -> Tuple[int, int, List[str], bool]:
        adjustments: List[str] = []

        try:
            effective_offset = int(offset)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            effective_offset = 0
            adjustments.append("default_offset")

        if effective_offset < 0:
            effective_offset = 0
            adjustments.append("normalized_negative_offset")

        try:
            effective_limit = int(limit)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            effective_limit = self._sample_cap
            adjustments.append("default_limit")

        if effective_limit <= 0:
            effective_limit = min(self._sample_cap, 50)
            adjustments.append("normalized_non_positive_limit")

        cap = self._sample_cap
        is_large = self._is_large_dataset(file_path)
        if is_large:
            cap = min(cap, self._large_dataset_preview_cap)

        if effective_limit > cap:
            effective_limit = cap
            adjustments.append("capped_limit_for_sample")

        return effective_offset, effective_limit, adjustments, is_large

    def _build_quality_report(self, frame: pd.DataFrame) -> Dict[str, Any]:
        # Handle named index (e.g. "Id") so it appears in the report
        if frame.index.name:
            frame = frame.reset_index(drop=False)

        raw_details: List[Optional[Dict[str, Any]]] = [
            self._build_column_detail(frame, column) for column in frame.columns
        ]
        column_details: List[Dict[str, Any]] = [
            detail for detail in raw_details if detail is not None
        ]

        missing_summary = [
            {
                "column": detail["name"],
                "missing_percentage": detail["missing_percentage"],
                "missing_count": detail["missing_count"],
            }
            for detail in column_details
            if detail["missing_percentage"] > 0.0
        ]

        recommendations = self._build_recommendations(column_details)
        text_categories = [
            detail["name"]
            for detail in column_details
            if detail.get("text_category") in {"categorical", "short_text"}
        ]

        quality_report: Dict[str, Any] = {
            "basic_metadata": {
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            },
            "quality_metrics": {
                "column_details": column_details,
            },
            "missing_data_summary": missing_summary,
            "recommendations": recommendations,
            "text_analysis_summary": {
                "categorical_text_columns": text_categories,
            },
            "sample_preview": {
                "columns": frame.columns.tolist(),
                "rows": JSONSafeSerializer.clean_for_json(frame.head(10).to_dict("records")),
            },
        }

        return quality_report

    def _build_column_detail(self, frame: pd.DataFrame, column: Any) -> Optional[Dict[str, Any]]:
        column_name = str(column).strip()
        if not column_name:
            return None

        series = frame[column]
        total_rows = frame.shape[0] or 1
        non_null_count = int(series.notna().sum())
        missing_count = int(total_rows - non_null_count)
        missing_pct = float((missing_count / total_rows) * 100.0) if total_rows else 0.0
        unique_non_missing = int(series.nunique(dropna=True)) if non_null_count else 0
        unique_pct = float((unique_non_missing / non_null_count) * 100.0) if non_null_count else 0.0

        dtype_label = self._normalize_dtype(series.dtype)

        text_category, avg_text_len = self._infer_text_metadata(series)

        detail = {
            "name": column_name,
            "column": column_name,
            "dtype": dtype_label,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 3),
            "null_percentage": round(missing_pct, 3),
            "unique_count": unique_non_missing,
            "unique_percentage": round(unique_pct, 3),
            "text_category": text_category,
            "avg_text_length": avg_text_len,
        }
        return detail

    def _normalize_dtype(self, dtype: Any) -> str:
        dtype_label = str(dtype).lower()
        if "int" in dtype_label:
            return "Int64"
        if "float" in dtype_label:
            return "float64"
        if "bool" in dtype_label:
            return "boolean"
        if "datetime" in dtype_label:
            return "datetime64[ns]"
        if "object" in dtype_label or "string" in dtype_label:
            return "string"
        if "category" in dtype_label:
            return "category"
        return str(dtype)

    def _infer_text_metadata(self, series: pd.Series) -> Tuple[Optional[str], Optional[float]]:
        is_text_like = (
            pd_types.is_object_dtype(series)
            or pd_types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_text_like:
            return None, None

        try:
            string_series = series.dropna().astype("string")
        except Exception:  # pragma: no cover - defensive
            return "categorical", None

        if string_series.empty:
            return "categorical", None

        avg_length = float(string_series.str.len().mean()) if not string_series.empty else None
        unique_ratio = (string_series.nunique(dropna=True) / max(len(string_series), 1)) if len(string_series) else 0.0

        if avg_length is not None and avg_length > 48:
            return "free_text", round(avg_length, 3)
        if unique_ratio > 0.7:
            return "high_cardinality", round(avg_length, 3) if avg_length is not None else None
        return "categorical", round(avg_length, 3) if avg_length is not None else None

    def _build_recommendations(self, column_details: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        recommendations: List[Dict[str, Any]] = []

        high_missing = [detail["name"] for detail in column_details if detail["missing_percentage"] >= 60.0]
        constant_columns = [detail["name"] for detail in column_details if detail["unique_count"] <= 1]

        if high_missing:
            recommendations.append(
                {
                    "title": "Drop columns with heavy missingness",
                    "description": "Columns above 60% missing values rarely contribute to models.",
                    "priority": "high",
                    "category": "data_quality",
                    "signal_type": "missing_data",
                    "columns": sorted(high_missing),
                    "tags": ["missing_data"],
                }
            )

        if constant_columns:
            recommendations.append(
                {
                    "title": "Remove constant columns",
                    "description": "Columns with a single unique value carry no predictive power.",
                    "priority": "medium",
                    "category": "data_quality",
                    "signal_type": "low_variance",
                    "columns": sorted(constant_columns),
                    "tags": ["low_variance"],
                }
            )

        return recommendations
