"""Data loading and preview utilities."""

import logging
from typing import Any, Dict, Literal, Tuple

from fastapi import HTTPException
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from core.feature_engineering.eda_fast import FeatureEngineeringEDAService
from core.feature_engineering.eda_fast.service import DEFAULT_SAMPLE_CAP
from core.feature_engineering.full_capture import FullDatasetCaptureService

logger = logging.getLogger(__name__)


def coerce_int(value: Any, fallback: int) -> int:
    """Safely coerce a value to int, returning fallback on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def resolve_sample_cap(requested_size: Any) -> int:
    """Ensure the EDA service sample cap honors larger caller requests."""
    try:
        resolved = int(requested_size)
    except (TypeError, ValueError):
        resolved = 0

    if resolved <= 0:
        return DEFAULT_SAMPLE_CAP

    return max(DEFAULT_SAMPLE_CAP, resolved)


def build_eda_service(session: AsyncSession, requested_size: Any) -> FeatureEngineeringEDAService:
    return FeatureEngineeringEDAService(session, sample_cap=resolve_sample_cap(requested_size))


def build_preview_frame(preview_payload: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preview_data = preview_payload.get("preview") or {}
    sample_rows = preview_data.get("sample_data") or []

    try:
        frame = pd.DataFrame(sample_rows)
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        return frame, preview_data

    return frame, preview_data


async def load_dataset_frame(
    session: AsyncSession,
    dataset_source_id: str,
    *,
    sample_size: int,
    mode: str = "head",
    execution_mode: Literal["auto", "sample", "full"] = "auto",
    allow_empty_sample: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve a dataset frame using sampling or full capture as needed."""

    normalized_id = dataset_source_id.strip() if isinstance(dataset_source_id, str) else str(dataset_source_id)
    if not normalized_id:
        return pd.DataFrame(), {}

    normalized_mode = (mode or "head").strip().lower() or "head"
    should_use_full_capture = (
        execution_mode == "full"
        or (execution_mode == "auto" and sample_size == 0 and not allow_empty_sample)
    )

    if should_use_full_capture:
        capture_service = FullDatasetCaptureService(session)
        result = await capture_service.capture_full_dataset(normalized_id)
        
        if result.success and result.frame is not None:
            return result.frame, {
                "total_rows": len(result.frame),
                "sample_size": len(result.frame),
                "is_full_capture": True,
            }
        
        # Fallback to sampling if full capture fails or returns empty
        logger.warning(f"Full capture failed for {normalized_id}, falling back to sampling")

    effective_sample = sample_size if sample_size > 0 else DEFAULT_SAMPLE_CAP
    if allow_empty_sample and sample_size == 0:
        effective_sample = 0
        
    eda_service = build_eda_service(session, effective_sample)
    preview_payload = await eda_service.preview_source(
        normalized_id,
        sample_size=effective_sample,
        mode=normalized_mode,
    )

    if not preview_payload.get("success"):
        return pd.DataFrame(), {}

    preview = preview_payload.get("preview") or {}
    sample_rows = preview.get("sample_data") or []

    try:
        frame = pd.DataFrame(sample_rows)
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        return frame, preview

    total_rows = coerce_int(preview.get("total_rows"), frame.shape[0])
    sample_rows_used = coerce_int(preview.get("sample_size"), frame.shape[0])

    return frame, {
        "total_rows": total_rows,
        "sample_size": sample_rows_used,
        "is_full_capture": False,
    }


async def load_dataset_frame(
    session: AsyncSession,
    dataset_source_id: str,
    *,
    sample_size: int,
    mode: str = "head",
    execution_mode: Literal["auto", "sample", "full"] = "auto",
    allow_empty_sample: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve a dataset frame using sampling or full capture as needed."""

    normalized_id = dataset_source_id.strip() if isinstance(dataset_source_id, str) else str(dataset_source_id)
    if not normalized_id:
        raise HTTPException(status_code=400, detail="dataset_source_id must not be empty")

    normalized_mode = (mode or "head").strip().lower() or "head"
    should_use_full_capture = (
        execution_mode == "full"
        or (execution_mode == "auto" and sample_size == 0 and not allow_empty_sample)
    )

    if should_use_full_capture:
        capture_service = FullDatasetCaptureService(session)
        try:
            frame, metadata = await capture_service.capture(normalized_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        total_rows = coerce_int(metadata.get("total_rows"), frame.shape[0])
        return frame, {
            "total_rows": total_rows,
            "sample_size": int(frame.shape[0]),
            "columns": metadata.get("columns") or frame.columns.tolist(),
            "dtypes": metadata.get("dtypes") or {},
            "mode": "full_capture",
            "sampling_adjustments": [],
            "large_dataset": False,
        }

    effective_sample = sample_size if sample_size > 0 else DEFAULT_SAMPLE_CAP
    if allow_empty_sample and sample_size == 0:
        effective_sample = 0
    eda_service = build_eda_service(session, effective_sample)
    preview_payload = await eda_service.preview_source(
        normalized_id,
        sample_size=effective_sample,
        mode=normalized_mode,
    )

    if not preview_payload.get("success"):
        detail = preview_payload.get("error") or preview_payload.get("message") or "Unable to preview dataset"
        raise HTTPException(status_code=400, detail=detail)

    preview = preview_payload.get("preview") or {}
    sample_rows = preview.get("sample_data") or []

    try:
        frame = pd.DataFrame(sample_rows)
    except Exception:
        frame = pd.DataFrame()

    if frame.empty:
        columns = preview.get("columns") or []
        if columns:
            frame = pd.DataFrame(columns=columns)

    total_rows = coerce_int(preview.get("total_rows"), frame.shape[0])
    sample_rows_used = coerce_int(preview.get("sample_size"), frame.shape[0])

    return frame, {
        "total_rows": total_rows,
        "sample_size": sample_rows_used,
        "columns": preview.get("columns") or frame.columns.tolist(),
        "dtypes": preview.get("dtypes") or {},
        "mode": preview.get("mode") or normalized_mode,
        "sampling_adjustments": preview.get("sampling_adjustments") or [],
        "large_dataset": bool(preview.get("large_dataset", False)),
    }

