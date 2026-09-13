"""Shared row limits for source previews, exports and job listings."""

from fastapi import HTTPException

from backend.config import get_settings

# Preserve the existing export ceiling for every explicit source sample.
# Connector fetch_data(None) remains the separate full-ingestion operation.
MAX_SAMPLE_ROWS = 50_000


def validate_limit(limit: int, maximum: int) -> int:
    """Reject non-positive or excessive row counts before readers or queries run."""
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be an integer between 1 and {maximum}")
    return limit


def validate_page_bounds(limit: int, skip: int = 0) -> int:
    """Validate job/source page bounds against the current configured page cap."""
    if type(skip) is not int or skip < 0:
        raise ValueError("skip must be a non-negative integer")
    return validate_limit(limit, get_settings().MAX_PAGE_SIZE)


def validate_query_limit(limit: int, maximum: int) -> int:
    """Report invalid HTTP row counts as 422 before dispatching to a service."""
    try:
        return validate_limit(limit, maximum)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
