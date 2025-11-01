"""
Standard Response Schemas

Common response models used across the API.
"""

from pydantic import BaseModel
from typing import Any, Optional, Dict, List
from datetime import datetime


class StandardResponse(BaseModel):
    """Standard API response format."""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    timestamp: datetime = datetime.now()


class PaginatedResponse(StandardResponse):
    """Response with pagination metadata."""
    data: List[Any]
    pagination: Dict[str, Any]
    
    @classmethod
    def create(
        cls,
        items: List[Any],
        total: int,
        skip: int,
        limit: int,
        success: bool = True,
        message: str = "Data retrieved successfully"
    ):
        return cls(
            success=success,
            message=message,
            data=items,
            pagination={
                "total": total,
                "skip": skip,
                "limit": limit,
                "has_next": skip + len(items) < total,
                "has_prev": skip > 0
            }
        )


class ErrorResponse(StandardResponse):
    """Error response format."""
    success: bool = False
    data: None = None
    
    @classmethod
    def create(cls, message: str, errors: Optional[List[str]] = None):
        return cls(
            success=False,
            message=message,
            errors=errors or []
        )