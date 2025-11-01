"""Base schemas for common patterns."""

from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat() if v else None},
    )


class TimestampMixin(BaseModel):
    """Mixin for models with timestamps."""
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class IdMixin(BaseModel):
    """Mixin for models with ID."""
    id: int = Field(..., description="Unique identifier", gt=0)