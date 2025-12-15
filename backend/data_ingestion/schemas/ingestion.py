from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class DataSourceCreate(BaseModel):
    name: str
    type: str = "file"
    config: Dict[str, Any] = {}
    description: Optional[str] = None


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    file_id: Optional[str] = None


class IngestionStatus(BaseModel):
    status: str  # pending, processing, completed, failed
    progress: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    updated_at: datetime


class DataSourceRead(BaseModel):
    id: int
    source_id: Optional[str]
    name: str
    type: str
    config: Dict[str, Any]
    is_active: bool
    test_status: str
    created_at: datetime
    updated_at: datetime
    source_metadata: Optional[Dict[str, Any]] = Field(None, alias="source_metadata")

    rows: Optional[int] = None
    columns: Optional[int] = None
    size_bytes: Optional[int] = None

    @model_validator(mode="after")
    def extract_metadata_fields(self):
        if self.source_metadata:
            self.rows = self.source_metadata.get("row_count")
            self.columns = self.source_metadata.get("column_count")
            self.size_bytes = self.source_metadata.get("file_size")
        return self

    class Config:
        from_attributes = True


class DataSourceListResponse(BaseModel):
    sources: list[DataSourceRead]


class DataSourceResponse(BaseModel):
    source: DataSourceRead


class DataSourceSampleResponse(BaseModel):
    data: list[Dict[str, Any]]
