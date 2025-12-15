from typing import Any, Dict, Optional


class DataIngestionException(Exception):
    """Base exception for data ingestion errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
