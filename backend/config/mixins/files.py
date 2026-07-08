"""File handling, upload, and directory settings."""

import logging
from pathlib import Path
from typing import Annotated

from pydantic_settings import NoDecode


class FilesMixin:
    """Upload limits, allowed extensions, and directory paths."""

    UPLOAD_DIR: str = "uploads/data"
    # Configurable via MAX_UPLOAD_SIZE env var (bytes).
    # Default 10 GB — large enough for real-world datasets while still
    # protecting against runaway uploads on under-resourced deployments.
    # Docker Compose users can override with: MAX_UPLOAD_SIZE=5368709120
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024 * 1024  # 10 GB
    # NoDecode: pydantic-settings must not JSON-decode this before field_validator runs.
    ALLOWED_EXTENSIONS: Annotated[list[str], NoDecode] = [
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".json",
        ".txt",
        ".pkl",
        ".pickle",
        ".feather",
        ".h5",
        ".hdf5",
    ]
    TRAINING_ARTIFACT_DIR: str = "uploads/models"
    TEMP_DIR: str = "temp/processing"
    EXPORT_DIR: str = "exports/data"
    MODELS_DIR: str = "uploads/models"

    # Data ingestion feature toggles
    ENABLE_LINEAGE: bool = True
    ENABLE_SCHEMA_DRIFT: bool = True
    ENABLE_RETENTION: bool = True

    def create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories: list[str | Path] = [
            self.UPLOAD_DIR,  # type: ignore[attr-defined]
            self.TEMP_DIR,  # type: ignore[attr-defined]
            self.EXPORT_DIR,  # type: ignore[attr-defined]
            self.MODELS_DIR,  # type: ignore[attr-defined]
            Path(self.LOG_FILE).parent,  # ty: ignore[unresolved-attribute]
            "logs",
            "exports/models",
        ]
        for directory in directories:
            target_path = directory if isinstance(directory, Path) else Path(directory)
            target_path.mkdir(parents=True, exist_ok=True)
        logging.getLogger(__name__).info("Created %d application directories", len(directories))
