"""File handling, upload, and directory settings."""

import logging
from pathlib import Path
from typing import List


class FilesMixin:
    """Upload limits, allowed extensions, and directory paths."""

    UPLOAD_DIR: str = "uploads/data"
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 1024  # 1 GB
    ALLOWED_EXTENSIONS: List[str] = [
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
        directories: List[str | Path] = [
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
