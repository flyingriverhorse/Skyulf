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
    # `.pkl`/`.pickle`/`.h5`/`.hdf5` are intentionally excluded: nothing in the
    # ingestion pipeline (LocalFileConnector.SUPPORTED_EXTENSIONS) actually
    # reads these formats today, and pickle/unsafe-HDF5 deserialization of an
    # arbitrary user-uploaded file is a latent RCE risk if a loader for these
    # formats is ever added on this same upload path. Model artifacts (which
    # do use `.pkl`/`.joblib`) are stored and loaded through a separate,
    # trusted path (backend/ml_pipeline/deployment), not this upload allow-list.
    ALLOWED_EXTENSIONS: Annotated[list[str], NoDecode] = [
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".json",
        ".txt",
        ".feather",
    ]
    TRAINING_ARTIFACT_DIR: str = "uploads/models"
    TEMP_DIR: str = "temp/processing"
    EXPORT_DIR: str = "exports/data"
    MODELS_DIR: str = "uploads/models"

    # Data ingestion feature toggles
    ENABLE_LINEAGE: bool = True
    ENABLE_SCHEMA_DRIFT: bool = True
    ENABLE_RETENTION: bool = True

    # When serializing large DataFrames/collections to JSON (see
    # data_ingestion/serialization.py), yield control back to the event loop
    # via `asyncio.sleep(0)` every N rows/items so a very large payload
    # doesn't block other requests for the whole serialization duration.
    SERIALIZATION_YIELD_THRESHOLD_ROWS: int = 1000

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
