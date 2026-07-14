import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiofiles
from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.schemas.ingestion import (
    DataSourceCreate,
    IngestionJobResponse,
)
from backend.data_ingestion.tasks import ingest_data_task
from backend.database.models import DataSource
from backend.exceptions.core import ForbiddenException, ResourceNotFoundException, SkyulfException
from backend.services.data_service import DataService

logger = logging.getLogger(__name__)


class DataIngestionService:
    def __init__(self, session: AsyncSession, upload_dir: str = "uploads/data"):
        self.session = session
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_service = DataService()

    async def list_sources(
        self, user_id: int | None = None, limit: int = 50, skip: int = 0
    ) -> Sequence[DataSource]:
        """
        List all data sources.
        """
        query = select(DataSource)
        if user_id:
            query = query.where(DataSource.created_by == user_id)

        # Order by created_at desc for consistent pagination
        query = query.order_by(DataSource.created_at.desc())
        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def list_usable_sources(self, user_id: int | None = None) -> Sequence[DataSource]:
        """
        List only successfully ingested data sources.
        """
        query = select(DataSource).where(DataSource.test_status == "success")
        if user_id:
            query = query.where(DataSource.created_by == user_id)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_source(self, source_id: int | str) -> DataSource | None:
        """
        Get a data source by ID (PK) or source_id (UUID).
        """
        stmt = select(DataSource)

        if isinstance(source_id, int):
            stmt = stmt.where(DataSource.id == source_id)
        elif isinstance(source_id, str):
            if source_id.isdigit():
                stmt = stmt.where(DataSource.id == int(source_id))
            else:
                stmt = stmt.where(DataSource.source_id == source_id)

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_data_source_by_id(self, ds_id: int) -> DataSource | None:
        """Alias for get_source with int ID, for backward compatibility if needed."""
        return await self.get_source(ds_id)

    async def delete_source(self, source_id: int | str) -> bool:
        """
        Delete a data source and its associated file if applicable.

        File removal is attempted before the DB row is deleted, but a
        failure to remove the file (permissions, file locked, already
        gone, etc.) does not block deleting the data source record — a
        stuck/missing file on disk shouldn't prevent a user from removing
        the source. If removal fails, we log a clear, discoverable ERROR
        that flags the file as orphaned (no DB row will reference it
        afterwards) so an operator can find and clean it up manually.
        """
        source = await self.get_source(source_id)
        if not source:
            return False

        # Delete file if it exists
        if source.type == "file" and source.config:
            file_path = source.config.get("file_path")
            if file_path:
                try:
                    path = Path(file_path)
                    if path.exists():
                        path.unlink()
                        logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(
                        "Orphaned file: failed to delete '%s' while deleting source %s "
                        "(the data source record will still be removed); manual cleanup "
                        "required. Error: %s",
                        file_path,
                        source_id,
                        e,
                    )

        await self.session.delete(source)
        await self.session.commit()
        return True

    async def cancel_ingestion(self, source_id: int | str) -> bool:
        """
        Cancel an ongoing ingestion job.

        Idempotent: returns True if the job is now in a non-running state
        (i.e. either we just cancelled it, or it had already finished /
        been cancelled). Returns False only when the source itself does
        not exist — that lets the router answer 404 cleanly without
        flagging benign races as 400 errors. The most common cause of a
        spurious 400 was the UI's 5 s poll racing the user's click on a
        job that completed in the same tick.
        """
        source = await self.get_source(source_id)
        if not source:
            return False

        metadata = dict(source.source_metadata or {})
        ingestion_status = metadata.get("ingestion_status", {})
        current_status = ingestion_status.get("status")

        if current_status in ["pending", "processing"]:
            metadata["ingestion_status"] = {
                "status": "cancelled",
                "progress": ingestion_status.get("progress", 0.0),
                "error": "Cancelled by user",
                "updated_at": datetime.now(UTC).isoformat(),
            }
            cast(Any, source).source_metadata = metadata
            await self.session.commit()

        # Already finished / cancelled / not yet started → desired state
        # already holds, so the cancel is a successful no-op.
        return True

    async def get_sample(self, source_id: int | str, limit: int = 5) -> list[dict]:
        """
        Get a sample of data from the source.
        """
        from backend.data_ingestion.connectors.file import LocalFileConnector
        from backend.data_ingestion.connectors.s3 import S3Connector

        source = await self.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        config: dict[str, Any] = cast(dict[str, Any], source.config) or {}
        # Normalize path retrieval: check 'file_path' then 'path'
        file_path = config.get("file_path") or config.get("path")

        connector: S3Connector | LocalFileConnector

        # Check for S3 path first, regardless of source type
        if file_path and str(file_path).startswith("s3://"):
            storage_options = config.get("storage_options", {})

            # Ensure options are strings for Polars
            # Polars expects 'aws_access_key_id', NOT 'key'
            # So we don't need to remap here, but we need to ensure they are strings
            str_options = {k: str(v) for k, v in storage_options.items() if v is not None}

            try:
                logger.info(
                    f"Fetching S3 sample from {file_path} with options keys: {list(str_options.keys())}"
                )
                connector = S3Connector(file_path, storage_options=str_options)
                await connector.connect()
                df = await connector.fetch_data(limit=limit)
                return df.to_dicts()
            except (ForbiddenException, ResourceNotFoundException) as e:
                # Typed exceptions raised by S3Connector for 403/404 provider
                # responses — classified there rather than string-matched here.
                logger.error(f"Failed to get S3 sample: {e}", exc_info=True)
                raise HTTPException(
                    status_code=400, detail="S3 access denied or resource not found"
                ) from e
            except Exception as e:
                logger.error(f"Failed to get S3 sample: {e}", exc_info=True)
                raise SkyulfException(message="Failed to read S3 data sample") from e

        if source.type in ["file", "csv", "txt"]:
            if not file_path:
                raise HTTPException(status_code=400, detail="Missing file path")

            # Defense-in-depth: this branch calls `data_service.get_sample`
            # directly instead of going through `LocalFileConnector`, so it
            # does NOT automatically benefit from the connector's
            # `resolve_safe_path` containment guard. Apply the same guard
            # explicitly here so a crafted/attacker-influenced `file_path`
            # (e.g. via source config) cannot resolve outside the configured
            # upload directory, even if this call site is ever reached with
            # an unvalidated path.
            try:
                abs_path = LocalFileConnector.resolve_safe_path(str(file_path))
            except PermissionError as e:
                logger.warning("Rejected out-of-bounds file path for sample: %s", file_path)
                raise HTTPException(status_code=400, detail="Invalid file path") from e

            try:
                if not abs_path.exists():
                    # Try relative to workspace if absolute fails
                    abs_path = LocalFileConnector.resolve_safe_path(str(Path.cwd() / file_path))

                return await self.data_service.get_sample(abs_path, limit=limit)
            except HTTPException:
                raise
            except PermissionError as e:
                logger.warning("Rejected out-of-bounds file path for sample: %s", file_path)
                raise HTTPException(status_code=400, detail="Invalid file path") from e
            except Exception as e:
                logger.error(f"Failed to get sample: {e}")
                raise SkyulfException(message="Failed to read data sample") from e

        elif source.type in ["s3", "parquet"]:
            # This block might be redundant now if file_path starts with s3://,
            # but kept for cases where type is explicit but path might not be standard s3:// (unlikely)
            # or for parquet files that are local.

            storage_options = config.get("storage_options", {})

            if not file_path:
                raise HTTPException(status_code=400, detail="Missing path")

            # If it's local parquet
            if not str(file_path).startswith("s3://"):
                try:
                    abs_path = Path(file_path).absolute()
                    connector = LocalFileConnector(str(abs_path))
                    await connector.connect()
                    df = await connector.fetch_data(limit=limit)
                    return df.to_dicts()
                except PermissionError as e:
                    # Raised by LocalFileConnector's containment guard when
                    # `abs_path` resolves outside the configured upload dir.
                    logger.warning("Rejected out-of-bounds parquet path for sample: %s", file_path)
                    raise HTTPException(status_code=400, detail="Invalid file path") from e
                except Exception:
                    logger.exception("Failed to read local parquet: %s", file_path)
                    raise SkyulfException(message="Failed to read local parquet file") from None

            try:
                logger.info(
                    f"Fetching S3 sample from {file_path} with options keys: {list(storage_options.keys())}"
                )
                # Ensure options are strings for Polars
                str_options = {k: str(v) for k, v in storage_options.items() if v is not None}
                connector = S3Connector(file_path, storage_options=str_options)
                await connector.connect()
                df = await connector.fetch_data(limit=limit)
                return df.to_dicts()
            except (ForbiddenException, ResourceNotFoundException) as e:
                logger.error(f"Failed to get S3 sample: {e}")
                raise HTTPException(
                    status_code=400, detail="S3 access denied or resource not found"
                ) from e
            except Exception as e:
                logger.error(f"Failed to get S3 sample: {e}")
                raise SkyulfException(message="Failed to read S3 data sample") from e

        # TODO: Handle other source types (SQL, etc.)
        return []

    # Source types created via `handle_create_source` (inline config, no file
    # upload). "file" sources go through `handle_file_upload` instead.
    _INLINE_SOURCE_TYPES = frozenset({"s3"})

    async def handle_create_source(
        self,
        payload: DataSourceCreate,
        user_id: int,
        background_tasks: BackgroundTasks | None = None,
    ) -> IngestionJobResponse:
        """
        Create a data source from an inline config (e.g. S3) and start ingestion.
        """
        if payload.type not in self._INLINE_SOURCE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported source type '{payload.type}'. Supported: "
                    f"{', '.join(sorted(self._INLINE_SOURCE_TYPES))}"
                ),
            )
        if payload.type == "s3":
            path = payload.config.get("path")
            if not path:
                raise HTTPException(
                    status_code=400, detail="Missing 'path' in config for s3 source"
                )
            # `get_source_sample`/connectors treat any non-`s3://` path for a
            # type="s3" source as a *local filesystem path* (no traversal or
            # extension checks, unlike /upload). Without this guard, this
            # endpoint would let any caller register an arbitrary server-local
            # file as a "s3" source and read it back via ingestion/sampling.
            if not str(path).startswith("s3://"):
                raise HTTPException(
                    status_code=400,
                    detail="'path' must be an s3:// URI for s3 sources",
                )

        source_id = str(uuid.uuid4())
        try:
            new_source = DataSource(
                source_id=source_id,
                name=payload.name,
                type=payload.type,
                config=payload.config,
                created_by=user_id,
                is_active=True,
                test_status="untested",
                source_metadata={
                    "ingestion_status": {
                        "status": "pending",
                        "progress": 0.0,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                    "description": payload.description,
                },
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)

            settings = get_settings()
            if settings.USE_CELERY:
                ingest_data_task.delay(new_source.id)
            elif background_tasks:
                background_tasks.add_task(ingest_data_task, new_source.id)
            else:
                # Fallback: Run in thread — retain a strong reference so the
                # task is not garbage-collected before it finishes.
                import asyncio

                _task = asyncio.create_task(asyncio.to_thread(ingest_data_task, new_source.id))
                _source_id = new_source.id

                def _on_done(t: asyncio.Task) -> None:
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.error("Ingestion task failed for source %s: %s", _source_id, exc)

                _task.add_done_callback(_on_done)

            return IngestionJobResponse(
                job_id=str(new_source.id),
                status="pending",
                message=f"'{payload.type}' source created and ingestion started",
                file_id=source_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise SkyulfException(message=f"Database error: {str(e)}") from e

    async def handle_file_upload(
        self,
        file: UploadFile,
        user_id: int,
        background_tasks: BackgroundTasks | None = None,
    ) -> IngestionJobResponse:
        """
        Handle file upload and create a data source entry.
        """
        settings = get_settings()

        # 0. Reject early via Content-Length if the client declares a size —
        #    avoids buffering a huge body only to discard it at the end.
        content_length = file.headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
                if declared_size > settings.MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: declared {declared_size / (1024**3):.1f} GB, "
                            f"limit is {settings.MAX_UPLOAD_SIZE / (1024**3):.0f} GB. "
                            "Set the MAX_UPLOAD_SIZE env var (bytes) to raise this limit."
                        ),
                    )
            except ValueError:
                pass  # Malformed header — let the streaming check catch it

        # 1. Validate filename — reject path traversal attempts
        raw_name = file.filename or "unknown"
        if ".." in raw_name or raw_name.startswith(("/", "\\")):
            raise HTTPException(status_code=400, detail="Invalid filename")

        # 2. Validate extension against allowlist
        file_ext = Path(raw_name).suffix.lower()
        allowed = [e.lower() for e in settings.ALLOWED_EXTENSIONS]
        if file_ext not in allowed:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed)}",
            )

        # 3. Generate unique filename and save, enforcing size limit
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}{file_ext}"
        file_path = self.upload_dir / safe_filename
        bytes_written = 0

        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    bytes_written += len(content)
                    if bytes_written > settings.MAX_UPLOAD_SIZE:
                        file_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File too large: {bytes_written / (1024**3):.1f} GB received, "
                                f"limit is {settings.MAX_UPLOAD_SIZE / (1024**3):.0f} GB. "
                                "Set the MAX_UPLOAD_SIZE env var (bytes) to raise this limit."
                            ),
                        )
                    await out_file.write(content)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise SkyulfException(message="Failed to save file") from e

        # 4. Create DataSource record
        try:
            new_source = DataSource(
                source_id=file_id,
                name=raw_name,
                type="file",
                config={"file_path": str(file_path.absolute())},
                created_by=user_id,
                is_active=True,
                test_status="untested",
                source_metadata={
                    "ingestion_status": {
                        "status": "pending",
                        "progress": 0.0,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                    "original_filename": raw_name,
                    "file_size": file_path.stat().st_size,
                },
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)

            # 5. Trigger Task
            if settings.USE_CELERY:
                ingest_data_task.delay(new_source.id)
            elif background_tasks:
                background_tasks.add_task(ingest_data_task, new_source.id)
            else:
                # Fallback: Run in thread — retain a strong reference so the
                # task is not garbage-collected before it finishes.
                import asyncio

                _task = asyncio.create_task(asyncio.to_thread(ingest_data_task, new_source.id))
                _source_id = new_source.id

                def _on_done(t: asyncio.Task) -> None:
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.error("Ingestion task failed for source %s: %s", _source_id, exc)

                _task.add_done_callback(_on_done)

            return IngestionJobResponse(
                job_id=str(new_source.id),  # Using source ID as job ID for now
                status="pending",
                message="File uploaded and ingestion started",
                file_id=file_id,
            )

        except Exception as e:
            logger.error(f"Database error: {e}")
            # Cleanup file if DB fails
            if file_path.exists():
                file_path.unlink()
            raise SkyulfException(message=f"Database error: {str(e)}") from e

    async def get_ingestion_status(self, source_id: int) -> dict[str, Any]:
        """
        Get the status of an ingestion job.
        """
        result = await self.session.execute(select(DataSource).where(DataSource.id == source_id))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="DataSource not found")

        metadata: dict[str, Any] = cast(dict[str, Any], source.source_metadata) or {}
        return cast(dict[str, Any], metadata.get("ingestion_status", {"status": "unknown"}))
