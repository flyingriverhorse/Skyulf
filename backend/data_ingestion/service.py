import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import aiofiles  # type: ignore
from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.connectors.file import LocalFileConnector
from backend.data_ingestion.connectors.s3 import S3Connector
from backend.data_ingestion.schemas.ingestion import (
    DataSourceCreate,
    IngestionJobResponse,
)
from backend.data_ingestion.tasks import ingest_data_task
from backend.database.models import DataSource
from backend.services.data_service import DataService

logger = logging.getLogger(__name__)


class DataIngestionService:
    def __init__(self, session: AsyncSession, upload_dir: str = "uploads/data"):
        self.session = session
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.data_service = DataService()

    async def list_sources(
        self, user_id: Optional[int] = None, limit: int = 50, skip: int = 0
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

    async def list_usable_sources(
        self, user_id: Optional[int] = None
    ) -> Sequence[DataSource]:
        """
        List only successfully ingested data sources.
        """
        query = select(DataSource).where(DataSource.test_status == "success")
        if user_id:
            query = query.where(DataSource.created_by == user_id)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_source(self, source_id: Union[int, str]) -> Optional[DataSource]:
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

    async def get_data_source_by_id(self, ds_id: int) -> Optional[DataSource]:
        """Alias for get_source with int ID, for backward compatibility if needed."""
        return await self.get_source(ds_id)

    async def delete_source(self, source_id: Union[int, str]) -> bool:
        """
        Delete a data source and its associated file if applicable.
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
                    logger.error(f"Failed to delete file {file_path}: {e}")

        await self.session.delete(source)
        await self.session.commit()
        return True

    async def cancel_ingestion(self, source_id: Union[int, str]) -> bool:
        """
        Cancel an ongoing ingestion job.
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
                "updated_at": datetime.utcnow().isoformat(),
            }
            cast(Any, source).source_metadata = metadata
            await self.session.commit()
            return True

        return False

    async def get_sample(
        self, source_id: Union[int, str], limit: int = 5
    ) -> list[dict]:
        """
        Get a sample of data from the source.
        """
        from backend.data_ingestion.connectors.file import LocalFileConnector
        from backend.data_ingestion.connectors.s3 import S3Connector
        
        source = await self.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        config: Dict[str, Any] = cast(Dict[str, Any], source.config) or {}
        # Normalize path retrieval: check 'file_path' then 'path'
        file_path = config.get("file_path") or config.get("path")
        
        connector: Union[S3Connector, LocalFileConnector]

        # Check for S3 path first, regardless of source type
        if file_path and str(file_path).startswith("s3://"):
            storage_options = config.get("storage_options", {})
            
            # Ensure options are strings for Polars
            # Polars expects 'aws_access_key_id', NOT 'key'
            # So we don't need to remap here, but we need to ensure they are strings
            str_options = {k: str(v) for k, v in storage_options.items() if v is not None}
            
            try:
                logger.info(f"Fetching S3 sample from {file_path} with options keys: {list(str_options.keys())}")
                connector = S3Connector(file_path, storage_options=str_options)
                await connector.connect()
                df = await connector.fetch_data(limit=limit)
                return cast(List[Dict[str, Any]], df.to_dicts())
            except Exception as e:
                logger.error(f"Failed to get S3 sample: {e}", exc_info=True)
                # If it's a 403/404 from S3, it might come as ValueError from connector
                if "403" in str(e) or "404" in str(e):
                     raise HTTPException(status_code=400, detail=f"S3 Access Error: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to read S3 data sample: {str(e)}"
                )

        if source.type in ["file", "csv", "txt"]:
            if not file_path:
                raise HTTPException(status_code=400, detail="Missing file path")

            try:
                # Ensure we use the absolute path
                abs_path = Path(file_path).absolute()
                if not abs_path.exists():
                    # Try relative to workspace if absolute fails
                    abs_path = Path.cwd() / file_path

                return await self.data_service.get_sample(abs_path, limit=limit)
            except Exception as e:
                logger.error(f"Failed to get sample: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to read data sample: {str(e)}"
                )
        
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
                    return cast(List[Dict[str, Any]], df.to_dicts())
                 except Exception as e:
                     raise HTTPException(status_code=500, detail=f"Failed to read local parquet: {e}")

            try:
                logger.info(f"Fetching S3 sample from {file_path} with options keys: {list(storage_options.keys())}")
                # Ensure options are strings for Polars
                str_options = {k: str(v) for k, v in storage_options.items() if v is not None}
                connector = S3Connector(file_path, storage_options=str_options)
                await connector.connect()
                df = await connector.fetch_data(limit=limit)
                return df.to_dicts()
            except Exception as e:
                logger.error(f"Failed to get S3 sample: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to read S3 data sample: {str(e)}"
                )

        # TODO: Handle other source types (SQL, etc.)
        return []

    async def handle_file_upload(
        self,
        file: UploadFile,
        user_id: int,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> IngestionJobResponse:
        """
        Handle file upload and create a data source entry.
        """
        # 1. Generate unique filename
        filename = file.filename or "unknown"
        file_ext = Path(filename).suffix
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}{file_ext}"
        file_path = self.upload_dir / safe_filename

        # 2. Save file
        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    await out_file.write(content)
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file")

        # 3. Create DataSource record
        try:
            new_source = DataSource(
                source_id=file_id,
                name=file.filename,
                type="file",
                config={"file_path": str(file_path.absolute())},
                created_by=user_id,
                is_active=True,
                test_status="untested",
                source_metadata={
                    "ingestion_status": {
                        "status": "pending",
                        "progress": 0.0,
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                    "original_filename": file.filename,
                    "file_size": file_path.stat().st_size,
                },
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)

            # 4. Trigger Task
            settings = get_settings()
            if settings.USE_CELERY:
                ingest_data_task.delay(new_source.id)
            elif background_tasks:
                background_tasks.add_task(ingest_data_task, new_source.id)
            else:
                # Fallback: Run in thread
                import asyncio

                asyncio.create_task(asyncio.to_thread(ingest_data_task, new_source.id))

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
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_database_source(
        self,
        data: DataSourceCreate,
        user_id: int,
        background_tasks: Optional[BackgroundTasks] = None,
    ) -> IngestionJobResponse:
        """
        Create a database source and trigger ingestion.
        """
        try:
            source_id = str(uuid.uuid4())
            new_source = DataSource(
                source_id=source_id,
                name=data.name,
                type=data.type,  # 'postgres', 'mysql', etc.
                config=data.config,
                created_by=user_id,
                is_active=True,
                test_status="untested",
                description=data.description,
                source_metadata={
                    "ingestion_status": {
                        "status": "pending",
                        "progress": 0.0,
                        "updated_at": datetime.utcnow().isoformat(),
                    }
                },
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)

            # Trigger ingestion
            settings = get_settings()
            if settings.USE_CELERY:
                ingest_data_task.delay(new_source.id)
            elif background_tasks:
                background_tasks.add_task(ingest_data_task, new_source.id)
            else:
                # Fallback: Run in thread
                import asyncio

                asyncio.create_task(asyncio.to_thread(ingest_data_task, new_source.id))

            return IngestionJobResponse(
                job_id=str(new_source.id),
                status="pending",
                message="Database source created and ingestion started",
            )
        except Exception as e:
            logger.error(f"Failed to create database source: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to create source: {str(e)}"
            )

    async def get_ingestion_status(self, source_id: int) -> Dict[str, Any]:
        """
        Get the status of an ingestion job.
        """
        result = await self.session.execute(
            select(DataSource).where(DataSource.id == source_id)
        )
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="DataSource not found")

        metadata: Dict[str, Any] = cast(Dict[str, Any], source.source_metadata) or {}
        return cast(
            Dict[str, Any], metadata.get("ingestion_status", {"status": "unknown"})
        )
