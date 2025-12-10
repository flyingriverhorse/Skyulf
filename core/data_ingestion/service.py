import os
import uuid
import aiofiles
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Union
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from core.database.models import DataSource, User
from core.data_ingestion.schemas.ingestion import DataSourceCreate, IngestionJobResponse
from core.data_ingestion.tasks import ingest_data_task
from core.data_ingestion.connectors.file import LocalFileConnector

logger = logging.getLogger(__name__)

class DataIngestionService:
    def __init__(self, session: AsyncSession, upload_dir: str = "uploads/data"):
        self.session = session
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def list_sources(self, user_id: int = None) -> list[DataSource]:
        """
        List all data sources.
        """
        query = select(DataSource)
        if user_id:
            query = query.where(DataSource.created_by == user_id)
        
        result = await self.session.execute(query)
        return result.scalars().all()

    async def list_usable_sources(self, user_id: int = None) -> list[DataSource]:
        """
        List only successfully ingested data sources.
        """
        query = select(DataSource).where(DataSource.test_status == 'success')
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
        if source.type == 'file' and source.config:
            file_path = source.config.get('file_path')
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
        ingestion_status = metadata.get('ingestion_status', {})
        current_status = ingestion_status.get('status')
        
        if current_status in ['pending', 'processing']:
            metadata['ingestion_status'] = {
                'status': 'cancelled',
                'progress': ingestion_status.get('progress', 0.0),
                'error': 'Cancelled by user',
                'updated_at': datetime.utcnow().isoformat()
            }
            source.source_metadata = metadata
            await self.session.commit()
            return True
            
        return False

    async def get_sample(self, source_id: Union[int, str], limit: int = 5) -> list[dict]:
        """
        Get a sample of data from the source.
        """
        source = await self.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        config = source.config or {}
        
        if source.type == 'file' or source.type == 'csv' or source.type == 'txt':
            file_path = config.get('file_path')
            if not file_path:
                raise HTTPException(status_code=400, detail="Missing file path")
            
            try:
                # Ensure we use the absolute path
                abs_path = Path(file_path).absolute()
                if not abs_path.exists():
                     # Try relative to workspace if absolute fails
                     abs_path = Path(os.getcwd()) / file_path
                
                connector = LocalFileConnector(str(abs_path))
                await connector.connect()
                # Pass limit to fetch_data to avoid loading full file if possible
                df = await connector.fetch_data(limit=limit)
                
                # Convert to list of dicts, handling potential serialization issues
                # Polars to_dicts handles basic types well
                return df.to_dicts()
            except Exception as e:
                logger.error(f"Failed to get sample: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to read data sample: {str(e)}")
        
        # TODO: Handle other source types (SQL, etc.)
        return []

    async def handle_file_upload(self, file: UploadFile, user_id: int) -> IngestionJobResponse:
        """
        Handle file upload and trigger ingestion task.
        """
        # 1. Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}{file_ext}"
        file_path = self.upload_dir / safe_filename

        # 2. Save file
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
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
                        "updated_at": datetime.utcnow().isoformat()
                    },
                    "original_filename": file.filename,
                    "file_size": os.path.getsize(file_path)
                }
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)
            
            # 4. Trigger Celery Task
            ingest_data_task.delay(new_source.id)
            
            return IngestionJobResponse(
                job_id=str(new_source.id), # Using source ID as job ID for now
                status="pending",
                message="File uploaded and ingestion started",
                file_id=file_id
            )

        except Exception as e:
            logger.error(f"Database error: {e}")
            # Cleanup file if DB fails
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_database_source(self, data: DataSourceCreate, user_id: int) -> IngestionJobResponse:
        """
        Create a database source and trigger ingestion.
        """
        try:
            source_id = str(uuid.uuid4())
            new_source = DataSource(
                source_id=source_id,
                name=data.name,
                type=data.type, # 'postgres', 'mysql', etc.
                config=data.config,
                created_by=user_id,
                is_active=True,
                test_status="untested",
                description=data.description,
                source_metadata={
                    "ingestion_status": {
                        "status": "pending",
                        "progress": 0.0,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            self.session.add(new_source)
            await self.session.commit()
            await self.session.refresh(new_source)
            
            # Trigger ingestion
            ingest_data_task.delay(new_source.id)
            
            return IngestionJobResponse(
                job_id=str(new_source.id),
                status="pending",
                message="Database source created and ingestion started"
            )
        except Exception as e:
            logger.error(f"Failed to create database source: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create source: {str(e)}")

    async def get_ingestion_status(self, source_id: int) -> dict:
        """
        Get the status of an ingestion job.
        """
        result = await self.session.execute(
            select(DataSource).where(DataSource.id == source_id)
        )
        source = result.scalar_one_or_none()
        
        if not source:
            raise HTTPException(status_code=404, detail="DataSource not found")
            
        metadata = source.source_metadata or {}
        return metadata.get("ingestion_status", {"status": "unknown"})
