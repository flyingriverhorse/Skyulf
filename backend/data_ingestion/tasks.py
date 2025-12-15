import asyncio
import logging
from datetime import datetime, timezone

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.data_ingestion.connectors.api import ApiConnector
from backend.data_ingestion.connectors.base import BaseConnector
from backend.data_ingestion.connectors.file import LocalFileConnector
from backend.data_ingestion.connectors.sql import DatabaseConnector
from backend.data_ingestion.engine.profiler import DataProfiler
from backend.database.models import DataSource

logger = logging.getLogger(__name__)

# Helper to get sync session


def get_db_session():
    settings = get_settings()
    if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
        sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    else:
        sync_url = settings.DATABASE_URL.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )

    engine = create_engine(sync_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


@shared_task(name="core.data_ingestion.tasks.ingest_data_task")
def ingest_data_task(source_id: int):
    """
    Background task to ingest data from any source (File, SQL, etc.).
    """
    logger.info(f"Starting ingestion for source {source_id}")
    session = get_db_session()

    try:
        # 1. Get DataSource
        data_source = (
            session.query(DataSource).filter(DataSource.id == source_id).first()
        )
        if not data_source:
            logger.error(f"DataSource {source_id} not found")
            return

        # Update status to processing
        metadata = dict(data_source.source_metadata or {})
        metadata["ingestion_status"] = {
            "status": "processing",
            "progress": 0.1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        data_source.source_metadata = metadata
        session.commit()

        # 2. Select Connector
        config = data_source.config or {}
        connector: BaseConnector

        if data_source.type == "file":
            file_path = config.get("file_path")
            if not file_path:
                raise ValueError("Missing file_path in config")
            connector = LocalFileConnector(file_path)

        elif data_source.type in ["postgres", "mysql", "sqlite", "snowflake"]:
            # For SQL, we expect connection_string in config (or built from credentials)
            # In a real app, we would decrypt credentials here.
            connection_string = config.get("connection_string")
            table_name = config.get("table_name")
            query = config.get("query")

            if not connection_string:
                raise ValueError("Missing connection_string in config")

            connector = DatabaseConnector(
                connection_string, table_name=table_name, query=query
            )

        elif data_source.type == "api":
            url = config.get("url")
            method = config.get("method", "GET")
            headers = config.get("headers")
            params = config.get("params")
            data_key = config.get("data_key")

            if not url:
                raise ValueError("Missing url in config")

            connector = ApiConnector(
                url, method=method, headers=headers, params=params, data_key=data_key
            )

        else:
            raise ValueError(f"Unsupported source type: {data_source.type}")

        # 3. Run Ingestion (Async wrapper)
        async def run_ingestion():
            await connector.connect()
            # Load full data to get accurate counts
            # Note: For very large SQL tables, we might want to avoid fetching everything just for metadata.
            # But for now, we follow the pattern.
            df = await connector.fetch_data()

            # Run Profiling
            profile = DataProfiler.profile(df)

            return profile

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        profile = loop.run_until_complete(run_ingestion())
        loop.close()

        # 4. Update Metadata with Success
        metadata["ingestion_status"] = {
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Flatten profile into metadata for easy access
        metadata["schema"] = {
            col: stats["type"] for col, stats in profile["columns"].items()
        }
        metadata["row_count"] = profile["row_count"]
        metadata["column_count"] = profile["column_count"]
        metadata["profile"] = profile  # Store full profile

        data_source.source_metadata = metadata
        data_source.test_status = "success"
        data_source.last_tested = datetime.now(timezone.utc)

        session.commit()
        logger.info(f"Ingestion completed for source {source_id}")

    except Exception as e:
        logger.error(f"Ingestion failed for source {source_id}: {str(e)}")
        if session:
            # Re-query to ensure session is valid
            data_source = (
                session.query(DataSource).filter(DataSource.id == source_id).first()
            )
            if data_source:
                metadata = dict(data_source.source_metadata or {})
                metadata["ingestion_status"] = {
                    "status": "failed",
                    "error": str(e),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                data_source.source_metadata = metadata
                data_source.test_status = "failed"
                session.commit()
    finally:
        session.close()
