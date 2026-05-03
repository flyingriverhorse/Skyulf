import logging

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.ml_pipeline.services.pipeline_execution_service import execute_pipeline

logger = logging.getLogger(__name__)


def get_db_session():
    settings = get_settings()
    if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
        sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    else:
        sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    engine = create_engine(sync_url)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)()


@shared_task(name="core.ml_pipeline.tasks.run_pipeline_task")
def run_pipeline_task(job_id: str, pipeline_config_dict: dict) -> None:
    """Celery entry point — unpacks args, delegates to the execution service."""
    session = get_db_session()
    try:
        execute_pipeline(job_id, pipeline_config_dict, session)
    finally:
        session.close()
