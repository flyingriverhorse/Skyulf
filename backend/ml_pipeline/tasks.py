import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Generator, List, Tuple

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.ml_pipeline._services.pipeline_execution_service import execute_pipeline

logger = logging.getLogger(__name__)

def _pipeline_span(job_id: str):
    """Return a Sentry transaction context manager, or a no-op if sentry-sdk is absent."""
    try:
        import sentry_sdk  # runtime check so tests can patch sys.modules without module reload

        @contextmanager
        def _span() -> Generator:
            with sentry_sdk.start_transaction(op="pipeline", name=f"run_pipeline/{job_id}") as tx:
                tx.set_tag("job_id", job_id)
                yield tx

        return _span()
    except ImportError:
        return nullcontext()


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
    with _pipeline_span(job_id):
        session = get_db_session()
        try:
            execute_pipeline(job_id, pipeline_config_dict, session)
        finally:
            session.close()


@shared_task(name="core.ml_pipeline.tasks.run_pipeline_batch_task")
def run_pipeline_batch_task(branches: List[Tuple[str, dict]]) -> None:
    """Run multiple pipeline branches in a single Celery task.

    Saves N-1 Redis round-trips vs submitting one task per branch.
    Branches run concurrently when N > 1.
    """

    def _run_one(job_id: str, config_dict: dict) -> None:
        with _pipeline_span(job_id):
            session = get_db_session()
            try:
                execute_pipeline(job_id, config_dict, session)
            finally:
                session.close()

    if len(branches) == 1:
        _run_one(*branches[0])
    else:
        with ThreadPoolExecutor(max_workers=len(branches)) as pool:
            futures = [pool.submit(_run_one, jid, pl) for jid, pl in branches]
            for f in futures:
                f.result()  # re-raises per-branch exceptions so Celery marks the task failed
