import logging
import threading
import traceback
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.ml_pipeline._execution.strategies import JobStrategyFactory
from backend.ml_pipeline._services.pipeline_execution_service import execute_pipeline

logger = logging.getLogger(__name__)


def _pipeline_span(job_id: str):
    """Return a Sentry transaction context manager, or a no-op if sentry-sdk is absent."""
    try:
        import sentry_sdk  # ty: ignore[unresolved-import]  # noqa: E501

        @contextmanager
        def _span() -> Generator:
            with sentry_sdk.start_transaction(op="pipeline", name=f"run_pipeline/{job_id}") as tx:
                tx.set_tag("job_id", job_id)
                yield tx

        return _span()
    except ImportError:
        return nullcontext()


# Module-level cache — building a SQLAlchemy engine on every task call is expensive.
# Celery workers are long-lived processes; one engine per worker is correct.
_sync_engine = None
_sync_session_factory = None
_engine_init_lock = threading.Lock()


def get_db_session():
    global _sync_engine, _sync_session_factory
    if _sync_session_factory is None:
        with _engine_init_lock:
            # Double-checked locking: another thread may have finished
            # initializing while we were waiting for the lock.
            if _sync_session_factory is None:
                settings = get_settings()
                if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
                    sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
                else:
                    sync_url = settings.DATABASE_URL.replace(
                        "postgresql+asyncpg://", "postgresql+psycopg2://"
                    )
                _sync_engine = create_engine(sync_url, pool_pre_ping=True)
                _sync_session_factory = sessionmaker(
                    autocommit=False, autoflush=False, bind=_sync_engine
                )
    return _sync_session_factory()


def _mark_job_failed_if_unrecorded(job_id: str, error_msg: str) -> None:
    """Best-effort fallback: mark the job row as failed when an exception
    escapes ``execute_pipeline`` itself (which normally records failures on
    its own). This only fires for infra-level failures — e.g. a bug in
    ``execute_pipeline`` or a DB/session failure — so it must never raise.
    """
    try:
        session = get_db_session()
        try:
            job, strategy = JobStrategyFactory.find_job(session, job_id)
            if job and strategy and job.status not in ("failed", "completed", "cancelled"):
                strategy.handle_failure(job, error_msg)
                session.commit()
        finally:
            session.close()
    except Exception:
        logger.exception("Failed to mark job %s as failed after unhandled task error", job_id)


@shared_task(name="core.ml_pipeline.tasks.run_pipeline_task")
def run_pipeline_task(job_id: str, pipeline_config_dict: dict) -> None:
    """Celery entry point — unpacks args, delegates to the execution service."""
    with _pipeline_span(job_id):
        session = get_db_session()
        try:
            execute_pipeline(job_id, pipeline_config_dict, session)
        except Exception:
            from backend.exceptions.handlers import record_pipeline_error

            record_pipeline_error(
                job_id, traceback.format_exc().splitlines()[-1], traceback.format_exc()
            )
            _mark_job_failed_if_unrecorded(job_id, traceback.format_exc().splitlines()[-1])
            raise
        finally:
            session.close()


@shared_task(name="core.ml_pipeline.tasks.run_pipeline_batch_task")
def run_pipeline_batch_task(branches: list[tuple[str, dict]]) -> None:
    """Run multiple pipeline branches in a single Celery task.

    Saves N-1 Redis round-trips vs submitting one task per branch.
    Branches run concurrently when N > 1.
    """

    def _run_one(job_id: str, config_dict: dict) -> None:
        with _pipeline_span(job_id):
            session = get_db_session()
            try:
                execute_pipeline(job_id, config_dict, session)
            except Exception:
                from backend.exceptions.handlers import record_pipeline_error

                record_pipeline_error(
                    job_id, traceback.format_exc().splitlines()[-1], traceback.format_exc()
                )
                _mark_job_failed_if_unrecorded(job_id, traceback.format_exc().splitlines()[-1])
                raise
            finally:
                session.close()

    if len(branches) == 1:
        _run_one(*branches[0])
    else:
        max_workers = min(len(branches), get_settings().MAX_PARALLEL_BRANCH_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_run_one, jid, pl) for jid, pl in branches]
            errors = []
            for f in futures:
                try:
                    f.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                # Re-raise the first error; all failures are already logged
                # inside _run_one so Celery receives a meaningful traceback.
                raise errors[0]
