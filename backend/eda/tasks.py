import asyncio
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.models import DataSource, EDAReport, get_database_session
from backend.services.data_service import DataService
from backend.utils.file_utils import extract_file_path_from_source
from skyulf.profiling.analyzer import EDAAnalyzer

logger = logging.getLogger(__name__)


def _resolve_file_path(data_source: DataSource) -> str | Path | None:
    """Resolve the file path for a DataSource, falling back to source_id if it looks like a CSV/Parquet path."""
    source_data_dict = {
        "config": data_source.config or {},
        "connection_info": data_source.source_metadata or {},
        "file_path": (data_source.config or {}).get("file_path"),
        "source_id": data_source.source_id,
    }

    file_path: str | Path | None = extract_file_path_from_source(source_data_dict)

    # Debug logging
    logger.info(
        f"Resolving path for DataSource {data_source.id} ({data_source.type}). Found: {file_path}"
    )

    # Last resort: check if source_id looks like a path
    if (
        not file_path
        and data_source.source_id
        and (
            str(data_source.source_id).endswith(".csv")
            or str(data_source.source_id).endswith(".parquet")
        )
    ):
        file_path = data_source.source_id

    return file_path


def _resolve_s3_credentials(data_source: DataSource) -> dict:
    """Resolve raw S3 credentials for a DataSource, trying explicit creds, then config, then env settings."""
    # 1. Try explicit credentials field
    creds = data_source.credentials or {}

    # 2. If empty, try config (sometimes stored there)
    if not creds:
        config_creds = data_source.config or {}
        creds = {
            "aws_access_key_id": config_creds.get("aws_access_key_id"),
            "aws_secret_access_key": config_creds.get("aws_secret_access_key"),
            "aws_session_token": config_creds.get("aws_session_token"),
            "endpoint_url": config_creds.get("endpoint_url"),
        }

    # 3. If still empty, try system environment variables (via Settings)
    if not creds.get("aws_access_key_id"):
        settings = get_settings()
        creds = {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "aws_session_token": settings.AWS_SESSION_TOKEN,
            "endpoint_url": None,  # Usually standard AWS
        }
        if creds.get("aws_access_key_id"):
            logger.info("Using AWS credentials from environment variables.")

    return creds


def _build_s3_storage_options(data_source: DataSource) -> dict:
    """Resolve S3 credentials for a DataSource into storage_options, mapped to the boto3/s3fs key names."""
    creds = _resolve_s3_credentials(data_source)

    # Map common boto3 keys to s3fs keys if necessary
    storage_options = {
        "key": creds.get("aws_access_key_id") or creds.get("key"),
        "secret": creds.get("aws_secret_access_key") or creds.get("secret"),
        "token": creds.get("aws_session_token") or creds.get("token"),
        "endpoint_url": creds.get("endpoint_url"),
    }
    # Remove None values
    storage_options = {k: v for k, v in storage_options.items() if v is not None}

    if storage_options:
        logger.info(f"Using S3 credentials with keys: {list(storage_options.keys())}")
    else:
        logger.warning(
            "S3 path detected but no credentials found in DataSource (credentials or config)."
        )

    return storage_options


def _run_eda_analyzer(df, report_config: dict | None):
    """Run EDAAnalyzer.analyze using the target/exclude/filter/task_type settings from the report config."""
    exclude_cols = report_config.get("exclude_cols") if report_config else None
    filters = report_config.get("filters") if report_config else None
    target_col = report_config.get("target_col") if report_config else None
    task_type = report_config.get("task_type") if report_config else None

    analyzer = EDAAnalyzer(df)
    return analyzer.analyze(
        target_col=target_col,
        exclude_cols=exclude_cols,
        filters=filters,
        task_type=task_type,
    )


async def _fail_report_safely(
    session: AsyncSession, report: EDAReport | None, report_id: int, error: Exception
):
    """Roll back the session and best-effort mark the report FAILED after an unexpected error."""
    try:
        await session.rollback()
        if report:
            report.status = "FAILED"
            report.error_message = str(error)
            await session.commit()
    except Exception:
        logger.warning("Failed to update report %s status to FAILED", report_id, exc_info=True)


async def _resolve_or_fail_file_path(
    session: AsyncSession, report: EDAReport, data_source: DataSource
) -> str | Path | None:
    """Resolve the dataset file path, marking the report FAILED and committing if not found."""
    file_path = _resolve_file_path(data_source)
    if not file_path:
        report.status = "FAILED"
        report.error_message = (
            f"File path not found for source {data_source.id}. Type: {data_source.type}"
        )
        await session.commit()
        return None
    return file_path


async def _load_dataframe_or_fail(
    session: AsyncSession,
    report: EDAReport,
    data_source: DataSource,
    data_service: DataService,
    file_path: str | Path,
) -> Any:
    """Load the dataset used for analysis, marking the report FAILED and committing on failure."""
    storage_options = None
    if file_path and str(file_path).startswith("s3://"):
        storage_options = _build_s3_storage_options(data_source)

    try:
        return await data_service.load_file(
            file_path, force_type="polars", storage_options=storage_options
        )
    except Exception as e:
        report.status = "FAILED"
        report.error_message = f"Failed to load data: {str(e)}"
        await session.commit()
        return None


async def _run_analysis_or_fail(session: AsyncSession, report: EDAReport, df: Any) -> Any:
    """Run the EDA analyzer, marking the report FAILED and committing on failure."""
    try:
        return _run_eda_analyzer(df, report.config)
    except Exception as e:
        report.status = "FAILED"
        report.error_message = f"Analysis failed: {str(e)}"
        await session.commit()
        return None


async def run_eda_analysis(report_id: int, session: AsyncSession):
    """
    Core logic to run EDA analysis.
    """
    report = None
    try:
        logger.info(f"Starting EDA analysis for report {report_id}")

        # 1. Fetch Report and DataSource
        report = await session.get(EDAReport, report_id)
        if not report:
            logger.error(f"EDAReport {report_id} not found.")
            return

        data_source = await session.get(DataSource, report.data_source_id)
        if not data_source:
            report.status = "FAILED"
            report.error_message = "DataSource not found."
            await session.commit()
            return

        # 2. Load Data
        # Use the robust file path extraction logic from file_utils
        file_path = await _resolve_or_fail_file_path(session, report, data_source)
        if not file_path:
            return

        data_service = DataService()

        # load_file is async
        df = await _load_dataframe_or_fail(session, report, data_source, data_service, file_path)
        if df is None:
            return

        # 3. Run Analysis
        # Run in thread pool if CPU bound? Polars releases GIL mostly, so it's fine.
        profile = await _run_analysis_or_fail(session, report, df)
        if profile is None:
            return

        # 4. Save Result
        # model_dump is Pydantic v2
        report.profile_data = profile.model_dump(mode="json")
        report.status = "COMPLETED"
        await session.commit()
        logger.info(f"EDA Analysis completed for report {report_id}")

    except Exception as e:
        logger.error(f"EDA Analysis failed for report {report_id}: {e}")
        await _fail_report_safely(session, report, report_id, e)


async def run_eda_background(report_id: int):
    """
    Entry point for FastAPI BackgroundTasks.
    Creates its own session.
    """
    async with get_database_session() as session:
        await run_eda_analysis(report_id, session)


# Celery Task Definition
# We import celery_app only if needed to avoid circular imports or if configured
try:
    from backend.celery_app import celery_app

    @celery_app.task(name="eda.generate_profile")
    def generate_profile_celery(report_id: int):
        """
        Entry point for Celery.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_eda_background(report_id))
        finally:
            asyncio.set_event_loop(None)
            loop.close()

except ImportError:
    pass
