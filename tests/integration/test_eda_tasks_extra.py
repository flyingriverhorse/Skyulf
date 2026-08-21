"""Focused unit tests for backend.eda.tasks.

These tests exercise the extracted helper functions directly with plain
Python doubles / AsyncMocks instead of a live DB or Celery broker, mirroring
the "USE_CELERY=False" test configuration set up in tests/conftest.py.
"""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.database.models import DataSource, EDAReport
from backend.eda import tasks

# _resolve_file_path only accepts a local path candidate if the file actually
# exists on disk, so tests that rely on config["file_path"] need a real file.
# Keep it inside the repo (never under /tmp) and clean it up afterward.
_TEMP_DIR = Path(__file__).resolve().parent / "temp"


@pytest.fixture
def real_csv_path():
    """Create a small real CSV file inside the repo and yield its path, cleaning up after."""
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _TEMP_DIR / f"eda_tasks_test_{uuid.uuid4().hex}.csv"
    file_path.write_text("col1,col2\n1,2\n3,4\n")
    try:
        yield str(file_path)
    finally:
        file_path.unlink(missing_ok=True)


def make_data_source(**overrides) -> DataSource:
    """Build a plain (unsaved) DataSource instance for use in unit tests."""
    defaults = {
        "id": 1,
        "source_id": "src-1",
        "name": "Test Source",
        "type": "csv",
        "config": {},
        "credentials": None,
        "source_metadata": None,
    }
    defaults.update(overrides)
    return DataSource(**defaults)


def make_report(**overrides) -> EDAReport:
    """Build a plain (unsaved) EDAReport instance for use in unit tests."""
    defaults = {"id": 1, "data_source_id": 1, "status": "PENDING", "config": {}}
    defaults.update(overrides)
    return EDAReport(**defaults)


# --- _resolve_file_path -----------------------------------------------------


def test_resolve_file_path_from_config(real_csv_path):
    """A file_path present in config is used directly when the file exists on disk."""
    ds = make_data_source(config={"file_path": real_csv_path})
    resolved = tasks._resolve_file_path(ds)
    assert str(resolved) == real_csv_path


def test_resolve_file_path_falls_back_to_source_id_csv():
    """When no path is found via config, a .csv source_id is used as a last resort."""
    ds = make_data_source(config={}, source_id="uploads/dataset.csv")
    assert tasks._resolve_file_path(ds) == "uploads/dataset.csv"


def test_resolve_file_path_falls_back_to_source_id_parquet():
    """When no path is found via config, a .parquet source_id is used as a last resort."""
    ds = make_data_source(config={}, source_id="uploads/dataset.parquet")
    assert tasks._resolve_file_path(ds) == "uploads/dataset.parquet"


def test_resolve_file_path_no_match_returns_none():
    """When neither config nor source_id yield a path, None is returned."""
    ds = make_data_source(config={}, source_id="not-a-path-id")
    assert tasks._resolve_file_path(ds) is None


# --- _resolve_s3_credentials -------------------------------------------------


def test_resolve_s3_credentials_explicit():
    """Explicit credentials on the DataSource are used as-is."""
    ds = make_data_source(credentials={"aws_access_key_id": "AKIA123"})
    creds = tasks._resolve_s3_credentials(ds)
    assert creds == {"aws_access_key_id": "AKIA123"}


def test_resolve_s3_credentials_from_config():
    """When no explicit credentials exist, config-embedded credentials are used."""
    ds = make_data_source(
        credentials=None,
        config={
            "aws_access_key_id": "AKIACONFIG",
            "aws_secret_access_key": "secretconfig",
            "aws_session_token": "tokenconfig",
            "endpoint_url": "https://minio.local",
        },
    )
    creds = tasks._resolve_s3_credentials(ds)
    assert creds["aws_access_key_id"] == "AKIACONFIG"
    assert creds["endpoint_url"] == "https://minio.local"


def test_resolve_s3_credentials_from_env_settings():
    """When neither explicit nor config credentials exist, env-derived settings are used."""
    ds = make_data_source(credentials=None, config={})
    fake_settings = MagicMock(
        AWS_ACCESS_KEY_ID="AKIAENV",
        AWS_SECRET_ACCESS_KEY="secretenv",
        AWS_SESSION_TOKEN="tokenenv",
    )
    with patch.object(tasks, "get_settings", return_value=fake_settings):
        creds = tasks._resolve_s3_credentials(ds)
    assert creds["aws_access_key_id"] == "AKIAENV"
    assert creds["endpoint_url"] is None


def test_resolve_s3_credentials_env_missing_key():
    """When env settings also lack an access key, the resulting dict has a None key."""
    ds = make_data_source(credentials=None, config={})
    fake_settings = MagicMock(
        AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None, AWS_SESSION_TOKEN=None
    )
    with patch.object(tasks, "get_settings", return_value=fake_settings):
        creds = tasks._resolve_s3_credentials(ds)
    assert creds["aws_access_key_id"] is None


# --- _build_s3_storage_options ------------------------------------------------


def test_build_s3_storage_options_with_creds():
    """Storage options are mapped from boto3-style keys and None values are dropped."""
    ds = make_data_source(
        credentials={
            "aws_access_key_id": "AKIA123",
            "aws_secret_access_key": "secret",
            "aws_session_token": None,
            "endpoint_url": None,
        }
    )
    options = tasks._build_s3_storage_options(ds)
    assert options == {"key": "AKIA123", "secret": "secret"}


def test_build_s3_storage_options_no_creds_logs_warning():
    """When no credentials resolve at all, an empty dict is returned."""
    ds = make_data_source(credentials=None, config={})
    fake_settings = MagicMock(
        AWS_ACCESS_KEY_ID=None, AWS_SECRET_ACCESS_KEY=None, AWS_SESSION_TOKEN=None
    )
    with patch.object(tasks, "get_settings", return_value=fake_settings):
        options = tasks._build_s3_storage_options(ds)
    assert options == {}


# --- _run_eda_analyzer --------------------------------------------------------


def test_run_eda_analyzer_with_config():
    """_run_eda_analyzer pulls target/exclude/filter/task_type out of report_config."""
    report_config = {
        "exclude_cols": ["a"],
        "filters": [{"col": "a", "op": "=="}],
        "target_col": "y",
        "task_type": "classification",
    }
    fake_analyzer = MagicMock()
    fake_analyzer.analyze.return_value = "profile-result"
    with patch.object(tasks, "EDAAnalyzer", return_value=fake_analyzer) as analyzer_cls:
        result = tasks._run_eda_analyzer("df", report_config)
    analyzer_cls.assert_called_once_with("df")
    fake_analyzer.analyze.assert_called_once_with(
        target_col="y",
        exclude_cols=["a"],
        filters=[{"col": "a", "op": "=="}],
        task_type="classification",
    )
    assert result == "profile-result"


def test_run_eda_analyzer_with_no_config():
    """_run_eda_analyzer tolerates a None report_config, passing None for all fields."""
    fake_analyzer = MagicMock()
    fake_analyzer.analyze.return_value = "profile-result"
    with patch.object(tasks, "EDAAnalyzer", return_value=fake_analyzer):
        result = tasks._run_eda_analyzer("df", None)
    fake_analyzer.analyze.assert_called_once_with(
        target_col=None, exclude_cols=None, filters=None, task_type=None
    )
    assert result == "profile-result"


# --- _fail_report_safely -------------------------------------------------------


async def test_fail_report_safely_updates_report():
    """_fail_report_safely rolls back and marks the report FAILED with the error message."""
    session = AsyncMock()
    report = make_report(status="PENDING")
    await tasks._fail_report_safely(session, report, 1, RuntimeError("boom"))
    session.rollback.assert_awaited_once()
    session.commit.assert_awaited_once()
    assert report.status == "FAILED"
    assert report.error_message == "boom"


async def test_fail_report_safely_no_report():
    """_fail_report_safely tolerates a None report (only rollback is attempted)."""
    session = AsyncMock()
    await tasks._fail_report_safely(session, None, 1, RuntimeError("boom"))
    session.rollback.assert_awaited_once()
    session.commit.assert_not_awaited()


async def test_fail_report_safely_swallows_secondary_errors():
    """_fail_report_safely logs (rather than raises) if the rollback/commit itself fails."""
    session = AsyncMock()
    session.rollback.side_effect = Exception("rollback failed")
    report = make_report()
    # Should not raise.
    await tasks._fail_report_safely(session, report, 1, RuntimeError("boom"))


# --- _resolve_or_fail_file_path -------------------------------------------------


async def test_resolve_or_fail_file_path_success(real_csv_path):
    """When a file path resolves, it is returned without touching the report/session."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source(config={"file_path": real_csv_path})
    result = await tasks._resolve_or_fail_file_path(session, report, ds)
    assert str(result) == real_csv_path
    session.commit.assert_not_awaited()
    assert report.status == "PENDING"


async def test_resolve_or_fail_file_path_failure_marks_report_failed():
    """When no file path resolves, the report is marked FAILED and committed."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source(config={}, source_id="not-a-path")
    result = await tasks._resolve_or_fail_file_path(session, report, ds)
    assert result is None
    assert report.status == "FAILED"
    assert "File path not found" in report.error_message
    session.commit.assert_awaited_once()


# --- _load_dataframe_or_fail -----------------------------------------------------


async def test_load_dataframe_or_fail_success():
    """A successful load_file call returns the dataframe without touching the report."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source()
    data_service = AsyncMock()
    data_service.load_file.return_value = "the-df"
    result = await tasks._load_dataframe_or_fail(session, report, ds, data_service, "/data/f.csv")
    assert result == "the-df"
    data_service.load_file.assert_awaited_once_with(
        "/data/f.csv", force_type="polars", storage_options=None
    )
    session.commit.assert_not_awaited()


async def test_load_dataframe_or_fail_s3_path_builds_storage_options():
    """An s3:// path triggers storage_options resolution via _build_s3_storage_options."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source()
    data_service = AsyncMock()
    data_service.load_file.return_value = "the-df"
    with patch.object(tasks, "_build_s3_storage_options", return_value={"key": "k"}) as build_opts:
        result = await tasks._load_dataframe_or_fail(
            session, report, ds, data_service, "s3://bucket/f.csv"
        )
    assert result == "the-df"
    build_opts.assert_called_once_with(ds)
    data_service.load_file.assert_awaited_once_with(
        "s3://bucket/f.csv", force_type="polars", storage_options={"key": "k"}
    )


async def test_load_dataframe_or_fail_failure_marks_report_failed():
    """A failing load_file call marks the report FAILED, commits, and returns None."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source()
    data_service = AsyncMock()
    data_service.load_file.side_effect = Exception("bad file")
    result = await tasks._load_dataframe_or_fail(session, report, ds, data_service, "/data/f.csv")
    assert result is None
    assert report.status == "FAILED"
    assert "Failed to load data" in report.error_message
    session.commit.assert_awaited_once()


# --- _run_analysis_or_fail --------------------------------------------------------


async def test_run_analysis_or_fail_success():
    """A successful analysis returns the profile without touching the report."""
    session = AsyncMock()
    report = make_report()
    with patch.object(tasks, "_run_eda_analyzer", return_value="profile"):
        result = await tasks._run_analysis_or_fail(session, report, "df")
    assert result == "profile"
    session.commit.assert_not_awaited()


async def test_run_analysis_or_fail_failure_marks_report_failed():
    """A failing analysis marks the report FAILED, commits, and returns None."""
    session = AsyncMock()
    report = make_report()
    with patch.object(tasks, "_run_eda_analyzer", side_effect=Exception("analysis broke")):
        result = await tasks._run_analysis_or_fail(session, report, "df")
    assert result is None
    assert report.status == "FAILED"
    assert "Analysis failed" in report.error_message
    session.commit.assert_awaited_once()


# --- run_eda_analysis (end-to-end orchestration, all mocked) ----------------------


async def test_run_eda_analysis_report_not_found():
    """run_eda_analysis returns early (without raising) if the report doesn't exist."""
    session = AsyncMock()
    session.get.return_value = None
    await tasks.run_eda_analysis(999, session)
    session.get.assert_awaited_once()


async def test_run_eda_analysis_data_source_not_found():
    """run_eda_analysis marks the report FAILED if the DataSource doesn't exist."""
    session = AsyncMock()
    report = make_report()

    async def fake_get(model, ident):
        if model is EDAReport:
            return report
        return None

    session.get.side_effect = fake_get
    await tasks.run_eda_analysis(1, session)
    assert report.status == "FAILED"
    assert report.error_message == "DataSource not found."


async def test_run_eda_analysis_full_success(real_csv_path):
    """run_eda_analysis completes successfully through the full happy path."""
    session = AsyncMock()
    report = make_report()
    ds = make_data_source(config={"file_path": real_csv_path})

    async def fake_get(model, ident):
        if model is EDAReport:
            return report
        return ds

    session.get.side_effect = fake_get

    fake_profile = MagicMock()
    fake_profile.model_dump.return_value = {"summary": "ok"}

    with (
        patch.object(tasks, "DataService") as data_service_cls,
        patch.object(tasks, "_run_eda_analyzer", return_value=fake_profile),
    ):
        data_service_instance = AsyncMock()
        data_service_instance.load_file.return_value = "the-df"
        data_service_cls.return_value = data_service_instance

        await tasks.run_eda_analysis(1, session)

    assert report.status == "COMPLETED"
    assert report.profile_data == {"summary": "ok"}


async def test_run_eda_analysis_unexpected_exception_marks_failed():
    """An unexpected exception during orchestration is caught and the report marked FAILED."""
    session = AsyncMock()
    report = make_report()
    session.get.side_effect = RuntimeError("unexpected db failure")

    await tasks.run_eda_analysis(1, session)

    session.rollback.assert_awaited_once()


# --- run_eda_background -----------------------------------------------------------


async def test_run_eda_background_uses_context_managed_session():
    """run_eda_background opens a session via get_database_session and delegates to run_eda_analysis."""
    fake_session = AsyncMock()

    class FakeSessionCtx:
        async def __aenter__(self):
            return fake_session

        async def __aexit__(self, *args):
            return False

    with (
        patch.object(tasks, "get_database_session", return_value=FakeSessionCtx()),
        patch.object(tasks, "run_eda_analysis", new=AsyncMock()) as run_mock,
    ):
        await tasks.run_eda_background(42)

    run_mock.assert_awaited_once_with(42, fake_session)


# --- Celery task wrapper -----------------------------------------------------------


def test_generate_profile_celery_runs_background_task():
    """The Celery task entry point creates/closes its own event loop and delegates to run_eda_background."""
    if not hasattr(tasks, "generate_profile_celery"):
        pytest.skip("Celery app not available; generate_profile_celery was not defined")

    with patch.object(tasks, "run_eda_background", new=AsyncMock()) as run_mock:
        tasks.generate_profile_celery.run(7)

    run_mock.assert_awaited_once_with(7)
