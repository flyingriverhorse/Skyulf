"""Focused unit tests for helper functions in backend.eda.router.

These target helper functions that were extracted during a pure-refactor
pass (extract-method complexity cleanup) and lost direct test coverage
attribution. We test them directly (bypassing the HTTP layer where the
underlying I/O is easy to mock) to restore coverage without depending on
external infra (S3, Celery broker, etc).
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from fastapi import BackgroundTasks, HTTPException

from backend.database.models import DataSource, EDAReport
from backend.eda.router import (
    AnalyzeRequest,
    DecompositionRequest,
    FilterRequest,
    _build_analysis_config,
    _build_decomposition_storage_options,
    _dispatch_analysis_job,
    _load_decomposition_dataframe,
    _prepare_decomposition_dataframe,
    _resolve_decomposition_file_path,
    _resolve_decomposition_s3_credentials,
    _run_decomposition_analysis,
)


def _make_ds(**kwargs) -> DataSource:
    """Build an in-memory (unsaved) DataSource with sensible defaults for tests."""
    defaults = {
        "name": "ds",
        "type": "csv",
        "config": {},
        "credentials": None,
        "source_metadata": None,
        "source_id": None,
    }
    defaults.update(kwargs)
    return DataSource(**defaults)


# ---------------------------------------------------------------------------
# _build_analysis_config
# ---------------------------------------------------------------------------


def test_build_analysis_config_none_body():
    """None body returns an empty config dict without inspecting attributes."""
    assert _build_analysis_config(None) == {}


def test_build_analysis_config_full_body():
    """All optional fields present on the request populate the config dict."""
    body = AnalyzeRequest(
        target_col="target",
        exclude_cols=["a", "b"],
        filters=[FilterRequest(column="a", operator="==", value=1)],
        task_type="Classification",
    )
    config = _build_analysis_config(body)
    assert config["target_col"] == "target"
    assert config["exclude_cols"] == ["a", "b"]
    assert config["filters"] == [{"column": "a", "operator": "==", "value": 1}]
    assert config["task_type"] == "Classification"


def test_build_analysis_config_empty_body():
    """A body with no fields set produces an empty config dict."""
    body = AnalyzeRequest()
    assert _build_analysis_config(body) == {}


# ---------------------------------------------------------------------------
# _dispatch_analysis_job
# ---------------------------------------------------------------------------


async def test_dispatch_analysis_job_uses_celery_when_enabled():
    """When USE_CELERY is on, the job is dispatched via Celery and its task id stored."""
    report = EDAReport(id=1, data_source_id=1, status="PENDING", config={})
    background_tasks = BackgroundTasks()
    session = AsyncMock()
    session.add = MagicMock()

    fake_settings = MagicMock()
    fake_settings.USE_CELERY = True

    with (
        patch("backend.eda.router.get_settings", return_value=fake_settings),
        patch("backend.eda.router.generate_profile_celery") as mock_celery_task,
    ):
        mock_celery_task.delay.return_value = MagicMock(id="celery-task-123")
        await _dispatch_analysis_job(report, background_tasks, session)

    assert report.config["celery_task_id"] == "celery-task-123"
    session.add.assert_called_once_with(report)
    session.commit.assert_awaited_once()


async def test_dispatch_analysis_job_uses_background_tasks_when_disabled():
    """When USE_CELERY is off, the job is scheduled via FastAPI BackgroundTasks."""
    report = EDAReport(id=42, data_source_id=1, status="PENDING", config={})
    background_tasks = BackgroundTasks()
    session = AsyncMock()

    fake_settings = MagicMock()
    fake_settings.USE_CELERY = False

    with patch("backend.eda.router.get_settings", return_value=fake_settings):
        await _dispatch_analysis_job(report, background_tasks, session)

    assert len(background_tasks.tasks) == 1
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# _resolve_decomposition_file_path
# ---------------------------------------------------------------------------


def test_resolve_decomposition_file_path_from_config(tmp_path):
    """A valid local file_path in config resolves directly."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    ds = _make_ds(config={"file_path": str(csv_file)})

    result = _resolve_decomposition_file_path(ds)
    assert Path(result) == csv_file


def test_resolve_decomposition_file_path_fallback_source_id():
    """If no path is resolvable from config, but source_id looks like a csv/parquet name, use it."""
    ds = _make_ds(config={}, source_id="some-uuid.csv")
    result = _resolve_decomposition_file_path(ds)
    assert result == "some-uuid.csv"


def test_resolve_decomposition_file_path_not_found_raises():
    """No resolvable path and a non-file-like source_id raises HTTP 400."""
    ds = _make_ds(config={}, source_id="not-a-file-id")
    with pytest.raises(HTTPException) as exc_info:
        _resolve_decomposition_file_path(ds)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _resolve_decomposition_s3_credentials / _build_decomposition_storage_options
# ---------------------------------------------------------------------------


def test_resolve_decomposition_s3_credentials_explicit():
    """Explicit ds.credentials take precedence over config and settings."""
    ds = _make_ds(credentials={"aws_access_key_id": "AKIA", "aws_secret_access_key": "secret"})
    creds = _resolve_decomposition_s3_credentials(ds)
    assert creds["aws_access_key_id"] == "AKIA"


def test_resolve_decomposition_s3_credentials_from_config():
    """When ds.credentials is empty, fall back to config-embedded credentials."""
    ds = _make_ds(
        credentials=None,
        config={
            "aws_access_key_id": "config-key",
            "aws_secret_access_key": "config-secret",
            "aws_session_token": "token",
            "endpoint_url": "http://minio",
        },
    )
    creds = _resolve_decomposition_s3_credentials(ds)
    assert creds["aws_access_key_id"] == "config-key"
    assert creds["endpoint_url"] == "http://minio"


def test_resolve_decomposition_s3_credentials_from_settings():
    """When neither ds.credentials nor config have keys, fall back to global settings."""
    ds = _make_ds(credentials=None, config={})
    fake_settings = MagicMock()
    fake_settings.AWS_ACCESS_KEY_ID = "settings-key"
    fake_settings.AWS_SECRET_ACCESS_KEY = "settings-secret"
    fake_settings.AWS_SESSION_TOKEN = None

    with patch("backend.eda.router.get_settings", return_value=fake_settings):
        creds = _resolve_decomposition_s3_credentials(ds)

    assert creds["aws_access_key_id"] == "settings-key"
    assert creds["endpoint_url"] is None


def test_build_decomposition_storage_options_filters_none():
    """None-valued entries are dropped from the resulting storage_options dict."""
    ds = _make_ds(credentials={"aws_access_key_id": "AKIA", "aws_secret_access_key": "secret"})
    options = _build_decomposition_storage_options(ds)
    assert options == {"key": "AKIA", "secret": "secret"}


# ---------------------------------------------------------------------------
# _load_decomposition_dataframe
# ---------------------------------------------------------------------------


async def test_load_decomposition_dataframe_pandas_conversion():
    """A pandas DataFrame returned by the data service is converted to Polars."""
    data_service = AsyncMock()
    data_service.load_file.return_value = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    df = await _load_decomposition_dataframe(data_service, "some.csv", None)
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["a", "b"]


async def test_load_decomposition_dataframe_polars_passthrough():
    """A Polars DataFrame returned by the data service is returned unchanged."""
    data_service = AsyncMock()
    expected = pl.DataFrame({"a": [1, 2]})
    data_service.load_file.return_value = expected

    df = await _load_decomposition_dataframe(data_service, "some.parquet", None)
    assert df.equals(expected)


async def test_load_decomposition_dataframe_fallback_on_conversion_error():
    """If pl.from_pandas fails initially, object columns are stringified and retried."""
    data_service = AsyncMock()
    pdf = pd.DataFrame({"a": [1, 2], "obj": [object(), object()]})
    data_service.load_file.return_value = pdf

    call_count = {"n": 0}
    real_from_pandas = pl.from_pandas

    def fake_from_pandas(frame):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("cannot convert")
        return real_from_pandas(frame)

    with patch("backend.eda.router.pl.from_pandas", side_effect=fake_from_pandas):
        df = await _load_decomposition_dataframe(data_service, "some.csv", None)

    assert isinstance(df, pl.DataFrame)
    assert call_count["n"] == 2


async def test_load_decomposition_dataframe_load_failure_raises():
    """A failure in data_service.load_file surfaces as a SkyulfException."""
    from backend.exceptions.core import SkyulfException

    data_service = AsyncMock()
    data_service.load_file.side_effect = RuntimeError("boom")

    with pytest.raises(SkyulfException):
        await _load_decomposition_dataframe(data_service, "some.csv", None)


# ---------------------------------------------------------------------------
# _prepare_decomposition_dataframe
# ---------------------------------------------------------------------------


async def test_prepare_decomposition_dataframe_local_path(tmp_path):
    """A local file path skips storage_options resolution entirely."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    ds = _make_ds(config={"file_path": str(csv_file)})

    with patch("backend.eda.router.DataService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.load_file = AsyncMock(return_value=pl.DataFrame({"a": [1], "b": [2]}))
        df = await _prepare_decomposition_dataframe(ds)

    instance.load_file.assert_awaited_once()
    call_args = instance.load_file.call_args
    assert call_args.kwargs["storage_options"] is None
    assert isinstance(df, pl.DataFrame)


async def test_prepare_decomposition_dataframe_s3_path_builds_storage_options():
    """An s3:// path triggers storage_options resolution from credentials/settings."""
    ds = _make_ds(
        config={"file_path": "s3://bucket/data.csv"},
        credentials={"aws_access_key_id": "AKIA", "aws_secret_access_key": "secret"},
    )

    with patch("backend.eda.router.DataService") as mock_service_cls:
        instance = mock_service_cls.return_value
        instance.load_file = AsyncMock(return_value=pl.DataFrame({"a": [1]}))
        df = await _prepare_decomposition_dataframe(ds)

    call_args = instance.load_file.call_args
    assert call_args.kwargs["storage_options"] == {"key": "AKIA", "secret": "secret"}
    assert isinstance(df, pl.DataFrame)


# ---------------------------------------------------------------------------
# _run_decomposition_analysis
# ---------------------------------------------------------------------------


def test_run_decomposition_analysis_normalizes_empty_split_col():
    """An empty-string split_col (sent by the frontend) is normalised to None."""
    body = DecompositionRequest(measure_col="m", measure_agg="sum", split_col="")

    with patch("backend.eda.router.EDAAnalyzer") as mock_analyzer_cls:
        instance = mock_analyzer_cls.return_value
        instance.get_decomposition_split.return_value = {"result": "ok"}
        result = _run_decomposition_analysis(pl.DataFrame({"a": [1]}), body)

    instance.get_decomposition_split.assert_called_once_with(
        measure_col="m", measure_agg="sum", split_col=None, filters=[]
    )
    assert result == {"result": "ok"}


def test_run_decomposition_analysis_with_filters_and_split_col():
    """Filters are converted to dicts and a real split_col is passed through unchanged."""
    body = DecompositionRequest(
        measure_col="m",
        measure_agg="mean",
        split_col="category",
        filters=[FilterRequest(column="a", operator=">", value=5)],
    )

    with patch("backend.eda.router.EDAAnalyzer") as mock_analyzer_cls:
        instance = mock_analyzer_cls.return_value
        instance.get_decomposition_split.return_value = {"result": "ok"}
        _run_decomposition_analysis(pl.DataFrame({"a": [1]}), body)

    instance.get_decomposition_split.assert_called_once_with(
        measure_col="m",
        measure_agg="mean",
        split_col="category",
        filters=[{"column": "a", "operator": ">", "value": 5}],
    )
