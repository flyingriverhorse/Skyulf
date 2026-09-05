from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from backend.config import get_settings
from backend.data_ingestion.service import DataIngestionService
from backend.data_ingestion.tasks import ingest_data_task
from backend.database.models import DataSource


@pytest.fixture
def mock_session():
    session = MagicMock()
    return session


@pytest.fixture
def mock_data_source():
    ds = DataSource(id=1, type="file", config={"file_path": "/tmp/test.csv"}, source_metadata={})
    return ds


@patch("backend.data_ingestion.tasks.get_db_session")
@patch("backend.data_ingestion.tasks.LocalFileConnector")
@patch("backend.data_ingestion.tasks.DataProfiler")
def test_ingest_file_task(
    mock_profiler, mock_connector_cls, mock_get_session, mock_session, mock_data_source
):
    # Setup mocks
    mock_get_session.return_value = mock_session
    mock_session.query.return_value.filter.return_value.first.return_value = mock_data_source

    mock_connector = AsyncMock()
    mock_connector_cls.return_value = mock_connector

    # Mock fetch_data to return a dummy DataFrame
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    mock_connector.fetch_data.return_value = df

    # Mock profiler
    mock_profiler.profile.return_value = {
        "row_count": 3,
        "column_count": 2,
        "columns": {"a": {"type": "Int64"}, "b": {"type": "Utf8"}},
    }

    # Run task
    ingest_data_task(1)

    # Verify interactions
    mock_connector.connect.assert_called_once()
    mock_connector.fetch_data.assert_called_once()
    mock_profiler.profile.assert_called_once()

    # Verify metadata update
    assert mock_data_source.test_status == "success"
    assert mock_data_source.source_metadata["ingestion_status"]["status"] == "completed"
    assert mock_data_source.source_metadata["row_count"] == 3
    assert mock_data_source.source_metadata["column_count"] == 2
    assert "profile" in mock_data_source.source_metadata


def test_upload_dir_defaults_to_settings_upload_dir(monkeypatch, tmp_path):
    """The service must write uploads where ``LocalFileConnector`` reads them.

    The connector resolves against ``settings.UPLOAD_DIR``; the service used to
    hardcode the literal ``"uploads/data"``, so configuring ``UPLOAD_DIR`` wrote
    files to a directory no connector would ever look in.
    """
    configured = tmp_path / "configured_uploads"
    monkeypatch.setattr(get_settings(), "UPLOAD_DIR", str(configured))

    service = DataIngestionService(session=MagicMock())

    assert service.upload_dir == configured
    assert configured.is_dir(), "the directory must still be created eagerly"


def test_upload_dir_explicit_argument_overrides_settings(monkeypatch, tmp_path):
    """An explicit ``upload_dir`` wins, which is how tests isolate their writes."""
    from_settings = tmp_path / "from_settings"
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(get_settings(), "UPLOAD_DIR", str(from_settings))

    service = DataIngestionService(session=MagicMock(), upload_dir=str(explicit))

    assert service.upload_dir == explicit
    assert not from_settings.exists(), "settings must not be consulted when overridden"
