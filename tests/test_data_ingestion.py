import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import polars as pl
from datetime import datetime
from backend.data_ingestion.tasks import ingest_data_task
from backend.database.models import DataSource

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def mock_data_source():
    ds = DataSource(
        id=1,
        type='file',
        config={'file_path': '/tmp/test.csv'},
        source_metadata={}
    )
    return ds

@patch('core.data_ingestion.tasks.get_db_session')
@patch('core.data_ingestion.tasks.LocalFileConnector')
@patch('core.data_ingestion.tasks.DataProfiler')
def test_ingest_file_task(mock_profiler, mock_connector_cls, mock_get_session, mock_session, mock_data_source):
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
        "columns": {
            "a": {"type": "Int64"},
            "b": {"type": "Utf8"}
        }
    }

    # Run task
    ingest_data_task(1)

    # Verify interactions
    mock_connector.connect.assert_called_once()
    mock_connector.fetch_data.assert_called_once()
    mock_profiler.profile.assert_called_once()
    
    # Verify metadata update
    assert mock_data_source.test_status == 'success'
    assert mock_data_source.source_metadata['ingestion_status']['status'] == 'completed'
    assert mock_data_source.source_metadata['row_count'] == 3
    assert mock_data_source.source_metadata['column_count'] == 2
    assert 'profile' in mock_data_source.source_metadata

@patch('core.data_ingestion.tasks.get_db_session')
@patch('core.data_ingestion.tasks.DatabaseConnector')
@patch('core.data_ingestion.tasks.DataProfiler')
def test_ingest_sql_task(mock_profiler, mock_connector_cls, mock_get_session, mock_session):
    # Setup SQL DataSource
    ds = DataSource(
        id=2,
        type='postgres',
        config={'connection_string': 'sqlite:///:memory:', 'table_name': 'users'},
        source_metadata={}
    )
    
    mock_get_session.return_value = mock_session
    mock_session.query.return_value.filter.return_value.first.return_value = ds
    
    mock_connector = AsyncMock()
    mock_connector_cls.return_value = mock_connector
    mock_connector.fetch_data.return_value = pl.DataFrame({"id": [1]})
    
    mock_profiler.profile.return_value = {
        "row_count": 1,
        "column_count": 1,
        "columns": {"id": {"type": "Int64"}}
    }

    # Run task
    ingest_data_task(2)

    # Verify
    mock_connector_cls.assert_called_with('sqlite:///:memory:', table_name='users', query=None)
    mock_connector.connect.assert_called_once()
    assert ds.test_status == 'success'

@patch('core.data_ingestion.tasks.get_db_session')
@patch('core.data_ingestion.tasks.ApiConnector')
@patch('core.data_ingestion.tasks.DataProfiler')
def test_ingest_api_task(mock_profiler, mock_connector_cls, mock_get_session, mock_session):
    # Setup API DataSource
    ds = DataSource(
        id=3,
        type='api',
        config={'url': 'https://api.example.com/data', 'method': 'GET'},
        source_metadata={}
    )
    
    mock_get_session.return_value = mock_session
    mock_session.query.return_value.filter.return_value.first.return_value = ds
    
    mock_connector = AsyncMock()
    mock_connector_cls.return_value = mock_connector
    mock_connector.fetch_data.return_value = pl.DataFrame({"id": [1]})
    
    mock_profiler.profile.return_value = {
        "row_count": 1,
        "column_count": 1,
        "columns": {"id": {"type": "Int64"}}
    }

    # Run task
    ingest_data_task(3)

    # Verify
    mock_connector_cls.assert_called_with('https://api.example.com/data', method='GET', headers=None, params=None, data_key=None)
    mock_connector.connect.assert_called_once()
    assert ds.test_status == 'success'
