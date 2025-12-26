import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from backend.ml_pipeline.execution.utils import resolve_dataset_name, get_dataset_map

@pytest.mark.asyncio
async def test_resolve_dataset_name():
    session = AsyncMock()
    
    # Mock execute result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = "Iris Dataset"
    session.execute.return_value = mock_result
    
    # Test with ID
    name = await resolve_dataset_name(session, "41")
    assert name == "Iris Dataset"
    
    # Test with UUID
    name = await resolve_dataset_name(session, "uuid-123")
    assert name == "Iris Dataset"
    
    # Test not found
    mock_result.scalar_one_or_none.return_value = None
    name = await resolve_dataset_name(session, "999")
    assert name == "Dataset 999"

@pytest.mark.asyncio
async def test_get_dataset_map():
    session = AsyncMock()
    
    # Mock execute result
    mock_result = MagicMock()
    # id, source_id, name
    mock_result.all.return_value = [
        (41, "uuid-41", "Iris"),
        (42, None, "Titanic")
    ]
    session.execute.return_value = mock_result
    
    ds_map = await get_dataset_map(session)
    
    assert ds_map["41"] == "Iris"
    assert ds_map["uuid-41"] == "Iris"
    assert ds_map["42"] == "Titanic"
