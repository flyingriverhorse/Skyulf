import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from backend.ml_pipeline.model_registry.service import ModelRegistryService
from backend.ml_pipeline.execution.basic_training_manager import BasicTrainingManager
from backend.data.catalog import SmartCatalog
from backend.database.models import DataSource, BasicTrainingJob

@pytest.mark.asyncio
async def test_get_job_artifacts_s3_creds():
    # Mock session
    session = AsyncMock()
    
    # Mock Job
    mock_job = MagicMock()
    mock_job.artifact_uri = "s3://my-bucket/artifacts/job-123"
    
    # Mock execute result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    session.execute.return_value = mock_result
    
    # Mock Settings
    # Patch get_settings where it is used in ArtifactFactory
    with patch("backend.ml_pipeline.artifacts.factory.get_settings") as mock_settings:
        mock_settings.return_value.AWS_ACCESS_KEY_ID = "test-key"
        mock_settings.return_value.AWS_SECRET_ACCESS_KEY = "test-secret"
        mock_settings.return_value.AWS_REGION = "us-east-1"
        
        # Mock S3ArtifactStore
        # We must patch where it is imported/used, which is inside ArtifactFactory
        with patch("backend.ml_pipeline.artifacts.factory.S3ArtifactStore") as MockS3Store:
            mock_store_instance = MockS3Store.return_value
            mock_store_instance.list_artifacts.return_value = ["model.joblib"]
            
            artifacts = await ModelRegistryService.get_job_artifacts(session, "job-123")
            
            assert artifacts.files == ["model.joblib"]
            assert artifacts.storage_type == "s3"
            assert artifacts.base_uri == "s3://my-bucket/artifacts/job-123"
            
            # Verify S3Store init with creds
            MockS3Store.assert_called_once()
            call_args = MockS3Store.call_args
            assert call_args.kwargs["bucket_name"] == "my-bucket"
            assert call_args.kwargs["prefix"] == "artifacts/job-123"
            assert call_args.kwargs["storage_options"]["aws_access_key_id"] == "test-key"

@pytest.mark.asyncio
async def test_get_training_job_name_resolution():
    session = AsyncMock()
    
    # Mock Job
    mock_job = MagicMock()
    mock_job.id = "job-123"
    mock_job.dataset_source_id = "41"
    
    # Mock execute results
    mock_result_job = MagicMock()
    mock_result_job.scalar_one_or_none.return_value = mock_job
    
    mock_result_ds = MagicMock()
    mock_result_ds.scalar_one_or_none.return_value = "Iris Dataset"
    
    # Side effect for execute
    async def side_effect(stmt):
        s = str(stmt)
        if "training_jobs" in s:
            return mock_result_job
        if "data_sources" in s:
            return mock_result_ds
        return MagicMock()
        
    session.execute.side_effect = side_effect
    
    # Mock map_training_job_to_info
    with patch("backend.ml_pipeline.execution.basic_training_manager.BasicTrainingManager.map_training_job_to_info") as mock_map:
        await BasicTrainingManager.get_training_job(session, "job-123")
        
        # Verify map called with correct name
        mock_map.assert_called_once()
        args = mock_map.call_args[0]
        assert args[1] == "Iris Dataset"

def test_smart_catalog_get_dataset_name():
    session = MagicMock()
    catalog = SmartCatalog(session=session)
    
    # Mock DB query
    mock_ds = MagicMock()
    mock_ds.name = "Titanic"
    session.query.return_value.filter.return_value.first.return_value = mock_ds
    
    name = catalog.get_dataset_name("101")
    assert name == "Titanic"
    
    # Verify query
    session.query.assert_called()
