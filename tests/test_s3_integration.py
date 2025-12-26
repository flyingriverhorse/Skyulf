import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from backend.ml_pipeline.deployment.service import DeploymentService
from backend.ml_pipeline.api import get_job_evaluation
from backend.database.models import TrainingJob

@pytest.mark.asyncio
async def test_deployment_predict_s3_creds():
    session = AsyncMock()
    
    # Mock active deployment
    mock_deployment = MagicMock()
    mock_deployment.artifact_uri = "s3://my-bucket/models/job-123"
    
    # Mock get_active_deployment
    with patch("backend.ml_pipeline.deployment.service.DeploymentService.get_active_deployment") as mock_get:
        mock_get.return_value = mock_deployment
        
        # Mock Settings - patch where it is used
        with patch("backend.ml_pipeline.deployment.service.get_settings") as mock_settings:
            mock_settings.return_value.AWS_ACCESS_KEY_ID = "test-key"
            mock_settings.return_value.AWS_SECRET_ACCESS_KEY = "test-secret"
            mock_settings.return_value.AWS_DEFAULT_REGION = "us-east-1"
            mock_settings.return_value.AWS_ENDPOINT_URL = None
            
            # Mock S3ArtifactStore
            with patch("backend.ml_pipeline.deployment.service.S3ArtifactStore") as MockS3Store:
                mock_store = MockS3Store.return_value
                # Mock artifact loading
                mock_model = MagicMock()
                mock_model.predict.return_value = [1, 0]
                mock_store.load.return_value = mock_model
                
                # Call predict
                preds = await DeploymentService.predict(session, [{"f1": 1}])
                
                assert preds == [1, 0]
                
                # Verify S3Store init
                MockS3Store.assert_called_once()
                call_args = MockS3Store.call_args
                assert call_args.kwargs["bucket_name"] == "my-bucket"
                assert call_args.kwargs["storage_options"]["aws_access_key_id"] == "test-key"
                assert call_args.kwargs["storage_options"]["region_name"] == "us-east-1"

@pytest.mark.asyncio
async def test_get_job_evaluation_s3():
    session = AsyncMock()
    job_id = "job-123"
    
    # Mock Job
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_job.artifact_uri = "s3://my-bucket/artifacts/job-123"
    mock_job.status = "completed"
    
    # Mock execute result
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    session.execute.return_value = mock_result
    
    # Mock Settings
    with patch("backend.ml_pipeline.api.get_settings") as mock_settings:
        mock_settings.return_value.AWS_ACCESS_KEY_ID = "test-key"
        mock_settings.return_value.AWS_SECRET_ACCESS_KEY = "test-secret"
        mock_settings.return_value.AWS_DEFAULT_REGION = "us-east-1"
        
        # Mock S3ArtifactStore - patch the class where it is defined, as it is imported inside the function
        with patch("backend.ml_pipeline.artifacts.s3.S3ArtifactStore") as MockS3Store:
            mock_store = MockS3Store.return_value
            mock_store.exists.return_value = True
            mock_store.load.return_value = {"job_id": job_id, "splits": {}}
            
            # Call API
            await get_job_evaluation(job_id, session)
            
            # Verify S3Store init
            MockS3Store.assert_called_once()
            call_args = MockS3Store.call_args
            assert call_args.kwargs["bucket_name"] == "my-bucket"
            assert call_args.kwargs["prefix"] == "artifacts/job-123"
            assert call_args.kwargs["storage_options"]["aws_access_key_id"] == "test-key"
