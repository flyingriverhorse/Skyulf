import pytest
from unittest.mock import MagicMock, patch
from backend.ml_pipeline.tasks import run_pipeline_task

@pytest.mark.asyncio
async def test_artifact_routing_logic():
    # We will mock everything around the logic we want to test
    
    with patch("backend.ml_pipeline.tasks.get_db_session") as mock_get_session, \
         patch("backend.ml_pipeline.artifacts.factory.get_settings") as mock_get_settings, \
         patch("backend.ml_pipeline.artifacts.factory.S3ArtifactStore") as MockS3Store, \
         patch("backend.ml_pipeline.artifacts.factory.LocalArtifactStore") as MockLocalStore, \
         patch("backend.ml_pipeline.tasks.PipelineEngine") as MockEngine, \
         patch("os.makedirs"):
         
        # Setup Session and Job
        session = MagicMock()
        mock_get_session.return_value = session
        job = MagicMock()
        # Ensure dataset_source_id is None to avoid query logic that returns a mock
        job.dataset_source_id = None
        session.query.return_value.filter.return_value.first.return_value = job
        
        # Setup Settings
        settings = MagicMock()
        settings.S3_ARTIFACT_BUCKET = "my-bucket"
        settings.AWS_ACCESS_KEY_ID = "key"
        settings.AWS_SECRET_ACCESS_KEY = "secret"
        settings.AWS_DEFAULT_REGION = "us-east-1"
        settings.TRAINING_ARTIFACT_DIR = "/tmp/artifacts"
        # Important: Explicitly set boolean flags to avoid MagicMock truthiness issues
        settings.SAVE_S3_ARTIFACTS_LOCALLY = False
        settings.UPLOAD_TO_S3_FOR_LOCAL_FILES = False
        
        mock_get_settings.return_value = settings
        
        # Scenario 1: S3 Source -> S3 Artifacts
        settings.UPLOAD_TO_S3_FOR_LOCAL_FILES = False
        pipeline_config_s3 = {
            "pipeline_id": "p1",
            "nodes": [
                {"node_id": "n1", "step_type": "data_loader", "params": {"dataset_id": "s3://bucket/data.csv"}, "inputs": []}
            ]
        }
        
        run_pipeline_task("job-1", pipeline_config_s3)
        
        # Verify S3 Store was used
        MockS3Store.assert_called()
        MockLocalStore.assert_not_called()
        
        # Reset mocks
        MockS3Store.reset_mock()
        MockLocalStore.reset_mock()
        
        # Scenario 2: Local Source + Config False -> Local Artifacts
        settings.UPLOAD_TO_S3_FOR_LOCAL_FILES = False
        pipeline_config_local = {
            "pipeline_id": "p1",
            "nodes": [
                {"node_id": "n1", "step_type": "data_loader", "params": {"dataset_id": "/local/data.csv"}, "inputs": []}
            ]
        }
        
        run_pipeline_task("job-2", pipeline_config_local)
        
        # Verify Local Store was used
        MockS3Store.assert_not_called()
        MockLocalStore.assert_called()
        
        # Reset mocks
        MockS3Store.reset_mock()
        MockLocalStore.reset_mock()
        
        # Scenario 3: Local Source + Config True -> S3 Artifacts
        settings.UPLOAD_TO_S3_FOR_LOCAL_FILES = True
        
        run_pipeline_task("job-3", pipeline_config_local)
        
        # Verify S3 Store was used
        MockS3Store.assert_called()
        MockLocalStore.assert_not_called()
        
        # Reset mocks
        MockS3Store.reset_mock()
        MockLocalStore.reset_mock()

        # Scenario 4: S3 Source + Config Force Local -> Local Artifacts
        settings.UPLOAD_TO_S3_FOR_LOCAL_FILES = False # Reset this
        settings.SAVE_S3_ARTIFACTS_LOCALLY = True
        
        run_pipeline_task("job-4", pipeline_config_s3)
        
        # Verify Local Store was used (despite S3 source)
        MockS3Store.assert_not_called()
        MockLocalStore.assert_called()
