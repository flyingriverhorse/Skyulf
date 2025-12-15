import pytest
import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime
from backend.ml_pipeline.tasks import run_pipeline_task
from backend.database.models import TrainingJob, HyperparameterTuningJob
from backend.ml_pipeline.execution.schemas import PipelineExecutionResult, NodeExecutionResult

# Mock data
MOCK_JOB_ID = str(uuid.uuid4())
MOCK_PIPELINE_CONFIG = {
    "pipeline_id": "test_pipeline",
    "nodes": [
        {
            "node_id": "node_1",
            "step_type": "data_loader",
            "params": {"dataset_id": "ds_1"},
            "inputs": []
        }
    ],
    "metadata": {}
}

@pytest.fixture
def mock_session():
    session = MagicMock()
    return session

@pytest.fixture
def mock_engine_class():
    with patch("backend.ml_pipeline.tasks.PipelineEngine") as mock:
        yield mock

@pytest.fixture
def mock_get_db_session():
    with patch("backend.ml_pipeline.tasks.get_db_session") as mock:
        yield mock

def test_run_pipeline_task_training_job(mock_get_db_session, mock_engine_class):
    # Setup Mock Session and Job
    session = MagicMock()
    mock_get_db_session.return_value = session
    
    job = TrainingJob(id=MOCK_JOB_ID, status="queued")
    
    # Configure query side effects
    # First query returns the job
    # Second query (in exception handler or re-query) also returns job
    session.query.return_value.filter.return_value.first.return_value = job
    
    # Setup Mock Engine
    engine_instance = mock_engine_class.return_value
    engine_instance.run.return_value = PipelineExecutionResult(
        pipeline_id="test_pipeline",
        status="success",
        node_results={
            "node_1": NodeExecutionResult(
                node_id="node_1",
                status="success",
                output_artifact_id="art_1",
                metrics={"accuracy": 0.95}
            )
        }
    )
    
    # Run Task
    run_pipeline_task(MOCK_JOB_ID, MOCK_PIPELINE_CONFIG)
    
    # Assertions
    assert job.status == "completed"
    assert job.progress == 100
    assert job.metrics == {"accuracy": 0.95}
    assert job.finished_at is not None
    
    # Verify Engine was called
    engine_instance.run.assert_called_once()

def test_run_pipeline_task_tuning_job(mock_get_db_session, mock_engine_class):
    # Setup Mock Session and Job
    session = MagicMock()
    mock_get_db_session.return_value = session
    
    job = HyperparameterTuningJob(id=MOCK_JOB_ID, status="queued")
    
    # Configure query side effects
    # First query (TrainingJob) returns None
    # Second query (HyperparameterTuningJob) returns job
    def side_effect(*args, **kwargs):
        query_mock = MagicMock()
        if args[0] == TrainingJob:
            query_mock.filter.return_value.first.return_value = None
        elif args[0] == HyperparameterTuningJob:
            query_mock.filter.return_value.first.return_value = job
        return query_mock
        
    session.query.side_effect = side_effect
    
    # Setup Mock Engine
    engine_instance = mock_engine_class.return_value
    engine_instance.run.return_value = PipelineExecutionResult(
        pipeline_id="test_pipeline",
        status="success",
        node_results={}
    )
    
    # Run Task
    run_pipeline_task(MOCK_JOB_ID, MOCK_PIPELINE_CONFIG)
    
    # Assertions
    assert job.status == "completed"
    assert job.progress == 100
    
    # Verify Engine was called
    engine_instance.run.assert_called_once()

def test_run_pipeline_task_failure(mock_get_db_session, mock_engine_class):
    # Setup Mock Session and Job
    session = MagicMock()
    mock_get_db_session.return_value = session
    
    job = TrainingJob(id=MOCK_JOB_ID, status="queued")
    session.query.return_value.filter.return_value.first.return_value = job
    
    # Setup Mock Engine to raise exception
    engine_instance = mock_engine_class.return_value
    engine_instance.run.side_effect = Exception("Pipeline crashed")
    
    # Run Task
    run_pipeline_task(MOCK_JOB_ID, MOCK_PIPELINE_CONFIG)
    
    # Assertions
    assert job.status == "failed"
    assert "Pipeline crashed" in job.error_message
    assert job.finished_at is not None
