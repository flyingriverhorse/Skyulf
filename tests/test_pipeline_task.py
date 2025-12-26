import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.database.models import HyperparameterTuningJob, TrainingJob, DataSource
from backend.ml_pipeline.execution.schemas import (
    NodeExecutionResult,
    PipelineExecutionResult,
)
from backend.ml_pipeline.tasks import run_pipeline_task

# Mock data
MOCK_JOB_ID = str(uuid.uuid4())
MOCK_PIPELINE_CONFIG = {
    "pipeline_id": "test_pipeline",
    "nodes": [
        {
            "node_id": "node_1",
            "step_type": "data_loader",
            "params": {"dataset_id": "ds_1"},
            "inputs": [],
        }
    ],
    "metadata": {},
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
                metrics={"accuracy": 0.95},
            )
        },
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
        pipeline_id="test_pipeline", status="success", node_results={}
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


def test_run_pipeline_task_resolves_dataset_id(mock_get_db_session, mock_engine_class):
    """
    Test that run_pipeline_task correctly resolves a numeric dataset_id (28)
    to a file path using the DataSource table.
    """
    # Setup Mock Session
    session = MagicMock()
    mock_get_db_session.return_value = session

    # Setup Job
    job = TrainingJob(id=MOCK_JOB_ID, status="queued")

    # Setup DataSource Mock
    mock_ds = MagicMock()
    mock_ds.to_dict.return_value = {
        "config": {"file_path": "uploads/data/resolved_path.csv"}
    }

    def query_side_effect(model):
        query_mock = MagicMock()
        # Check by name to avoid import mismatch issues
        model_name = getattr(model, "__name__", str(model))
        print(f"DEBUG: query called with {model} (name={model_name})")

        if model_name == "TrainingJob":
            query_mock.filter.return_value.first.return_value = job
        elif model_name == "DataSource":
            print("DEBUG: Returning mock_ds for DataSource")
            query_mock.filter.return_value.first.return_value = mock_ds
        return query_mock

    session.query.side_effect = query_side_effect

    # Config with numeric ID
    config_with_numeric_id = {
        "pipeline_id": "test_pipeline",
        "nodes": [
            {
                "node_id": "node_1",
                "step_type": "data_loader",
                "params": {"dataset_id": "28"},  # Numeric ID as string
                "inputs": [],
            }
        ],
        "metadata": {},
    }

    # Setup Mock Engine
    engine_instance = mock_engine_class.return_value
    engine_instance.run.return_value = PipelineExecutionResult(
        pipeline_id="test_pipeline",
        status="success",
        node_results={},
    )

    # Patch extract_file_path_from_source to avoid file existence check
    # Note: With SmartCatalog, this patch might be needed inside the catalog's method if we were testing catalog execution,
    # but here we are just testing tasks.py flow.
    # However, SmartCatalog imports it locally, so we might need to patch it where it's used.
    # But wait, tasks.py doesn't call it anymore! SmartCatalog calls it.
    # And we are mocking PipelineEngine, so engine.run() is a mock.
    # So SmartCatalog.load() is NEVER CALLED in this test!
    # So we don't need to patch extract_file_path_from_source for this test anymore.
    
    # Run Task
    run_pipeline_task(MOCK_JOB_ID, config_with_numeric_id)

    # Verify Engine was initialized with the SmartCatalog
    # and the config still has the original ID (because SmartCatalog handles resolution)
    run_call_args = engine_instance.run.call_args
    passed_config = run_call_args[0][0]

    # The node in the passed config should have the ORIGINAL ID "28"
    assert passed_config.nodes[0].params["dataset_id"] == "28"
    
    # Verify SmartCatalog was used
    # We can check the catalog passed to PipelineEngine constructor
    init_call_args = mock_engine_class.call_args
    passed_catalog = init_call_args[1]["catalog"]
    
    # It should be a SmartCatalog instance
    from backend.data.catalog import SmartCatalog
    assert isinstance(passed_catalog, SmartCatalog)
