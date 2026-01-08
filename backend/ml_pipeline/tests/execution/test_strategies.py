import pytest
from unittest.mock import MagicMock, AsyncMock
from backend.ml_pipeline.execution.strategies import BasicTrainingStrategy, AdvancedTuningStrategy
from backend.ml_pipeline.execution.schemas import NodeConfig
from backend.ml_pipeline.constants import StepType

@pytest.fixture
def mock_session():
    return AsyncMock()

@pytest.fixture
def mock_node_config():
    return NodeConfig(
        node_id="test_node",
        step_type=StepType.BASIC_TRAINING,
        params={"target_column": "target", "model_type": "rf"}
    )

@pytest.mark.asyncio
async def test_basic_training_strategy_execute(mock_session, mock_node_config):
    # Mock the manager
    with pytest.MonkeyPatch.context() as m:
        mock_manager = AsyncMock()
        mock_manager.create_training_job.return_value = "job_123"
        m.setattr("backend.ml_pipeline.execution.strategies.BasicTrainingManager", mock_manager)

        strategy = BasicTrainingStrategy()
        job_id = await strategy.execute("pipeline_1", mock_node_config, {}, mock_session)
        
        assert job_id == "job_123"
        mock_manager.create_training_job.assert_called_once()

@pytest.mark.asyncio
async def test_advanced_tuning_strategy_execute(mock_session, mock_node_config):
    # Mock the manager
    with pytest.MonkeyPatch.context() as m:
        mock_manager = AsyncMock()
        mock_manager.create_tuning_job.return_value = "job_456"
        m.setattr("backend.ml_pipeline.execution.strategies.AdvancedTuningManager", mock_manager)

        strategy = AdvancedTuningStrategy()
        job_id = await strategy.execute("pipeline_1", mock_node_config, {}, mock_session)
        
        assert job_id == "job_456"
        mock_manager.create_tuning_job.assert_called_once()
