import pytest
from unittest.mock import MagicMock
from backend.ml_pipeline.execution.strategies import (
    BasicTrainingStrategy,
    AdvancedTuningStrategy,
    JobStrategyFactory,
)
from backend.database.models import BasicTrainingJob, AdvancedTuningJob


def test_basic_training_strategy_get_job_model() -> None:
    strategy = BasicTrainingStrategy()
    assert strategy.get_job_model() is BasicTrainingJob


def test_basic_training_strategy_initial_log() -> None:
    strategy = BasicTrainingStrategy()
    job = MagicMock()
    job.version = 3
    log = strategy.get_initial_log(job)
    assert "Training Job Version: 3" in log


def test_advanced_tuning_strategy_get_job_model() -> None:
    strategy = AdvancedTuningStrategy()
    assert strategy.get_job_model() is AdvancedTuningJob


def test_advanced_tuning_strategy_initial_log() -> None:
    strategy = AdvancedTuningStrategy()
    job = MagicMock()
    job.run_number = 5
    log = strategy.get_initial_log(job)
    assert "Tuning Job Run: 5" in log


def test_strategy_factory_basic() -> None:
    strategy = JobStrategyFactory.get_strategy("basic_training")
    assert isinstance(strategy, BasicTrainingStrategy)


def test_strategy_factory_advanced() -> None:
    strategy = JobStrategyFactory.get_strategy("advanced_tuning")
    assert isinstance(strategy, AdvancedTuningStrategy)
