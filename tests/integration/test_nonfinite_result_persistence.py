"""Exercise durable finite-or-null JSON writes through EDA and job completion paths."""

import json
import math
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import polars as pl
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from backend.database.models import Base, DataSource, EDAReport, TrainingJob
from backend.eda import tasks
from backend.ml_pipeline._execution.advanced_tuning_manager import AdvancedTuningManager
from backend.ml_pipeline._execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline._execution.schemas import (
    JobStatus,
    NodeExecutionResult,
    PipelineExecutionResult,
)
from backend.ml_pipeline._execution.strategies import AdvancedTuningStrategy, BasicTrainingStrategy
from backend.ml_pipeline._services import pipeline_execution_service
from skyulf.profiling.schemas import (
    DatasetProfile,
    SeasonalityStats,
    TimeSeriesAnalysis,
    TimeSeriesPoint,
)


@pytest.fixture
def database(tmp_path):
    """Keep the durable SQL write/reload isolated from application database state."""
    path = tmp_path / "finite-json.sqlite"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    yield engine, path
    engine.dispose()


def make_job(run_mode="fixed", model_type="preview"):
    """Provide required real job columns for the same completion path used by preview jobs."""
    return TrainingJob(
        id="finite-job",
        pipeline_id="finite-pipeline",
        node_id="preview",
        dataset_source_id="data",
        model_type=model_type,
        run_mode=run_mode,
        graph={},
        status="running",
    )


@pytest.mark.parametrize("strategy", [BasicTrainingStrategy(), AdvancedTuningStrategy()])
def test_pipeline_success_persists_finite_metrics_and_tuning_fields(
    database, monkeypatch, strategy
):
    """The real completion commit must not persist Infinity or mutate the engine result."""
    engine, _ = database
    monkeypatch.setattr(pipeline_execution_service, "publish_job_event", lambda event: None)
    metrics: dict[str, Any] = {
        "score": float("inf"),
        "details": [float("nan"), {"minimum": float("-inf"), "count": 3}],
        "label": "Infinity",
        "best_score": float("-inf"),
        "best_params": {"limit": float("inf")},
        "trials": [{"score": float("nan"), "params": {"limit": float("inf")}}],
        "decision_thresholds": {"yes": 0.5},
        "decision_threshold_metric": "f1",
    }
    result = PipelineExecutionResult(
        pipeline_id="finite-pipeline",
        status="success",
        node_results={
            "preview": NodeExecutionResult(
                node_id="preview", status="success", metrics=metrics, execution_time=float("inf")
            ),
        },
    )
    with Session(engine) as session:
        job = make_job(strategy.run_mode)
        session.add(job)
        session.commit()
        pipeline_execution_service._write_pipeline_result(
            session, job, strategy, job.id, result, "artifact"
        )
    with Session(engine) as session:
        saved = session.get(TrainingJob, "finite-job")
        assert saved is not None and saved.metrics is not None
        assert saved.tuned_thresholds is not None
        assert saved.status == "completed"
        assert saved.metrics["score"] is None
        assert saved.metrics["details"] == [None, {"minimum": None, "count": 3}]
        assert saved.metrics["node_timings"][0]["execution_time"] is None
        assert saved.metrics["label"] == "Infinity"
        assert saved.tuned_thresholds["thresholds"] == {"yes": 0.5}
        if strategy.run_mode == "tuned":
            assert saved.best_score is None
            assert saved.best_params == {"limit": None}
            assert saved.results == [{"score": None, "params": {"limit": None}}]
        raw = session.execute(
            text("SELECT metrics FROM training_jobs WHERE id = 'finite-job'")
        ).scalar_one()
        assert json.loads(json.dumps(json.loads(raw), allow_nan=False)) == saved.metrics
    assert math.isinf(metrics["score"])
    detail = metrics["details"][0]
    assert isinstance(detail, float) and math.isnan(detail)


@pytest.mark.parametrize("manager", [BasicTrainingManager, AdvancedTuningManager])
def test_manager_status_writes_normalize_numeric_scalars_without_mutating_callers(
    database, manager
):
    """Worker status updates must share the completion path's finite JSON contract."""
    engine, _ = database
    metrics = {
        "score": float("inf"),
        "numpy": np.array([1.0, float("nan")]),
        "decimal": Decimal("Infinity"),
    }
    result = {
        "metrics": metrics,
        "score": float("inf"),
        "best_score": float("inf"),
        "best_params": {"v": float("nan")},
    }
    mode = "fixed" if manager is BasicTrainingManager else "tuned"
    with Session(engine) as session:
        session.add(make_job(mode, "random_forest_classifier"))
        session.commit()
        assert manager.update_status_sync(
            session, "finite-job", status=JobStatus.COMPLETED, result=result
        )
    with Session(engine) as session:
        saved = session.get(TrainingJob, "finite-job")
        assert saved is not None and saved.metrics is not None
        values = saved.metrics if mode == "fixed" else saved.metrics["metrics"]
        assert values == {"score": None, "numpy": [1.0, None], "decimal": None}
        assert json.loads(json.dumps(saved.metrics, allow_nan=False)) == saved.metrics
    assert math.isinf(metrics["score"])
    assert np.isnan(metrics["numpy"][1])
    assert metrics["decimal"].is_infinite()


async def test_eda_task_commits_finite_typed_and_nested_profile_json(
    database, tmp_path, monkeypatch
):
    """EDA's actual model_dump-to-JSON-column write must survive a fresh SQLite session."""
    engine, path = database
    csv_path = tmp_path / "profile.csv"
    csv_path.write_text("value\n1\n2\n", encoding="utf-8")
    profile = DatasetProfile(
        row_count=2,
        column_count=1,
        duplicate_rows=0,
        missing_cells_percentage=0.0,
        memory_usage_mb=1.0,
        columns={},
        vif={"value": float("inf")},
        sample_data=[{"value": float("nan"), "label": "NaN"}],
        timeseries=TimeSeriesAnalysis(
            date_col="date",
            trend=[TimeSeriesPoint(date="2026-01-01", values={"value": float("-inf")})],
            seasonality=SeasonalityStats(day_of_week=[], month_of_year=[]),
        ),
    )
    with Session(engine) as session:
        source = DataSource(name="finite profile", type="csv", config={"file_path": str(csv_path)})
        session.add(source)
        session.flush()
        report = EDAReport(data_source_id=source.id, config={})
        session.add(report)
        session.commit()
        report_id = report.id
    monkeypatch.setattr(
        tasks.DataService, "load_file", AsyncMock(return_value=pl.DataFrame({"value": [1, 2]}))
    )
    monkeypatch.setattr(tasks, "_run_eda_analyzer", lambda frame, config: profile)
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        async with AsyncSession(async_engine) as session:
            await tasks.run_eda_analysis(report_id, session)
    finally:
        await async_engine.dispose()
    with Session(engine) as session:
        saved = session.get(EDAReport, report_id)
        assert saved is not None and saved.profile_data is not None
        assert saved.status == "COMPLETED", saved.error_message
        assert saved.profile_data["vif"] == {"value": None}
        assert saved.profile_data["sample_data"] == [{"value": None, "label": "NaN"}]
        assert saved.profile_data["timeseries"]["trend"][0]["values"] == {"value": None}
        assert json.loads(json.dumps(saved.profile_data, allow_nan=False)) == saved.profile_data
    assert profile.vif is not None and math.isinf(profile.vif["value"])
