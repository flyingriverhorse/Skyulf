"""Tests for ThresholdTuningService: preview/save/toggle/clear."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, TrainingJob
from backend.ml_pipeline._services.threshold_tuning_service import (
    ThresholdTuningError,
    ThresholdTuningService,
)

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    """Provides an in-memory async SQLite session with all tables created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


async def _insert_job(session: AsyncSession, job_id: str) -> None:
    """Inserts a minimal `training_jobs` row via raw SQL (ORM defaults don't apply to raw INSERT)."""
    await session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": "pipe-1",
            "node_id": "node-1",
            "ds_id": "ds-1",
            "user_id": None,
            "status": "completed",
            "run_mode": "fixed",
            "version": 1,
            "model_type": "random_forest",
            "graph": "{}",
            "artifact_uri": job_id,
            "error_message": None,
            "progress": 100,
            "current_step": None,
            "started_at": None,
            "finished_at": None,
        },
    )
    await session.commit()


def _fake_evaluation_data() -> dict:
    """Builds a raw (undecoded) evaluation payload matching EvaluationService's real shape."""
    return {
        "job_id": "job-1",
        "problem_type": "classification",
        "splits": {
            "validation": {
                "y_true": [0, 1, 2, 2, 1],
                "y_pred": [0, 1, 2, 2, 0],
                "y_proba": {
                    "classes": ["0", "1", "2"],
                    "values": [
                        [0.5, 0.3, 0.2],
                        [0.2, 0.6, 0.2],
                        [0.34, 0.33, 0.33],
                        [0.1, 0.1, 0.8],
                        [0.4, 0.4, 0.2],
                    ],
                },
            },
            "test": None,
        },
    }


def _fake_binary_evaluation_data() -> dict:
    """Builds a raw (undecoded) 2-class evaluation payload for roc_auc coverage."""
    return {
        "job_id": "job-1",
        "problem_type": "classification",
        "splits": {
            "validation": {
                "y_true": [0, 1, 0, 1, 1, 0, 1, 0],
                "y_pred": [0, 1, 0, 0, 1, 0, 1, 1],
                "y_proba": {
                    "classes": ["0", "1"],
                    "values": [
                        [0.9, 0.1],
                        [0.2, 0.8],
                        [0.7, 0.3],
                        [0.55, 0.45],
                        [0.1, 0.9],
                        [0.6, 0.4],
                        [0.3, 0.7],
                        [0.4, 0.6],
                    ],
                },
            },
            "test": None,
        },
    }


@pytest.mark.asyncio
async def test_preview_returns_thresholds_for_valid_job(async_session):
    """preview() returns thresholds/classes/metric/split_used using the validation split."""
    await _insert_job(async_session, "job-1")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(_fake_evaluation_data(), None)),
    ):
        result = await ThresholdTuningService.preview(async_session, "job-1", metric="f1")

    assert result["metric"] == "f1"
    assert result["split_used"] == "validation"
    assert set(result["classes"]) == {0, 1, 2}
    assert set(result["thresholds"].keys()) == {"0", "1", "2"}
    assert all(isinstance(v, float) for v in result["thresholds"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", ["precision", "recall", "balanced_accuracy"])
async def test_preview_returns_thresholds_for_other_metrics(async_session, metric):
    """preview() works end-to-end for precision/recall/balanced_accuracy metrics too."""
    await _insert_job(async_session, "job-1")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(_fake_evaluation_data(), None)),
    ):
        result = await ThresholdTuningService.preview(async_session, "job-1", metric=metric)

    assert result["metric"] == metric
    assert set(result["classes"]) == {0, 1, 2}
    assert set(result["thresholds"].keys()) == {"0", "1", "2"}
    assert all(isinstance(v, float) for v in result["thresholds"].values())


@pytest.mark.asyncio
async def test_preview_roc_auc_works_for_binary_classification(async_session):
    """preview() succeeds with roc_auc for a binary (2-class) job."""
    await _insert_job(async_session, "job-1")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(_fake_binary_evaluation_data(), None)),
    ):
        result = await ThresholdTuningService.preview(async_session, "job-1", metric="roc_auc")

    assert result["metric"] == "roc_auc"
    assert set(result["classes"]) == {0, 1}
    assert set(result["thresholds"].keys()) == {"0", "1"}


@pytest.mark.asyncio
async def test_preview_roc_auc_raises_threshold_tuning_error_for_multiclass(async_session):
    """preview() raises ThresholdTuningError (not a raw ValueError) for roc_auc + 3+ classes.

    optimize_thresholds() always scores hard, post-threshold class predictions
    (never probability scores), and roc_auc_score() on discrete multiclass
    labels raises ValueError internally (it needs a 2D probability matrix for
    multi_class="ovr"/"ovo"), so this must be guarded against explicitly.
    """
    await _insert_job(async_session, "job-1")

    with (
        patch(
            "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
            "._load_raw_evaluation_data",
            new=AsyncMock(return_value=(_fake_evaluation_data(), None)),
        ),
        pytest.raises(ThresholdTuningError),
    ):
        await ThresholdTuningService.preview(async_session, "job-1", metric="roc_auc")


@pytest.mark.asyncio
async def test_preview_falls_back_to_test_split_when_no_validation(async_session):
    """preview() silently falls back to the test split when validation is absent."""
    await _insert_job(async_session, "job-1")
    data = _fake_evaluation_data()
    data["splits"]["test"] = data["splits"].pop("validation")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(data, None)),
    ):
        result = await ThresholdTuningService.preview(async_session, "job-1", metric="accuracy")

    assert result["split_used"] == "test"


@pytest.mark.asyncio
async def test_preview_raises_for_missing_job(async_session):
    """preview() raises ThresholdTuningError when the job doesn't exist."""
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.preview(async_session, "nonexistent", metric="f1")


@pytest.mark.asyncio
async def test_preview_raises_for_unsupported_metric(async_session):
    """preview() raises ThresholdTuningError for a metric outside the supported set."""
    await _insert_job(async_session, "job-1")
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.preview(async_session, "job-1", metric="not_a_metric")


@pytest.mark.asyncio
async def test_preview_raises_when_no_splits_available(async_session):
    """preview() raises ThresholdTuningError when neither validation nor test splits exist."""
    await _insert_job(async_session, "job-1")
    data = {"job_id": "job-1", "problem_type": "classification", "splits": {"train": {}}}

    with (
        patch(
            "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
            "._load_raw_evaluation_data",
            new=AsyncMock(return_value=(data, None)),
        ),
        pytest.raises(ThresholdTuningError),
    ):
        await ThresholdTuningService.preview(async_session, "job-1", metric="f1")


@pytest.mark.asyncio
async def test_save_toggle_clear_round_trip(async_session):
    """save() persists+enables thresholds; toggle() flips the flag; clear() removes them."""
    await _insert_job(async_session, "job-2")

    saved = await ThresholdTuningService.save(
        async_session,
        "job-2",
        thresholds={"0": 0.6, "1": 0.5, "2": 0.3},
        classes=[0, 1, 2],
        metric="f1",
        split_used="validation",
    )
    assert saved is True

    job = (
        await async_session.execute(select(TrainingJob).where(TrainingJob.id == "job-2"))
    ).scalar_one()
    assert job.tuned_thresholds_enabled is True
    assert job.tuned_thresholds["thresholds"] == {"0": 0.6, "1": 0.5, "2": 0.3}

    await ThresholdTuningService.toggle(async_session, "job-2", enabled=False)
    await async_session.refresh(job)
    assert job.tuned_thresholds_enabled is False

    await ThresholdTuningService.clear(async_session, "job-2")
    await async_session.refresh(job)
    assert job.tuned_thresholds is None
    assert job.tuned_thresholds_enabled is False


@pytest.mark.asyncio
async def test_toggle_raises_when_no_saved_thresholds(async_session):
    """toggle() raises ThresholdTuningError when the job has no saved tuned thresholds yet."""
    await _insert_job(async_session, "job-3")
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.toggle(async_session, "job-3", enabled=True)


@pytest.mark.asyncio
async def test_get_saved_returns_empty_shell_when_nothing_saved(async_session):
    """get_saved() returns an all-None/disabled shell for a job with no saved thresholds."""
    await _insert_job(async_session, "job-4")

    result = await ThresholdTuningService.get_saved(async_session, "job-4")

    assert result == {
        "thresholds": None,
        "classes": None,
        "metric": None,
        "split_used": None,
        "computed_at": None,
        "enabled": False,
    }


@pytest.mark.asyncio
async def test_get_saved_reflects_saved_and_toggled_state(async_session):
    """get_saved() reflects the saved thresholds and current enabled flag after save()/toggle()."""
    await _insert_job(async_session, "job-5")

    await ThresholdTuningService.save(
        async_session,
        "job-5",
        thresholds={"0": 0.6, "1": 0.5, "2": 0.3},
        classes=[0, 1, 2],
        metric="f1",
        split_used="validation",
    )

    result = await ThresholdTuningService.get_saved(async_session, "job-5")
    assert result["thresholds"] == {"0": 0.6, "1": 0.5, "2": 0.3}
    assert result["classes"] == [0, 1, 2]
    assert result["metric"] == "f1"
    assert result["split_used"] == "validation"
    assert result["computed_at"] is not None
    assert result["enabled"] is True

    await ThresholdTuningService.toggle(async_session, "job-5", enabled=False)
    result = await ThresholdTuningService.get_saved(async_session, "job-5")
    assert result["enabled"] is False
    # Thresholds themselves stay intact even when disabled — only the flag flips.
    assert result["thresholds"] == {"0": 0.6, "1": 0.5, "2": 0.3}


@pytest.mark.asyncio
async def test_get_saved_raises_for_missing_job(async_session):
    """get_saved() raises ThresholdTuningError when the job doesn't exist."""
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.get_saved(async_session, "nonexistent")


@pytest.mark.asyncio
async def test_save_toggle_clear_raise_for_missing_job(async_session):
    """save()/toggle()/clear() all raise ThresholdTuningError for an unknown job id."""
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.save(
            async_session,
            "nonexistent",
            thresholds={"0": 0.5},
            classes=[0],
            metric="f1",
            split_used="validation",
        )
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.toggle(async_session, "nonexistent", enabled=True)
    with pytest.raises(ThresholdTuningError):
        await ThresholdTuningService.clear(async_session, "nonexistent")
