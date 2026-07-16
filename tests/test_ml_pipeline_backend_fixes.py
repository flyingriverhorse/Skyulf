"""Regression tests for the r5 audit fixes in `backend/ml_pipeline/`:

1. POST /deployment/predict rejects requests over
   `Settings.MAX_PREDICT_REQUEST_ROWS` (enforced in deployment/api.py, mirroring
   the MAX_UPLOAD_SIZE pattern in data_ingestion).
2. `EvaluationService.get_job_evaluation` raises instead of silently
   serving mismatched job_id evaluation data.
3. `tasks.get_db_session()` lazy-init is race-free under concurrent calls.
4. `run_pipeline_task` / `run_pipeline_batch_task` mark the job as "failed"
   in the DB when an exception escapes `execute_pipeline` itself (an
   infra-level failure the pipeline's own internal handler didn't catch).
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.config import get_settings
from backend.database.models import BasicTrainingJob
from backend.ml_pipeline._execution.strategies import JobStrategyFactory
from backend.ml_pipeline.deployment.api import predict
from backend.ml_pipeline.deployment.schemas import PredictionRequest
from backend.ml_pipeline.tasks import run_pipeline_batch_task, run_pipeline_task


def _make_request() -> Request:
    """Build a minimal real Starlette Request so slowapi's rate limiter decorator
    (which requires an actual Request instance, not a mock) can inspect it."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/deployment/predict",
        "headers": [],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# 1. Unbounded /predict payload
# ---------------------------------------------------------------------------


async def test_prediction_request_accepts_up_to_max_row_limit():
    """A request at the configured row limit should pass the guard and proceed."""
    max_rows = get_settings().MAX_PREDICT_REQUEST_ROWS
    data = [{"col": i} for i in range(max_rows)]
    request = PredictionRequest(data=data)

    with patch(
        "backend.ml_pipeline.deployment.api.DeploymentService.get_active_deployment",
        new=AsyncMock(return_value=None),
    ):
        # No active deployment -> 404, but this proves the row-count guard
        # did NOT reject the at-limit payload (it would raise 413 first otherwise).
        with pytest.raises(HTTPException) as exc_info:
            await predict(_make_request(), request, MagicMock())
        assert exc_info.value.status_code == 404


async def test_prediction_request_rejects_oversized_batch():
    """A request exceeding the configured row limit should be rejected with 413."""
    max_rows = get_settings().MAX_PREDICT_REQUEST_ROWS
    data = [{"col": i} for i in range(max_rows + 1)]
    request = PredictionRequest(data=data)

    with pytest.raises(HTTPException) as exc_info:
        await predict(_make_request(), request, MagicMock())
    assert exc_info.value.status_code == 413


# ---------------------------------------------------------------------------
# 2. Evaluation job_id mismatch
# ---------------------------------------------------------------------------


async def test_get_job_evaluation_raises_on_job_id_mismatch():
    """A stale/foreign job_id embedded in the loaded artifact must not be
    silently served back to the caller."""
    from backend.ml_pipeline._services.evaluation_service import EvaluationService

    job_id = "job-a"
    mock_job = MagicMock()
    mock_job.artifact_uri = "some/uri"
    mock_job.node_id = "node-1"
    mock_job.status = "completed"

    mock_store = MagicMock()
    mock_store.exists.side_effect = lambda key: key == f"{job_id}_evaluation_data"
    mock_store.load.return_value = {"job_id": "job-b", "splits": {}}

    with (
        patch(
            "backend.ml_pipeline._services.evaluation_service.JobService.get_job_by_id",
            new=AsyncMock(return_value=mock_job),
        ),
        patch(
            "backend.ml_pipeline._services.evaluation_service.ArtifactFactory.get_artifact_store",
            return_value=mock_store,
        ),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await EvaluationService.get_job_evaluation(MagicMock(), job_id)

        # The RuntimeError should wrap the original mismatch ValueError.
        assert "mismatch" in str(exc_info.value.__cause__).lower()


async def test_get_job_evaluation_succeeds_when_job_id_matches():
    """Sanity check: matching job_id data is still returned normally."""
    from backend.ml_pipeline._services.evaluation_service import EvaluationService

    job_id = "job-a"
    mock_job = MagicMock()
    mock_job.artifact_uri = "some/uri"
    mock_job.node_id = "node-1"
    mock_job.status = "completed"

    mock_store = MagicMock()
    mock_store.exists.side_effect = lambda key: key == f"{job_id}_evaluation_data"
    mock_store.load.return_value = {"job_id": job_id, "splits": {}}

    with (
        patch(
            "backend.ml_pipeline._services.evaluation_service.JobService.get_job_by_id",
            new=AsyncMock(return_value=mock_job),
        ),
        patch(
            "backend.ml_pipeline._services.evaluation_service.ArtifactFactory.get_artifact_store",
            return_value=mock_store,
        ),
    ):
        result = await EvaluationService.get_job_evaluation(MagicMock(), job_id)
        assert result["job_id"] == job_id


async def test_get_job_evaluation_decodes_reference_crosstab_labels():
    """A clustering reference column that was label-encoded upstream (e.g.
    species name -> 0/1/2) should have its crosstab keys decoded back to the
    original text, not left as numeric-looking strings, when a matching
    LabelEncoder is present in the bundled feature engineer."""
    from sklearn.preprocessing import LabelEncoder

    from backend.ml_pipeline._services.evaluation_service import EvaluationService

    job_id = "job-a"
    mock_job = MagicMock()
    mock_job.artifact_uri = "some/uri"
    mock_job.node_id = "node-1"
    mock_job.status = "completed"

    species_encoder = LabelEncoder()
    species_encoder.fit(["setosa", "versicolor", "virginica"])

    mock_feature_engineer = MagicMock()
    mock_feature_engineer.fitted_steps = [
        {
            "type": "LabelEncoder",
            "artifact": {"encoders": {"species": species_encoder}},
        }
    ]

    evaluation_data = {
        "job_id": job_id,
        "problem_type": "clustering",
        "splits": {
            "train": {
                "clustering": {
                    "reference_column": "species",
                    "reference_crosstab": {
                        "0": {"0": 46, "1": 2},
                        "1": {"1": 44},
                        "2": {"2": 50},
                    },
                }
            }
        },
    }

    mock_store = MagicMock()
    mock_store.exists.side_effect = lambda key: (
        key
        in {
            f"{job_id}_evaluation_data",
            job_id,
        }
    )
    mock_store.load.side_effect = lambda key: (
        evaluation_data
        if key == f"{job_id}_evaluation_data"
        else {"feature_engineer": mock_feature_engineer, "target_column": ""}
    )

    with (
        patch(
            "backend.ml_pipeline._services.evaluation_service.JobService.get_job_by_id",
            new=AsyncMock(return_value=mock_job),
        ),
        patch(
            "backend.ml_pipeline._services.evaluation_service.ArtifactFactory.get_artifact_store",
            return_value=mock_store,
        ),
    ):
        result = await EvaluationService.get_job_evaluation(MagicMock(), job_id)

        crosstab = result["splits"]["train"]["clustering"]["reference_crosstab"]
        assert crosstab == {
            "0": {"setosa": 46, "versicolor": 2},
            "1": {"versicolor": 44},
            "2": {"virginica": 50},
        }


# ---------------------------------------------------------------------------
# 3. get_db_session lazy-init race
# ---------------------------------------------------------------------------


def test_get_db_session_concurrent_init_creates_single_engine():
    """Simulate many threads racing the lazy-init check; only one engine
    should ever be created (double-checked locking prevents the leak)."""
    import backend.ml_pipeline.tasks as tasks_module

    # Reset module globals to force re-initialization.
    tasks_module._sync_engine = None
    tasks_module._sync_session_factory = None

    created_engines = []
    create_engine_call_count = 0
    lock = threading.Lock()

    def fake_create_engine(*args, **kwargs):
        nonlocal create_engine_call_count
        # Simulate work between the "is None" check and engine creation,
        # widening the race window so concurrent threads are likely to
        # collide without the lock.
        with lock:
            create_engine_call_count += 1
        engine = MagicMock()
        created_engines.append(engine)
        return engine

    with (
        patch("backend.ml_pipeline.tasks.create_engine", side_effect=fake_create_engine),
        patch("backend.ml_pipeline.tasks.sessionmaker", return_value=MagicMock()),
        patch("backend.ml_pipeline.tasks.get_settings") as mock_settings,
    ):
        mock_settings.return_value.DATABASE_URL = "sqlite+aiosqlite:///test.db"

        barrier = threading.Barrier(20)

        def call_get_db_session():
            barrier.wait()
            return tasks_module.get_db_session()

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(call_get_db_session) for _ in range(20)]
            for f in futures:
                f.result()

    assert create_engine_call_count == 1
    assert len(created_engines) == 1
    assert tasks_module._sync_engine is created_engines[0]

    # Reset globals again so this test doesn't leak state into other tests.
    tasks_module._sync_engine = None
    tasks_module._sync_session_factory = None


# ---------------------------------------------------------------------------
# 4. Escaped exception from execute_pipeline must still mark the job failed
# ---------------------------------------------------------------------------


def test_run_pipeline_task_marks_job_failed_when_execute_pipeline_raises_unexpectedly():
    """If execute_pipeline itself raises (violating its own 'raises nothing'
    contract, e.g. an infra-level bug), the job's status must still end up
    as 'failed' instead of being stuck at 'running'/'queued'."""
    job_id = str(uuid.uuid4())
    job = BasicTrainingJob(id=job_id, status="running")

    session = MagicMock()

    with (
        patch("backend.ml_pipeline.tasks.get_db_session", return_value=session),
        patch(
            "backend.ml_pipeline.tasks.execute_pipeline",
            side_effect=RuntimeError("infra failure"),
        ),
        patch("backend.exceptions.handlers.record_pipeline_error"),
        patch.object(
            JobStrategyFactory,
            "find_job",
            return_value=(job, JobStrategyFactory.get_strategy_by_job(job)),
        ),
        pytest.raises(RuntimeError),
    ):
        run_pipeline_task(job_id, {"nodes": []})

    assert job.status == "failed"


def test_run_pipeline_batch_task_marks_job_failed_when_execute_pipeline_raises_unexpectedly():
    """Same fallback behavior for the batch-task's `_run_one` inner function."""
    job_id = str(uuid.uuid4())
    job = BasicTrainingJob(id=job_id, status="running")

    session = MagicMock()

    from backend.ml_pipeline._execution.strategies import JobStrategyFactory

    with (
        patch("backend.ml_pipeline.tasks.get_db_session", return_value=session),
        patch(
            "backend.ml_pipeline.tasks.execute_pipeline",
            side_effect=RuntimeError("infra failure"),
        ),
        patch("backend.exceptions.handlers.record_pipeline_error"),
        patch.object(
            JobStrategyFactory,
            "find_job",
            return_value=(job, JobStrategyFactory.get_strategy_by_job(job)),
        ),
        pytest.raises(RuntimeError),
    ):
        run_pipeline_batch_task([(job_id, {"nodes": []})])

    assert job.status == "failed"
