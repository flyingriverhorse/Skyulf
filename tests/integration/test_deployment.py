from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService


class _IdentityEngineer:
    def __init__(self, target_label_encoder):
        self.fitted_steps = [
            {
                "name": "label_encode_target",
                "type": "LabelEncoder",
                "applier": None,
                "artifact": {"encoders": {"__target__": target_label_encoder}, "columns": []},
            }
        ]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class _FixedPredictor:
    def __init__(self, preds):
        self._preds = preds

    def predict(self, X):
        return np.asarray(self._preds)


# Use an in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_deployment_flow(async_session, tmp_path):
    # 1. Setup: Create a dummy model artifact
    pipeline_id = "test_pipeline_deploy"
    node_id = "test_node_deploy"
    job_id = "test_job_123"

    # Create a simple model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([2, 4, 6])
    model.fit(X, y)

    # Create directory structure matching what DeploymentService expects
    # It expects exports/models/{pipeline_id} relative to os.getcwd()
    # We will mock os.getcwd() to return tmp_path

    models_dir = tmp_path / "exports" / "models" / pipeline_id
    models_dir.mkdir(parents=True, exist_ok=True)

    store = LocalArtifactStore(str(models_dir))
    # Save with job_id as key, as DeploymentService expects job_id in the URI
    store.save(job_id, model)

    # 2. Create Job in DB
    # We need to insert manually because TrainingJob might have required fields not in init
    await async_session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": pipeline_id,
            "node_id": node_id,
            "ds_id": "ds1",
            "user_id": None,
            "status": "completed",
            "run_mode": "fixed",
            "version": 1,
            "model_type": "linear_regression",
            "graph": "{}",
            "artifact_uri": job_id,  # The engine saves it as job_id now
            "error_message": None,
            "progress": 0,
            "current_step": None,
            "started_at": None,
            "finished_at": None,
        },
    )
    await async_session.commit()

    # Patch os.getcwd to return tmp_path
    with patch("os.getcwd", return_value=str(tmp_path)):
        # 3. Deploy
        deployment = await DeploymentService.deploy_model(async_session, job_id)
        assert deployment.is_active == True
        # The service constructs the URI as pipeline_id/job_id
        assert deployment.artifact_uri == f"{pipeline_id}/{job_id}"

        # 4. Predict
        data = [{"a": 4}]
        preds, thresholds_applied = await DeploymentService.predict(async_session, data)
        assert len(preds) == 1
        assert abs(preds[0] - 8.0) < 0.001
        assert thresholds_applied is None


@pytest.mark.asyncio
async def test_deployment_predict_decodes_label_encoded_target(async_session, tmp_path):
    pipeline_id = "test_pipeline_deploy_decode"
    job_id = "test_job_decode_123"

    from sklearn.preprocessing import LabelEncoder

    target_le = LabelEncoder()
    target_le.fit(["cat", "dog"])

    engineer = _IdentityEngineer(target_le)
    predictor = _FixedPredictor([0, 1])

    models_dir = tmp_path / "exports" / "models" / pipeline_id
    models_dir.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(str(models_dir))
    store.save(job_id, {"feature_engineer": engineer, "model": predictor, "job_id": job_id})

    await async_session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": pipeline_id,
            "node_id": "node_decode",
            "ds_id": "ds1",
            "user_id": None,
            "status": "completed",
            "run_mode": "fixed",
            "version": 1,
            "model_type": "dummy_classifier",
            "graph": "{}",
            "artifact_uri": job_id,
            "error_message": None,
            "progress": 0,
            "current_step": None,
            "started_at": None,
            "finished_at": None,
        },
    )
    await async_session.commit()

    with patch("os.getcwd", return_value=str(tmp_path)):
        await DeploymentService.deploy_model(async_session, job_id)
        preds, thresholds_applied = await DeploymentService.predict(async_session, [{"a": 1}])
        assert preds == ["cat", "dog"]
        assert thresholds_applied is None


class _PassthroughEngineer:
    """Feature engineer that returns its input unchanged and has no target encoder."""

    fitted_steps: list = []

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class _ProbaPredictor:
    """Classifier stub with fixed predict_proba, exposing classes_ for threshold logic."""

    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes)
        self._proba = np.asarray(proba, dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(self._proba, axis=1)]

    def predict_proba(self, X):
        return np.tile(self._proba, (len(X), 1))


async def _insert_training_job(session, job_id, pipeline_id, tuned=None, enabled=False):
    """Inserts a minimal completed training_jobs row, optionally with tuned thresholds."""
    import json

    await session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, tuned_thresholds, tuned_thresholds_enabled, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :tuned_thresholds, :tuned_thresholds_enabled, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": pipeline_id,
            "node_id": f"node_{job_id}",
            "ds_id": "ds1",
            "user_id": None,
            "status": "completed",
            "run_mode": "fixed",
            "version": 1,
            "model_type": "dummy_classifier",
            "graph": "{}",
            "artifact_uri": job_id,
            "error_message": None,
            "progress": 0,
            "current_step": None,
            "tuned_thresholds": json.dumps(tuned) if tuned is not None else None,
            "tuned_thresholds_enabled": 1 if enabled else 0,
            "started_at": None,
            "finished_at": None,
        },
    )
    await session.commit()


async def _deploy_proba_artifact(session, tmp_path, pipeline_id, job_id, tuned=None, enabled=False):
    """Saves a bundled proba-classifier artifact and deploys it, returning nothing."""
    engineer = _PassthroughEngineer()
    # proba [0.6, 0.4] -> default argmax picks class 0
    predictor = _ProbaPredictor([0, 1], [[0.6, 0.4]])

    models_dir = tmp_path / "exports" / "models" / pipeline_id
    models_dir.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(str(models_dir))
    store.save(job_id, {"feature_engineer": engineer, "model": predictor, "job_id": job_id})

    await _insert_training_job(session, job_id, pipeline_id, tuned=tuned, enabled=enabled)
    with patch("os.getcwd", return_value=str(tmp_path)):
        await DeploymentService.deploy_model(session, job_id)


@pytest.mark.asyncio
async def test_predict_override_thresholds_mismatch_raises(async_session, tmp_path):
    from backend.ml_pipeline.deployment.service import OverrideThresholdMismatch

    await _deploy_proba_artifact(async_session, tmp_path, "pipe_mismatch", "job_mismatch")
    with (
        patch("os.getcwd", return_value=str(tmp_path)),
        pytest.raises(OverrideThresholdMismatch),
    ):
        await DeploymentService.predict(
            async_session,
            [{"feature_1": 1.0}],
            override_thresholds={"nonexistent_class": 0.5},
        )


@pytest.mark.asyncio
async def test_predict_saved_enabled_thresholds_change_prediction(async_session, tmp_path):
    # No thresholds -> default argmax over [0.6, 0.4] -> class 0
    await _deploy_proba_artifact(async_session, tmp_path, "pipe_saved_off", "job_saved_off")
    with patch("os.getcwd", return_value=str(tmp_path)):
        preds_default, applied_default = await DeploymentService.predict(
            async_session, [{"feature_1": 1.0}]
        )
    assert preds_default == [0]
    assert applied_default is None

    # Saved + enabled thresholds that provably flip the prediction to class 1
    tuned = {
        "thresholds": {"0": 0.9, "1": 0.1},
        "classes": [0, 1],
        "metric": "f1",
        "split_used": "validation",
        "computed_at": "2026-01-01T00:00:00+00:00",
    }
    await _deploy_proba_artifact(
        async_session, tmp_path, "pipe_saved_on", "job_saved_on", tuned=tuned, enabled=True
    )
    with patch("os.getcwd", return_value=str(tmp_path)):
        preds_tuned, applied_tuned = await DeploymentService.predict(
            async_session, [{"feature_1": 1.0}]
        )
    assert preds_tuned == [1]
    assert applied_tuned == {"0": 0.9, "1": 0.1}


@pytest.mark.asyncio
async def test_predict_saved_thresholds_disabled_not_applied(async_session, tmp_path):
    tuned = {
        "thresholds": {"0": 0.9, "1": 0.1},
        "classes": [0, 1],
        "metric": "f1",
        "split_used": "validation",
        "computed_at": "2026-01-01T00:00:00+00:00",
    }
    # enabled=False -> thresholds must NOT be applied
    await _deploy_proba_artifact(
        async_session, tmp_path, "pipe_disabled", "job_disabled", tuned=tuned, enabled=False
    )
    with patch("os.getcwd", return_value=str(tmp_path)):
        preds, applied = await DeploymentService.predict(async_session, [{"feature_1": 1.0}])
    assert preds == [0]
    assert applied is None


@pytest.mark.asyncio
async def test_predict_override_takes_priority_over_saved(async_session, tmp_path):
    # Saved thresholds would flip to class 1; override keeps class 0.
    tuned = {
        "thresholds": {"0": 0.9, "1": 0.1},
        "classes": [0, 1],
        "metric": "f1",
        "split_used": "validation",
        "computed_at": "2026-01-01T00:00:00+00:00",
    }
    await _deploy_proba_artifact(
        async_session, tmp_path, "pipe_priority", "job_priority", tuned=tuned, enabled=True
    )
    with patch("os.getcwd", return_value=str(tmp_path)):
        preds, applied = await DeploymentService.predict(
            async_session,
            [{"feature_1": 1.0}],
            override_thresholds={"0": 0.1, "1": 0.9},
        )
    assert preds == [0]
    assert applied == {"0": 0.1, "1": 0.9}


@pytest_asyncio.fixture
async def http_async_session():
    """In-memory async SQLite session with all tables created (for HTTP client tests)."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def deployment_client(http_async_session):
    """TestClient wired to override the deployment router's session dependency."""
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.ml_pipeline.deployment import api as deployment_api

    async def _override():
        yield http_async_session

    app.dependency_overrides[deployment_api.get_async_session] = _override
    with TestClient(app, base_url="http://testserver") as client:
        yield client
    app.dependency_overrides.pop(deployment_api.get_async_session, None)


@pytest.mark.asyncio
async def test_predict_endpoint_override_mismatch_returns_422(
    http_async_session, deployment_client, tmp_path
):
    await _deploy_proba_artifact(http_async_session, tmp_path, "pipe_http_422", "job_http_422")
    with patch("os.getcwd", return_value=str(tmp_path)):
        response = deployment_client.post(
            "/api/deployment/predict",
            json={
                "data": [{"feature_1": 1.0}],
                "override_thresholds": {"nonexistent_class": 0.5},
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_saved_thresholds_returns_thresholds_applied(
    http_async_session, deployment_client, tmp_path
):
    tuned = {
        "thresholds": {"0": 0.9, "1": 0.1},
        "classes": [0, 1],
        "metric": "f1",
        "split_used": "validation",
        "computed_at": "2026-01-01T00:00:00+00:00",
    }
    await _deploy_proba_artifact(
        http_async_session, tmp_path, "pipe_http_ok", "job_http_ok", tuned=tuned, enabled=True
    )
    with patch("os.getcwd", return_value=str(tmp_path)):
        response = deployment_client.post(
            "/api/deployment/predict",
            json={"data": [{"feature_1": 1.0}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [1]
    assert body["thresholds_applied"] == {"0": 0.9, "1": 0.1}
