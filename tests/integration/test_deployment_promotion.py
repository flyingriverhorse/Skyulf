"""Exercise promotion failures and legacy artifact paths against real local storage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.config import get_settings
from backend.database.models import Base, Deployment, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment import api as deployment_api
from backend.ml_pipeline.deployment.service import DeploymentService
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest_asyncio.fixture
async def promotion_session(tmp_path, monkeypatch):
    """Keep jobs and artifacts isolated while retaining production path containment."""
    monkeypatch.setattr(get_settings(), "TRAINING_ARTIFACT_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(get_settings(), "TESTING", False)
    database = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'promotion.db'}")
    try:
        async with database.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(database, expire_on_commit=False) as session:
            yield session
    finally:
        await database.dispose()


def _model():
    """Fit a real predictor with an independently checkable doubled-input result."""
    return LinearRegression().fit(pd.DataFrame({"value": [1.0, 2.0, 3.0]}), [2.0, 4.0, 6.0])


async def _job(session, job_id, artifact="valid", legacy=False):
    """Persist candidate jobs with usable, absent, corrupt, or malformed artifacts."""
    store = LocalArtifactStore(str(Path(get_settings().TRAINING_ARTIFACT_DIR) / "pipeline"))
    uri = store.get_artifact_uri(job_id)
    if artifact == "valid":
        store.save(job_id, (_model(), {"score": 1.0}))
    elif artifact == "corrupt":
        Path(uri).write_bytes(b"invalid joblib artifact")
    elif artifact == "invalid":
        store.save(job_id, {"metadata": "not a predictor"})
    elif artifact == "unfitted":
        store.save(job_id, LinearRegression())
    elif artifact == "invalid_engineer":
        store.save(job_id, {"feature_engineer": None, "model": _model()})
    elif artifact == "unfitted_engineer":
        store.save(job_id, {"feature_engineer": StandardScaler(), "model": _model()})
    elif artifact == "invalid_model":
        store.save(job_id, {"feature_engineer": FeatureEngineer([]), "model": None})
    session.add(
        TrainingJob(
            id=job_id,
            pipeline_id="pipeline",
            node_id="model",
            dataset_source_id="source",
            status="completed",
            run_mode="fixed",
            model_type="linear_regression",
            graph={},
            artifact_uri=job_id if legacy else uri,
        )
    )
    await session.commit()


@pytest.mark.parametrize(
    "artifact",
    [
        "missing",
        "corrupt",
        "invalid",
        "unfitted",
        "invalid_engineer",
        "unfitted_engineer",
        "invalid_model",
    ],
)
async def test_invalid_candidate_preserves_working_deployment(promotion_session, artifact):
    """An unusable replacement must leave the serving model and deployment history intact."""
    session = promotion_session
    await _job(session, "working")
    working = await DeploymentService.deploy_model(session, "working")
    working_id = working.id
    await _job(session, "candidate", artifact)
    with pytest.raises(ValueError):
        await DeploymentService.deploy_model(session, "candidate")
    active = await DeploymentService.get_active_deployment(session)
    assert active is not None and active.id == working_id
    assert len(await DeploymentService.list_deployments(session)) == 1
    predictions, thresholds = await DeploymentService.predict(session, [{"value": 4.0}])
    np.testing.assert_allclose(predictions, [8.0])
    assert thresholds is None


async def test_promotion_write_failure_rolls_back_deactivation(promotion_session):
    """A failed replacement insert must roll back the active-model update in the same transaction."""
    session = promotion_session
    await _job(session, "working")
    working = await DeploymentService.deploy_model(session, "working")
    working_id = working.id
    await _job(session, "candidate")

    def fail_insert(mapper, connection, target):
        """Inject failure at the database insertion boundary after the real UPDATE."""
        raise RuntimeError("replacement insert failed")

    event.listen(Deployment, "before_insert", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="replacement insert failed"):
            await DeploymentService.deploy_model(session, "candidate")
    finally:
        event.remove(Deployment, "before_insert", fail_insert)
    active = await DeploymentService.get_active_deployment(session)
    assert active is not None and active.id == working_id
    assert len((await session.scalars(select(Deployment))).all()) == 1


@pytest.mark.parametrize("root", ["uploads/models", "configured/artifacts"])
async def test_legacy_reference_uses_configured_root(
    promotion_session, tmp_path, monkeypatch, root
):
    """Legacy jobs must load, expose their schema, and predict inside the permitted artifact root."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(get_settings(), "TRAINING_ARTIFACT_DIR", root)
    session = promotion_session
    await _job(session, "legacy", legacy=True)
    deployment = await DeploymentService.deploy_model(session, "legacy")
    details = await DeploymentService.get_deployment_details(session, deployment)
    assert details["input_schema"] == [{"name": "value", "type": "unknown"}]
    predictions, _ = await DeploymentService.predict(session, [{"value": 4.0}])
    np.testing.assert_allclose(predictions, [8.0])
    assert deployment.artifact_uri == "pipeline/legacy"


async def test_artifact_details_cannot_load_outside_root(promotion_session, tmp_path):
    """Schema inspection must enforce the same root containment as prediction loading."""
    store = LocalArtifactStore(str(tmp_path / "outside"))
    store.save("outside", _model())
    with pytest.raises(PermissionError, match="outside"):
        DeploymentService._load_artifact_for_details(store.get_artifact_uri("outside"))
    assert not (Path(get_settings().TRAINING_ARTIFACT_DIR) / "outside").exists()


async def test_deploy_endpoint_rejects_missing_replacement(promotion_session):
    """A missing candidate must return HTTP 400 while the previous deployment still predicts."""
    session = promotion_session
    await _job(session, "working")
    await DeploymentService.deploy_model(session, "working")
    await _job(session, "candidate", "missing")
    app = FastAPI()
    app.include_router(deployment_api.router, prefix="/api")

    async def database_session():
        """Share the temporary database with the actual deployment routes."""
        yield session

    app.dependency_overrides[deployment_api.get_async_session] = database_session
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/api/deployment/deploy/candidate")
        assert response.status_code == 400
        prediction = await client.post("/api/deployment/predict", json={"data": [{"value": 4.0}]})
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == "working"
    np.testing.assert_allclose(prediction.json()["predictions"], [8.0])
