import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from core.database.models import Base, TrainingJob, Deployment
from core.ml_pipeline.deployment.service import DeploymentService
import os
import pandas as pd
import joblib
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from unittest.mock import patch

# Use an in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
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
    store.save(node_id, model)
    
    # 2. Create Job in DB
    # We need to insert manually because TrainingJob might have required fields not in init
    await async_session.execute(
        text("""
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """),
        {
            "id": job_id,
            "pipeline_id": pipeline_id,
            "node_id": node_id,
            "ds_id": "ds1",
            "user_id": None,
            "status": "completed",
            "version": 1,
            "model_type": "linear_regression",
            "graph": "{}",
            "artifact_uri": node_id, # The engine saves it as node_id
            "error_message": None,
            "progress": 0,
            "current_step": None,
            "started_at": None,
            "finished_at": None
        }
    )
    await async_session.commit()
    
    # Patch os.getcwd to return tmp_path
    with patch("os.getcwd", return_value=str(tmp_path)):
        # 3. Deploy
        deployment = await DeploymentService.deploy_model(async_session, job_id)
        assert deployment.is_active == True
        # The service constructs the URI as pipeline_id/node_id if it was just node_id
        assert deployment.artifact_uri == f"{pipeline_id}/{node_id}"
        
        # 4. Predict
        data = [{"a": 4}]
        preds = await DeploymentService.predict(async_session, data)
        assert len(preds) == 1
        assert abs(preds[0] - 8.0) < 0.001
