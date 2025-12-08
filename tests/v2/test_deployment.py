import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from core.database.models import Base, TrainingJob, Deployment
from core.ml_pipeline.deployment.service import DeploymentService
import os
import pandas as pd
import joblib
from core.ml_pipeline.artifacts.local import LocalArtifactStore

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
async def test_deployment_flow(async_session):
    # 1. Setup: Create a dummy model artifact
    pipeline_id = "test_pipeline_deploy"
    node_id = "test_node_deploy"
    
    # Create a simple model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([2, 4, 6])
    model.fit(X, y)
    
    # Save it
    base_path = os.path.join(os.getcwd(), "exports", "models", pipeline_id)
    os.makedirs(base_path, exist_ok=True)
    
    joblib.dump(model, os.path.join(base_path, f"{node_id}.joblib"))
    
    # 2. Create Job in DB
    job = TrainingJob(
        id="test_job_123",
        pipeline_id=pipeline_id,
        node_id=node_id,
        dataset_source_id="ds1",
        status="completed",
        model_type="linear_regression",
        artifact_uri=node_id, # The engine saves it as node_id
        graph={}
    )
    async_session.add(job)
    await async_session.commit()
    
    # 3. Deploy
    deployment = await DeploymentService.deploy_model(async_session, "test_job_123")
    assert deployment.is_active == True
    assert deployment.artifact_uri == f"{pipeline_id}/{node_id}"
    
    # 4. Predict
    data = [{"a": 4}]
    preds = await DeploymentService.predict(async_session, data)
    assert len(preds) == 1
    assert abs(preds[0] - 8.0) < 0.001
    
    # Cleanup
    import shutil
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
