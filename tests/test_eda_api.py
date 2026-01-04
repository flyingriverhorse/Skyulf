import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from backend.database.models import DataSource, EDAReport, Base
from backend.main import app
from backend.config import get_settings

# Setup async engine for tests
settings = get_settings()
# Use the same DB as the app (sqlite)
DATABASE_URL = f"sqlite+aiosqlite:///{settings.DB_PATH}"
engine = create_async_engine(DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

@pytest.fixture
async def db_session():
    async with TestingSessionLocal() as session:
        yield session

@pytest.fixture
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c

@pytest.mark.asyncio
async def test_trigger_analysis_creates_report(client, db_session):
    # 1. Create a dummy data source
    # We need to use a unique source_id to avoid conflicts if DB is persistent
    import uuid
    import os
    unique_id = str(uuid.uuid4())
    
    # Create a dummy CSV file in temp
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(temp_dir, f"test_eda_{unique_id}.csv"))
    with open(file_path, "w") as f:
        f.write("col1,col2\n1,2\n3,4")

    ds = DataSource(
        name="Test Dataset",
        type="csv",
        source_id=unique_id,
        test_status="untested",
        config={"format": "csv", "size_bytes": 100, "rows": 10, "columns": 2, "file_path": file_path}
    )
    db_session.add(ds)
    await db_session.commit()
    await db_session.refresh(ds)
    
    # 2. Trigger analysis
    # TestClient is sync, so we don't await it
    response = client.post(f"/api/eda/{ds.id}/analyze")
    
    # 3. Verify response
    print(f"Response: {response.status_code} - {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PENDING"
    job_id = data["job_id"]
    
    # 4. Verify DB entry
    report = await db_session.get(EDAReport, job_id)
    assert report is not None
    assert report.data_source_id == ds.id
    if report.status == "FAILED":
        print(f"Report Failed: {report.error_message}")
    
    # It might be PENDING, RUNNING or COMPLETED depending on speed
    assert report.status in ["PENDING", "RUNNING", "COMPLETED"]
    assert report.config == {}
    assert report.is_active is True
    assert report.test_status == "untested"
