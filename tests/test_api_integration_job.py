import pytest
from fastapi.testclient import TestClient
from core.main import app
import os
import shutil
import json

@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c

def test_run_pipeline_job(client):
    # Create a dummy CSV for testing
    os.makedirs("temp_test_data", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data/iris_job.csv")
    
    with open(csv_path, "w") as f:
        f.write("sepal_length,sepal_width,petal_length,petal_width,species\n")
        f.write("5.1,3.5,1.4,0.2,setosa\n")
        f.write("4.9,3.0,1.4,0.2,setosa\n")

    try:
        payload = {
            "pipeline_id": "test_job_001",
            "nodes": [
                {
                    "node_id": "node_1",
                    "step_type": "data_loader",
                    "params": {
                        "source_id": "csv", 
                        "path": csv_path
                    },
                    "inputs": []
                }
            ],
            "metadata": {}
        }
        
        # 1. Start Job
        response = client.post("/api/pipeline/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        job_id = data["job_id"]
        
        # 2. Poll Status
        # Since BackgroundTasks run in the same process/thread in TestClient usually, 
        # it might finish immediately or we might need to wait.
        # But TestClient executes background tasks after the response is sent.
        
        import time
        max_retries = 5
        for _ in range(max_retries):
            status_response = client.get(f"/api/pipeline/jobs/{job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            if status_data["status"] in ["completed", "failed"]:
                break
            time.sleep(0.5)
            
        assert status_data["status"] == "completed"
        assert status_data["pipeline_id"] == "test_job_001"
        
    finally:
        if os.path.exists("temp_test_data"):
            shutil.rmtree("temp_test_data")
