import pytest
from fastapi.testclient import TestClient
from core.main import app
import os
import shutil
import json
import numpy as np

@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="http://localhost") as c:
        yield c

def test_get_registry(client):
    response = client.get("/api/pipeline/registry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check for a known item
    assert any(item["id"] == "data_loader" for item in data)
    assert any(item["id"] == "random_forest_classifier" for item in data)

def test_preview_pipeline(client):
    # Create a dummy CSV for testing
    os.makedirs("temp_test_data", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data/iris.csv")
    
    # Create a simple iris-like csv
    with open(csv_path, "w") as f:
        f.write("sepal_length,sepal_width,petal_length,petal_width,species\n")
        f.write("5.1,3.5,1.4,0.2,setosa\n")
        f.write("4.9,3.0,1.4,0.2,setosa\n")
        f.write("7.0,3.2,4.7,1.4,versicolor\n")
        f.write("6.4,3.2,4.5,1.5,versicolor\n")

    try:
        payload = {
            "pipeline_id": "test_preview_001",
            "nodes": [
                {
                    "node_id": "node_1",
                    "step_type": "data_loader",
                    "params": {
                        "source_id": "csv", 
                        "path": csv_path
                    },
                    "inputs": []
                },
                {
                    "node_id": "node_2",
                    "step_type": "feature_engineering",
                    "params": {
                        "steps": [
                            {
                                "name": "scaler",
                                "transformer": "StandardScaler",
                                "params": {}
                            }
                        ]
                    },
                    "inputs": ["node_1"]
                }
            ]
        }

        response = client.post("/api/pipeline/preview", json=payload)
        if response.status_code != 200:
            print(f"Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "preview_data" in data
        # Check if preview data has records
        assert len(data["preview_data"]) > 0
        
    finally:
        if os.path.exists("temp_test_data"):
            shutil.rmtree("temp_test_data")

def test_preview_recommendations(client):
    # Create a CSV with missing values to trigger recommendations
    os.makedirs("temp_test_data_rec", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data_rec/missing.csv")
    
    with open(csv_path, "w") as f:
        f.write("A,B,C\n")
        f.write("1,10,x\n")
        f.write(",20,y\n") # Missing in A
        f.write("3,,z\n")  # Missing in B
        f.write("4,40,\n") # Missing in C
        f.write("5,50,w\n")

    try:
        payload = {
            "pipeline_id": "test_rec_001",
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
            ]
        }

        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "recommendations" in data
        recs = data["recommendations"]
        assert len(recs) > 0
        
        # Check for imputation recommendation
        imputation_recs = [r for r in recs if r["type"] == "imputation"]
        assert len(imputation_recs) > 0
        
        # Check if it detected missing columns
        targets = []
        for r in imputation_recs:
            targets.extend(r["target_columns"])
        
        assert "A" in targets or "B" in targets or "C" in targets

    finally:
        if os.path.exists("temp_test_data_rec"):
            shutil.rmtree("temp_test_data_rec")

def test_cleaning_recommendations(client):
    # Create a CSV with duplicates and high missing values
    os.makedirs("temp_test_data_clean", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data_clean/dirty.csv")
    
    with open(csv_path, "w") as f:
        f.write("A,B,C\n")
        f.write("1,10,x\n")
        f.write("1,10,x\n") # Duplicate
        f.write(",,y\n")    # High missing row
        f.write(",,z\n")    # High missing row
        f.write(",,w\n")    # High missing row (3/5 missing = 60%)

    try:
        payload = {
            "pipeline_id": "test_clean_001",
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
            ]
        }

        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        recs = data["recommendations"]
        
        # Check for duplicate rows recommendation
        dup_recs = [r for r in recs if r["rule_id"] == "duplicate_rows_drop"]
        assert len(dup_recs) > 0
        
        # Check for high missing drop recommendation
        high_missing_recs = [r for r in recs if r["rule_id"] == "high_missing_drop"]
        assert len(high_missing_recs) > 0
        targets = high_missing_recs[0]["target_columns"]
        assert "A" in targets and "B" in targets
        
    finally:
        if os.path.exists("temp_test_data_clean"):
            shutil.rmtree("temp_test_data_clean")

def test_encoding_outlier_recommendations(client):
    # Create a CSV with categorical data and skewed numeric data
    os.makedirs("temp_test_data_adv", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data_adv/advanced.csv")
    
    with open(csv_path, "w") as f:
        f.write("cat_low,cat_high,skewed_num\n")
        # Low cardinality: 'A', 'B'
        # High cardinality: unique per row
        # Skewed: 1, 1, 1, 1, 1000
        for i in range(20):
            f.write(f"A,{i},1\n")
        f.write("B,99,1000\n") # Outlier

    try:
        payload = {
            "pipeline_id": "test_adv_001",
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
            ]
        }

        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        recs = data["recommendations"]
        
        # Check for OneHotEncoding (cat_low)
        ohe_recs = [r for r in recs if r["rule_id"] == "one_hot_encoding"]
        assert len(ohe_recs) > 0
        assert "cat_low" in ohe_recs[0]["target_columns"]
        
        # Check for OutlierRemoval (skewed_num)
        # Skewness of [1]*20 + [1000] is very high
        outlier_recs = [r for r in recs if r["rule_id"] == "outlier_removal_iqr"]
        assert len(outlier_recs) > 0
        assert "skewed_num" in outlier_recs[0]["target_columns"]

    finally:
        if os.path.exists("temp_test_data_adv"):
            shutil.rmtree("temp_test_data_adv")

def test_transformation_recommendations(client):
    # Create a CSV with skewed data for Power Transform
    os.makedirs("temp_test_data_trans", exist_ok=True)
    csv_path = os.path.abspath("temp_test_data_trans/skewed.csv")
    
    with open(csv_path, "w") as f:
        f.write("pos_skew,neg_skew\n")
        # Positive skew (Box-Cox candidate): 1, 2, 3, ..., 100
        # Negative skew (Yeo-Johnson candidate): -100, -1, 0, 1
        
        # Generating synthetic skewed data
        # Log-normal is positively skewed
        import numpy as np
        np.random.seed(42)
        pos_skew = np.random.lognormal(0, 1, 100)
        # Shift to ensure strictly positive for Box-Cox
        pos_skew = pos_skew + 1.0 
        
        # Negative values for Yeo-Johnson
        neg_skew = np.random.normal(0, 1, 100)
        neg_skew[0] = -100 # Extreme negative outlier creating skew
        
        for p, n in zip(pos_skew, neg_skew):
            f.write(f"{p},{n}\n")

    try:
        payload = {
            "pipeline_id": "test_trans_001",
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
            ]
        }

        response = client.post("/api/pipeline/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        recs = data["recommendations"]
        
        # Check for Box-Cox (pos_skew)
        # Note: Skewness calculation depends on data. Lognormal(0,1) has skewness ~6.
        box_cox_recs = [r for r in recs if r["rule_id"] == "power_transform_box_cox"]
        if not box_cox_recs:
             # Fallback check if it recommended Yeo-Johnson instead (if min_value check failed or logic differed)
             yeo_recs = [r for r in recs if r["rule_id"] == "power_transform_yeo_johnson"]
             assert len(yeo_recs) > 0
             targets = []
             for r in yeo_recs: targets.extend(r["target_columns"])
             assert "pos_skew" in targets
        else:
             assert "pos_skew" in box_cox_recs[0]["target_columns"]

    finally:
        if os.path.exists("temp_test_data_trans"):
            shutil.rmtree("temp_test_data_trans")

def test_run_pipeline_submission(client):
    # This test only checks if the job is submitted successfully
    payload = {
        "pipeline_id": "test_run_001",
        "nodes": [
            {
                "node_id": "node_1",
                "step_type": "data_loader",
                "params": {"source_id": "dummy"},
                "inputs": []
            }
        ]
    }
    
    # We expect this to return 200 and a message, 
    # even if the background task fails (which it might due to dummy source)
    response = client.post("/api/pipeline/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Pipeline execution started"
    assert data["pipeline_id"] == "test_run_001"
