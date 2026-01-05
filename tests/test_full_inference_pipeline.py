import logging
import os
import shutil
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(r"c:\Users\Murat\Desktop\skyulf-mlflow")

from skyulf.preprocessing.pipeline import FeatureEngineer

from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.data.catalog import FileSystemCatalog

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_inference_pipeline():
    # 1. Setup
    base_path = r"c:\Users\Murat\Desktop\skyulf-mlflow\temp_test_artifacts_full"
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    store = LocalArtifactStore(base_path)
    catalog = FileSystemCatalog()
    engine = PipelineEngine(store, catalog=catalog)

    # 2. Create Complex Dummy Data
    # - 'age': Numeric, has outliers (150), has missing (NaN)
    # - 'income': Numeric, needs scaling
    # - 'city': Categorical, needs encoding
    # - 'gender': Categorical, needs encoding
    # - 'target': Binary

    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 150, 40, np.nan, 22, 28, 33, 45],
            "income": [
                50000,
                60000,
                70000,
                80000,
                90000,
                55000,
                45000,
                65000,
                75000,
                85000,
            ],
            "city": ["NY", "LA", "NY", "SF", "LA", "SF", "NY", "LA", "SF", "NY"],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    data_path = os.path.join(base_path, "data.csv")
    df.to_csv(data_path, index=False)

    # 3. Define Pipeline
    nodes = []

    # Node 1: Loader
    nodes.append(
        NodeConfig(
            node_id="loader", step_type="data_loader", params={"path": data_path}
        )
    )

    # Node 2: Value Replacement (Cleaning)
    # Replace 'SF' with 'San Francisco'
    nodes.append(
        NodeConfig(
            node_id="clean_city",
            step_type="ValueReplacement",
            params={"to_replace": {"SF": "San Francisco"}, "columns": ["city"]},
            inputs=["loader"],
        )
    )

    # Node 3: Simple Imputer (Imputation)
    # Fill missing age with median
    nodes.append(
        NodeConfig(
            node_id="impute_age",
            step_type="SimpleImputer",
            params={"strategy": "median", "columns": ["age"]},
            inputs=["clean_city"],
        )
    )

    # Node 4: Manual Bounds (Outliers)
    # Clip age to 0-100
    nodes.append(
        NodeConfig(
            node_id="clip_age",
            step_type="ManualBounds",
            params={"bounds": {"age": {"lower": 0, "upper": 100}}},
            inputs=["impute_age"],
        )
    )

    # Node 5: One Hot Encoder (Encoding)
    # Encode city and gender
    nodes.append(
        NodeConfig(
            node_id="encode_cats",
            step_type="OneHotEncoder",
            params={"columns": ["city", "gender"], "handle_unknown": "ignore"},
            inputs=["clip_age"],
        )
    )

    # Node 6: Standard Scaler (Scaling)
    # Scale income
    nodes.append(
        NodeConfig(
            node_id="scale_income",
            step_type="StandardScaler",
            params={"columns": ["income"]},
            inputs=["encode_cats"],
        )
    )

    # Node 7: Variance Threshold (Feature Selection)
    # Remove low variance features (dummy check)
    nodes.append(
        NodeConfig(
            node_id="select_features",
            step_type="VarianceThreshold",
            params={"threshold": 0.0},
            inputs=["scale_income"],
        )
    )

    # Node 8: Splitter
    nodes.append(
        NodeConfig(
            node_id="splitter",
            step_type="TrainTestSplitter",
            params={"test_size": 0.2, "random_state": 42},
            inputs=["select_features"],
        )
    )

    # Node 9: Model
    nodes.append(
        NodeConfig(
            node_id="model",
            step_type="basic_training",
            params={"algorithm": "logistic_regression", "target_column": "target"},
            inputs=["splitter"],
        )
    )

    config = PipelineConfig(pipeline_id="full_test_pipeline", nodes=nodes)

    # 4. Run Engine (Training)
    print("Running Training Pipeline...")
    try:
        result = engine.run(config)
    except Exception as e:
        print(f"Engine run failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return

    if result.status == "failed":
        print("Pipeline Failed!")
        for nid, res in result.node_results.items():
            if res.status == "failed":
                print(f"Node {nid} failed: {res.error}")
        # Continue to check artifact anyway if possible
    else:
        print("Pipeline Succeeded.")

    # 5. Load Artifact
    try:
        model_artifact = store.load("model")
    except Exception as e:
        print(f"Could not load model artifact: {e}")
        return

    # 6. Verify Artifact Structure
    print("Verifying Artifact Structure...")
    if not isinstance(model_artifact, dict):
        print(f"Model artifact is not a dict: {type(model_artifact)}")
        return

    plan = model_artifact.get("transformer_plan", [])
    print(f"Plan Length: {len(plan)}")

    # 7. Simulate Inference
    print("Simulating Inference...")

    # New data
    # - Row 1: Valid
    #   - 'age': 50
    #   - 'income': 50000
    #   - 'city': 'SF'
    #   - 'gender': 'M'
    # - Row 2: Invalid (Outlier)
    #   - 'age': 120 (should be dropped by ManualBounds)
    #   - 'income': 50000
    #   - 'city': 'NY'
    #   - 'gender': 'F'

    new_data = pd.DataFrame(
        {
            "age": [50, 120],
            "income": [50000, 50000],
            "city": ["SF", "NY"],
            "gender": ["M", "F"],
        }
    )

    transformers = model_artifact.get("transformers", [])

    t_objs = {}
    for t in transformers:
        t_node = t.get("node_id")
        t_name = t.get("transformer_name")
        t_col = t.get("column_name")
        t_objs[(t_node, t_name, t_col)] = t.get("transformer")

    current_df = new_data.copy()

    for step in plan:
        node_id = step.get("node_id")
        t_name = step.get("transformer_name")
        t_col = step.get("column_name")
        t_type = step.get("transformer_type")

        # Skip Splitter in inference usually, but let's see if it runs (it should just pass through or warn)
        if t_type == "TrainTestSplitter":
            continue

        print(f"Applying {t_name} ({t_type}) on {t_col}...")

        obj = t_objs.get((node_id, t_name, t_col))

        ApplierCls = None
        try:
            # Use FeatureEngineer factory to get the correct applier class
            temp_engineer = FeatureEngineer([])
            _, applier_instance = temp_engineer._get_transformer_components(t_type)
            ApplierCls = type(applier_instance)
        except ValueError:
            print(f"Unknown transformer type: {t_type}")

        if ApplierCls:
            applier = ApplierCls()
            params = {}
            if isinstance(obj, dict):
                params = obj.copy()

            # Inject object into common keys for Appliers that need the raw object (like OneHotEncoder)
            # Only inject if not already present (to avoid overwriting the real object with the wrapper dict)
            if obj is not None:
                if "encoder_object" not in params:
                    params["encoder_object"] = obj
                if "scaler_object" not in params:
                    params["scaler_object"] = obj
                if "imputer_object" not in params:
                    params["imputer_object"] = obj
                if "transformer_object" not in params:
                    params["transformer_object"] = obj

            try:
                res = applier.apply(current_df, params)
                if isinstance(res, tuple):
                    current_df = res[0]
                else:
                    current_df = res
            except Exception as e:
                print(f"Error applying {t_type}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"Warning: No Applier for {t_type}")

    print("Transformed Data:")
    print(current_df)

    # Verifications
    # 1. Row count?
    if len(current_df) == 1:
        print("SUCCESS: Outlier row dropped (1 row remaining)")
    else:
        print(f"FAILURE: Expected 1 row, got {len(current_df)}")

    # 2. Age check
    if "age" in current_df.columns:
        val = current_df["age"].iloc[0]
        if val == 50:
            print("SUCCESS: Valid age preserved")
        else:
            print(f"FAILURE: Unexpected age value: {val}")

    # 3. City encoded?
    # 'SF' -> 'San Francisco' -> OneHotEncoded
    # We expect columns like 'city_San Francisco' or similar depending on OHE implementation
    cols = current_df.columns.tolist()
    print(f"Columns: {cols}")

    # Check for OHE columns
    ohe_cols = [c for c in cols if "city" in c or "gender" in c]
    if len(ohe_cols) > 0:
        print(f"SUCCESS: Categorical columns encoded: {ohe_cols}")
    else:
        print("FAILURE: No encoded columns found")

    # 4. Income scaled?
    if "income" in current_df.columns:
        val = current_df["income"].iloc[0]
        # 50000 was the min in training data (approx), so scaled value should be around -1.5 or so (StandardScaler)
        # Mean of income ~ 68000, Std ~ 15000. (50000 - 68000)/15000 ~ -1.2
        print(f"Scaled Income: {val}")
        if -2.0 < val < 2.0:
            print("SUCCESS: Income scaled reasonably")
        else:
            print("WARNING: Income scaling seems off (or maybe not scaled?)")


if __name__ == "__main__":
    test_full_inference_pipeline()
