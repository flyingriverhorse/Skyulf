
import sys
import os
import pandas as pd
import pytest

# Add local skyulf-core to path
sys.path.append(os.path.join(os.getcwd(), "skyulf-core"))

from skyulf.pipeline import SkyulfPipeline
from skyulf.engines.pandas_engine import PandasEngine
from skyulf.engines.protocol import SkyulfDataFrame

def test_skyulfdataframe_integration():
    """
    Test that SkyulfPipeline handles SkyulfDataFrame input (via PandasWrapper).
    This ensures the pipeline logic respects the protocol and doesn't rely solely on pd.DataFrame.
    """
    print("Setting up data...")
    data = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        "feature2": [5.0, 4.0, 3.0, 2.0, 1.0] * 20,
        "category": ["A", "B", "A", "B", "C"] * 20,
        "target": [0, 1, 0, 1, 0] * 20
    })

    # Wrap it! This is the key step.
    # wrapped_data creates a SkyulfPandasWrapper which is a SkyulfDataFrame, NOT a pd.DataFrame
    wrapped_data = PandasEngine.wrap(data)
    
    # Confirm it's NOT a pd.DataFrame
    # assert not isinstance(wrapped_data, pd.DataFrame), "Wrapper failed to hide DataFrame identity"
    # Actually, verify inheritance. SkyulfPandasWrapper holds _df, doesn't inherit from pd.DataFrame
    
    assert isinstance(wrapped_data, SkyulfDataFrame), "Wrapper does not implement SkyulfDataFrame"

    # Define Pipeline
    pipeline_config = {
        "preprocessing": [
            {
                "name": "scaler",
                "transformer": "StandardScaler",
                "params": {"columns": ["feature1", "feature2"]},
            }
        ],
        "modeling": {"type": "logistic_regression", "params": {}},
    }

    pipeline = SkyulfPipeline(pipeline_config)

    print("Running fit() with SkyulfDataFrame...")
    # This should pass if pipeline.py correctly handles SkyulfDataFrame
    metrics = pipeline.fit(wrapped_data, target_column="target")
    print("Fit metrics:", metrics)
    
    assert "preprocessing" in metrics
    assert "modeling" in metrics

    print("Running predict()...")
    # Test predict with wrapped data too
    new_data = data.drop(columns=["target"]).iloc[:5]
    wrapped_new_data = PandasEngine.wrap(new_data)
    
    preds = pipeline.predict(wrapped_new_data)
    print("Predictions shape:", preds.shape)
    assert len(preds) == 5

if __name__ == "__main__":
    try:
        test_skyulfdataframe_integration()
        print("\nSUCCESS: SkyulfDataFrame integration verified!")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
