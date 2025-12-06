
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer

logging.basicConfig(level=logging.DEBUG)

def test_elliptic_envelope():
    print("\n--- Testing Elliptic Envelope ---")
    
    # Create data with outliers
    np.random.seed(42)
    X = 0.3 * np.random.randn(100, 2)
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X + 2, X - 2, X_outliers]
    df = pd.DataFrame(X, columns=['col1', 'col2'])
    
    print(f"Original shape: {df.shape}")
    
    pipeline_config = [
        {
            "name": "outlier_node",
            "transformer": "EllipticEnvelope",
            "params": {
                "columns": ["col1", "col2"],
                "contamination": 0.1
            }
        }
    ]
    
    pipeline = FeatureEngineer(pipeline_config)
    
    try:
        df_out, metrics = pipeline.fit_transform(df)
        print(f"Output shape: {df_out.shape}")
        print("Metrics:", metrics)
        
        if "rows_removed" in metrics:
            print(f"Rows removed: {metrics['rows_removed']}")
        else:
            print("ERROR: 'rows_removed' metric missing!")
            
        if "rows_total" in metrics:
             print(f"Rows total: {metrics['rows_total']}")
        else:
             print("ERROR: 'rows_total' metric missing!")

        if "contamination" in metrics:
             print(f"Contamination: {metrics['contamination']}")
        else:
             print("ERROR: 'contamination' metric missing!")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_elliptic_envelope()
