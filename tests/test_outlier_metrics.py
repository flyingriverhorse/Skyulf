
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ml_pipeline.preprocessing.pipeline import FeatureEngineer

logging.basicConfig(level=logging.DEBUG)

def test_outlier_metrics():
    print("\n--- Testing Outlier Metrics ---")
    
    # Create data with clear outliers
    np.random.seed(42)
    # Normal data: mean=10, std=1
    data = np.random.normal(10, 1, 100)
    # Outliers: 100, 200
    data = np.append(data, [100, 200, -50])
    
    df = pd.DataFrame(data, columns=['col1'])
    
    print(f"Original shape: {df.shape}")
    
    # Test IQR
    print("\n--- Testing IQR ---")
    pipeline_config_iqr = [
        {
            "name": "iqr_node",
            "transformer": "IQR",
            "params": {
                "columns": ["col1"],
                "multiplier": 1.5
            }
        }
    ]
    
    pipeline_iqr = FeatureEngineer(pipeline_config_iqr)
    df_out_iqr, metrics_iqr = pipeline_iqr.fit_transform(df.copy())
    print(f"IQR Output shape: {df_out_iqr.shape}")
    print("IQR Metrics:", metrics_iqr)
    
    if metrics_iqr.get("rows_removed") == 3:
        print("SUCCESS: IQR removed 3 rows as expected.")
    else:
        print(f"FAILURE: IQR removed {metrics_iqr.get('rows_removed')} rows, expected 3.")

    # Test ZScore
    print("\n--- Testing ZScore ---")
    pipeline_config_zscore = [
        {
            "name": "zscore_node",
            "transformer": "ZScore",
            "params": {
                "columns": ["col1"],
                "threshold": 3.0
            }
        }
    ]
    
    pipeline_zscore = FeatureEngineer(pipeline_config_zscore)
    df_out_zscore, metrics_zscore = pipeline_zscore.fit_transform(df.copy())
    print(f"ZScore Output shape: {df_out_zscore.shape}")
    print("ZScore Metrics:", metrics_zscore)
    
    if metrics_zscore.get("rows_removed") == 3:
        print("SUCCESS: ZScore removed 3 rows as expected.")
    else:
        print(f"FAILURE: ZScore removed {metrics_zscore.get('rows_removed')} rows, expected 3.")

if __name__ == "__main__":
    test_outlier_metrics()
