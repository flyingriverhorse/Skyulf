
import time
import pandas as pd
import numpy as np
import pytest
from skyulf.preprocessing.feature_generation import FeatureGenerationApplier

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def test_feature_generation_speed():
    if not HAS_POLARS:
        pytest.skip("Polars not installed")

    # Create a large dataset
    N = 50_000 # Reduced for similarity test speed
    data = {
        "date_col": ["2023-01-01"] * N,
        "text_a": ["hello world"] * N,
        "text_b": ["hello python"] * N,
        "num_a": np.random.rand(N),
        "num_b": np.random.rand(N),
    }
    
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)

    applier = FeatureGenerationApplier()

    # 1. Datetime Extract (Native Polars vs Pandas)
    print("\n--- Benchmarking Datetime Extract ---")
    config_dt = {
        "operations": [{
            "operation_type": "datetime_extract",
            "input_columns": ["date_col"],
            "datetime_features": ["year", "month", "day"]
        }]
    }
    
    start = time.time()
    applier.apply(df_pd, config_dt)
    pd_dt_time = time.time() - start
    print(f"Pandas: {pd_dt_time:.4f}s")

    start = time.time()
    applier.apply(df_pl, config_dt)
    pl_dt_time = time.time() - start
    print(f"Polars: {pl_dt_time:.4f}s")
    print(f"Speedup: {pd_dt_time / pl_dt_time:.2f}x")

    # 2. Arithmetic (Native Polars vs Pandas)
    print("\n--- Benchmarking Arithmetic ---")
    config_math = {
        "operations": [{
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["num_a"],
            "secondary_columns": ["num_b"],
            "output_column": "sum_ab"
        }]
    }

    start = time.time()
    applier.apply(df_pd, config_math)
    pd_math_time = time.time() - start
    print(f"Pandas: {pd_math_time:.4f}s")

    start = time.time()
    applier.apply(df_pl, config_math)
    pl_math_time = time.time() - start
    print(f"Polars: {pl_math_time:.4f}s")
    if pl_math_time > 0:
        print(f"Speedup: {pd_math_time / pl_math_time:.2f}x")
    else:
        print("Speedup: Infinite (Polars took 0s)")

    # 3. Similarity (Python Loop vs map_elements)
    print("\n--- Benchmarking Similarity ---")
    config_sim = {
        "operations": [{
            "operation_type": "similarity",
            "method": "ratio",
            "input_columns": ["text_a"],
            "secondary_columns": ["text_b"],
            "output_column": "sim_score"
        }]
    }

    start = time.time()
    applier.apply(df_pd, config_sim)
    pd_sim_time = time.time() - start
    print(f"Pandas: {pd_sim_time:.4f}s")

    start = time.time()
    applier.apply(df_pl, config_sim)
    pl_sim_time = time.time() - start
    print(f"Polars: {pl_sim_time:.4f}s")
    print(f"Speedup: {pd_sim_time / pl_sim_time:.2f}x")

if __name__ == "__main__":
    test_feature_generation_speed()
