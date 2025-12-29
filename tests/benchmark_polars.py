import time
import numpy as np
import pandas as pd
import polars as pl
from skyulf import SkyulfPipeline
from skyulf.engines import get_engine

def create_large_dataset(n_rows=100_000):
    print(f"Generating {n_rows} rows with 20 columns...")
    np.random.seed(42)
    
    data = {}
    
    # 10 Numeric Columns
    for i in range(10):
        data[f"num_{i}"] = np.random.normal(0, 1, n_rows)
        
    # 5 Categorical Columns (High Cardinality)
    for i in range(5):
        data[f"cat_high_{i}"] = np.random.randint(0, 1000, n_rows).astype(str)
        
    # 5 Categorical Columns (Low Cardinality)
    for i in range(5):
        data[f"cat_low_{i}"] = np.random.choice(["A", "B", "C", "D", "E"], n_rows)
        
    # Target
    data["target"] = np.random.randint(0, 2, n_rows)
    
    # Add missing values to num_0 and cat_low_0
    data["num_0"][0:int(n_rows*0.1)] = np.nan
    data["cat_low_0"][0:int(n_rows*0.1)] = None
    
    return pd.DataFrame(data)

def run_benchmark():
    # 1. Setup Data
    n_rows = 2_000_000 # 2 Million rows
    print(f"Creating dataset with {n_rows:,} rows...")
    df_pandas = create_large_dataset(n_rows)
    print("Converting to Polars...")
    df_polars = pl.from_pandas(df_pandas)
    
    # 2. Define Pipeline
    config = {
        "preprocessing": [
            {
                "name": "impute_numeric",
                "transformer": "SimpleImputer",
                "params": {"columns": ["num_0"], "strategy": "mean"}
            },
            {
                "name": "scale_numeric",
                "transformer": "StandardScaler",
                "params": {"columns": [f"num_{i}" for i in range(10)]}
            },
            {
                "name": "encode_cat",
                "transformer": "OneHotEncoder",
                "params": {"columns": [f"cat_low_{i}" for i in range(5)]}
            },
             {
                "name": "hash_cat",
                "transformer": "HashEncoder",
                "params": {"columns": [f"cat_high_{i}" for i in range(5)], "n_features": 100}
            }
        ],
        "modeling": {} 
    }
    
    pipeline = SkyulfPipeline(config)
    
    # --- Benchmark Pandas ---
    print("\n--- Benchmarking Pandas Engine ---")
    
    # Fit first (on small sample to be fast)
    print("Fitting pipeline (Pandas)...")
    pipeline.fit(df_pandas.head(10000), target_column="target")
    
    # Measure Transform
    print("Running Transform (Pandas)...")
    t0 = time.time()
    _ = pipeline.feature_engineer.transform(df_pandas)
    pandas_time = time.time() - t0
    print(f"Pandas Transform Time: {pandas_time:.4f} seconds")
    
    # --- Benchmark Polars ---
    print("\n--- Benchmarking Polars Engine ---")
    
    # Measure Transform
    print("Running Transform (Polars)...")
    t0 = time.time()
    _ = pipeline.feature_engineer.transform(df_polars)
    polars_time = time.time() - t0
    print(f"Polars Transform Time: {polars_time:.4f} seconds")
    
    # --- Results ---
    speedup = pandas_time / polars_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("✅ Polars is faster!")
    else:
        print("⚠️ Pandas is faster (or overhead dominates).")

if __name__ == "__main__":
    run_benchmark()
