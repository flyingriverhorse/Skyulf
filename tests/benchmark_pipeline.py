
import time
import pandas as pd
import numpy as np
import sys
import os

# Add local skyulf-core to path
sys.path.append(os.path.join(os.getcwd(), "skyulf-core"))

from skyulf.pipeline import SkyulfPipeline
from skyulf.engines.pandas_engine import PandasEngine

# Try importing Polars
try:
    import polars as pl
    from skyulf.engines.polars_engine import PolarsEngine
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def generate_data(n_rows=50000):
    np.random.seed(42)
    return pd.DataFrame({
        "numeric_1": np.random.randn(n_rows),
        "numeric_2": np.random.rand(n_rows) * 100,
        "categorical": np.random.choice(["A", "B", "C", "D"], n_rows),
        "target": np.random.randint(0, 2, n_rows)
    })

def run_benchmark():
    print("Generating synthetic data (50,000 rows)...")
    df = generate_data(50000)
    
    pipeline_config = {
        "preprocessing": [
            {
                "name": "imputer", 
                "transformer": "SimpleImputer",
                "params": {"strategy": "mean"}
            },
            {
                "name": "scaler",
                "transformer": "StandardScaler",
                "params": {"columns": ["numeric_1", "numeric_2"]}
            },
            {
                "name": "encoder",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["categorical"]}
            }
        ],
        "modeling": {
            "type": "logistic_regression",
            "params": {}
        }
    }
    
    results = []

    # 1. Standard Pandas (Baseline)
    print("\n--- Benchmarking Standard Pandas (Direct) ---")
    pipe_pandas = SkyulfPipeline(pipeline_config)
    
    start = time.time()
    pipe_pandas.fit(df, target_column="target")
    fit_time = time.time() - start
    
    # Drop target for prediction to avoid mismatch (model expects X only)
    df_predict = df.drop(columns=["target"])

    start = time.time()
    pipe_pandas.predict(df_predict)
    predict_time = time.time() - start
    
    print(f"Fit: {fit_time:.4f}s")
    print(f"Predict: {predict_time:.4f}s")
    results.append({"Mode": "Pandas (Direct)", "Fit": fit_time, "Predict": predict_time})

    # 2. Wrapped Pandas (SkyulfDataFrame Protocol)
    print("\n--- Benchmarking SkyulfDataFrame Wrapper (Pandas) ---")
    pipe_wrapped = SkyulfPipeline(pipeline_config)
    # We copy to ensure fair comparison (no cache hits if any)
    df_wrapped = PandasEngine.wrap(df.copy())
    
    start = time.time()
    pipe_wrapped.fit(df_wrapped, target_column="target")
    fit_time = time.time() - start
    
    # We need to wrap predict data too, and drop target 
    # (though pipeline handles extra cols usually, Model expects exact feature match)
    df_predict_wrapped = PandasEngine.wrap(df.drop(columns=["target"]).copy())

    start = time.time()
    pipe_wrapped.predict(df_predict_wrapped)
    predict_time = time.time() - start
    
    print(f"Fit: {fit_time:.4f}s")
    print(f"Predict: {predict_time:.4f}s")
    results.append({"Mode": "Pandas (Wrapped)", "Fit": fit_time, "Predict": predict_time})

    # 3. Polars (if available) - Current State check
    if HAS_POLARS:
        print("\n--- Benchmarking Polars (via PolarsEngine) ---")
        try:
            df_polars = pl.from_pandas(df)
            df_wrapped_polars = PolarsEngine.wrap(df_polars)
            
            pipe_polars = SkyulfPipeline(pipeline_config)
            
            start = time.time()
            pipe_polars.fit(df_wrapped_polars, target_column="target")
            fit_time = time.time() - start
            
            df_polars_predict = PolarsEngine.wrap(df_polars.drop("target"))

            start = time.time()
            pipe_polars.predict(df_polars_predict)
            predict_time = time.time() - start
            
            print(f"Fit: {fit_time:.4f}s")
            print(f"Predict: {predict_time:.4f}s")
            results.append({"Mode": "Polars (Wrapped)", "Fit": fit_time, "Predict": predict_time})
        except Exception as e:
            print(f"Polars benchmark failed: {e}")

    # Summary
    print("\n\n=== RESULTS ===")
    res_df = pd.DataFrame(results)
    res_df["Total"] = res_df["Fit"] + res_df["Predict"]
    # print(res_df.to_markdown(index=False)) # markdown requires optional dependency
    print(res_df.to_string(index=False))

    # Check overhead
    baseline = res_df[res_df["Mode"] == "Pandas (Direct)"].iloc[0]["Total"]
    wrapped = res_df[res_df["Mode"] == "Pandas (Wrapped)"].iloc[0]["Total"]
    overhead = (wrapped - baseline) / baseline * 100
    print(f"\nWrapper Overhead: {overhead:.2f}%")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Benchmark failure: {e}")
