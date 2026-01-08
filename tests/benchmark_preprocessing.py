import time
import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, Any, List

# --- Import All Nodes ---
from skyulf.preprocessing.drop_and_missing import DropMissingRowsCalculator, DropMissingRowsApplier
from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
from skyulf.preprocessing.outliers import ZScoreCalculator, ZScoreApplier
from skyulf.preprocessing.cleaning import TextCleaningCalculator, TextCleaningApplier
from skyulf.preprocessing.bucketing import KBinsDiscretizerCalculator, KBinsDiscretizerApplier
from skyulf.preprocessing.feature_generation import PolynomialFeaturesCalculator, PolynomialFeaturesApplier
from skyulf.preprocessing.feature_selection import VarianceThresholdCalculator, VarianceThresholdApplier
from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
from skyulf.preprocessing.encoding import OneHotEncoderCalculator, OneHotEncoderApplier


def generate_dataset(n_rows: int = 100_000):
    print(f"Generating synthetic dataset with {n_rows:,} rows...")
    
    np.random.seed(42)
    
    # Dirty categories (spaces, mixed case)
    categories = [' Red ', 'blue', 'GREEN', ' Yellow', 'Purple']
    zones = ['North', 'South', 'East', 'West']
    
    data = {
        "age": np.random.normal(35, 12, n_rows),
        "income": np.random.normal(55000, 15000, n_rows),
        "score": np.random.uniform(0, 1, n_rows),
        "color": np.random.choice(categories, n_rows),
        "zone": np.random.choice(zones, n_rows),
        "noise_1": np.random.choice([1.0, 1.000001], n_rows), # Low variance
    }
    
    # Introduce missing values (5%)
    mask = np.random.rand(n_rows) < 0.05
    data["age"][mask] = np.nan
    data["income"][mask] = np.nan
    
    # Introduce Outliers for Income
    outlier_mask = np.random.rand(n_rows) < 0.01
    data["income"][outlier_mask] = data["income"][outlier_mask] * 10
    
    # Create DataFrames
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    print(f"Dataset Size in Memory (Pandas): {df_pd.memory_usage(index=True).sum() / 1024**2:.2f} MB")
    print("-" * 60)
    return df_pd, df_pl

def run_pipeline(df, engine_name: str):
    """
    Runs a Comprehensive Pipeline:
    1. Drop Missing Rows (Threshold)
    2. Imputation (Mean)
    3. Outlier Removal (Z-Score > 3)
    4. Text Cleaning (Trim, Lowercase)
    5. Bucketing (Age -> 5 bins)
    6. Feature Generation (Poly Features interact income/score)
    7. Feature Selection (Variance Threshold)
    8. Scaling (Standard)
    9. Encoding (OneHot)
    """
    start_total = time.time()
    times = {}

    def measure(name, calc_cls, app_cls, params):
        t0 = time.time()
        calc = calc_cls()
        app = app_cls()
        try:
            fit_res = calc.fit(df, params)
            # Some appliers might return None if filter drops everything, but unrelated here
            res_df = app.apply(df, fit_res)
        except Exception as e:
            print(f"Error in {name}: {e}")
            raise e
        times[name] = time.time() - t0
        return res_df

    # 1. Drop Missing Rows (if > 50% missing, generally not applicable here, but lets drop strict nulls in subset)
    # Using small threshold to drop rows with any missing in 'zone' (none currently) or strict drop on age
    # NOTE: To be fair, let's just drop rows where 'zone' is null (0 rows) to test overhead, 
    # or drop rows with > 2 missing values.
    # Let's simple use "Drop Missing Rows" where "age" is missing to clean up dataset first? 
    # Actually standard flow is Impute first usually. 
    # Let's Drop Missing Rows where 'noise_1' is NaN (none) to test speed of "check".
    df = measure("DropMissing", DropMissingRowsCalculator, DropMissingRowsApplier, 
                 {"columns": ["noise_1"], "threshold": 0}) 

    # 2. Imputation (Age, Income)
    df = measure("Imputation", SimpleImputerCalculator, SimpleImputerApplier, 
                 {"columns": ["age", "income"], "strategy": "mean"})

    # 3. Outlier Removal (Income)
    df = measure("Outliers", ZScoreCalculator, ZScoreApplier, 
                 {"columns": ["income"], "threshold": 3.0, "method": "drop"})

    # 4. Text Cleaning (Color)
    # Correct format is list of dicts: [{"op": "trim"}, {"op": "lowercase"}]
    df = measure("Cleaning", TextCleaningCalculator, TextCleaningApplier, 
                 {"columns": ["color"], "operations": [{"op": "trim"}, {"op": "lowercase"}]})

    # 5. Bucketing (Age)
    df = measure("Bucketing", KBinsDiscretizerCalculator, KBinsDiscretizerApplier, 
                 {"columns": ["age"], "n_bins": 5, "strategy": "quantile"})

    # 6. Feature Generation (Poly on Income/Score)
    # Degree 2 interaction
    df = measure("FeatGen", PolynomialFeaturesCalculator, PolynomialFeaturesApplier, 
                 {"columns": ["income", "score"], "degree": 2, "interaction_only": True})

    # 7. Feature Selection (Drop Low Variance 'noise_1')
    df = measure("FeatSelect", VarianceThresholdCalculator, VarianceThresholdApplier, 
                 {"threshold": 0.01, "drop_columns": True})

    # 8. Scaling (Income, Score)
    df = measure("Scaling", StandardScalerCalculator, StandardScalerApplier, 
                 {"columns": ["income", "score"]})

    # 9. Encoding (Color, Zone)
    # Note: Bucketing produced a new column usually, but we haven't selected it. 
    # Bucketing modifies in place or renames? Default implies modifying or new col.
    # KBins usually maps to ordinal integers.
    df = measure("Encoding", OneHotEncoderCalculator, OneHotEncoderApplier, 
                 {"columns": ["color", "zone"]})
    
    total_time = time.time() - start_total
    
    print(f"[{engine_name}] Pipeline completed in {total_time:.4f}s. Final Shape: {df.shape}")
    return times, total_time

def main():
    # 1. Setup
    ROWS = 500_000
    df_pd, df_pl = generate_dataset(ROWS)
    
    # 2. Run Benchmarks
    print("\nRunning PANDAS Pipeline...")
    pd_times, pd_total = run_pipeline(df_pd, "Pandas")
    
    print("\nRunning POLARS Pipeline...")
    pl_times, pl_total = run_pipeline(df_pl, "Polars")
    
    # 3. Generate Versus Table
    print("\n" + "="*75)
    print(f"{'OPERATION':<20} | {'PANDAS (s)':<12} | {'POLARS (s)':<12} | {'SPEEDUP':<10}")
    print("-" * 75)
    
    nodes = ["DropMissing", "Imputation", "Outliers", "Cleaning", 
             "Bucketing", "FeatGen", "FeatSelect", "Scaling", "Encoding"]
             
    for op in nodes:
        t_pd = pd_times.get(op, 0)
        t_pl = pl_times.get(op, 0)
        speedup = t_pd / t_pl if t_pl > 0 else 0.0
        print(f"{op:<20} | {t_pd:<12.4f} | {t_pl:<12.4f} | {speedup:<9.2f}x")
        
    print("-" * 75)
    speedup_total = pd_total / pl_total if pl_total > 0 else 0.0
    print(f"{'TOTAL PIPELINE':<20} | {pd_total:<12.4f} | {pl_total:<12.4f} | {speedup_total:<9.2f}x")
    print("="*75)

if __name__ == "__main__":
    main()
