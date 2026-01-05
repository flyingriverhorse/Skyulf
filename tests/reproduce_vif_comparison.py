
import polars as pl
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from skyulf.profiling.analyzer import EDAAnalyzer

def test_vif_comparison():
    print("--- VIF Comparison Test ---")
    
    # 1. Create Synthetic Data (Multicollinear)
    # x1 = random
    # x2 = 2*x1 + noise (High correlation with x1)
    # x3 = random
    np.random.seed(42)
    n = 1000
    x1 = np.random.normal(0, 1, n)
    x2 = 2 * x1 + np.random.normal(0, 0.1, n) # High correlation
    x3 = np.random.normal(0, 1, n)
    
    data = {
        "x1": x1,
        "x2": x2,
        "x3": x3
    }
    
    df_pl = pl.DataFrame(data)
    df_pd = pd.DataFrame(data)
    
    print("Dataset created with x2 = 2*x1 + noise (Expect high VIF for x1 and x2)")
    
    # 2. Calculate VIF using Statsmodels (Standard)
    print("\n[Statsmodels Calculation]")
    X = add_constant(df_pd)
    vif_sm = {}
    for i in range(1, X.shape[1]): # Skip constant
        col = X.columns[i]
        val = variance_inflation_factor(X.values, i)
        vif_sm[col] = val
        print(f"  {col}: {val:.4f}")
        
    # 3. Calculate VIF using Skyulf Analyzer (Polars/NumPy)
    print("\n[Skyulf Analyzer Calculation]")
    analyzer = EDAAnalyzer(df_pl)
    # We need to access the private method or run analyze()
    # Let's use the private method for direct comparison
    vif_skyulf = analyzer._calculate_vif(["x1", "x2", "x3"])
    
    if vif_skyulf:
        for col, val in vif_skyulf.items():
            print(f"  {col}: {val:.4f}")
    else:
        print("  Failed to calculate VIF")

    # 4. Compare
    print("\n[Comparison]")
    for col in ["x1", "x2", "x3"]:
        sm_val = vif_sm.get(col, 0)
        sky_val = vif_skyulf.get(col, 0) if vif_skyulf else 0
        diff = abs(sm_val - sky_val)
        match = diff < 0.001
        print(f"  {col}: Statsmodels={sm_val:.4f} | Skyulf={sky_val:.4f} | Diff={diff:.6f} | Match={match}")

if __name__ == "__main__":
    test_vif_comparison()
