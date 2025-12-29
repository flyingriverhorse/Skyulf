import pytest
import pandas as pd
import numpy as np
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
from skyulf.engines import get_engine

@pytest.mark.skipif(not HAS_POLARS, reason="Polars not installed")
def test_standard_scaler_parity():
    print("Starting Parity Test...")
    # Data
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
    df_pd = pd.DataFrame(data)
    df_pl = pl.DataFrame(data)
    
    config = {"columns": ["a", "b"], "with_mean": True, "with_std": True}
    
    # Fit Pandas
    print("Fitting Pandas...")
    calc = StandardScalerCalculator()
    params_pd = calc.fit(df_pd, config)
    
    # Fit Polars
    print("Fitting Polars...")
    params_pl = calc.fit(df_pl, config)
    
    # Assert Params Equality
    print("Checking Params...")
    assert np.allclose(params_pd["mean"], params_pl["mean"])
    assert np.allclose(params_pd["scale"], params_pl["scale"])
    
    # Apply Pandas
    print("Applying Pandas...")
    applier = StandardScalerApplier()
    res_pd = applier.apply(df_pd, params_pd)
    
    # Apply Polars
    print("Applying Polars...")
    res_pl = applier.apply(df_pl, params_pl)
    
    # Convert Polars result to Pandas for comparison
    res_pl_pd = res_pl.to_pandas()
    
    # Assert Result Equality
    print("Checking Results...")
    pd.testing.assert_frame_equal(res_pd, res_pl_pd)
    
    print("StandardScaler Parity Test Passed!")

if __name__ == "__main__":
    print(f"Has Polars: {HAS_POLARS}")
    if HAS_POLARS:
        test_standard_scaler_parity()
    else:
        print("Skipping test because Polars is not installed.")
