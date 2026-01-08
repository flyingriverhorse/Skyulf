
import unittest
import polars as pl
import numpy as np
import pandas as pd
from skyulf.preprocessing.transformations import (
    SimpleTransformationCalculator, SimpleTransformationApplier,
    GeneralTransformationCalculator, GeneralTransformationApplier,
    PowerTransformerCalculator, PowerTransformerApplier
)

class TestTransformationsPolars(unittest.TestCase):

    def test_simple_transformation(self):
        print("\n--- Testing Simple Transformations (Polars) ---")
        df = pl.DataFrame({
            "A": [1.0, 10.0, 100.0, 1000.0],
            "B": [-1.0, 0.0, 1.0, 2.0],
        })
        
        calc = SimpleTransformationCalculator()
        applier = SimpleTransformationApplier()
        
        params = {"transformations": [{"column": "A", "method": "log"}]}
        fit_res = calc.fit(df, params)
        res = applier.apply(df, fit_res)
        
        expected_A = np.log1p([1.0, 10.0, 100.0, 1000.0])
        curr_A = res["A"].to_numpy()
        np.testing.assert_allclose(curr_A, expected_A, err_msg="Log transformation failed")
        print("Simple Transformation (Log) passed.")

    def test_general_transformation_boxcox(self):
        print("\n--- Testing General Transformation Box-Cox (Polars) ---")
        # strictly positive for Box-Cox
        df_pos = pl.DataFrame({
            "X": [0.1, 1.2, 5.5, 12.0, 0.5, 3.3],
            "Y": [1, 2, 3, 4, 5, 6]
        })
        
        calc_gen = GeneralTransformationCalculator()
        applier_gen = GeneralTransformationApplier()
        
        params_gen = {"transformations": [{"column": "X", "method": "box-cox"}]}
        fit_res_gen = calc_gen.fit(df_pos, params_gen)
        
        # check if params saved
        trans_list = fit_res_gen["transformations"]
        self.assertEqual(len(trans_list), 1)
        self.assertIn("lambdas", trans_list[0])
        print(f"Box-Cox fitted lambda: {trans_list[0]['lambdas']}")
        
        res_gen = applier_gen.apply(df_pos, fit_res_gen)
        self.assertEqual(res_gen["X"].null_count(), 0)
        print("General Transformation (Box-Cox) passed.")

    def test_power_transformer(self):
        print("\n--- Testing Power Transformer Yeo-Johnson (Polars) ---")
        calc_pow = PowerTransformerCalculator()
        applier_pow = PowerTransformerApplier()
        
        # Yeo-Johnson supports negative values
        df_yj = pl.DataFrame({
            "Z": [-2.0, -0.5, 0.0, 0.5, 2.0]
        })
        
        params_pow = {"method": "yeo-johnson", "columns": ["Z"]}
        fit_res_pow = calc_pow.fit(df_yj, params_pow)
        self.assertIn("lambdas", fit_res_pow)
        self.assertEqual(len(fit_res_pow["lambdas"]), 1)
        
        res_pow = applier_pow.apply(df_yj, fit_res_pow)
        self.assertEqual(res_pow["Z"].null_count(), 0)
        print("PowerTransformer (Yeo-Johnson) passed.")

if __name__ == '__main__':
    unittest.main()
