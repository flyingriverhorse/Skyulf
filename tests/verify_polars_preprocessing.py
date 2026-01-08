
import unittest
import numpy as np
import pandas as pd
import polars as pl
import logging

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import nodes
from skyulf.preprocessing.bucketing import KBinsDiscretizerCalculator, KBinsDiscretizerApplier
from skyulf.preprocessing.cleaning import (
    TextCleaningCalculator, TextCleaningApplier, 
    InvalidValueReplacementCalculator, InvalidValueReplacementApplier,
    ValueReplacementCalculator, ValueReplacementApplier,
    AliasReplacementCalculator, AliasReplacementApplier
)
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationCalculator, FeatureGenerationApplier,
    PolynomialFeaturesCalculator, PolynomialFeaturesApplier
)
from skyulf.preprocessing.inspection import DatasetProfileCalculator

class TestPolarsPreprocessingFull(unittest.TestCase):
    
    def setUp(self):
        # Create a standard Polars DataFrame for testing
        self.df = pl.DataFrame({
            "A_num": [1.0, 2.0, 3.0, 100.0, None], # None for nulls!
            "B_cat": ["apple", "banana", "APPLE", " pear ", None],
            "C_txt": ["Yes", "no", "True", "0", "bad"],
            "D_neg": [-10, 0, 10, 20, 30]
        })

    # --- 1. Bucketing ---
    def test_bucketing_polars(self):
        print("\n[Bucketing] Testing Output...")
        # Test Quantile Bucketing (qcut)
        calc = KBinsDiscretizerCalculator() # Alias for Bucketizer in this context
        applier = KBinsDiscretizerApplier()
        
        try:
            params = {"strategy": "quantile", "n_bins": 2, "columns": ["A_num"]}
            fit_res = calc.fit(self.df, params)
            res = applier.apply(self.df, fit_res)
            
            # Check for bin_edges instead of bins
            self.assertIn("bin_edges", fit_res)
            
            print("  v Quantile extraction passed")
        except Exception as e:
            self.fail(f"Bucketing failed: {e}")

    # --- 2. Cleaning: Text ---
    def test_text_cleaning_polars(self):
        print("\n[Cleaning] Testing TextCleaning...")
        calc = TextCleaningCalculator()
        applier = TextCleaningApplier()
        
        # Trim and Lowercase B_cat
        config = {
            "columns": ["B_cat"],
            "operations": [
                {"op": "trim", "mode": "both"},
                {"op": "case", "mode": "lower"}
            ]
        }
        res_fit = calc.fit(self.df, config)
        res = applier.apply(self.df, res_fit)
        
        # " pear " -> "pear", "APPLE" -> "apple"
        vals = res["B_cat"].to_list()
        self.assertIn("pear", vals)
        self.assertIn("apple", vals)
        self.assertNotIn("APPLE", vals)
        self.assertNotIn(" pear ", vals)
        print("  v Text operations passed")

    # --- 3. Cleaning: Alias ---
    def test_alias_replacement_polars(self):
        print("\n[Cleaning] Testing AliasReplacement...")
        calc = AliasReplacementCalculator()
        applier = AliasReplacementApplier()
        
        # "Yes"->"True", "no"->"False", "0"->"False"
        # Binds standard boolean map
        config = {"columns": ["C_txt"], "alias_type": "boolean"}
        res_fit = calc.fit(self.df, config)
        res = applier.apply(self.df, res_fit)
        
        vals = res["C_txt"].to_list()
        # default mapping: Yes->Yes, no->No. Wait, check CONSTANTS in code.
        # COMMON_BOOLEAN_ALIASES map: "yes"->"Yes", "no"->"No", "0"->"No"
        self.assertIn("Yes", vals)
        self.assertIn("No", vals)
        # "bad" is not in mapping, should remain "bad"
        self.assertIn("bad", vals)
        print("  v Alias mapping passed")

    # --- 4. Cleaning: Value / Invalid ---
    def test_value_replacement_polars(self):
        print("\n[Cleaning] Testing ValueReplacement...")
        calc = InvalidValueReplacementCalculator()
        applier = InvalidValueReplacementApplier()
        
        # Replace negative values in D_neg with NaN
        config = {
            "columns": ["D_neg"],
            "rule": "negative",
            "replacement": None # NaN
        }
        res_fit = calc.fit(self.df, config)
        res = applier.apply(self.df, res_fit)
        
        # -10 should be null
        d_vals = res["D_neg"]
        # In polars, null is None in to_list usually
        self.assertIsNone(d_vals[0]) 
        self.assertEqual(d_vals[1], 0)
        print("  v Invalid value replacement passed")

    # --- 5. Feature Generation: Math ---
    def test_math_formula_polars(self):
        print("\n[Feature Gen] Testing FeatureGeneration (Math)...")
        calc = FeatureGenerationCalculator()
        applier = FeatureGenerationApplier()
        
        # Create E = D_neg * 2
        # Param structure based on inspection:
        # operations: [{'operation_type': 'arithmetic', 'method': 'multiply', 'input_columns': ['D_neg'], 'constants': [2], 'output_column': 'E'}]
        config = {
            "operations": [
                {
                    "operation_type": "arithmetic",
                    "method": "multiply",
                    "input_columns": ["D_neg"],
                    "constants": [2],
                    "output_column": "E"
                }
            ]
        }
        res_fit = calc.fit(self.df, config)
        res = applier.apply(self.df, res_fit)
        
        self.assertIn("E", res.columns)
        # -10 * 2 = -20
        self.assertEqual(res["E"][0], -20.0)
        print("  v Math formula passed")

    # --- 6. Feature Generation: Polynomial ---
    def test_polynomial_polars(self):
        print("\n[Feature Gen] Testing Polynomial...")
        calc = PolynomialFeaturesCalculator()
        applier = PolynomialFeaturesApplier()
        
        # Poly features for D_neg (degree 2)
        # This uses the hybrid path (conversion)
        config = {
            "columns": ["D_neg"],
            "degree": 2,
            "interaction_only": False
        }
        
        # D_neg has negative values, ignore NaN in other cols
        df_clean = self.df.drop_nulls() 
        
        res_fit = calc.fit(df_clean, config)
        res = applier.apply(df_clean, res_fit)
        
        # Check generated column names
        cols = res.columns
        self.assertTrue(len(cols) > len(df_clean.columns))
        print("  v Polynomial conversion passed")

    # --- 7. Inspection ---
    def test_inspection_polars(self):
        print("\n[Inspection] Testing Profile...")
        calc = DatasetProfileCalculator()
        
        config = {}
        res = calc.fit(self.df, config)
        
        profile = res["profile"]
        self.assertEqual(profile["rows"], 5)
        self.assertEqual(profile["columns"], 4)
        # Check missing count for A_num (1 null)
        self.assertEqual(profile["missing"]["A_num"], 1)
        
        print("  v Inspection profile passed")

if __name__ == '__main__':
    unittest.main()
