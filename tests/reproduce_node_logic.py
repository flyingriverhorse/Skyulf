
import sys
import unittest
import numpy as np
import pandas as pd
import polars as pl
from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
from skyulf.preprocessing.outliers import IQRCalculator, IQRApplier
from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
from skyulf.preprocessing.encoding import (
    OneHotEncoderCalculator, OneHotEncoderApplier,
    OrdinalEncoderCalculator, OrdinalEncoderApplier,
    LabelEncoderCalculator, LabelEncoderApplier
)
from skyulf.preprocessing.drop_and_missing import (
    DropMissingRowsCalculator, DropMissingRowsApplier,
    DeduplicateCalculator, DeduplicateApplier
)
from skyulf.preprocessing.split import (
    SplitCalculator, SplitApplier,
    FeatureTargetSplitCalculator, FeatureTargetSplitApplier
)

class TestPreprocessingNodes(unittest.TestCase):
    def test_standard_scaler_polars(self):
        """Test Standard Scaler with Polars Engine"""
        print("\nTesting StandardScalerCalculator (Polars)...")
        
        # Setup data
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Fit
        calc = StandardScalerCalculator()
        config = {"columns": ["a", "b"]}
        params = calc.fit(df, config)
        
        print(f"Fit Params: {params}")
        self.assertEqual(params["type"], "standard_scaler")
        # Check mean of [1,2,3,4,5] is 3.0
        self.assertAlmostEqual(params["mean"][0], 3.0) 
        
        # Apply
        applier = StandardScalerApplier()
        res = applier.apply(df, params)
        
        # Depending on implementation, it might return just DF (if input was DF)
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
        
        # Check result
        # (val - mean) / std
        # (3 - 3) / std = 0
        a_col = out_df.get_column("a").to_numpy()
        self.assertAlmostEqual(a_col[2], 0.0)
        
        print("StandardScaler Polars Test Passed!")

    def test_iqr_outlier_polars(self):
        """Test IQR Outlier Removal with Polars Engine (Logic Check)"""
        print("\nTesting IQRCalculator (Polars)...")
        
        # Setup data with outliers
        df = pl.DataFrame({
            "val": [1, 2, 3, 4, 5, 100],  # 100 is outlier
            "other": ["a", "b", "c", "d", "e", "f"]
        })
        
        # Fit
        calc = IQRCalculator()
        config = {"columns": ["val"], "multiplier": 1.5}
        params = calc.fit(df, config)
        
        print(f"Fit Params: {params}")
        
        # Apply
        applier = IQRApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
        
        print(f"Original Row Count: {df.height}")
        print(f"Filtered Row Count: {out_df.height}")
        
        # Should remove 1 row
        self.assertEqual(out_df.height, 5)
        self.assertNotIn(100, out_df.get_column("val").to_list())
        
        print("IQR Polars Test Passed!")

    def test_iqr_outlier_with_y_polars(self):
        """Test IQR Outlier Removal with Target Variable (y) in Polars"""
        print("\nTesting IQR with Target Variable (Polars)...")
        
        X = pl.DataFrame({"val": [1, 2, 3, 100]})
        y = pl.Series("target", [0, 0, 0, 1])
        
        calc = IQRCalculator()
        params = calc.fit(X, {"columns": ["val"]})
        
        applier = IQRApplier()
        # Pass as tuple (X, y)
        res = applier.apply((X, y), params)
        
        # Expect tuple back
        self.assertIsInstance(res, tuple)
        out_X, out_y = res
        
        self.assertEqual(out_X.height, 3)
        self.assertEqual(out_y.len(), 3)
        self.assertEqual(out_y.to_list(), [0, 0, 0]) # the '1' target should be removed
        
        print("IQR with y (Polars) Test Passed!")

    def test_simple_imputer_polars(self):
        """Test Simple Imputer (Mean & Constant) with Polars"""
        print("\nTesting SimpleImputer (Polars)...")
        
        # Create DF with nulls
        df = pl.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0], # Mean should be 3.0 (sum 12 / 4)
            "b": [None, 10.0, 10.0, 10.0, 10.0]
        })
        
        # 1. Test Mean Strategy
        calc = SimpleImputerCalculator()
        config = {"columns": ["a"], "strategy": "mean"}
        params = calc.fit(df, config)
        
        print(f"Simple Imputer Params: {params}")
        self.assertEqual(params["strategy"], "mean")
        self.assertAlmostEqual(params["fill_values"]["a"], 3.0)
        
        applier = SimpleImputerApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
             
        # Check filled value
        filled_a = out_df.get_column("a").to_list()
        self.assertEqual(filled_a[2], 3.0) # The None at index 2 should be 3.0
        
        # 2. Test Constant Strategy
        config_const = {"columns": ["b"], "strategy": "constant", "fill_value": 999.0}
        params_const = calc.fit(df, config_const)
        res_const = applier.apply(df, params_const)
        
        if isinstance(res_const, tuple):
             out_df_const = res_const[0]
        else:
             out_df_const = res_const
             
        filled_b = out_df_const.get_column("b").to_list()
        self.assertEqual(filled_b[0], 999.0)
        
        print("SimpleImputer Polars Test Passed!")

    def test_onehot_encoder_polars(self):
        """Test OneHotEncoder with Polars"""
        print("\nTesting OneHotEncoder (Polars)...")
        
        df = pl.DataFrame({
            "color": ["red", "blue", "red", "green"],
            "size": ["S", "M", "L", "S"]
        })
        
        calc = OneHotEncoderCalculator()
        config = {"columns": ["color"], "drop_first": False}
        params = calc.fit(df, config)
        
        print(f"OneHot Params: {params.keys()}")
        self.assertIn("color_red", params["feature_names"])
        self.assertIn("color_blue", params["feature_names"])
        # green might accept sorted order: blue, green, red
        
        applier = OneHotEncoderApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
             
        # columns should include color_blue, color_green, color_red (assuming auto sort)
        # and original 'color' should be dropped by default
        self.assertIn("color_red", out_df.columns)
        self.assertNotIn("color", out_df.columns)
        
        # Check execution correctness: Row 0 is red
        # color_red should be 1
        red_col = out_df.get_column("color_red").to_list()
        self.assertEqual(red_col[0], 1)
        self.assertEqual(red_col[1], 0) # blue
        
        print("OneHotEncoder Polars Test Passed!")

    def test_ordinal_encoder_polars(self):
        """Test OrdinalEncoder with Polars"""
        print("\nTesting OrdinalEncoder (Polars)...")
        
        df = pl.DataFrame({
             "grade": ["A", "B", "C", "A"]
        })
        
        calc = OrdinalEncoderCalculator()
        params = calc.fit(df, {"columns": ["grade"]})
        
        applier = OrdinalEncoderApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
             
        grade_col = out_df.get_column("grade").to_list()
        # A should be encoded to same integer (0.0 probably)
        self.assertEqual(grade_col[0], grade_col[3])
        # A != B
        self.assertNotEqual(grade_col[0], grade_col[1])
        
        print("OrdinalEncoder Polars Test Passed!")

    def test_label_encoder_polars(self):
        """Test LabelEncoder (Target) in Polars"""
        print("\nTesting LabelEncoder Target (Polars)...")
        
        X = pl.DataFrame({"feature": [1, 2, 3]})
        y = pl.Series("target", ["no", "yes", "no"])
        
        calc = LabelEncoderCalculator()
        # No columns means fit on target
        params = calc.fit((X, y), {})
        
        applier = LabelEncoderApplier()
        res = applier.apply((X, y), params)
        
        self.assertIsInstance(res, tuple)
        out_X, out_y = res
        
        # X should be unchanged
        self.assertEqual(out_X.width, 1)
        
        # y should be integers
        y_list = out_y.to_list()
        self.assertEqual(y_list[0], y_list[2]) # no == no
        self.assertNotEqual(y_list[0], y_list[1]) # no != yes
        self.assertTrue(isinstance(y_list[0], int))
        
        print("LabelEncoder Polars Test Passed!")

    def test_drop_missing_rows_polars(self):
        """Test DropMissingRows in Polars"""
        print("\nTesting DropMissingRows (Polars)...")
        
        # Row 1 has null in 'a'
        df = pl.DataFrame({
            "a": [1, None, 3],
            "b": [1, 2, 3]
        })
        
        calc = DropMissingRowsCalculator()
        config = {"subset": ["a"], "how": "any"}
        params = calc.fit(df, config)
        
        applier = DropMissingRowsApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
             
        self.assertEqual(out_df.height, 2)
        self.assertEqual(out_df["a"].to_list(), [1, 3])
        
        print("DropMissingRows Polars Test Passed!")
        
    def test_deduplicate_polars(self):
        """Test Deduplicate in Polars"""
        print("\nTesting Deduplicate (Polars)...")
        
        df = pl.DataFrame({
            "id": [1, 1, 2, 3],
            "val": [10, 20, 30, 40]
        })
        
        calc = DeduplicateCalculator()
        config = {"subset": ["id"], "keep": "first"} # Should keep val=10
        params = calc.fit(df, config)
        
        applier = DeduplicateApplier()
        res = applier.apply(df, params)
        
        if isinstance(res, tuple):
             out_df = res[0]
        else:
             out_df = res
        
        self.assertEqual(out_df.height, 3)
        self.assertEqual(out_df.filter(pl.col("id") == 1)["val"][0], 10)
        
        print("Deduplicate Polars Test Passed!")

    def test_train_test_split_polars(self):
        """Test TrainTestSplit algorithm with Polars input"""
        print("\nTesting TrainTestSplit (Polars)...")
        
        # 10 rows
        df = pl.DataFrame({
            "id": range(10),
            "target": [0,0,0,0,0, 1,1,1,1,1]
        })
        
        calc = SplitCalculator()
        config = {"test_size": 0.2, "random_state": 42}
        params = calc.fit(df, config)
        
        applier = SplitApplier()
        res = applier.apply(df, params)
        
        # Result should be SplitDataset (not tuple)
        # However, BaseApplier usually returns data structures.
        # Check SplitDataset structure
        self.assertTrue(hasattr(res, "train"))
        self.assertTrue(hasattr(res, "test"))
        
        train_df, train_y = res.train
        test_df, test_y = res.test
        
        # train_y/test_y might be None if input was just DF and no target split logic inside split itself
        # Actually SplitApplier splits rows. It returns (X_train, y_train) IF input was (X, y).
        # If input was DF, it returns (train_df, None) usually?
        # Let's check Splitter logic.
        # It calls train_test_split(df_pd). Returns train, test DFs.
        # So res.train is (train_df, None) or just train_df?
        # SplitDataset definition: train: Tuple[Any, Any]
        
        # Logic in split.py: 
        # return SplitDataset(train=(X_train, y_train), ...)
        # If input was DF, X_train is the train DF, y_train is None? 
        # Ah, in split.py:
        # X_train, y_train = X_train_val, y_train_val (where y_train_val is None if y_pd was None)
        # So yes, (df, None).
        
        self.assertEqual(train_df.height, 8)
        self.assertEqual(test_df.height, 2)
        
        print("TrainTestSplit Polars Test Passed!")

    def test_feature_target_split_polars(self):
        """Test FeatureTargetSplit in Polars"""
        print("\nTesting FeatureTargetSplit (Polars)...")
        
        df = pl.DataFrame({
            "f1": [1, 2],
            "target": [0, 1]
        })
        
        calc = FeatureTargetSplitCalculator()
        config = {"target_column": "target"}
        params = calc.fit(df, config)
        
        applier = FeatureTargetSplitApplier()
        X, y = applier.apply(df, params)
        
        self.assertNotIn("target", X.columns)
        self.assertEqual(y.name, "target")
        self.assertEqual(y.len(), 2)
        
        print("FeatureTargetSplit Polars Test Passed!")

    def test_bucketing_polars(self):
        """Test binning in Polars."""
        import polars as pl
        from skyulf.preprocessing.bucketing import GeneralBinningCalculator, GeneralBinningApplier
        
        df_pl = pl.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            "B": [10, 20, 30, 40, 50, 100]
        })
        
        # Test 1: Equal Width Binning (fit converts to pandas internally, but apply should stay in Polars)
        calc = GeneralBinningCalculator()
        config = {
            "columns": ["A"],
            "strategy": "equal_width",
            "n_bins": 2,
            "output_suffix": "_bin"
        }
        params = calc.fit(df_pl, config)
        
        # Apply
        applier = GeneralBinningApplier()
        out_pl = applier.apply(df_pl, params)
        
        if isinstance(out_pl, tuple):
             out_pl = out_pl[0]

        self.assertIsInstance(out_pl, pl.DataFrame)
        self.assertIn("A_bin", out_pl.columns)
        
        # Check values.
        # [1, 10], 2 bins -> midpoint 5.5.
        # <= 5.5 -> bin 0
        # > 5.5 -> bin 1
        res = out_pl.select("A_bin").to_series().to_list()
        self.assertEqual(res[0], 0)
        self.assertEqual(res[-1], 1)
        
        # Test 2: Custom Binning + Labels
        manual_params = {
           "bin_edges": {"B": [0, 25, 50, 1000]},
           "output_suffix": "_cat",
           "custom_labels": {"B": ["Low", "Medium", "High"]},
           "label_format": "ordinal" # Should be ignored if labels provided
        }
        
        out_custom = applier.apply(df_pl, manual_params)
        if isinstance(out_custom, tuple):
             out_custom = out_custom[0]
             
        res_custom = out_custom.select("B_cat").to_series().to_list()
        self.assertEqual(res_custom[0], "Low")  # 10
        self.assertEqual(res_custom[2], "Medium") # 30
        self.assertEqual(res_custom[-1], "High") # 100

        print("Bucketing Polars Test Passed!")

    def test_feature_generation_polars(self):
        """Test Feature Generation (Math) and PolynomialFeatures in Polars."""
        import polars as pl
        from skyulf.preprocessing.feature_generation import (
             FeatureGenerationCalculator, FeatureGenerationApplier,
             PolynomialFeaturesCalculator, PolynomialFeaturesApplier
        )
        
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "dt": ["2023-01-01", "2023-01-02", "2023-06-15"]
        })
        
        # 1. Test Feature Math (Arithmetic)
        calc = FeatureGenerationCalculator()
        ops = [
            {
                "operation_type": "arithmetic",
                "method": "add",
                "input_columns": ["a", "b"],
                "output_column": "sum_ab"
            },
            {
                "operation_type": "datetime_extract",
                "input_columns": ["dt"],
                "datetime_features": ["month"]
            }
        ]
        config = {"operations": ops}
        params = calc.fit(df, config)
        
        applier = FeatureGenerationApplier()
        res = applier.apply(df, params)
        if isinstance(res, tuple): res = res[0]
        
        # Check sum
        # 1+10=11, 2+20=22
        sum_col = res.get_column("sum_ab").to_list()
        self.assertEqual(sum_col[0], 11.0)
        
        # Check datetime
        # 2023-01 -> month 1. 2023-06 -> month 6.
        month_col = res.get_column("dt_month").to_list()
        self.assertEqual(month_col[0], 1)
        self.assertEqual(month_col[2], 6)
        
        # 2. Test Polynomial Features
        # Poly degree 2: [1, a, b, a^2, ab, b^2]
        poly_calc = PolynomialFeaturesCalculator()
        poly_config = {
             "columns": ["a", "b"],
             "degree": 2,
             "include_bias": False,
             "include_input_features": False # exclude a, b (already in X)
        }
        poly_params = poly_calc.fit(df, poly_config)
        
        poly_applier = PolynomialFeaturesApplier()
        poly_res = poly_applier.apply(df, poly_params)
        if isinstance(poly_res, tuple): poly_res = poly_res[0]
        
        # Check columns exist
        cols = poly_res.columns
        # Should have poly_a_pow_2, poly_b_pow_2, poly_a_b
        
        # Check values. a=2, b=20. a*b = 40.
        # Find column for a*b. likely poly_a_b
        ab_col_name = "poly_a_b" 
        if ab_col_name in cols:
             ab_val = poly_res.get_column(ab_col_name)[1]
             self.assertEqual(ab_val, 40.0)
        
        print("Feature Generation Polars Test Passed!")

    def test_feature_selection_polars(self):
        """Test Feature Selection (Variance, Corr, Univariate) in Polars"""
        import polars as pl
        from skyulf.preprocessing.feature_selection import (
             VarianceThresholdCalculator, VarianceThresholdApplier,
             CorrelationThresholdCalculator, CorrelationThresholdApplier,
             UnivariateSelectionCalculator, UnivariateSelectionApplier
        )
        
        # 1. Variance Threshold
        # 'const' col has 0 variance. 'a' has variance.
        df_var = pl.DataFrame({
             "a": [1, 2, 3, 4],
             "const": [1, 1, 1, 1]
        })
        
        var_calc = VarianceThresholdCalculator()
        params = var_calc.fit(df_var, {"threshold": 0.0})
        
        self.assertIn("const", params["candidate_columns"])
        self.assertNotIn("const", params["selected_columns"])
        
        var_applier = VarianceThresholdApplier()
        out_var = var_applier.apply(df_var, params)
        if isinstance(out_var, tuple): out_var = out_var[0]
        
        self.assertNotIn("const", out_var.columns)
        self.assertIn("a", out_var.columns)
        
        # 2. Correlation Threshold
        # 'a' and 'b' perfectly correlated (corr=1.0)
        df_corr = pl.DataFrame({
             "a": [1, 2, 3, 4],
             "b": [2, 4, 6, 8],
             "c": [1, 5, 2, 9]
        })
        
        corr_calc = CorrelationThresholdCalculator()
        # Thresh 0.9. b should be dropped as it correlates with a.
        params_corr = corr_calc.fit(df_corr, {"threshold": 0.9})
        
        self.assertIn("b", params_corr["columns_to_drop"])
        
        app_corr = CorrelationThresholdApplier()
        out_corr = app_corr.apply(df_corr, params_corr)
        if isinstance(out_corr, tuple): out_corr = out_corr[0]
        
        self.assertNotIn("b", out_corr.columns)
        self.assertIn("a", out_corr.columns)
        
        # 3. Univariate select
        # Select features best predicting 'target'
        # 'noise' is random. 'signal' correlates with 'target'.
        df_uni = pl.DataFrame({
             "signal": [1, 2, 3, 4, 1, 2, 3, 4],
             "noise":  [5, 1, 8, 2, 5, 1, 8, 2],
             "target": [0, 0, 1, 1, 0, 0, 1, 1]
        })
        
        uni_calc = UnivariateSelectionCalculator()
        config_uni = {
             "target_column": "target",
             "k": 1,
             "method": "select_k_best"
        }
        params_uni = uni_calc.fit(df_uni, config_uni)
        
        self.assertIn("signal", params_uni["selected_columns"])
        self.assertNotIn("noise", params_uni["selected_columns"])
        
        print("Feature Selection Polars Test Passed!")

    def test_resampling_polars(self):
        """Test Oversampling/Undersampling in Polars"""
        import polars as pl
        from skyulf.preprocessing.resampling import (
            OversamplingCalculator, OversamplingApplier,
            UndersamplingCalculator, UndersamplingApplier
        )
        
        # Class 1 is minority (2 instances), Class 0 is majority (4 instances)
        # X must be numeric.
        df_im = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
            "target": [0, 0, 0, 0, 1, 1]
        })
        
        # Oversampling (SMOTE)
        calc = OversamplingCalculator()
        config = {
             "method": "smote", 
             "target_column": "target",
             "k_neighbors": 1
        }
        params = calc.fit(df_im, config)
        
        applier = OversamplingApplier()
        
        try:
             import imblearn
        except ImportError:
             print("Skipping Resampling test: imblearn not installed")
             return

        res = applier.apply(df_im, params)
        # Returns DF with target because input was DF
        self.assertIsInstance(res, pl.DataFrame)
        self.assertIn("target", res.columns)
        
        # Check if balanced. 4 class 0, 2 class 1. Should oversample class 1 to 4. Total 8.
        self.assertEqual(res.filter(pl.col("target") == 1).height, 4)
        self.assertEqual(res.height, 8)
        
        # Undersampling
        # Random under sampling of majority class
        calc_under = UndersamplingCalculator()
        config_under = {
            "method": "random_under_sampling",
            "target_column": "target"
        }
        params_under = calc_under.fit(df_im, config_under)
        
        applier_under = UndersamplingApplier()
        res_under = applier_under.apply(df_im, params_under)
        
        # Majority (0) has 4. Minority (1) has 2.
        # Undersample 0 to 2. Total 4.
        self.assertEqual(res_under.height, 4)
        self.assertEqual(res_under.filter(pl.col("target") == 0).height, 2)
        
        print("Resampling Polars Test Passed!")

if __name__ == '__main__':
    unittest.main()
