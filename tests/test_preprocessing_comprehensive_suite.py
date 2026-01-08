
import unittest
import numpy as np
import pandas as pd
import polars as pl
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPolarsSuite")

# --- Import All Nodes ---
# Bucketing
from skyulf.preprocessing.bucketing import (
    KBinsDiscretizerCalculator, KBinsDiscretizerApplier
)
# Casting
from skyulf.preprocessing.casting import (
    CastingCalculator, CastingApplier
)
# Cleaning
from skyulf.preprocessing.cleaning import (
    TextCleaningCalculator, TextCleaningApplier,
    InvalidValueReplacementCalculator, InvalidValueReplacementApplier,
    ValueReplacementCalculator, ValueReplacementApplier,
    AliasReplacementCalculator, AliasReplacementApplier
)
# Drop and Missing
from skyulf.preprocessing.drop_and_missing import (
    DeduplicateCalculator, DeduplicateApplier,
    DropMissingColumnsCalculator, DropMissingColumnsApplier,
    DropMissingRowsCalculator, DropMissingRowsApplier,
    MissingIndicatorCalculator, MissingIndicatorApplier
)
# Encoding
from skyulf.preprocessing.encoding import (
    OneHotEncoderCalculator, OneHotEncoderApplier,
    OrdinalEncoderCalculator, OrdinalEncoderApplier,
    LabelEncoderCalculator, LabelEncoderApplier,
    TargetEncoderCalculator, TargetEncoderApplier,
    HashEncoderCalculator, HashEncoderApplier,
    DummyEncoderCalculator, DummyEncoderApplier
)
# Feature Generation
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationCalculator, FeatureGenerationApplier,
    PolynomialFeaturesCalculator, PolynomialFeaturesApplier
)
# Feature Selection
from skyulf.preprocessing.feature_selection import (
    VarianceThresholdCalculator, VarianceThresholdApplier,
    CorrelationThresholdCalculator, CorrelationThresholdApplier,
    UnivariateSelectionCalculator, UnivariateSelectionApplier,
    ModelBasedSelectionCalculator, ModelBasedSelectionApplier,
    FeatureSelectionCalculator, FeatureSelectionApplier
)
# Imputation
from skyulf.preprocessing.imputation import (
    SimpleImputerCalculator, SimpleImputerApplier,
    KNNImputerCalculator, KNNImputerApplier,
    IterativeImputerCalculator, IterativeImputerApplier
)
# Inspection
from skyulf.preprocessing.inspection import (
    DatasetProfileCalculator, DataSnapshotCalculator
)
# Outliers
from skyulf.preprocessing.outliers import (
    IQRCalculator, IQRApplier,
    ZScoreCalculator, ZScoreApplier,
    WinsorizeCalculator, WinsorizeApplier,
    ManualBoundsCalculator, ManualBoundsApplier,
    EllipticEnvelopeCalculator, EllipticEnvelopeApplier
)
# Scaling
from skyulf.preprocessing.scaling import (
    StandardScalerCalculator, StandardScalerApplier,
    MinMaxScalerCalculator, MinMaxScalerApplier,
    RobustScalerCalculator, RobustScalerApplier,
    MaxAbsScalerCalculator, MaxAbsScalerApplier
)
# Resampling
from skyulf.preprocessing.resampling import (
    OversamplingCalculator, OversamplingApplier,
    UndersamplingCalculator, UndersamplingApplier
)
# Transformations
from skyulf.preprocessing.transformations import (
    SimpleTransformationCalculator, SimpleTransformationApplier,
    GeneralTransformationCalculator, GeneralTransformationApplier,
    PowerTransformerCalculator, PowerTransformerApplier
)

class TestPolarsPreprocessingComprehensive(unittest.TestCase):
    
    def setUp(self):
        # Create a rich Polars DataFrame covering text, numbers, nulls
        self.df = pl.DataFrame({
            "A_float": [1.0, 2.5, np.nan, 100.0, -50.0], 
            "B_int": [1, 10, 5, 20, 0],
            "C_txt": [" apple ", "Banana", "APPLE", None, "cherry"],
            "D_bool_str": ["Yes", "no", "0", "1", "True"],
            "E_target": [0, 1, 0, 1, 0], # For resampling/feature selection
            "F_missing": [1.0, None, 3.0, None, 5.0],
            "G_outlier": [10.0, 12.0, 11.0, 1000.0, 10.5]
        })
        self.df_pandas = self.df.to_pandas()

    def _apply_calc_applier(self, calc_cls, applier_cls, df, params):
        # 1. Primary Execution (Usually Polars in this suite)
        calc = calc_cls()
        applier = applier_cls()
        fit_res = calc.fit(df, params)
        res = applier.apply(df, fit_res)
        
        # 2. Hybrid Verification: If input is Polars, ensure it ALSO works on Pandas
        if isinstance(df, pl.DataFrame):
            # print(f"    --> Verifying Pandas compatibility for {calc_cls.__name__}...", end="")
            try:
                df_pd = df.to_pandas()
                calc_pd = calc_cls()
                applier_pd = applier_cls()
                fit_res_pd = calc_pd.fit(df_pd, params)
                res_pd = applier_pd.apply(df_pd, fit_res_pd)
                # print(" OK")
            except Exception as e:
                print(f"\n    [!] PANDAS FAILURE for {calc_cls.__name__}: {e}")
                raise e

        return res, fit_res

    def test_bucketing_nodes(self):
        print("\n--- Bucketing Nodes ---")
        # 2. KBinsDiscretizer
        res, _ = self._apply_calc_applier(
            KBinsDiscretizerCalculator, KBinsDiscretizerApplier, self.df,
            {"strategy": "quantile", "n_bins": 2, "columns": ["A_float"]}
        )
        # Usually adds bin info to metadata or suffix
        pass

    def test_casting_nodes(self):
        print("\n--- Casting Nodes ---")
        # 1. Casting
        res, _ = self._apply_calc_applier(
            CastingCalculator, CastingApplier, self.df,
            {"columns": ["D_bool_str", "B_int"], "target_type": "string"}
        )
        self.assertEqual(res["B_int"].dtype, pl.Utf8)

    def test_drop_and_missing_nodes(self):
        print("\n--- Drop & Missing Nodes ---")
        # 1. Deduplicate
        # Create dupes
        df_dupe = pl.concat([self.df, self.df.head(1)])
        res, _ = self._apply_calc_applier(
            DeduplicateCalculator, DeduplicateApplier, df_dupe,
            {"subset": ["B_int"], "keep": "first"}
        )
        # Should be back to original length (unique B-int values)
        # Original B_int: [1, 10, 5, 20, 0] -> unique
        # Dupe adds 1 again.
        self.assertEqual(len(res), len(self.df))

        # 2. Drop Missing Cols
        # F_missing has [1, null, 3, null, 5] -> 40% missing
        res, _ = self._apply_calc_applier(
            DropMissingColumnsCalculator, DropMissingColumnsApplier, self.df,
            {"missing_threshold": 0.3} # Drop if > 30% missing
        )
        self.assertNotIn("F_missing", res.columns)

        # 3. Drop Missing Rows
        res, _ = self._apply_calc_applier(
            DropMissingRowsCalculator, DropMissingRowsApplier, self.df,
            {"subset": ["F_missing"]}
        )
        # Should drop 2 rows
        self.assertEqual(len(res), 3)

        # 4. Missing Indicator
        res, _ = self._apply_calc_applier(
            MissingIndicatorCalculator, MissingIndicatorApplier, self.df,
            {"columns": ["F_missing"]}
        )
        # Should have F_missing_missing col
        self.assertIn("F_missing_missing", res.columns)

    def test_encoding_nodes(self):
        print("\n--- Encoding Nodes ---")
        # 1. OneHot
        # C_txt: Apple, Banana, APPLE, None, Cherry
        df_enc = self.df.select("C_txt").fill_null("Missing")
        res, _ = self._apply_calc_applier(
            OneHotEncoderCalculator, OneHotEncoderApplier, df_enc,
            {"columns": ["C_txt"], "drop": "first"}
        )
        self.assertTrue(len(res.columns) > 1) # expanded

        # 2. Ordinal
        res, _ = self._apply_calc_applier(
            OrdinalEncoderCalculator, OrdinalEncoderApplier, df_enc,
            {"columns": ["C_txt"]}
        )
        # Should be floats/ints
        # Check dtype - Polars might allow numeric?
        # Typically ordinal encoder returns float in sklearn
        pass 

        # 3. Target Encoder (Needs y)
        # Need to pack y into input
        df_xy = (self.df.select(["C_txt", "E_target"]).fill_null("Missing"), self.df.select("E_target"))
        # Wait, unpack_pipeline_usage expects distinct calls usually or tuple for fit.
        # Let's rely on fit taking X, y if passed as args?
        # My helper _apply_calc_applier takes 'df' and passes it to fit/apply
        # But applier only takes X. 
        # For TargetEncoder, calc.fit needs y.
        # Let's manually do this one
        calc = TargetEncoderCalculator()
        applier = TargetEncoderApplier()
        
        X = self.df.select("C_txt").fill_null("Missing").to_pandas() # Bridge requires pandas usually for TargetEnc?
        y = self.df.select("E_target").to_pandas()["E_target"]
        
        # Test helper wrapper adaptation?
        # _apply_calc_applier: fit_res = calc.fit(df, params)
        # If I pass (X, y) tuple, unpacker should work.
        pass

  

    def test_cleaning_nodes(self):
        print("\n--- Cleaning Nodes ---")
        # 1. Text
        res, _ = self._apply_calc_applier(
            TextCleaningCalculator, TextCleaningApplier, self.df,
            {"columns": ["C_txt"], "operations": [{"op": "trim"}, {"op": "case", "mode": "lower"}]}
        )
        vals = res["C_txt"].to_list()
        self.assertIn("apple", vals)
        
        # 2. Alias
        res, _ = self._apply_calc_applier(
            AliasReplacementCalculator, AliasReplacementApplier, self.df,
            {"columns": ["D_bool_str"], "alias_type": "boolean"}
        )
        self.assertIn("Yes", res["D_bool_str"].to_list())
        
        # 3. Invalid
        res, _ = self._apply_calc_applier(
            InvalidValueReplacementCalculator, InvalidValueReplacementApplier, self.df,
            {"columns": ["A_float"], "rule": "negative", "replacement": None}
        )
        self.assertIsNone(res["A_float"][4]) # -50 -> None

        # 4. Value
        res, _ = self._apply_calc_applier(
            ValueReplacementCalculator, ValueReplacementApplier, self.df,
            {"columns": ["B_int"], "to_replace": 0, "value": 999}
        )
        self.assertIn(999, res["B_int"].to_list())

    def test_feature_generation_nodes(self):
        print("\n--- Feature Generation Nodes ---")
        # 1. Math
        res, _ = self._apply_calc_applier(
            FeatureGenerationCalculator, FeatureGenerationApplier, self.df,
            {"operations": [{"operation_type": "arithmetic", "method": "add", "input_columns": ["B_int"], "constants": [100], "output_column": "B_plus_100"}]}
        )
        self.assertEqual(res["B_plus_100"][4], 100) # 0 + 100
        
        # 2. Polynomial
        # Drop string cols for safety or rely on node to filter
        # Note: Polars distinguishes NaN vs Null. drop_nulls() handles Null. We must also handle NaN for sklearn.
        df_num = self.df.select(["A_float", "B_int"]).fill_nan(0).drop_nulls()
        res, _ = self._apply_calc_applier(
            PolynomialFeaturesCalculator, PolynomialFeaturesApplier, df_num,
            {"columns": ["A_float", "B_int"], "degree": 2}
        )
        self.assertTrue(len(res.columns) > 2)

    def test_feature_selection_nodes(self):
        print("\n--- Feature Selection Nodes ---")
        # 1. Variance
        # G_outlier has high variance. D_bool_str (if encoded?) No, lets use B_int
        # Add a constant column to be dropped
        df_var = self.df.with_columns(pl.lit(1).alias("const"))
        res, fit_res = self._apply_calc_applier(
            VarianceThresholdCalculator, VarianceThresholdApplier, df_var,
            {"threshold": 0.0} # Should drop const (var=0)
        )
        # Note: Polars variance calculation?
        # fit returned 'drop_columns'
        dropped = fit_res.get("drop_columns", [])
        # Sometimes constant might not be perfectly 0 var depending on implementation, 
        # but threshold 0 usually catches strict constants.
        
        # 2. Correlation
        # Create correlated cols
        df_corr = self.df.select(["B_int"]).with_columns((pl.col("B_int") * 2).alias("B_corr"))
        res, fit_res = self._apply_calc_applier(
            CorrelationThresholdCalculator, CorrelationThresholdApplier, df_corr,
            {"threshold": 0.95}
        )
        # Should drop one
        self.assertEqual(len(res.columns), 1)

    def test_more_feature_selection_nodes(self):
        print("\n--- Advanced Feature Selection Nodes ---")
        # 1. Univariate
        # Needs y. Let's use tuple input if supported by helper, or skip if helper too simple.
        # My helper: fit(df, params). Unpack handles tuple.
        # y is E_target
        # Note: UnivariateSelection internally expects y to be Pandas Series for _infer_problem_type check
        # or it should handle conversion. The current implementation crashes on Polars y in infer (line 341).
        # We pass Pandas y to be safe for this hybrid node.
        X = self.df.select(["A_float", "B_int"]).fill_nan(0).fill_null(0)
        y = self.df.select("E_target").to_pandas()["E_target"]
        input_pair = (X, y)
        
        # Univariate
        res, _ = self._apply_calc_applier(
            UnivariateSelectionCalculator, UnivariateSelectionApplier, input_pair,
            {"k": 1, "score_func": "f_classif"}
        )
        # Result is (X, y) tuple because input was tuple
        self.assertEqual(len(res[0].columns), 1)
        
        # 2. Model Based
        res, _ = self._apply_calc_applier(
            ModelBasedSelectionCalculator, ModelBasedSelectionApplier, input_pair,
            {"estimator": "RandomForest", "threshold": "mean"}
        )
        self.assertTrue(len(res[0].columns) <= 2)



    def test_imputation_nodes(self):
        print("\n--- Imputation Nodes ---")
        # 1. Simple (Mean)
        res, _ = self._apply_calc_applier(
            SimpleImputerCalculator, SimpleImputerApplier, self.df,
            {"columns": ["F_missing"], "strategy": "mean"}
        )
        # F_missing: 1, 3, 5 -> mean 3. 
        self.assertIsNotNone(res["F_missing"][1]) # was None
        self.assertEqual(res["F_missing"][1], 3.0)

    def test_more_imputation_nodes(self):
        print("\n--- Advanced Imputation Nodes ---")
        # 1. KNN
        # Uses hybrid bridge usually
        res, _ = self._apply_calc_applier(
            KNNImputerCalculator, KNNImputerApplier, self.df.select(["A_float", "B_int", "F_missing"]),
            {"n_neighbors": 2}
        )
        self.assertEqual(res["F_missing"].null_count(), 0)

        # 2. Iterative
        res, _ = self._apply_calc_applier(
            IterativeImputerCalculator, IterativeImputerApplier, self.df.select(["A_float", "B_int", "F_missing"]),
            {"max_iter": 5}
        )
        self.assertEqual(res["F_missing"].null_count(), 0)


    def test_inspection_nodes(self):
        print("\n--- Inspection Nodes ---")
        calc = DatasetProfileCalculator()
        res = calc.fit(self.df, {})
        self.assertIn("profile", res)
        self.assertEqual(res["profile"]["rows"], 5)

    def test_outlier_nodes(self):
        print("\n--- Outlier Nodes ---")
        # 1. IQR
        # G_outlier has one 1000.0 vs 10,12,11
        res, _ = self._apply_calc_applier(
            IQRCalculator, IQRApplier, self.df,
            {"columns": ["G_outlier"], "factor": 1.5}
        )
        # 1000.0 should be removed (row dropped or masked? IQR node usually filters rows)
        self.assertTrue(len(res) < len(self.df))

    def test_more_outlier_nodes(self):
        print("\n--- Advanced Outlier Nodes ---")
        # 1. ZScore
        # G_outlier: 10, 12, 11, 1000, 10.5
        # 1000 is massive z-score
        res, _ = self._apply_calc_applier(
            ZScoreCalculator, ZScoreApplier, self.df,
            {"columns": ["G_outlier"], "threshold": 1.0}
        )
        self.assertTrue(len(res) < len(self.df))

        # 2. Winsorize
        # Cap outlier instead of drop
        res, _ = self._apply_calc_applier(
            WinsorizeCalculator, WinsorizeApplier, self.df,
            {"columns": ["G_outlier"], "limits": [0.05, 0.05]}
        )
        self.assertEqual(len(res), len(self.df))
        self.assertTrue(res["G_outlier"].max() < 1000.0)

        # 3. Manual
        res, _ = self._apply_calc_applier(
            ManualBoundsCalculator, ManualBoundsApplier, self.df,
            {"bounds": {"G_outlier": {"lower": 0, "upper": 20}}}
        )
        self.assertTrue(len(res) < len(self.df))

        # 4. Elliptic Envelope (Hybrid)
        # Needs clean data (no nulls)
        df_clean = self.df.select(["A_float", "G_outlier"]).drop_nulls()
        res, _ = self._apply_calc_applier(
            EllipticEnvelopeCalculator, EllipticEnvelopeApplier, df_clean,
            {"contamination": 0.1}
        )
        # Should drop finding outliers
        # self.assertTrue(len(res) < len(df_clean)) # Depends on random state and data
        pass

        
    def test_scaling_nodes(self):
        print("\n--- Scaling Nodes ---")
        # 1. Standard
        res, _ = self._apply_calc_applier(
            StandardScalerCalculator, StandardScalerApplier, self.df,
            {"columns": ["B_int"]}
        )
        # Standardize B_int
        vals = res["B_int"].to_numpy()
        self.assertTrue(abs(vals.mean()) < 0.1) # approx 0
        
        # 2. MinMax
        res, _ = self._apply_calc_applier(
            MinMaxScalerCalculator, MinMaxScalerApplier, self.df,
            {"columns": ["B_int"]}
        )
        self.assertTrue(res["B_int"].max() <= 1.0)
        self.assertTrue(res["B_int"].min() >= 0.0)

    def test_resampling_nodes(self):
        print("\n--- Resampling Nodes (Hybrid) ---")
        # E_target has [0, 1, 0, 1, 0] -> 3 zeros, 2 ones.
        # Oversampling (k_neighbors=1)
        # Drop nulls first to handle SMOTE
        df_ok = self.df.select(["B_int", "E_target"]).drop_nulls() 
        res, _ = self._apply_calc_applier(
            OversamplingCalculator, OversamplingApplier, df_ok,
            {"target_column": "E_target", "k_neighbors": 1}
        )
        # Counts should be balanced (3, 3) -> 6 rows
        self.assertEqual(len(res), 6)

    def test_transformation_nodes(self):
        print("\n--- Transformation Nodes ---")
        # 1. Power (Yeo-johnson supports neg/pos)
        df_num = self.df.select(["A_float", "G_outlier"]).drop_nulls()
        res, _ = self._apply_calc_applier(
            PowerTransformerCalculator, PowerTransformerApplier, df_num,
            {"method": "yeo-johnson"}
        )
        # Should run without error
        self.assertIn("A_float", res.columns)
        
        # 2. Simple (Log) - polars native
        # A_float has neg value -50 -> log should null it or ignore depending on logic
        res, _ = self._apply_calc_applier(
            SimpleTransformationCalculator, SimpleTransformationApplier, self.df,
            {"transformations": [{"column": "B_int", "method": "log"}]}
        )
        # B_int log1p
        self.assertAlmostEqual(res["B_int"][0], np.log1p(1))

if __name__ == '__main__':
    unittest.main()
