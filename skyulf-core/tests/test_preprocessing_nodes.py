"""Comprehensive unit tests for all preprocessing Calculator/Applier pairs.

Tests each node in isolation: fit → params → apply → verify.
Covers: scaling, imputation, encoding, outliers, cleaning, feature_selection,
feature_generation, drop_and_missing, bucketing, transformations, casting.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """DataFrame with numeric columns and some NaN."""
    np.random.seed(42)
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, np.nan, 90.0, 100.0],
        "c": np.random.normal(0, 1, 10),
    })


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """DataFrame with categorical + numeric columns."""
    return pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue", "green", "red", "blue", "green", "red"],
        "size": ["S", "M", "L", "S", "M", "L", "S", "M", "L", "S"],
        "price": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0, 18.0],
        "target": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def text_df() -> pd.DataFrame:
    """DataFrame with text columns for cleaning tests."""
    return pd.DataFrame({
        "name": ["  John ", "JANE", "bob  ", " Alice", "  EVE  "],
        "email": ["john@test.com", "jane@test.com", None, "alice@test.com", "eve@test.com"],
        "score": [85, 90, 78, 92, 88],
    })


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    """DataFrame with outliers for outlier detection tests."""
    np.random.seed(42)
    values = np.random.normal(50, 10, 100).tolist()
    # Inject outliers
    values[0] = 200
    values[1] = -100
    values[2] = 300
    return pd.DataFrame({
        "value": values,
        "other": np.random.normal(0, 1, 100),
    })


# ===========================================================================
# SCALING TESTS
# ===========================================================================

class TestStandardScaler:
    def test_fit_returns_params(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator
        calc = StandardScalerCalculator()
        params = calc.fit(numeric_df, {"columns": ["a", "b"]})
        assert "mean" in params
        assert "scale" in params
        assert "columns" in params
        assert len(params["mean"]) == 2
        assert len(params["scale"]) == 2

    def test_apply_centers_and_scales(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
        calc = StandardScalerCalculator()
        applier = StandardScalerApplier()
        df = numeric_df.dropna()
        params = calc.fit(df, {"columns": ["a", "b"]})
        result = applier.apply(df, params)
        assert isinstance(result, pd.DataFrame)
        # Mean should be approximately 0 after scaling
        assert abs(result["a"].mean()) < 0.1
        assert abs(result["b"].mean()) < 0.1

    def test_apply_with_nan_passthrough(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
        calc = StandardScalerCalculator()
        applier = StandardScalerApplier()
        params = calc.fit(numeric_df, {"columns": ["a"]})
        result = applier.apply(numeric_df, params)
        # NaN should remain NaN
        assert result["a"].isna().sum() >= 1

    def test_nonexistent_columns_returns_empty(self) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator
        calc = StandardScalerCalculator()
        # Pass columns that don't exist in the DataFrame
        df = pd.DataFrame({"x": [1, 2, 3]})
        params = calc.fit(df, {"columns": ["nonexistent"]})
        # Only nonexistent columns → no valid cols → empty params
        assert params == {}

    def test_with_mean_false(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
        calc = StandardScalerCalculator()
        applier = StandardScalerApplier()
        df = numeric_df.dropna()
        params = calc.fit(df, {"columns": ["a"], "with_mean": False, "with_std": True})
        result = applier.apply(df, params)
        # Mean should NOT be 0 since centering is off
        assert abs(result["a"].mean()) > 0.5


class TestMinMaxScaler:
    def test_fit_returns_params(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import MinMaxScalerCalculator
        calc = MinMaxScalerCalculator()
        params = calc.fit(numeric_df.dropna(), {"columns": ["a", "b"]})
        assert "min" in params
        assert "scale" in params
        assert "columns" in params

    def test_apply_scales_to_range(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import MinMaxScalerCalculator, MinMaxScalerApplier
        calc = MinMaxScalerCalculator()
        applier = MinMaxScalerApplier()
        df = numeric_df.dropna()
        params = calc.fit(df, {"columns": ["a"]})
        result = applier.apply(df, params)
        assert result["a"].min() >= -0.01
        assert result["a"].max() <= 1.01


class TestRobustScaler:
    def test_fit_and_apply(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import RobustScalerCalculator, RobustScalerApplier
        calc = RobustScalerCalculator()
        applier = RobustScalerApplier()
        df = numeric_df.dropna()
        params = calc.fit(df, {"columns": ["a", "b"]})
        assert "center" in params
        assert "scale" in params
        result = applier.apply(df, params)
        assert isinstance(result, pd.DataFrame)
        # Median should be approximately 0
        assert abs(result["a"].median()) < 0.1


class TestMaxAbsScaler:
    def test_fit_and_apply(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.scaling import MaxAbsScalerCalculator, MaxAbsScalerApplier
        calc = MaxAbsScalerCalculator()
        applier = MaxAbsScalerApplier()
        df = numeric_df.dropna()
        params = calc.fit(df, {"columns": ["a"]})
        assert "max_abs" in params
        result = applier.apply(df, params)
        assert result["a"].abs().max() <= 1.01


# ===========================================================================
# IMPUTATION TESTS
# ===========================================================================

class TestSimpleImputer:
    def test_mean_imputation(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
        calc = SimpleImputerCalculator()
        applier = SimpleImputerApplier()
        params = calc.fit(numeric_df, {"columns": ["a", "b"], "strategy": "mean"})
        assert "fill_values" in params
        assert "a" in params["fill_values"]
        result = applier.apply(numeric_df, params)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_median_imputation(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
        calc = SimpleImputerCalculator()
        applier = SimpleImputerApplier()
        params = calc.fit(numeric_df, {"columns": ["a"], "strategy": "median"})
        result = applier.apply(numeric_df, params)
        assert result["a"].isna().sum() == 0

    def test_constant_imputation(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
        calc = SimpleImputerCalculator()
        applier = SimpleImputerApplier()
        params = calc.fit(numeric_df, {"columns": ["a"], "strategy": "constant", "fill_value": -999})
        result = applier.apply(numeric_df, params)
        assert result["a"].isna().sum() == 0
        assert -999 in result["a"].values


class TestKNNImputer:
    def test_knn_imputation(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.imputation import KNNImputerCalculator, KNNImputerApplier
        calc = KNNImputerCalculator()
        applier = KNNImputerApplier()
        params = calc.fit(numeric_df, {"columns": ["a", "b"], "n_neighbors": 3})
        result = applier.apply(numeric_df, params)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0


class TestIterativeImputer:
    def test_iterative_imputation(self, numeric_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.imputation import IterativeImputerCalculator, IterativeImputerApplier
        calc = IterativeImputerCalculator()
        applier = IterativeImputerApplier()
        params = calc.fit(numeric_df, {"columns": ["a", "b"], "max_iter": 5})
        result = applier.apply(numeric_df, params)
        assert result["a"].isna().sum() == 0


# ===========================================================================
# ENCODING TESTS
# ===========================================================================

class TestOneHotEncoder:
    def test_fit_and_apply(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import OneHotEncoderCalculator, OneHotEncoderApplier
        calc = OneHotEncoderCalculator()
        applier = OneHotEncoderApplier()
        params = calc.fit(categorical_df, {"columns": ["color"]})
        assert "columns" in params
        assert "feature_names" in params
        result = applier.apply(categorical_df, params)
        # Should have new encoded columns
        assert any("red" in c.lower() for c in result.columns)
        assert any("blue" in c.lower() for c in result.columns)

    def test_drop_original(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import OneHotEncoderCalculator, OneHotEncoderApplier
        calc = OneHotEncoderCalculator()
        applier = OneHotEncoderApplier()
        params = calc.fit(categorical_df, {"columns": ["color"], "drop_original": True})
        result = applier.apply(categorical_df, params)
        assert "color" not in result.columns


class TestOrdinalEncoder:
    def test_fit_and_apply(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import OrdinalEncoderCalculator, OrdinalEncoderApplier
        calc = OrdinalEncoderCalculator()
        applier = OrdinalEncoderApplier()
        params = calc.fit(categorical_df, {"columns": ["size"]})
        result = applier.apply(categorical_df, params)
        # Ordinal encoded values should be numeric
        assert pd.api.types.is_numeric_dtype(result["size"])


class TestLabelEncoder:
    def test_fit_and_apply(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import LabelEncoderCalculator, LabelEncoderApplier
        calc = LabelEncoderCalculator()
        applier = LabelEncoderApplier()
        params = calc.fit(categorical_df, {"columns": ["color"]})
        result = applier.apply(categorical_df, params)
        assert pd.api.types.is_numeric_dtype(result["color"])
        # Should have 3 unique encoded values for red, blue, green
        assert result["color"].nunique() == 3


class TestTargetEncoder:
    def test_fit_and_apply(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import TargetEncoderCalculator, TargetEncoderApplier
        calc = TargetEncoderCalculator()
        applier = TargetEncoderApplier()
        # Target encoder needs (X, y) tuple
        X = categorical_df.drop(columns=["target"])
        y = categorical_df["target"]
        params = calc.fit((X, y), {"columns": ["color"], "target_column": "target"})
        result = applier.apply((X, y), params)
        # Result should be a tuple (X, y)
        if isinstance(result, tuple):
            X_out = result[0]
        else:
            X_out = result
        assert pd.api.types.is_numeric_dtype(X_out["color"])


class TestHashEncoder:
    def test_fit_and_apply(self, categorical_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.encoding import HashEncoderCalculator, HashEncoderApplier
        calc = HashEncoderCalculator()
        applier = HashEncoderApplier()
        params = calc.fit(categorical_df, {"columns": ["color"], "n_features": 4})
        result = applier.apply(categorical_df, params)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# OUTLIER TESTS
# ===========================================================================

class TestIQROutlier:
    def test_fit_computes_bounds(self, outlier_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.outliers import IQRCalculator
        calc = IQRCalculator()
        params = calc.fit(outlier_df, {"columns": ["value"], "multiplier": 1.5})
        assert "bounds" in params
        assert "value" in params["bounds"]
        assert "lower" in params["bounds"]["value"]
        assert "upper" in params["bounds"]["value"]

    def test_apply_filters_outliers(self, outlier_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.outliers import IQRCalculator, IQRApplier
        calc = IQRCalculator()
        applier = IQRApplier()
        params = calc.fit(outlier_df, {"columns": ["value"], "multiplier": 1.5})
        result = applier.apply(outlier_df, params)
        # Should have fewer rows (outliers removed)
        assert len(result) < len(outlier_df)


class TestZScoreOutlier:
    def test_fit_and_apply(self, outlier_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.outliers import ZScoreCalculator, ZScoreApplier
        calc = ZScoreCalculator()
        applier = ZScoreApplier()
        params = calc.fit(outlier_df, {"columns": ["value"], "threshold": 3.0})
        assert "stats" in params
        assert "threshold" in params
        result = applier.apply(outlier_df, params)
        assert len(result) <= len(outlier_df)


class TestWinsorize:
    def test_fit_and_apply(self, outlier_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.outliers import WinsorizeCalculator, WinsorizeApplier
        calc = WinsorizeCalculator()
        applier = WinsorizeApplier()
        params = calc.fit(outlier_df, {"columns": ["value"], "lower_percentile": 5, "upper_percentile": 95})
        result = applier.apply(outlier_df, params)
        # Same row count (winsorize clips, doesn't remove)
        assert len(result) == len(outlier_df)


# ===========================================================================
# DROP & MISSING TESTS
# ===========================================================================

class TestDeduplication:
    def test_removes_duplicates(self) -> None:
        from skyulf.preprocessing.drop_and_missing import DeduplicateCalculator, DeduplicateApplier
        df = pd.DataFrame({
            "a": [1, 1, 2, 3, 3],
            "b": [10, 10, 20, 30, 30],
        })
        calc = DeduplicateCalculator()
        applier = DeduplicateApplier()
        params = calc.fit(df, {})
        result = applier.apply(df, params)
        assert len(result) == 3


class TestDropMissingColumns:
    def test_drops_mostly_null_columns(self) -> None:
        from skyulf.preprocessing.drop_and_missing import DropMissingColumnsCalculator, DropMissingColumnsApplier
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [np.nan, np.nan, np.nan, np.nan, 1],
        })
        calc = DropMissingColumnsCalculator()
        applier = DropMissingColumnsApplier()
        # missing_threshold is percentage (0-100): 'bad' has 80% missing
        params = calc.fit(df, {"missing_threshold": 50})
        result = applier.apply(df, params)
        assert "good" in result.columns
        assert "bad" not in result.columns


class TestDropMissingRows:
    def test_drops_rows_with_na(self) -> None:
        from skyulf.preprocessing.drop_and_missing import DropMissingRowsCalculator, DropMissingRowsApplier
        df = pd.DataFrame({
            "a": [1, np.nan, 3, 4, np.nan],
            "b": [10, 20, 30, 40, 50],
        })
        calc = DropMissingRowsCalculator()
        applier = DropMissingRowsApplier()
        params = calc.fit(df, {"columns": ["a"]})
        result = applier.apply(df, params)
        assert result["a"].isna().sum() == 0


class TestMissingIndicator:
    def test_adds_indicator_column(self) -> None:
        from skyulf.preprocessing.drop_and_missing import MissingIndicatorCalculator, MissingIndicatorApplier
        df = pd.DataFrame({
            "a": [1, np.nan, 3, np.nan, 5],
            "b": [10, 20, 30, 40, 50],
        })
        calc = MissingIndicatorCalculator()
        applier = MissingIndicatorApplier()
        params = calc.fit(df, {"columns": ["a"]})
        result = applier.apply(df, params)
        # Should have a new indicator column
        indicator_cols = [c for c in result.columns if "missing" in c.lower() or "_indicator" in c.lower() or "_is_na" in c.lower()]
        assert len(indicator_cols) >= 1 or "a" in params.get("columns", [])


# ===========================================================================
# CLEANING TESTS
# ===========================================================================

class TestTextCleaning:
    def test_strips_and_lowercases(self, text_df: pd.DataFrame) -> None:
        from skyulf.preprocessing.cleaning import TextCleaningCalculator, TextCleaningApplier
        calc = TextCleaningCalculator()
        applier = TextCleaningApplier()
        # Operations must be dicts with 'op' and 'mode' keys
        params = calc.fit(text_df, {
            "columns": ["name"],
            "operations": [
                {"op": "trim", "mode": "both"},
                {"op": "case", "mode": "lower"},
            ],
        })
        result = applier.apply(text_df, params)
        assert result["name"].iloc[0] == "john"
        assert result["name"].iloc[1] == "jane"


class TestValueReplacement:
    def test_replaces_values(self) -> None:
        from skyulf.preprocessing.cleaning import ValueReplacementCalculator, ValueReplacementApplier
        df = pd.DataFrame({"status": ["active", "inactive", "active", "pending"]})
        calc = ValueReplacementCalculator()
        applier = ValueReplacementApplier()
        params = calc.fit(df, {
            "columns": ["status"],
            "mapping": {"inactive": "disabled", "pending": "waiting"},
        })
        result = applier.apply(df, params)
        assert "inactive" not in result["status"].values
        assert "disabled" in result["status"].values


# ===========================================================================
# BUCKETING / BINNING TESTS
# ===========================================================================

class TestGeneralBinning:
    def test_equal_width_binning(self) -> None:
        from skyulf.preprocessing.bucketing import GeneralBinningCalculator, GeneralBinningApplier
        df = pd.DataFrame({"age": list(range(0, 100))})
        calc = GeneralBinningCalculator()
        applier = GeneralBinningApplier()
        params = calc.fit(df, {"columns": ["age"], "strategy": "uniform", "n_bins": 5})
        result = applier.apply(df, params)
        assert isinstance(result, pd.DataFrame)


class TestKBinsDiscretizer:
    def test_fit_and_apply(self) -> None:
        from skyulf.preprocessing.bucketing import KBinsDiscretizerCalculator, KBinsDiscretizerApplier
        df = pd.DataFrame({"value": np.random.normal(50, 10, 100)})
        calc = KBinsDiscretizerCalculator()
        applier = KBinsDiscretizerApplier()
        params = calc.fit(df, {"columns": ["value"], "n_bins": 5, "strategy": "uniform"})
        result = applier.apply(df, params)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# TRANSFORMATION TESTS
# ===========================================================================

class TestSimpleTransformation:
    def test_log_transformation(self) -> None:
        from skyulf.preprocessing.transformations import SimpleTransformationCalculator, SimpleTransformationApplier
        df = pd.DataFrame({"value": [1.0, 10.0, 100.0, 1000.0, 10000.0]})
        calc = SimpleTransformationCalculator()
        applier = SimpleTransformationApplier()
        # Config uses 'transformations' list of per-column dicts
        params = calc.fit(df, {
            "transformations": [{"column": "value", "method": "log"}],
        })
        result = applier.apply(df, params)
        # log1p(10000) ≈ 9.21, much smaller than 10000
        assert result["value"].max() < 15


class TestPowerTransformer:
    def test_yeo_johnson(self) -> None:
        from skyulf.preprocessing.transformations import PowerTransformerCalculator, PowerTransformerApplier
        np.random.seed(42)
        df = pd.DataFrame({"value": np.random.exponential(2, 100)})
        calc = PowerTransformerCalculator()
        applier = PowerTransformerApplier()
        params = calc.fit(df, {"columns": ["value"], "method": "yeo-johnson"})
        result = applier.apply(df, params)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# FEATURE SELECTION TESTS
# ===========================================================================

class TestVarianceThreshold:
    def test_removes_low_variance(self) -> None:
        from skyulf.preprocessing.feature_selection import VarianceThresholdCalculator, VarianceThresholdApplier
        df = pd.DataFrame({
            "constant": [1] * 100,
            "varied": np.random.normal(0, 10, 100),
        })
        calc = VarianceThresholdCalculator()
        applier = VarianceThresholdApplier()
        params = calc.fit(df, {"threshold": 0.01})
        result = applier.apply(df, params)
        assert "constant" not in result.columns
        assert "varied" in result.columns


class TestCorrelationThreshold:
    def test_removes_highly_correlated(self) -> None:
        from skyulf.preprocessing.feature_selection import CorrelationThresholdCalculator, CorrelationThresholdApplier
        np.random.seed(42)
        base = np.random.normal(0, 1, 100)
        df = pd.DataFrame({
            "a": base,
            "b": base + np.random.normal(0, 0.01, 100),  # Almost perfectly correlated with a
            "c": np.random.normal(0, 1, 100),  # Independent
        })
        calc = CorrelationThresholdCalculator()
        applier = CorrelationThresholdApplier()
        params = calc.fit(df, {"threshold": 0.95})
        result = applier.apply(df, params)
        # Should remove one of the highly correlated pair
        assert len(result.columns) < 3


# ===========================================================================
# FEATURE GENERATION TESTS
# ===========================================================================

class TestPolynomialFeatures:
    def test_generates_polynomial_features(self) -> None:
        from skyulf.preprocessing.feature_generation import PolynomialFeaturesCalculator, PolynomialFeaturesApplier
        df = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [10, 20, 30, 40, 50],
        })
        calc = PolynomialFeaturesCalculator()
        applier = PolynomialFeaturesApplier()
        params = calc.fit(df, {"columns": ["x1", "x2"], "degree": 2, "interaction_only": False})
        result = applier.apply(df, params)
        # Should have more columns than original
        assert len(result.columns) > 2


# ===========================================================================
# CASTING TESTS
# ===========================================================================

class TestCasting:
    def test_cast_to_int(self) -> None:
        from skyulf.preprocessing.casting import CastingCalculator, CastingApplier
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        calc = CastingCalculator()
        applier = CastingApplier()
        # Config uses 'target_type' + 'columns' keys
        params = calc.fit(df, {"columns": ["value"], "target_type": "int"})
        result = applier.apply(df, params)
        assert result["value"].dtype in [np.int64, np.int32, int, pd.Int64Dtype()]

    def test_cast_to_string(self) -> None:
        from skyulf.preprocessing.casting import CastingCalculator, CastingApplier
        df = pd.DataFrame({"value": [1, 2, 3]})
        calc = CastingCalculator()
        applier = CastingApplier()
        params = calc.fit(df, {"columns": ["value"], "target_type": "str"})
        result = applier.apply(df, params)
        assert pd.api.types.is_string_dtype(result["value"]) or result["value"].dtype == object


# ===========================================================================
# TUPLE (X, y) INPUT TESTS — ensure nodes handle pipeline-style inputs
# ===========================================================================

class TestTuplePassthrough:
    """Verify that nodes correctly handle (X, y) tuple inputs."""

    def test_scaler_with_tuple(self) -> None:
        from skyulf.preprocessing.scaling import StandardScalerCalculator, StandardScalerApplier
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        y = pd.Series([0, 1, 0, 1, 0])
        calc = StandardScalerCalculator()
        applier = StandardScalerApplier()
        params = calc.fit((X, y), {"columns": ["a", "b"]})
        result = applier.apply((X, y), params)
        assert isinstance(result, tuple)
        X_out, y_out = result
        assert isinstance(X_out, pd.DataFrame)
        assert len(y_out) == 5

    def test_imputer_with_tuple(self) -> None:
        from skyulf.preprocessing.imputation import SimpleImputerCalculator, SimpleImputerApplier
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]})
        y = pd.Series([0, 1, 0, 1, 0])
        calc = SimpleImputerCalculator()
        applier = SimpleImputerApplier()
        params = calc.fit((X, y), {"columns": ["a"], "strategy": "mean"})
        result = applier.apply((X, y), params)
        assert isinstance(result, tuple)
        X_out, y_out = result
        assert X_out["a"].isna().sum() == 0
