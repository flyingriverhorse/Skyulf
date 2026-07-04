"""Unit tests for Oversampling and Undersampling resampling nodes.

Covers: fit artifact, apply with imbalanced data, helper functions,
edge cases (balanced input, 1-sample class, non-numeric columns, missing target).
"""

import importlib.util
from typing import Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Skip entire module if imbalanced-learn is not installed.
pytest.importorskip(
    "imblearn", reason="imbalanced-learn not installed — pip install imbalanced-learn"
)

from skyulf.preprocessing.resampling import (
    OversamplingApplier,
    OversamplingCalculator,
    UndersamplingApplier,
    UndersamplingCalculator,
    _extract_y_pandas,
    _extract_y_polars,
    _finalize_resampled,
    _to_pandas_y,
    _validate_numeric,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def imbalanced_pandas() -> Tuple[pd.DataFrame, pd.Series]:
    """Imbalanced binary dataset: 80 majority / 20 minority samples."""
    rng = np.random.default_rng(0)
    n_maj, n_min = 80, 20
    X = pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(5, 1, n_min)]),
            "f2": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(5, 1, n_min)]),
        }
    )
    y = pd.Series([0] * n_maj + [1] * n_min, name="target")
    return X, y


@pytest.fixture
def balanced_pandas() -> Tuple[pd.DataFrame, pd.Series]:
    """Already-balanced binary dataset: 50 samples per class."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 100), "f2": rng.normal(0, 1, 100)})
    y = pd.Series([0] * 50 + [1] * 50, name="target")
    return X, y


@pytest.fixture
def imbalanced_with_target_col() -> pd.DataFrame:
    """Imbalanced dataset with the target embedded as a column ('label')."""
    rng = np.random.default_rng(2)
    n_maj, n_min = 60, 15
    return pd.DataFrame(
        {
            "f1": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(5, 1, n_min)]),
            "f2": np.concatenate([rng.normal(0, 1, n_maj), rng.normal(5, 1, n_min)]),
            "label": [0] * n_maj + [1] * n_min,
        }
    )


# ---------------------------------------------------------------------------
# Oversampling — fit()
# ---------------------------------------------------------------------------


class TestOversamplingFit:
    def test_fit_returns_artifact_type(self, imbalanced_pandas: Any) -> None:
        """OversamplingCalculator.fit() must return an artifact with type='oversampling'."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {})
        assert art["type"] == "oversampling"

    def test_fit_stores_method(self, imbalanced_pandas: Any) -> None:
        """Configured method must be persisted in the artifact."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote"})
        assert art["method"] == "smote"

    def test_fit_stores_defaults(self, imbalanced_pandas: Any) -> None:
        """Default params (random_state=42, k_neighbors=5, sampling_strategy='auto') are persisted."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {})
        assert art["random_state"] == 42
        assert art["k_neighbors"] == 5
        assert art["sampling_strategy"] == "auto"

    def test_fit_respects_custom_params(self, imbalanced_pandas: Any) -> None:
        """Custom config values must override defaults in the artifact."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit(
            (X, y), {"method": "smote", "random_state": 7, "k_neighbors": 3}
        )
        assert art["random_state"] == 7
        assert art["k_neighbors"] == 3


# ---------------------------------------------------------------------------
# Oversampling — apply()
# ---------------------------------------------------------------------------


class TestOversamplingApply:
    def test_smote_increases_minority_count(self, imbalanced_pandas: Any) -> None:
        """SMOTE must increase minority class count to match (or approach) majority."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 3})
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        counts = y_res.value_counts()
        # After oversampling with strategy='auto', minority should equal majority
        assert counts[1] >= 20  # at minimum, minority can only grow
        assert counts[1] >= counts[0] * 0.9  # approximately balanced

    def test_smote_preserves_feature_columns(self, imbalanced_pandas: Any) -> None:
        """SMOTE must not add or remove feature columns."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 3})
        X_res, _ = OversamplingApplier().apply((X, y), art)
        assert list(X_res.columns) == list(X.columns)

    def test_smote_result_is_larger_than_input(self, imbalanced_pandas: Any) -> None:
        """Oversampling must strictly increase total row count when data is imbalanced."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 3})
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        assert len(X_res) > len(X)
        assert len(y_res) == len(X_res)

    def test_smote_target_column_extraction(self, imbalanced_with_target_col: Any) -> None:
        """SMOTE works when target is embedded in the DataFrame via target_column param."""
        df = imbalanced_with_target_col
        art = OversamplingCalculator().fit(
            df, {"method": "smote", "target_column": "label", "k_neighbors": 3}
        )
        result = OversamplingApplier().apply(df, art)
        # result is a plain DataFrame (not a tuple) because input was not a tuple
        assert isinstance(result, pd.DataFrame)
        # label column should still be present (extracted and reattached internally)
        # but the shape should be larger than original
        assert len(result) > len(df)

    def test_smote_already_balanced_data_no_change(self, balanced_pandas: Any) -> None:
        """SMOTE on already-balanced data should not change total row count significantly."""
        X, y = balanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 3})
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        # Already balanced → no rows added
        assert len(X_res) == len(X)
        assert y_res.value_counts()[0] == y_res.value_counts()[1]

    def test_smote_unknown_method_returns_unchanged(self, imbalanced_pandas: Any) -> None:
        """An unknown method name must return the data unchanged (no sampler built)."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote"})
        art["method"] = "totally_unknown_method"
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        assert len(X_res) == len(X)

    def test_smote_non_numeric_raises(self) -> None:
        """Non-numeric feature columns must raise ValueError before reaching imblearn."""
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
        y = pd.Series([0, 0, 1, 1], name="target")
        art = OversamplingCalculator().fit((X, y), {"method": "smote"})
        with pytest.raises(ValueError, match="non-numeric"):
            OversamplingApplier().apply((X, y), art)

    def test_smote_missing_target_returns_unchanged(self, imbalanced_pandas: Any) -> None:
        """If target cannot be resolved (no y and no target_column), apply returns unchanged."""
        X, _ = imbalanced_pandas
        # Pass X alone (no y, no target_column param)
        art = OversamplingCalculator().fit(X, {"method": "smote"})
        result = OversamplingApplier().apply(X, art)
        pd.testing.assert_frame_equal(result, X)

    def test_smote_polars_input_returns_polars(self, imbalanced_pandas: Any) -> None:
        """Polars input must return polars output after SMOTE (via pandas round-trip)."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 3})
        pl_X = pl.from_pandas(X)
        pl_y = pl.from_pandas(y.to_frame()).to_series()
        X_res, y_res = OversamplingApplier().apply((pl_X, pl_y), art)
        assert isinstance(X_res, pl.DataFrame)


# ---------------------------------------------------------------------------
# Undersampling — fit()
# ---------------------------------------------------------------------------


class TestUndersamplingFit:
    def test_fit_returns_artifact_type(self, imbalanced_pandas: Any) -> None:
        """UndersamplingCalculator.fit() must return artifact with type='undersampling'."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {})
        assert art["type"] == "undersampling"

    def test_fit_stores_method(self, imbalanced_pandas: Any) -> None:
        """Configured method must be persisted in the artifact."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        assert art["method"] == "random_under_sampling"

    def test_fit_stores_defaults(self, imbalanced_pandas: Any) -> None:
        """Default random_state and replacement must be stored."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {})
        assert art["random_state"] == 42
        assert art["replacement"] is False


# ---------------------------------------------------------------------------
# Undersampling — apply()
# ---------------------------------------------------------------------------


class TestUndersamplingApply:
    def test_random_undersampling_reduces_majority(self, imbalanced_pandas: Any) -> None:
        """RandomUnderSampler must reduce majority class count to match minority."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        counts = y_res.value_counts()
        # Both classes must now have equal counts
        assert counts[0] == counts[1]

    def test_random_undersampling_shrinks_dataset(self, imbalanced_pandas: Any) -> None:
        """Undersampling must strictly reduce total row count when data is imbalanced."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        assert len(X_res) < len(X)
        assert len(y_res) == len(X_res)

    def test_random_undersampling_preserves_columns(self, imbalanced_pandas: Any) -> None:
        """Undersampling must not add or remove feature columns."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        X_res, _ = UndersamplingApplier().apply((X, y), art)
        assert list(X_res.columns) == list(X.columns)

    def test_random_undersampling_already_balanced(self, balanced_pandas: Any) -> None:
        """Undersampling already-balanced data must leave class counts unchanged."""
        X, y = balanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        assert y_res.value_counts()[0] == y_res.value_counts()[1]

    def test_undersampling_target_column_extraction(self, imbalanced_with_target_col: Any) -> None:
        """RandomUnderSampler works when target is embedded via target_column param."""
        df = imbalanced_with_target_col
        art = UndersamplingCalculator().fit(
            df, {"method": "random_under_sampling", "target_column": "label"}
        )
        result = UndersamplingApplier().apply(df, art)
        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(df)

    def test_undersampling_unknown_method_returns_unchanged(self, imbalanced_pandas: Any) -> None:
        """An unknown method name must return data unchanged (no sampler built)."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        art["method"] = "unknown_sampler"
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        assert len(X_res) == len(X)

    def test_undersampling_non_numeric_raises(self) -> None:
        """Non-numeric feature columns must raise ValueError before reaching imblearn."""
        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "cat": ["a", "b"] * 3})
        y = pd.Series([0, 0, 0, 0, 1, 1], name="target")
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        with pytest.raises(ValueError, match="non-numeric"):
            UndersamplingApplier().apply((X, y), art)

    def test_undersampling_missing_target_returns_unchanged(self, imbalanced_pandas: Any) -> None:
        """If target cannot be resolved, apply must return the frame unchanged."""
        X, _ = imbalanced_pandas
        art = UndersamplingCalculator().fit(X, {"method": "random_under_sampling"})
        result = UndersamplingApplier().apply(X, art)
        pd.testing.assert_frame_equal(result, X)

    def test_undersampling_polars_input_returns_polars(self, imbalanced_pandas: Any) -> None:
        """Polars input must return polars output after undersampling."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        pl_X = pl.from_pandas(X)
        pl_y = pl.from_pandas(y.to_frame()).to_series()
        X_res, y_res = UndersamplingApplier().apply((pl_X, pl_y), art)
        assert isinstance(X_res, pl.DataFrame)

    def test_undersampling_result_y_is_series(self, imbalanced_pandas: Any) -> None:
        """The resampled y must be a pandas Series with the original name."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        _, y_res = UndersamplingApplier().apply((X, y), art)
        assert isinstance(y_res, pd.Series)
        assert y_res.name == "target"


# ---------------------------------------------------------------------------
# Edge case: single-sample minority class
# ---------------------------------------------------------------------------


class TestSingleSampleMinorityClass:
    def test_smote_with_one_minority_sample_raises(self) -> None:
        """SMOTE requires at least k_neighbors+1 minority samples; 1-sample class must raise."""
        rng = np.random.default_rng(99)
        X = pd.DataFrame({"f1": rng.normal(0, 1, 11), "f2": rng.normal(0, 1, 11)})
        y = pd.Series([0] * 10 + [1], name="target")  # only 1 minority sample
        art = OversamplingCalculator().fit((X, y), {"method": "smote", "k_neighbors": 5})
        with pytest.raises(Exception):
            # imblearn raises ValueError when k_neighbors >= minority sample count
            OversamplingApplier().apply((X, y), art)

    def test_random_undersampling_with_one_minority_sample(self) -> None:
        """RandomUnderSampler with 1 minority sample should return 1 sample per class."""
        rng = np.random.default_rng(77)
        X = pd.DataFrame({"f1": rng.normal(0, 1, 11), "f2": rng.normal(0, 1, 11)})
        y = pd.Series([0] * 10 + [1], name="target")
        art = UndersamplingCalculator().fit((X, y), {"method": "random_under_sampling"})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        assert y_res.value_counts()[0] == 1
        assert y_res.value_counts()[1] == 1


# ---------------------------------------------------------------------------
# Shared helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_extract_y_pandas_from_tuple(self) -> None:
        """_extract_y_pandas returns X and y unchanged when y is already provided."""
        X = pd.DataFrame({"f": [1.0, 2.0]})
        y = pd.Series([0, 1])
        X_out, y_out = _extract_y_pandas(X, y, target_col="f")
        pd.testing.assert_frame_equal(X_out, X)
        pd.testing.assert_series_equal(y_out, y)

    def test_extract_y_pandas_from_column(self) -> None:
        """_extract_y_pandas extracts y from the DataFrame when y is None."""
        df = pd.DataFrame({"f": [1.0, 2.0], "label": [0, 1]})
        X_out, y_out = _extract_y_pandas(df, None, target_col="label")
        assert "label" not in X_out.columns
        pd.testing.assert_series_equal(y_out, df["label"])

    def test_extract_y_pandas_missing_column_returns_none_y(self) -> None:
        """_extract_y_pandas returns (X, None) when target_col is absent from the frame."""
        df = pd.DataFrame({"f": [1.0, 2.0]})
        X_out, y_out = _extract_y_pandas(df, None, target_col="no_such_col")
        assert y_out is None
        pd.testing.assert_frame_equal(X_out, df)

    def test_extract_y_polars_from_column(self) -> None:
        """_extract_y_polars extracts y from a polars DataFrame when y is None."""
        df = pl.DataFrame({"f": [1.0, 2.0], "label": [0, 1]})
        X_out, y_out = _extract_y_polars(df, None, target_col="label")
        assert "label" not in X_out.columns
        assert y_out is not None

    def test_extract_y_polars_no_target_col(self) -> None:
        """_extract_y_polars returns (X, None) when target_col is not in the frame."""
        df = pl.DataFrame({"f": [1.0, 2.0]})
        X_out, y_out = _extract_y_polars(df, None, target_col="missing")
        assert y_out is None

    def test_to_pandas_y_from_polars_series(self) -> None:
        """_to_pandas_y must convert a polars Series to a pandas Series."""
        pl_y = pl.Series("target", [0, 1, 0])
        pd_y = _to_pandas_y(pl_y)
        assert isinstance(pd_y, pd.Series)
        np.testing.assert_array_equal(pd_y.values, [0, 1, 0])

    def test_to_pandas_y_none_returns_none(self) -> None:
        """_to_pandas_y(None) must return None."""
        assert _to_pandas_y(None) is None

    def test_to_pandas_y_already_pandas(self) -> None:
        """_to_pandas_y on an already-pandas Series returns it unchanged."""
        y = pd.Series([1, 2, 3])
        result = _to_pandas_y(y)
        pd.testing.assert_series_equal(result, y)

    def test_validate_numeric_passes_on_numeric(self) -> None:
        """_validate_numeric must not raise on all-numeric DataFrames."""
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        _validate_numeric(df)  # should not raise

    def test_validate_numeric_raises_on_categorical(self) -> None:
        """_validate_numeric must raise ValueError when non-numeric columns are present."""
        df = pd.DataFrame({"f1": [1.0, 2.0], "cat": ["a", "b"]})
        with pytest.raises(ValueError, match="non-numeric"):
            _validate_numeric(df)

    def test_finalize_resampled_wraps_numpy_arrays(self) -> None:
        """_finalize_resampled must wrap raw numpy arrays into DataFrame/Series."""
        X_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_np = np.array([0, 1])
        columns = pd.Index(["a", "b"])
        X_df, y_s = _finalize_resampled(X_np, y_np, columns, "target")
        assert isinstance(X_df, pd.DataFrame)
        assert isinstance(y_s, pd.Series)
        assert list(X_df.columns) == ["a", "b"]
        assert y_s.name == "target"

    def test_finalize_resampled_preserves_existing(self) -> None:
        """_finalize_resampled must return existing DataFrame/Series instances unchanged."""
        X_df = pd.DataFrame({"a": [1.0]})
        y_s = pd.Series([0], name="lbl")
        X_out, y_out = _finalize_resampled(X_df, y_s, X_df.columns, "fallback")
        pd.testing.assert_frame_equal(X_out, X_df)
        pd.testing.assert_series_equal(y_out, y_s)


# ---------------------------------------------------------------------------
# NearMiss and TomekLinks undersampling (additional sampler builder paths)
# ---------------------------------------------------------------------------


class TestNearMissUndersampling:
    def test_nearmiss_reduces_majority(self, imbalanced_pandas: Any) -> None:
        """NearMiss must balance classes by removing majority samples."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "nearmiss", "version": 1})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        counts = y_res.value_counts()
        assert counts[0] == counts[1]
        assert len(X_res) < len(X)


class TestTomekLinksUndersampling:
    def test_tomek_links_does_not_raise(self, imbalanced_pandas: Any) -> None:
        """TomekLinks must complete without error on imbalanced data."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit((X, y), {"method": "tomek_links"})
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        # TomekLinks only removes borderline pairs — row count may or may not change
        assert len(X_res) <= len(X)
        assert len(y_res) == len(X_res)

    def test_edited_nearest_neighbours_does_not_raise(self, imbalanced_pandas: Any) -> None:
        """EditedNearestNeighbours must complete without raising on well-separated classes."""
        X, y = imbalanced_pandas
        art = UndersamplingCalculator().fit(
            (X, y), {"method": "edited_nearest_neighbours", "n_neighbors": 3}
        )
        X_res, y_res = UndersamplingApplier().apply((X, y), art)
        assert len(y_res) == len(X_res)


# ---------------------------------------------------------------------------
# Additional oversampler builder paths (ADASYN, BorderlineSMOTE)
# ---------------------------------------------------------------------------


class TestAdditionalOversamplers:
    def test_adasyn_balances_classes(self) -> None:
        """ADASYN must increase minority class count on overlapping class data."""
        # ADASYN requires class overlap — perfectly separated data triggers a RuntimeError.
        rng = np.random.default_rng(5)
        n_maj, n_min = 60, 20
        X = pd.DataFrame(
            {
                "f1": np.concatenate([rng.normal(0, 2, n_maj), rng.normal(1, 2, n_min)]),
                "f2": np.concatenate([rng.normal(0, 2, n_maj), rng.normal(1, 2, n_min)]),
            }
        )
        y = pd.Series([0] * n_maj + [1] * n_min, name="target")
        art = OversamplingCalculator().fit((X, y), {"method": "adasyn", "k_neighbors": 3})
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        # ADASYN generates synthetic samples — minority class must grow
        assert y_res.value_counts()[1] > n_min

    def test_borderline_smote_balances_classes(self, imbalanced_pandas: Any) -> None:
        """BorderlineSMOTE must balance classes without error on well-separated data."""
        X, y = imbalanced_pandas
        art = OversamplingCalculator().fit(
            (X, y),
            {"method": "borderline_smote", "k_neighbors": 3, "m_neighbors": 5},
        )
        X_res, y_res = OversamplingApplier().apply((X, y), art)
        assert y_res.value_counts()[1] >= 20
