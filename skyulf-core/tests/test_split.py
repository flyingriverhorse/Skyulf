"""Unit tests for the DataSplitter / SplitApplier / FeatureTargetSplit nodes.

Covers: DataSplitter.split and split_xy row counts and proportions,
validation-set carve-out, stratification (including the "too few members"
fallback), reproducibility with a fixed random_state, the SplitApplier /
SplitCalculator node contract (returns SplitDataset), and
FeatureTargetSplitApplier/Calculator.
"""

import typing
from typing import Any, Dict

import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.split import (
    DataSplitter,
    FeatureTargetSplitApplier,
    FeatureTargetSplitCalculator,
    SplitApplier,
    SplitCalculator,
    _safe_stratify,
)


def _frame(n: int = 100) -> pd.DataFrame:
    """Build a simple 100-row DataFrame with a numeric feature and a target."""
    return pd.DataFrame({"feature": range(n), "target": [i % 2 for i in range(n)]})


# ---------------------------------------------------------------------------
# DataSplitter.split — row counts / proportions
# ---------------------------------------------------------------------------


def test_split_train_test_row_counts() -> None:
    """A 100-row frame split 80/20 must yield 80 train rows and 20 test rows."""
    df = _frame(100)
    splitter = DataSplitter(test_size=0.2, random_state=42)
    result = splitter.split(df)
    assert isinstance(result, SplitDataset)
    assert len(result.train) == 80
    assert len(result.test) == 20
    assert result.validation is None


def test_split_with_validation_carves_correct_proportions() -> None:
    """test_size=0.2 and validation_size=0.1 of a 100-row frame yields 70/20/10."""
    df = _frame(100)
    splitter = DataSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    result = splitter.split(df)
    assert result.validation is not None
    assert len(result.test) == 20
    assert len(result.validation) == 10
    assert len(result.train) == 70


def test_split_no_row_overlap_between_train_test_validation() -> None:
    """Train/test/validation partitions must be disjoint and cover all rows."""
    df = _frame(100)
    splitter = DataSplitter(test_size=0.3, validation_size=0.2, random_state=1)
    result = splitter.split(df)
    assert isinstance(result.train, pd.DataFrame)
    assert isinstance(result.test, pd.DataFrame)
    assert isinstance(result.validation, pd.DataFrame)
    train_idx = set(result.train.index)
    test_idx = set(result.test.index)
    val_idx = set(result.validation.index)
    assert train_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(val_idx)
    assert test_idx.isdisjoint(val_idx)
    assert train_idx | test_idx | val_idx == set(df.index)


def test_split_reproducible_with_fixed_random_state() -> None:
    """Two splits with the same random_state must produce identical partitions."""
    df = _frame(100)
    r1 = DataSplitter(test_size=0.25, random_state=7).split(df)
    r2 = DataSplitter(test_size=0.25, random_state=7).split(df)
    assert isinstance(r1.train, pd.DataFrame)
    assert isinstance(r2.train, pd.DataFrame)
    assert isinstance(r1.test, pd.DataFrame)
    assert isinstance(r2.test, pd.DataFrame)
    pd.testing.assert_frame_equal(r1.train, r2.train)
    pd.testing.assert_frame_equal(r1.test, r2.test)


def test_split_different_random_state_yields_different_partition() -> None:
    """Different random_state values must (almost certainly) yield different splits."""
    df = _frame(100)
    r1 = DataSplitter(test_size=0.25, random_state=1).split(df)
    r2 = DataSplitter(test_size=0.25, random_state=2).split(df)
    assert isinstance(r1.test, pd.DataFrame)
    assert isinstance(r2.test, pd.DataFrame)
    assert list(r1.test.index) != list(r2.test.index)


def test_split_shuffle_false_is_sequential() -> None:
    """shuffle=False must produce a contiguous, ordered train/test split."""
    df = _frame(10)
    splitter = DataSplitter(test_size=0.3, shuffle=False, random_state=42)
    result = splitter.split(df)
    # sklearn keeps original order when shuffle=False: train = first 7, test = last 3.
    assert isinstance(result.train, pd.DataFrame)
    assert isinstance(result.test, pd.DataFrame)
    assert list(result.train["feature"]) == list(range(7))
    assert list(result.test["feature"]) == list(range(7, 10))


# ---------------------------------------------------------------------------
# DataSplitter.split — stratification
# ---------------------------------------------------------------------------


def test_split_stratified_preserves_class_ratio() -> None:
    """Stratified split on a balanced target must preserve the 50/50 class ratio."""
    df = _frame(100)
    splitter = DataSplitter(test_size=0.2, random_state=42, stratify_col="target")
    result = splitter.split(df)
    assert isinstance(result.test, pd.DataFrame)
    test_ratio = result.test["target"].mean()
    assert test_ratio == pytest.approx(0.5, abs=0.05)


def test_safe_stratify_disables_when_class_too_small() -> None:
    """_safe_stratify must return None if any class has fewer than 2 members."""
    y = pd.Series([0, 0, 0, 1])  # class 1 has only 1 member
    assert _safe_stratify(y, "test") is None


def test_safe_stratify_keeps_y_when_all_classes_sufficient() -> None:
    """_safe_stratify must return y unchanged when every class has >= 2 members."""
    y = pd.Series([0, 0, 1, 1])
    result = _safe_stratify(y, "test")
    assert result is y


def test_safe_stratify_none_input_returns_none() -> None:
    """_safe_stratify must pass through None without raising."""
    assert _safe_stratify(None, "test") is None


def test_split_stratify_falls_back_gracefully_for_rare_class(caplog: Any) -> None:
    """A target with a singleton class must not raise; stratification is disabled."""
    df = pd.DataFrame({"feature": range(10), "target": [0] * 9 + [1]})
    splitter = DataSplitter(test_size=0.3, random_state=42, stratify_col="target")
    result = splitter.split(df)  # Should not raise despite the rare class.
    assert len(result.train) + len(result.test) == 10


# ---------------------------------------------------------------------------
# DataSplitter.split_xy
# ---------------------------------------------------------------------------


def test_split_xy_row_counts_and_shapes() -> None:
    """split_xy on (X, y) must produce matching row counts for X and y in each split."""
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([i % 2 for i in range(100)])
    splitter = DataSplitter(test_size=0.2, random_state=42)
    result = splitter.split_xy(X, y)
    X_train, y_train = result.train
    X_test, y_test = result.test
    assert len(X_train) == len(y_train) == 80
    assert len(X_test) == len(y_test) == 20


def test_split_xy_with_validation() -> None:
    """split_xy with validation_size must carve out the correct validation size."""
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([i % 2 for i in range(100)])
    splitter = DataSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    result = splitter.split_xy(X, y)
    assert result.validation is not None
    X_val, y_val = result.validation
    assert len(X_val) == len(y_val) == 10


# ---------------------------------------------------------------------------
# SplitCalculator / SplitApplier — node contract
# ---------------------------------------------------------------------------


def test_split_calculator_fit_stores_config_params() -> None:
    """SplitCalculator.fit must copy only known split-related config keys."""
    df = _frame(10)
    params = SplitCalculator().fit(df, {"test_size": 0.3, "random_state": 5, "extra": "ignored"})
    assert params["test_size"] == 0.3
    assert params["random_state"] == 5
    assert "extra" not in params
    assert params["type"] == "split"


def test_split_applier_returns_split_dataset_for_frame() -> None:
    """SplitApplier.apply on a plain DataFrame must return a SplitDataset."""
    df = _frame(50)
    params: Dict[str, Any] = {"test_size": 0.2, "random_state": 42}
    result = SplitApplier().apply(df, params)
    assert isinstance(result, SplitDataset)
    assert len(result.train) == 40
    assert len(result.test) == 10


def test_split_applier_splits_target_column_when_configured() -> None:
    """SplitApplier must split X/y apart when target_column is present in the frame."""
    df = _frame(50)
    params: Dict[str, Any] = {
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "target",
    }
    result = SplitApplier().apply(df, params)
    assert isinstance(result.train, tuple)
    X_train, y_train = result.train
    assert "target" not in X_train.columns
    assert len(X_train) == len(y_train)


def test_split_applier_accepts_xy_tuple_input() -> None:
    """SplitApplier must accept a pre-split (X, y) tuple as input."""
    X = pd.DataFrame({"feature": range(50)})
    y = pd.Series(range(50))
    params: Dict[str, Any] = {"test_size": 0.2, "random_state": 42}
    result = SplitApplier().apply((X, y), params)
    assert isinstance(result, SplitDataset)
    X_train, y_train = result.train
    assert len(X_train) == 40


def test_split_calculator_infer_output_schema_passes_through() -> None:
    """SplitCalculator.infer_output_schema must return the input schema unchanged."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(["a"], {"a": "int64"})
    result = SplitCalculator().infer_output_schema(schema, {})
    assert result is schema


# ---------------------------------------------------------------------------
# FeatureTargetSplit
# ---------------------------------------------------------------------------


def test_feature_target_split_calculator_fit_stores_target_column() -> None:
    """FeatureTargetSplitCalculator.fit must store the configured target_column."""
    params = FeatureTargetSplitCalculator().fit(_frame(5), {"target_column": "target"})
    assert params["target_column"] == "target"
    assert params["type"] == "feature_target_split"


def test_feature_target_split_applier_splits_single_frame() -> None:
    """FeatureTargetSplitApplier on a plain frame must return an (X, y) tuple."""
    df = _frame(10)
    result = FeatureTargetSplitApplier().apply(df, {"target_column": "target"})
    X, y = typing.cast("tuple[pd.DataFrame, pd.Series]", result)
    assert "target" not in X.columns
    assert list(y) == [i % 2 for i in range(10)]


def test_feature_target_split_applier_raises_without_target_column() -> None:
    """Missing target_column config must raise a ValueError."""
    with pytest.raises(ValueError):
        FeatureTargetSplitApplier().apply(_frame(5), {})


def test_feature_target_split_applier_raises_when_column_missing_from_frame() -> None:
    """A target_column absent from the frame must raise a ValueError."""
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError):
        FeatureTargetSplitApplier().apply(df, {"target_column": "does_not_exist"})


def test_feature_target_split_applier_handles_split_dataset_input() -> None:
    """FeatureTargetSplitApplier must split every member of an input SplitDataset."""
    df = _frame(50)
    split_result = SplitApplier().apply(df, {"test_size": 0.2, "random_state": 42})
    result = FeatureTargetSplitApplier().apply(split_result, {"target_column": "target"})
    assert isinstance(result, SplitDataset)
    X_train, y_train = result.train
    assert "target" not in X_train.columns
    assert len(X_train) == len(y_train)


def test_feature_target_split_applier_passes_through_tuple_input() -> None:
    """An already-split (X, y) tuple must pass through the Applier unchanged."""
    X = pd.DataFrame({"feature": [1, 2]})
    y = pd.Series([0, 1])
    result = FeatureTargetSplitApplier().apply((X, y), {"target_column": "target"})
    assert isinstance(result, tuple)
    assert result[0] is X
    assert result[1] is y
