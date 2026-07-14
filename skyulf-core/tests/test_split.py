"""Unit tests for the DataSplitter / SplitApplier / FeatureTargetSplit nodes.

Covers: DataSplitter.split and split_xy row counts and proportions,
validation-set carve-out, stratification (including the "too few members"
fallback), reproducibility with a fixed random_state, the SplitApplier /
SplitCalculator node contract (returns SplitDataset), and
FeatureTargetSplitApplier/Calculator.
"""

import logging
import typing
from typing import Any

import pandas as pd
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.split import (
    DataSplitter,
    FeatureTargetSplitApplier,
    FeatureTargetSplitCalculator,
    SplitApplier,
    SplitCalculator,
    _back_to_engine,
    _safe_stratify,
    _to_pandas_remember_engine,
)


def _frame(n: int = 100) -> pd.DataFrame:
    """Build a simple 100-row DataFrame with a numeric feature and a target."""
    return pd.DataFrame({"feature": range(n), "target": [i % 2 for i in range(n)]})


_validation_cases = TestCaseLoader("preprocessing/feature_target_split_validation").load()


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
    X_train, y_train = typing.cast(tuple[pd.DataFrame, pd.Series], result.train)
    X_test, y_test = typing.cast(tuple[pd.DataFrame, pd.Series], result.test)
    assert len(X_train) == len(y_train) == 80
    assert len(X_test) == len(y_test) == 20


def test_split_xy_with_validation() -> None:
    """split_xy with validation_size must carve out the correct validation size."""
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([i % 2 for i in range(100)])
    splitter = DataSplitter(test_size=0.2, validation_size=0.1, random_state=42)
    result = splitter.split_xy(X, y)
    assert result.validation is not None
    X_val, y_val = typing.cast(tuple[pd.DataFrame, pd.Series], result.validation)
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
    params: dict[str, Any] = {"test_size": 0.2, "random_state": 42}
    result = SplitApplier().apply(df, params)
    assert isinstance(result, SplitDataset)
    assert len(result.train) == 40
    assert len(result.test) == 10


def test_split_applier_splits_target_column_when_configured() -> None:
    """SplitApplier must split X/y apart when target_column is present in the frame."""
    df = _frame(50)
    params: dict[str, Any] = {
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "target",
    }
    result = SplitApplier().apply(df, params)
    assert isinstance(result.train, tuple)
    X_train, y_train = typing.cast(tuple[pd.DataFrame, pd.Series], result.train)
    assert "target" not in X_train.columns
    assert len(X_train) == len(y_train)


def test_split_applier_accepts_xy_tuple_input() -> None:
    """SplitApplier must accept a pre-split (X, y) tuple as input."""
    X = pd.DataFrame({"feature": range(50)})
    y = pd.Series(range(50))
    params: dict[str, Any] = {"test_size": 0.2, "random_state": 42}
    result = SplitApplier().apply((X, y), params)
    assert isinstance(result, SplitDataset)
    X_train, y_train = typing.cast(tuple[pd.DataFrame, pd.Series], result.train)
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


class TestValidation:
    """Config validation errors — scenarios loaded from
    ``tests/test_cases/preprocessing/feature_target_split_validation.json``.
    """

    @pytest.mark.parametrize(*_validation_cases)
    def test_invalid_config_raises(self, target_column: str | None, error_match: str) -> None:
        config: dict[str, Any] = {} if target_column is None else {"target_column": target_column}
        with pytest.raises(ValueError, match=error_match):
            FeatureTargetSplitApplier().apply(_frame(), config)


def test_feature_target_split_applier_handles_split_dataset_input() -> None:
    """FeatureTargetSplitApplier must split every member of an input SplitDataset."""
    df = _frame(50)
    split_result = SplitApplier().apply(df, {"test_size": 0.2, "random_state": 42})
    result = FeatureTargetSplitApplier().apply(split_result, {"target_column": "target"})
    assert isinstance(result, SplitDataset)
    X_train, y_train = typing.cast(tuple[pd.DataFrame, pd.Series], result.train)
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


# ---------------------------------------------------------------------------
# Polars engine — _to_pandas_remember_engine / _back_to_engine round trip
# ---------------------------------------------------------------------------


def test_data_splitter_split_polars_round_trips_back_to_polars() -> None:
    """split() on a Polars frame must convert to pandas internally and back to Polars."""
    df = pl.DataFrame({"feature": range(20), "target": [i % 2 for i in range(20)]})
    splitter = DataSplitter(test_size=0.2, random_state=42)
    result = splitter.split(typing.cast(Any, df))
    assert isinstance(result.train, pl.DataFrame)
    assert isinstance(result.test, pl.DataFrame)
    assert result.train.height == 16
    assert result.test.height == 4


def test_data_splitter_split_polars_with_validation_round_trips() -> None:
    """split() with validation on Polars input must return Polars frames for all splits."""
    df = pl.DataFrame({"feature": range(20), "target": [i % 2 for i in range(20)]})
    splitter = DataSplitter(test_size=0.2, validation_size=0.2, random_state=42)
    result = splitter.split(typing.cast(Any, df))
    assert isinstance(result.validation, pl.DataFrame)
    assert result.validation.height == 4


def test_data_splitter_split_xy_polars_round_trips_back_to_polars() -> None:
    """split_xy() on Polars X/y must return Polars frames/series in every split member."""
    X = pl.DataFrame({"feature": range(20)})
    y = pl.Series("target", [i % 2 for i in range(20)])
    splitter = DataSplitter(test_size=0.2, random_state=42)
    result = splitter.split_xy(typing.cast(Any, X), typing.cast(Any, y))
    X_train, y_train = result.train
    X_test, y_test = result.test
    assert isinstance(X_train, pl.DataFrame)
    assert isinstance(y_train, pl.Series)
    assert isinstance(X_test, pl.DataFrame)
    assert isinstance(y_test, pl.Series)
    assert X_train.height == len(y_train) == 16
    assert X_test.height == len(y_test) == 4


def test_data_splitter_split_xy_polars_with_validation_round_trips() -> None:
    """split_xy() with validation on Polars input must return Polars members throughout."""
    X = pl.DataFrame({"feature": range(20)})
    y = pl.Series("target", [i % 2 for i in range(20)])
    splitter = DataSplitter(test_size=0.2, validation_size=0.2, random_state=42)
    result = splitter.split_xy(typing.cast(Any, X), typing.cast(Any, y))
    assert result.validation is not None
    X_val, y_val = result.validation
    assert isinstance(X_val, pl.DataFrame)
    assert isinstance(y_val, pl.Series)
    assert X_val.height == len(y_val) == 4


# ---------------------------------------------------------------------------
# _build_splitter — implicit stratify sentinel
# ---------------------------------------------------------------------------


def test_split_applier_stratify_true_without_target_column_uses_sentinel() -> None:
    """stratify=True with no target_column must fall back to the implicit sentinel
    and still stratify successfully on a supplied (X, y) tuple."""
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([i % 2 for i in range(100)])
    params: dict[str, Any] = {"test_size": 0.2, "random_state": 42, "stratify": True}
    result = SplitApplier().apply((X, y), params)
    assert isinstance(result, SplitDataset)
    _, y_test = typing.cast(tuple[pd.DataFrame, pd.Series], result.test)
    assert y_test.mean() == pytest.approx(0.5, abs=0.05)


def test_split_applier_frame_stratify_without_target_column_warns_and_splits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """stratify=True with no target_column on a plain-frame input has no column
    to stratify on; this must emit a warning (not fail silently) and still
    complete as a plain (non-stratified) shuffle split."""
    df = pd.DataFrame({"feature": range(100)})
    params: dict[str, Any] = {"test_size": 0.2, "random_state": 42, "stratify": True}
    with caplog.at_level(logging.WARNING):
        result = SplitApplier().apply(df, params)
    assert isinstance(result, SplitDataset)
    assert isinstance(result.test, pd.DataFrame)
    assert any(
        "stratify" in record.message.lower() and "target_column" in record.message
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# FeatureTargetSplitApplier — polars single-frame split (_split_xy_one_polars)
# ---------------------------------------------------------------------------


def test_feature_target_split_applier_splits_single_polars_frame() -> None:
    """FeatureTargetSplitApplier on a Polars frame must return (X, y) Polars members."""
    df = pl.DataFrame({"feature": range(10), "target": [i % 2 for i in range(10)]})
    result = FeatureTargetSplitApplier().apply(typing.cast(Any, df), {"target_column": "target"})
    X, y = typing.cast("tuple[pl.DataFrame, pl.Series]", result)
    assert isinstance(X, pl.DataFrame)
    assert isinstance(y, pl.Series)
    assert "target" not in X.columns
    assert y.to_list() == [i % 2 for i in range(10)]


def test_feature_target_split_applier_polars_raises_when_column_missing() -> None:
    """A target_column absent from a Polars frame must raise a ValueError."""
    df = pl.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError):
        FeatureTargetSplitApplier().apply(typing.cast(Any, df), {"target_column": "does_not_exist"})


def test_feature_target_split_applier_handles_polars_split_dataset_input() -> None:
    """FeatureTargetSplitApplier must split every Polars member of an input SplitDataset."""
    df = pl.DataFrame({"feature": range(50), "target": [i % 2 for i in range(50)]})
    split_result = SplitApplier().apply(df, {"test_size": 0.2, "random_state": 42})
    result = FeatureTargetSplitApplier().apply(split_result, {"target_column": "target"})
    assert isinstance(result, SplitDataset)
    X_train, y_train = typing.cast("tuple[pl.DataFrame, pl.Series]", result.train)
    assert "target" not in X_train.columns
    assert X_train.height == len(y_train)


# ---------------------------------------------------------------------------
# _maybe_split_xy_member — already-split tuple members and missing-target fallback
# ---------------------------------------------------------------------------


def test_feature_target_split_applier_split_dataset_member_already_split() -> None:
    """A SplitDataset member that is already an (X, y) tuple with non-None y must
    be returned unchanged instead of being re-split."""
    X_train = pd.DataFrame({"feature": [1, 2]})
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame({"feature": [3, 4], "target": [1, 0]})
    split_dataset = SplitDataset(train=(X_train, y_train), test=(X_test, None))
    result = FeatureTargetSplitApplier().apply(split_dataset, {"target_column": "target"})
    assert isinstance(result, SplitDataset)
    # train member was already split -> passed through untouched.
    assert result.train[0] is X_train
    assert result.train[1] is y_train
    # test member had no y but does have the target column -> gets split.
    X_test_out, y_test_out = typing.cast(tuple[pd.DataFrame, pd.Series], result.test)
    assert "target" not in X_test_out.columns
    assert list(y_test_out) == [1, 0]


def test_feature_target_split_applier_split_dataset_member_tuple_without_target_col() -> None:
    """A SplitDataset (X, None) member whose X lacks the target column must pass through
    unchanged rather than raising."""
    X = pd.DataFrame({"feature": [1, 2, 3]})
    split_dataset = SplitDataset(train=(X, None), test=(X.copy(), None))
    result = FeatureTargetSplitApplier().apply(split_dataset, {"target_column": "target"})
    assert isinstance(result, SplitDataset)
    assert result.train[0] is X
    assert result.train[1] is None


def test_feature_target_split_calculator_infer_output_schema_passes_through() -> None:
    """FeatureTargetSplitCalculator.infer_output_schema must return the input schema
    unchanged (target column stays visible downstream in the y slot)."""
    from skyulf.core.schema import SkyulfSchema

    schema = SkyulfSchema.from_columns(
        ["feature", "target"], {"feature": "int64", "target": "int64"}
    )
    result = FeatureTargetSplitCalculator().infer_output_schema(schema, {"target_column": "target"})
    assert result is schema


# ---------------------------------------------------------------------------
# _to_pandas_remember_engine / _back_to_engine — private helper unit tests
# ---------------------------------------------------------------------------


def test_to_pandas_remember_engine_none_passthrough() -> None:
    """None input must pass through as (None, False) without conversion."""
    assert _to_pandas_remember_engine(None) == (None, False)


def test_to_pandas_remember_engine_pandas_passthrough() -> None:
    """A pandas DataFrame must pass through unconverted with was_polars=False."""
    df = pd.DataFrame({"a": [1, 2]})
    data, was_polars = _to_pandas_remember_engine(df)
    assert data is df
    assert was_polars is False


def test_back_to_engine_none_input_returns_none() -> None:
    """None input must pass through _back_to_engine unchanged regardless of was_polars."""
    assert _back_to_engine(None, True) is None
    assert _back_to_engine(None, False) is None


def test_back_to_engine_not_was_polars_returns_data_unchanged() -> None:
    """When was_polars is False, data must be returned unchanged even if convertible."""
    df = pd.DataFrame({"a": [1, 2]})
    assert _back_to_engine(df, False) is df


def test_back_to_engine_non_frame_data_with_was_polars_passthrough() -> None:
    """A was_polars=True conversion must fall through unchanged for non pandas types."""
    assert _back_to_engine(42, True) == 42


# ---------------------------------------------------------------------------
# Real-shaped dataset integration check
# ---------------------------------------------------------------------------


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample.

    Verifies that SplitCalculator / SplitApplier handles a real-shaped dataset
    with missing values and mixed dtypes: row counts add up and the train/test
    index sets are disjoint and cover the entire dataset.
    """

    def test_split_on_customers_preserves_all_rows(self) -> None:
        df = load_sample_dataset("customers")
        params: dict[str, Any] = {"test_size": 0.2, "random_state": 42}
        result = SplitApplier().apply(df, params)
        assert isinstance(result, SplitDataset)
        assert isinstance(result.train, pd.DataFrame)
        assert isinstance(result.test, pd.DataFrame)
        train_idx = set(result.train.index)
        test_idx = set(result.test.index)
        assert train_idx.isdisjoint(test_idx)
        assert train_idx | test_idx == set(df.index)
        assert len(result.train) + len(result.test) == len(df)
