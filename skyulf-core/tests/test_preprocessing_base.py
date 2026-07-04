"""Tests for skyulf.preprocessing.base (StatefulTransformer fit_transform/transform paths)."""

import typing

import pandas as pd
import pytest

from skyulf.core.schema import SkyulfSchema
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, StatefulTransformer


class _AddOneCalculator(BaseCalculator):
    """Calculator that just remembers a constant increment."""

    def fit(self, df, config):
        """Return a static params dict (no data-dependent fitting needed)."""
        return {"increment": config.get("increment", 1)}


class _AddOneApplier(BaseApplier):
    """Applier that adds `increment` to every value in column 'a'."""

    def apply(self, df, params):
        """Add params['increment'] to column 'a' and return the modified frame."""
        if isinstance(df, tuple):
            X, y = df
            X = X.copy()
            X["a"] = X["a"] + params["increment"]
            return (X, y)
        df = df.copy()
        df["a"] = df["a"] + params["increment"]
        return df


class _SplitReturningApplier(BaseApplier):
    """Applier that (incorrectly) returns a SplitDataset to trigger the guard error."""

    def apply(self, df, params):
        """Return a SplitDataset regardless of input, to test the error guard."""
        return SplitDataset(train=df, test=df)


def _transformer():
    """Build a StatefulTransformer using the AddOne calculator/applier pair."""
    return StatefulTransformer(_AddOneCalculator(), _AddOneApplier(), node_id="add_one")


def test_fit_transform_on_plain_dataframe():
    """fit_transform on a bare DataFrame should fit + apply directly (no splits)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    result = transformer.fit_transform(df, {"increment": 5})
    assert list(result["a"]) == [6, 7, 8]
    assert transformer.params == {"increment": 5}


def test_fit_transform_on_tuple_input():
    """fit_transform on an (X, y) tuple should pass the tuple through untouched in shape."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    transformer = _transformer()
    result = transformer.fit_transform((X, y), {"increment": 2})
    assert isinstance(result, tuple)
    assert list(result[0]["a"]) == [3, 4, 5]


def test_fit_transform_on_split_dataset_applies_to_all_splits():
    """fit_transform on a SplitDataset should fit on train and apply to all three splits."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    val = pd.DataFrame({"a": [100, 200]})
    dataset = SplitDataset(train=train, test=test, validation=val)
    transformer = _transformer()
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert list(result.train["a"]) == [2, 3]
    assert list(result.test["a"]) == [11, 21]
    assert list(result.validation["a"]) == [101, 201]


def test_fit_transform_skips_test_apply_when_disabled():
    """apply_on_test=False should leave the test split untouched by the applier."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(
        _AddOneCalculator(), _AddOneApplier(), node_id="add_one", apply_on_test=False
    )
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert isinstance(result, SplitDataset)
    assert isinstance(result.test, pd.DataFrame)
    assert isinstance(result.train, pd.DataFrame)
    assert list(result.test["a"]) == [10, 20]  # unchanged
    assert list(result.train["a"]) == [2, 3]


def test_fit_transform_skips_validation_when_none():
    """A dataset with validation=None should not invoke the applier on validation."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = _transformer()
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert result.validation is None


def test_fit_transform_records_profiling_metrics():
    """fit_transform should record fit_time, rows_in, and rows_out."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    transformer = _transformer()
    transformer.fit_transform(df, {"increment": 1})
    assert transformer.fit_time >= 0.0
    assert transformer.rows_in == 4
    assert transformer.rows_out == 4


def test_fit_transform_raises_when_applier_returns_split_dataset_on_train():
    """If the Applier illegally returns a SplitDataset for train, a TypeError should be raised."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(_AddOneCalculator(), _SplitReturningApplier(), node_id="bad")
    with pytest.raises(TypeError, match="not supported"):
        transformer.fit_transform(dataset, {})


def test_transform_on_plain_dataframe_reuses_stored_params():
    """transform() on a bare DataFrame should reuse previously fitted params."""
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    transformer.fit_transform(train_df, {"increment": 10})

    new_df = pd.DataFrame({"a": [0, 0]})
    result = transformer.transform(new_df)
    assert list(result["a"]) == [10, 10]


def test_transform_on_tuple_reuses_stored_params():
    """transform() on an (X, y) tuple should reuse the stored params."""
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    transformer.fit_transform(train_df, {"increment": 3})

    X = pd.DataFrame({"a": [0, 0]})
    y = pd.Series([1, 1])
    result = transformer.transform((X, y))
    assert list(result[0]["a"]) == [3, 3]


def test_transform_on_split_dataset_applies_to_all_splits():
    """transform() on a SplitDataset should apply the stored params to every split."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    val = pd.DataFrame({"a": [100, 200]})
    dataset = SplitDataset(train=train, test=test, validation=val)
    transformer = _transformer()
    transformer.fit_transform(dataset, {"increment": 1})

    new_dataset = SplitDataset(train=train.copy(), test=test.copy(), validation=val.copy())
    result = transformer.transform(new_dataset)
    assert list(result.train["a"]) == [2, 3]
    assert list(result.test["a"]) == [11, 21]
    assert list(result.validation["a"]) == [101, 201]


def test_transform_skips_test_when_disabled():
    """transform() with apply_on_test=False should leave the test split untouched."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(
        _AddOneCalculator(), _AddOneApplier(), node_id="add_one", apply_on_test=False
    )
    transformer.fit_transform(dataset, {"increment": 1})

    new_dataset = SplitDataset(train=train.copy(), test=test.copy(), validation=None)
    result = transformer.transform(new_dataset)
    assert isinstance(result, SplitDataset)
    assert isinstance(result.test, pd.DataFrame)
    assert list(result.test["a"]) == [10, 20]


def test_transform_raises_when_applier_returns_split_dataset():
    """transform() must guard against an Applier illegally returning a SplitDataset."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(_AddOneCalculator(), _SplitReturningApplier(), node_id="bad")
    transformer.params = {}
    with pytest.raises(TypeError, match="not supported"):
        transformer.transform(dataset)


def test_base_calculator_infer_output_schema_defaults_to_none():
    """The default infer_output_schema on BaseCalculator should return None."""
    calc = _AddOneCalculator()
    assert calc.infer_output_schema(typing.cast(SkyulfSchema, None), {}) is None
