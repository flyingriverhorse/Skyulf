"""Tests for skyulf.modeling.sklearn_wrapper (SklearnCalculator / SklearnApplier)."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from skyulf.modeling.sklearn_wrapper import SklearnApplier, SklearnCalculator


@pytest.fixture
def clf_data():
    """Small deterministic binary classification data."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 40), "f2": rng.normal(0, 1, 40)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X, y


def test_fit_logs_via_log_callback(clf_data):
    """A supplied log_callback should be invoked with the initialization message."""
    X, y = clf_data
    calc = SklearnCalculator(LogisticRegression, {}, "classification")
    messages = []
    calc.fit(X, y, {}, log_callback=messages.append)
    assert len(messages) == 1
    assert "LogisticRegression" in messages[0]


def test_fit_drops_unsupported_flat_params_with_warning(clf_data, caplog):
    """Unsupported flat config keys should be dropped with a logged warning, not passed to sklearn."""
    X, y = clf_data
    calc = SklearnCalculator(LogisticRegression, {}, "classification")
    with caplog.at_level(logging.WARNING, logger="skyulf.modeling.sklearn_wrapper"):
        model = calc.fit(X, y, {"C": 2.0, "definitely_not_a_param": 123})
    assert isinstance(model, LogisticRegression)
    assert model.C == 2.0
    assert any("Dropped parameters" in record.message for record in caplog.records)


def test_fit_accepts_kwargs_constructor_without_filtering():
    """A model class whose constructor accepts **kwargs should receive all params unfiltered."""

    class _KwargsModel(LogisticRegression):
        def __init__(self, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k != "custom_flag"})
            self.custom_flag = kwargs.get("custom_flag")

    rng = np.random.RandomState(1)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 30)})
    y = pd.Series((X["f1"] > 0).astype(int))
    calc = SklearnCalculator(_KwargsModel, {}, "classification")
    model = calc.fit(X, y, {"custom_flag": "yes"})
    assert model.custom_flag == "yes"


def test_fit_merges_nested_params_dict(clf_data):
    """A config with an explicit 'params' key should be treated as the source of truth."""
    X, y = clf_data
    calc = SklearnCalculator(LogisticRegression, {"C": 1.0}, "classification")
    model = calc.fit(X, y, {"params": {"C": 5.0}})
    assert model.C == 5.0


def test_predict_returns_pandas_series_preserving_index(clf_data):
    """predict() should return a pandas Series preserving the original DataFrame index."""
    X, y = clf_data
    X = X.set_index(pd.RangeIndex(start=100, stop=100 + len(X)))
    model = LogisticRegression().fit(X, y)
    preds = SklearnApplier().predict(X, model)
    assert isinstance(preds, pd.Series)
    assert list(preds.index) == list(X.index)


def test_predict_on_polars_frame_uses_default_index(clf_data):
    """predict() on a polars frame (no pandas .index, has .to_pandas) should still work."""
    X, y = clf_data
    model = LogisticRegression().fit(X, y)
    X_pl = pl.from_pandas(X)
    preds = SklearnApplier().predict(X_pl, model)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)


def test_predict_proba_returns_none_when_unsupported():
    """predict_proba should return None for a model without a predict_proba method."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"f1": rng.normal(0, 1, 20)})
    y = pd.Series(X["f1"] + rng.normal(0, 0.1, 20))
    model = LinearRegression().fit(X, y)
    result = SklearnApplier().predict_proba(X, model)
    assert result is None


def test_predict_proba_returns_dataframe_with_classes_as_columns(clf_data):
    """predict_proba should return a DataFrame whose columns are the model's classes_."""
    X, y = clf_data
    model = SVC(probability=True).fit(X, y)
    result = SklearnApplier().predict_proba(X, model)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(model.classes_)
