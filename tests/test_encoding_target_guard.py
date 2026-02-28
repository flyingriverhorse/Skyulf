"""Tests for target-column safety guard in encoding nodes.

Column-destroying encoders (OneHot, Dummy, Hash, TargetEncoder) must
automatically exclude the target column from processing so that downstream
Feature/Target Split and model training nodes can still find it.
"""

import logging

import pandas as pd
import pytest
from skyulf.preprocessing.encoding import (
    DummyEncoderCalculator,
    HashEncoderCalculator,
    LabelEncoderCalculator,
    OneHotEncoderCalculator,
    OrdinalEncoderCalculator,
    TargetEncoderCalculator,
    _exclude_target_column,
)


@pytest.fixture
def iris_like_df() -> pd.DataFrame:
    """Small dataset mimicking Iris where Species is categorical target."""
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 7.0, 6.3, 5.8, 7.1],
            "color": ["red", "blue", "red", "green", "blue", "green"],
            "region": ["north", "south", "north", "south", "north", "south"],
            "Species": ["setosa", "setosa", "versicolor", "virginica", "versicolor", "virginica"],
        }
    )


@pytest.fixture
def iris_Xy(iris_like_df: pd.DataFrame):
    """Return (X, y) tuple with Species as target."""
    X = iris_like_df.drop(columns=["Species"])
    y = iris_like_df["Species"]
    return X, y


# ---------- Unit tests for _exclude_target_column helper ---------- #


class TestExcludeTargetColumn:
    """Direct tests for the _exclude_target_column helper function."""

    def test_removes_target_from_unsafe_encoder(self):
        cols = ["color", "Species", "region"]
        config: dict = {"target_column": "Species"}
        result = _exclude_target_column(cols, config, "OneHotEncoder")
        assert result == ["color", "region"]

    def test_detects_target_from_y_name(self):
        y = pd.Series(["a", "b", "c"], name="Species")
        cols = ["color", "Species"]
        result = _exclude_target_column(cols, {}, "DummyEncoder", y=y)
        assert "Species" not in result

    def test_no_op_for_safe_encoder(self):
        cols = ["color", "Species"]
        config: dict = {"target_column": "Species"}
        result = _exclude_target_column(cols, config, "LabelEncoder")
        assert result == ["color", "Species"]

    def test_no_op_when_target_not_in_columns(self):
        cols = ["color", "region"]
        config: dict = {"target_column": "Species"}
        result = _exclude_target_column(cols, config, "OneHotEncoder")
        assert result == ["color", "region"]

    def test_logs_warning_when_excluding(self, caplog: pytest.LogCaptureFixture):
        cols = ["color", "Species"]
        config: dict = {"target_column": "Species"}
        with caplog.at_level(logging.WARNING):
            _exclude_target_column(cols, config, "HashEncoder")
        assert "Excluding target column 'Species'" in caplog.text
        assert "LabelEncoder or OrdinalEncoder" in caplog.text


# ---------- Integration tests for UNSAFE encoders ---------- #


class TestOneHotExcludesTarget:
    """OneHotEncoder must skip target column when passed as (X, y) or via config."""

    def test_auto_detection_excludes_target_via_y(self, iris_Xy):
        X, y = iris_Xy
        # Put Species back so auto-detection would pick it up
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = OneHotEncoderCalculator()
        params = calc.fit((X_with_target, y), {"columns": []})
        # Species must NOT appear in encoded_columns
        if params:
            assert "Species" not in params.get("columns", [])

    def test_explicit_columns_excludes_target(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = OneHotEncoderCalculator()
        # User explicitly lists Species — guard should remove it
        params = calc.fit(
            (X_with_target, y),
            {"columns": ["color", "Species"]},
        )
        if params:
            assert "Species" not in params.get("columns", [])


class TestDummyExcludesTarget:
    """DummyEncoder must skip target column."""

    def test_explicit_columns_excludes_target(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = DummyEncoderCalculator()
        params = calc.fit(
            (X_with_target, y),
            {"columns": ["color", "Species"], "drop_first": False},
        )
        if params:
            assert "Species" not in params.get("columns", [])


class TestHashExcludesTarget:
    """HashEncoder must skip target column."""

    def test_explicit_columns_excludes_target(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = HashEncoderCalculator()
        params = calc.fit(
            (X_with_target, y),
            {"columns": ["color", "Species"], "n_features": 4},
        )
        if params:
            assert "Species" not in params.get("columns", [])


class TestTargetEncoderExcludesTarget:
    """TargetEncoder must skip the target column itself."""

    def test_explicit_columns_excludes_target(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = TargetEncoderCalculator()
        params = calc.fit(
            (X_with_target, y),
            {"columns": ["color", "Species"], "target_column": "Species"},
        )
        if params:
            assert "Species" not in params.get("columns", [])


# ---------- SAFE encoders should NOT remove target ---------- #


class TestLabelEncoderKeepsTarget:
    """LabelEncoder is target-safe — it should encode the target column fine."""

    def test_encodes_target_column(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = LabelEncoderCalculator()
        params = calc.fit(X_with_target, {"columns": ["Species"]})
        assert params.get("type") == "label_encoder"
        assert "Species" in params.get("columns", []) or "Species" in params.get("mappings", {})


class TestOrdinalEncoderKeepsTarget:
    """OrdinalEncoder is target-safe — it should encode the target column fine."""

    def test_encodes_target_column(self, iris_Xy):
        X, y = iris_Xy
        X_with_target = X.copy()
        X_with_target["Species"] = y.values
        calc = OrdinalEncoderCalculator()
        params = calc.fit(X_with_target, {"columns": ["Species"]})
        assert params.get("type") == "ordinal"
