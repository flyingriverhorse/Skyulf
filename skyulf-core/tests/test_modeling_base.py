"""Tests for skyulf.modeling.base (BaseModelCalculator, BaseModelApplier, StatefulEstimator)."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import BaseModelApplier, BaseModelCalculator, StatefulEstimator
from skyulf.modeling.classification import (
    LogisticRegressionApplier,
    LogisticRegressionCalculator,
)
from skyulf.modeling.regression import (
    RandomForestRegressorApplier,
    RandomForestRegressorCalculator,
)

# ---------------------------------------------------------------------------
# Minimal concrete implementations for ABC tests
# ---------------------------------------------------------------------------


class _DummyCalculator(BaseModelCalculator):
    """Minimal calculator that echoes a constant artifact."""

    @property
    def problem_type(self) -> str:
        """Returns classification."""
        return "classification"

    def fit(self, X, y, config, progress_callback=None, log_callback=None, validation_data=None):
        """Return a simple dict as the model artifact."""
        return {"fitted": True, "n_samples": len(X)}


class _DummyApplier(BaseModelApplier):
    """Minimal applier that returns all-zeros predictions."""

    def predict(self, df, model_artifact) -> pd.Series:
        """Return zeros for every row."""
        return pd.Series(np.zeros(len(df), dtype=int))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _classification_dataset() -> tuple:
    """Binary classification split (160 train / 40 test)."""
    X_arr, y_arr = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])  # ty: ignore[invalid-argument-type]
    df["target"] = y_arr
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None), df


def _regression_dataset() -> tuple:
    """Regression split (160 train / 40 test)."""
    X_arr, y_arr = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])  # ty: ignore[invalid-argument-type]
    df["target"] = y_arr
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None), df


# ---------------------------------------------------------------------------
# BaseModelCalculator / BaseModelApplier contracts
# ---------------------------------------------------------------------------


def test_base_calculator_abstract_raises():
    """Instantiating BaseModelCalculator directly should raise TypeError."""
    with pytest.raises(TypeError):
        BaseModelCalculator()  # type: ignore[abstract]


def test_base_applier_abstract_raises():
    """Instantiating BaseModelApplier directly should raise TypeError."""
    with pytest.raises(TypeError):
        BaseModelApplier()  # type: ignore[abstract]


def test_dummy_calculator_problem_type():
    """Concrete calculator should return the declared problem_type."""
    calc = _DummyCalculator()
    assert calc.problem_type == "classification"


def test_dummy_calculator_default_params_empty():
    """Default params should be empty dict unless overridden."""
    calc = _DummyCalculator()
    assert calc.default_params == {}


def test_dummy_applier_predict_zeros():
    """Dummy applier should return a zero-filled Series."""
    appl = _DummyApplier()
    X = pd.DataFrame({"a": [1, 2, 3]})
    preds = appl.predict(X, model_artifact=None)
    assert list(preds) == [0, 0, 0]


def test_base_applier_predict_proba_default_none():
    """Default predict_proba should return None."""
    appl = _DummyApplier()
    X = pd.DataFrame({"a": [1]})
    assert appl.predict_proba(X, model_artifact=None) is None


def test_base_calculator_prepare_tuning_params_noop():
    """prepare_tuning_params should return None by default."""
    calc = _DummyCalculator()
    assert calc.prepare_tuning_params({}) is None


def test_base_calculator_build_tuning_search_space_empty():
    """build_tuning_search_space should return empty dict by default."""
    calc = _DummyCalculator()
    assert calc.build_tuning_search_space({}, "grid") == {}


# ---------------------------------------------------------------------------
# StatefulEstimator._extract_xy
# ---------------------------------------------------------------------------


def test_extract_xy_from_dataframe_with_target():
    """_extract_xy should split DataFrame into features and target."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    X, y = estimator._extract_xy(df, "target")
    assert "target" not in X.columns
    assert list(y) == [0, 1]


def test_extract_xy_missing_target_raises():
    """_extract_xy should raise ValueError if target column is absent."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in data"):
        estimator._extract_xy(df, "missing_col")


def test_extract_xy_from_tuple_xy():
    """_extract_xy with a (X, y) tuple should return it unchanged."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    X = pd.DataFrame({"a": [1, 2]})
    y = pd.Series([0, 1])
    X_out, y_out = estimator._extract_xy((X, y), "target")
    assert len(X_out) == 2
    assert list(y_out) == [0, 1]


def test_extract_xy_from_tuple_y_none_target_in_columns():
    """_extract_xy with (X, None) where target is embedded in X should recurse and split it."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    X_out, y_out = estimator._extract_xy((df, None), "target")
    assert "target" not in X_out.columns
    assert list(y_out) == [0, 1]


def test_extract_xy_polars_dataframe():
    """_extract_xy should support Polars DataFrames via the Polars engine branch."""
    import polars as pl

    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pl.DataFrame({"a": [1, 2], "target": [0, 1]})
    X_out, y_out = estimator._extract_xy(df, "target")
    assert "target" not in X_out.columns
    assert list(y_out) == [0, 1]


def test_extract_xy_polars_missing_target_raises():
    """_extract_xy on a Polars DataFrame without the target column should raise ValueError."""
    import polars as pl

    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in data"):
        estimator._extract_xy(df, "target")


class _DropKwargUnsupportedFrame:
    """Fake frame exposing 'columns' + a non-pandas-style attribute fallback.

    Its ``drop`` method rejects the ``columns=`` keyword (raising TypeError) but has
    no positional fallback either, forcing _extract_xy into the attribute-access path.
    """

    def __init__(self, target_name: str, target_values: pd.Series):
        self.columns = ["a", target_name]
        setattr(self, target_name, target_values)

    def drop(self, *args, **kwargs):
        raise TypeError("columns kwarg not supported")


def test_extract_xy_typeerror_falls_back_to_attribute_access():
    """When drop(columns=...) raises TypeError, _extract_xy should fall back to getattr."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )
    target_values = pd.Series([0, 1])
    fake = _DropKwargUnsupportedFrame("target", target_values)
    X_out, y_out = estimator._extract_xy(fake, "target")
    assert X_out is fake
    assert y_out is target_values


def test_extract_xy_unexpected_type_raises():
    """_extract_xy should raise ValueError for a data type it cannot interpret."""
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="n1"
    )

    class _Unsupported:
        pass

    with pytest.raises(ValueError, match="Unexpected data type"):
        estimator._extract_xy(_Unsupported(), "target")


# ---------------------------------------------------------------------------
# StatefulEstimator.fit_predict
# ---------------------------------------------------------------------------


def test_fit_predict_train_test_split():
    """fit_predict should return predictions for both train and test splits."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e1",
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert "train" in preds
    assert "test" in preds
    assert len(preds["train"]) == 160
    assert len(preds["test"]) == 40


def test_fit_predict_stores_model():
    """After fit_predict the model attribute should be populated."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e2",
    )
    estimator.fit_predict(dataset, "target", config={})
    assert estimator.model is not None


def test_fit_predict_with_validation_split():
    """fit_predict should return validation predictions when validation split is present."""
    dataset, df = _classification_dataset()
    val_df = df.sample(20, random_state=99)
    dataset_with_val = SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=val_df)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e3",
    )
    preds = estimator.fit_predict(dataset_with_val, "target", config={})
    assert "validation" in preds
    assert len(preds["validation"]) == 20


def test_fit_predict_raw_dataframe_input():
    """fit_predict should accept a raw DataFrame (no test split) and return train preds."""
    _, df = _classification_dataset()
    train_only = df.iloc[:80].copy()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e4",
    )
    preds = estimator.fit_predict(train_only, "target", config={})
    assert "train" in preds
    assert len(preds["train"]) == 80


def test_fit_predict_regression():
    """fit_predict for a regression problem should return numeric predictions."""
    dataset, _ = _regression_dataset()
    estimator = StatefulEstimator(
        calculator=RandomForestRegressorCalculator(),
        applier=RandomForestRegressorApplier(),
        node_id="e5",
    )
    preds = estimator.fit_predict(dataset, "target", config={"params": {"n_estimators": 5}})
    assert "train" in preds
    assert pd.api.types.is_float_dtype(preds["train"]) or pd.api.types.is_numeric_dtype(
        preds["train"]
    )


# ---------------------------------------------------------------------------
# StatefulEstimator.evaluate
# ---------------------------------------------------------------------------


def test_evaluate_before_fit_raises():
    """evaluate() before fit_predict() should raise ValueError."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e6",
    )
    with pytest.raises(ValueError, match="Model has not been trained"):
        estimator.evaluate(dataset, "target")


def test_evaluate_classification_returns_report():
    """evaluate() on classification should include 'accuracy' in test metrics."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e7",
    )
    estimator.fit_predict(dataset, "target", config={})
    report = estimator.evaluate(dataset, "target")
    assert report["problem_type"] == "classification"
    assert "accuracy" in report["splits"]["test"].metrics


def test_evaluate_regression_returns_report():
    """evaluate() on regression should include 'mse' in test metrics."""
    dataset, _ = _regression_dataset()
    estimator = StatefulEstimator(
        calculator=RandomForestRegressorCalculator(),
        applier=RandomForestRegressorApplier(),
        node_id="e8",
    )
    estimator.fit_predict(dataset, "target", config={"params": {"n_estimators": 5}})
    report = estimator.evaluate(dataset, "target")
    assert report["problem_type"] == "regression"
    assert "mse" in report["splits"]["test"].metrics


# ---------------------------------------------------------------------------
# StatefulEstimator.refit
# ---------------------------------------------------------------------------


def test_refit_without_validation_is_noop_fit():
    """refit() without a validation split should still train the model."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e9",
    )
    estimator.refit(dataset, "target", config={})
    assert estimator.model is not None


def test_refit_with_validation_combines_train_val():
    """refit() with validation should train on train+val combined."""
    dataset, df = _classification_dataset()
    val_df = df.sample(20, random_state=77)
    dataset_with_val = SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=val_df)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e10",
    )
    estimator.fit_predict(dataset_with_val, "target", config={})
    estimator.refit(dataset_with_val, "target", config={})
    assert estimator.model is not None


def test_fit_predict_validation_predictions_support_polars_dataset():
    """fit_predict()'s validation-prediction branch must support polars data.

    Regression test for a bug where the non-tuple validation branch called
    ``dataset.validation.drop(columns=[target_column])`` without the
    try/except TypeError -> polars fallback used by the otherwise-identical
    test-prediction branch a few lines above it, causing a TypeError for
    polars validation splits.
    """
    import polars as pl

    _, df = _classification_dataset()
    train_pl = pl.from_pandas(df.iloc[:160].reset_index(drop=True))
    val_pl = pl.from_pandas(df.iloc[160:180].reset_index(drop=True))
    test_pl = pl.from_pandas(df.iloc[180:].reset_index(drop=True))
    dataset_pl = SplitDataset(train=train_pl, test=test_pl, validation=val_pl)

    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e10c",
    )
    predictions = estimator.fit_predict(dataset_pl, "target", config={})
    assert "validation" in predictions


def test_refit_with_validation_supports_polars_dataset():
    """refit() must combine train+val for polars data too, not just pandas.

    Regression test for a bug where refit() unconditionally used pd.concat,
    which raises a TypeError on polars DataFrames/Series (_extract_xy returns
    polars objects when the engine is polars).
    """
    import polars as pl

    _, df = _classification_dataset()
    train_pl = pl.from_pandas(df.iloc[:160].reset_index(drop=True))
    val_pl = pl.from_pandas(df.iloc[160:180].reset_index(drop=True))
    test_pl = pl.from_pandas(df.iloc[180:].reset_index(drop=True))
    dataset_pl = SplitDataset(train=train_pl, test=test_pl, validation=val_pl)

    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e10b",
    )
    estimator.fit_predict(dataset_pl, "target", config={})
    estimator.refit(dataset_pl, "target", config={})
    assert estimator.model is not None


# ---------------------------------------------------------------------------
# StatefulEstimator.cross_validate
# ---------------------------------------------------------------------------


def test_cross_validate_via_stateful_estimator():
    """cross_validate() on StatefulEstimator should return aggregated_metrics."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="e11",
    )
    result = estimator.cross_validate(dataset, "target", config={}, n_folds=3)
    assert "aggregated_metrics" in result
    assert len(result["folds"]) == 3


# ---------------------------------------------------------------------------
# StatefulEstimator.fit_predict — raw tuple dataset handling
# ---------------------------------------------------------------------------


def test_fit_predict_tuple_train_test_dataframes():
    """Passing (train_df, test_df) tuple should be treated as an explicit split."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "target": [0, 1, 0, 1, 0, 1]})
    train_df, test_df = df.iloc[:4], df.iloc[4:]
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t1"
    )
    preds = estimator.fit_predict((train_df, test_df), "target", config={})
    assert len(preds["train"]) == 4
    assert len(preds["test"]) == 2


def test_fit_predict_tuple_xy_fallback_warns():
    """Passing a plain (X, y) tuple (no target embedded) should log a leakage warning."""
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t2"
    )
    messages = []
    preds = estimator.fit_predict((X, y), "target", config={}, log_callback=messages.append)
    assert len(preds["train"]) == 4
    assert any("No test set provided" in m for m in messages)


def test_fit_predict_test_split_as_tuple_with_y():
    """dataset.test provided as an (X, y) tuple with y present should be predicted directly."""
    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_X = pd.DataFrame({"a": [5, 6]})
    test_y = pd.Series([0, 1])
    dataset = SplitDataset(train=train_df, test=(test_X, test_y), validation=None)
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t3"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["test"]) == 2


def test_fit_predict_test_split_as_tuple_y_none_target_embedded():
    """dataset.test as (X, None) where target is embedded in X should be dropped before predict."""
    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_X = pd.DataFrame({"a": [5, 6], "target": [0, 1]})
    dataset = SplitDataset(train=train_df, test=(test_X, None), validation=None)
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t4"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["test"]) == 2


def test_fit_predict_test_split_as_polars_tuple_typeerror_fallback():
    """A Polars (X, None) test tuple should use the drop([col]) TypeError fallback path."""
    import polars as pl

    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_pl = pl.DataFrame({"a": [5, 6], "target": [0, 1]})
    dataset = SplitDataset(train=train_df, test=cast(Any, (test_pl, None)), validation=None)
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t5"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["test"]) == 2


def test_fit_predict_test_split_polars_non_tuple_is_empty_and_drop():
    """A Polars test split (non-tuple) should use is_empty() and drop() fallback."""
    import polars as pl

    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_pl = pl.DataFrame({"a": [5, 6], "target": [0, 1]})
    dataset = SplitDataset(train=train_df, test=cast(Any, test_pl), validation=None)
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t6"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["test"]) == 2


def test_fit_predict_validation_split_as_tuple_y_none_target_embedded():
    """dataset.validation as (X, None) with the target embedded should be dropped before predict."""
    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    val_X = pd.DataFrame({"a": [7, 8], "target": [0, 1]})
    dataset = SplitDataset(train=train_df, test=pd.DataFrame(), validation=(val_X, None))
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t7"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["validation"]) == 2


def test_fit_predict_validation_split_as_polars_tuple_typeerror_fallback():
    """A Polars (X, None) validation tuple should use the drop([col]) TypeError fallback path."""
    import polars as pl

    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    val_pl = pl.DataFrame({"a": [7, 8], "target": [0, 1]})
    dataset = SplitDataset(
        train=train_df, test=pd.DataFrame(), validation=cast(Any, (val_pl, None))
    )
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t8"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["validation"]) == 2


def test_fit_predict_test_split_dataframe_no_target_column():
    """A non-tuple dataset.test missing the target column should be used as-is."""
    train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    test_df = pd.DataFrame({"a": [5, 6]})
    dataset = SplitDataset(train=train_df, test=test_df, validation=None)
    estimator = StatefulEstimator(
        calculator=_DummyCalculator(), applier=_DummyApplier(), node_id="t10"
    )
    preds = estimator.fit_predict(dataset, "target", config={})
    assert len(preds["test"]) == 2


# NOTE: base.py line 287 (`X_val = dataset.validation` when the target column is
# *not* present in a non-tuple validation split) is unreachable through the public
# fit_predict() API: `_extract_xy(dataset.validation, target_column)` is called
# unconditionally at the top of fit_predict() (base.py:210) for any non-tuple
# validation split and raises ValueError there if the target column is missing,
# before the code can ever reach the later re-check at line 287. This appears to
# be defensive dead code kept for symmetry with the analogous `test` split branch.


# ---------------------------------------------------------------------------
# StatefulEstimator.evaluate — split-shape edge cases
# ---------------------------------------------------------------------------


class _UnknownProblemTypeCalculator(LogisticRegressionCalculator):
    """Reuses real LogisticRegression fit/predict, but reports an unsupported problem type."""

    @property
    def problem_type(self) -> str:
        """Return an unrecognized problem type to exercise the else/raise branch."""
        return "clustering"


def test_evaluate_unknown_problem_type_raises():
    """evaluate() should raise ValueError when problem_type is neither classification nor regression."""
    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=_UnknownProblemTypeCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u1",
    )
    estimator.fit_predict(dataset, "target", config={})
    with pytest.raises(ValueError, match="Unknown problem type"):
        estimator.evaluate(dataset, "target")


def test_evaluate_train_split_as_tuple_y_none_target_embedded():
    """evaluate() should extract y from an (X, None) train tuple when the target is embedded."""
    dataset, df = _classification_dataset()
    train_tuple_dataset = SplitDataset(
        train=(df.iloc[:160], None), test=df.iloc[160:], validation=None
    )
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u2",
    )
    estimator.fit_predict(train_tuple_dataset, "target", config={})
    report = estimator.evaluate(train_tuple_dataset, "target")
    assert report["splits"]["train"] is not None
    assert "accuracy" in report["splits"]["train"].metrics


def test_evaluate_train_split_tuple_missing_target_returns_none():
    """evaluate_split should return None for a tuple split whose X has no target and y is None."""
    dataset, df = _classification_dataset()
    X_no_target = df.drop(columns=["target"]).iloc[:160]
    train_tuple_dataset = SplitDataset(
        train=(X_no_target, None), test=df.iloc[160:], validation=None
    )
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u3",
    )
    # Fit directly on properly-shaped data so the model is trained, then evaluate the
    # malformed tuple dataset to trigger the "cannot evaluate without target" branch.
    good_dataset, _ = _classification_dataset()
    estimator.fit_predict(good_dataset, "target", config={})
    report = estimator.evaluate(train_tuple_dataset, "target")
    assert report["splits"]["train"] is None


def test_evaluate_train_split_dataframe_missing_target_returns_none():
    """evaluate_split should return None for a plain DataFrame split missing the target column."""
    dataset, df = _classification_dataset()
    X_no_target = df.drop(columns=["target"]).iloc[:160]
    train_dataset = SplitDataset(train=X_no_target, test=df.iloc[160:], validation=None)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u4",
    )
    good_dataset, _ = _classification_dataset()
    estimator.fit_predict(good_dataset, "target", config={})
    report = estimator.evaluate(train_dataset, "target")
    assert report["splits"]["train"] is None


def test_evaluate_train_split_unsupported_type_returns_none():
    """evaluate_split should return None for a train split of an unsupported type."""
    dataset, _ = _classification_dataset()
    unsupported_dataset = SplitDataset(
        train=cast(Any, object()), test=dataset.test, validation=None
    )
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u5",
    )
    good_dataset, _ = _classification_dataset()
    estimator.fit_predict(good_dataset, "target", config={})
    report = estimator.evaluate(unsupported_dataset, "target")
    assert report["splits"]["train"] is None


def test_evaluate_test_split_as_tuple_triggers_has_test_branch():
    """evaluate() should recognize a non-empty (X, y) test tuple via the has_test tuple branch."""
    dataset, df = _classification_dataset()
    test_X = df.drop(columns=["target"]).iloc[160:]
    test_y = df["target"].iloc[160:]
    tuple_test_dataset = SplitDataset(train=df.iloc[:160], test=(test_X, test_y), validation=None)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u6",
    )
    estimator.fit_predict(tuple_test_dataset, "target", config={})
    report = estimator.evaluate(tuple_test_dataset, "target")
    assert "test" in report["splits"]
    assert report["splits"]["test"] is not None


def test_evaluate_validation_split_as_tuple_triggers_has_val_branch():
    """evaluate() should recognize a non-empty (X, y) validation tuple via the has_val tuple branch."""
    dataset, df = _classification_dataset()
    val_X = df.drop(columns=["target"]).iloc[160:180]
    val_y = df["target"].iloc[160:180]
    tuple_val_dataset = SplitDataset(
        train=df.iloc[:160], test=df.iloc[180:], validation=(val_X, val_y)
    )
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u7",
    )
    estimator.fit_predict(tuple_val_dataset, "target", config={})
    report = estimator.evaluate(tuple_val_dataset, "target")
    assert "validation" in report["splits"]
    assert report["splits"]["validation"] is not None


def test_evaluate_train_split_as_polars_tuple_typeerror_fallback():
    """evaluate_split should use the drop([col]) TypeError fallback for a Polars (X, None) tuple."""
    import polars as pl

    dataset, df = _classification_dataset()
    train_pl = pl.DataFrame(df.iloc[:160].reset_index(drop=True))
    train_tuple_dataset = SplitDataset(
        train=cast(Any, (train_pl, None)), test=df.iloc[160:], validation=None
    )
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u9",
    )
    estimator.fit_predict(train_tuple_dataset, "target", config={})
    report = estimator.evaluate(train_tuple_dataset, "target")
    assert report["splits"]["train"] is not None
    assert "accuracy" in report["splits"]["train"].metrics


def test_evaluate_validation_dataframe_triggers_has_val_dataframe_branch():
    """evaluate() should recognize a non-empty plain-DataFrame validation split."""
    dataset, df = _classification_dataset()
    val_df = df.iloc[160:180]
    dataframe_val_dataset = SplitDataset(train=df.iloc[:160], test=df.iloc[180:], validation=val_df)
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=LogisticRegressionApplier(),
        node_id="u10",
    )
    estimator.fit_predict(dataframe_val_dataset, "target", config={})
    report = estimator.evaluate(dataframe_val_dataset, "target")
    assert "validation" in report["splits"]
    assert report["splits"]["validation"] is not None

    """evaluate() should unpack a (model, meta) tuple stored on self.model before evaluating."""
    from skyulf.modeling._tuning.engine import TuningApplier

    dataset, _ = _classification_dataset()
    estimator = StatefulEstimator(
        calculator=LogisticRegressionCalculator(),
        applier=TuningApplier(base_applier=LogisticRegressionApplier()),
        node_id="u8",
    )
    estimator.fit_predict(dataset, "target", config={})
    # Simulate a Tuner-style artifact: (model, extra_metadata) — TuningApplier.predict()
    # unpacks this tuple itself, matching the real self.model shape produced by the tuner.
    estimator.model = (estimator.model, {"best_params": {}})
    report = estimator.evaluate(dataset, "target")
    assert report["splits"]["train"] is not None
