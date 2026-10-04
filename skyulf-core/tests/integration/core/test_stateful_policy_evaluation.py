"""Public estimator evaluation must match deployed predictions and fixed CV policies."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningApplier, TuningCalculator
from skyulf.modeling.base import StatefulEstimator
from skyulf.modeling.classification import LogisticRegressionApplier, LogisticRegressionCalculator
from skyulf.modeling.regression import RidgeRegressionApplier, RidgeRegressionCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_nested_threshold_evaluation_matches_saved_predictions(engine):
    """Reported metrics and confusion matrices must describe the deployed decision cutoff."""
    X, y = make_classification(
        n_samples=180,
        n_features=4,
        n_informative=2,
        weights=[0.8, 0.2],
        class_sep=0.5,
        random_state=17,
    )
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = np.where(y, "yes", "no")
    train, held = frame.iloc[:120], frame.iloc[120:]
    dataset = SplitDataset(
        train=pl.from_pandas(train) if engine == "polars" else train,
        test=pl.from_pandas(held) if engine == "polars" else held,
        validation=None,
    )
    estimator = StatefulEstimator(
        TuningCalculator(LogisticRegressionCalculator()),
        TuningApplier(LogisticRegressionApplier()),
        "threshold",
    )
    predictions = estimator.fit_predict(
        dataset,
        "target",
        {
            "strategy": "grid",
            "search_space": {"C": [0.2]},
            "metric": "f1",
            "cv_type": "nested_cv",
            "cv_folds": 2,
            "cv_inner_folds": 2,
            "tune_threshold": True,
        },
    )
    report = estimator.evaluate(dataset, "target")
    assert isinstance(estimator.model, tuple)
    model, result = estimator.model
    held_x = held.drop(columns="target")
    expected = np.asarray(predictions["test"])
    truth = np.asarray(held["target"])
    assert result.decision_thresholds is not None
    assert np.any(expected != model.predict(held_x.to_numpy()))
    np.testing.assert_array_equal(report["raw_data"]["splits"]["test"]["y_pred"], expected)
    assert report["splits"]["test"].metrics["accuracy"] == pytest.approx(
        accuracy_score(truth, expected)
    )
    assert (
        report["splits"]["test"].classification.confusion_matrix.matrix
        == confusion_matrix(truth, expected, labels=model.classes_).tolist()
    )
    assert report["splits"]["test"].metrics["roc_auc"] == pytest.approx(
        roc_auc_score(truth == "yes", model.predict_proba(held_x.to_numpy())[:, 1])
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["group_k_fold", "time_series_split"])
def test_fixed_policy_cv_matches_independent_metadata_free_reference(engine, policy):
    """Direct fixed CV must exclude identifiers and use the requested group or time partitions."""
    rng = np.random.default_rng(71)
    X = pd.DataFrame({"x": rng.normal(size=60), "z": rng.normal(size=60)})
    y = pd.Series(2 * X.x - X.z + rng.normal(size=60) * 0.1)
    groups = np.repeat([f"entity_{i:02d}" for i in range(12)], 5)
    X["split_identifier"] = (
        groups if policy == "group_k_fold" else pd.date_range("2026-01-01", periods=60)
    )
    kwargs: dict[str, Any] = (
        {"group_column": "split_identifier"}
        if policy == "group_k_fold"
        else {"time_column": "split_identifier", "gap": 2, "test_size": 10, "max_train_size": 20}
    )
    splitter = (
        GroupKFold(3)
        if policy == "group_k_fold"
        else TimeSeriesSplit(3, gap=2, test_size=10, max_train_size=20)
    )
    partitions = splitter.split(X, y, groups) if policy == "group_k_fold" else splitter.split(X, y)
    expected = []
    for train, test in partitions:
        model = Ridge(alpha=0.7).fit(X.iloc[train][["x", "z"]], y.iloc[train])
        expected.append(mean_squared_error(y.iloc[test], model.predict(X.iloc[test][["x", "z"]])))
    frame = X.assign(target=y)
    if engine == "polars":
        frame = pl.from_pandas(frame)
    estimator = StatefulEstimator(RidgeRegressionCalculator(), RidgeRegressionApplier(), "fixed")
    report = estimator.cross_validate(
        SplitDataset(train=frame, test=frame.head(0)),
        "target",
        {"params": {"alpha": 0.7}},
        n_folds=3,
        cv_type=policy,
        shuffle=False,
        **kwargs,
    )
    assert report["split_policy"]["method"] == policy
    assert [fold["metrics"]["mse"] for fold in report["folds"]] == pytest.approx(expected)
