"""Comprehensive unit tests for all modeling Calculator/Applier wrappers.

Tests each model: instantiate → fit → predict → verify metrics.
Covers classification (9 models) and regression (11 models).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_dataset() -> SplitDataset:
    """Binary classification dataset split into train/test."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


@pytest.fixture
def regression_dataset() -> SplitDataset:
    """Regression dataset split into train/test."""
    X, y = make_regression(
        n_samples=200, n_features=5, n_informative=3, noise=0.1, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


# ===========================================================================
# CLASSIFICATION MODELS
# ===========================================================================

class TestLogisticRegression:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import LogisticRegressionCalculator, LogisticRegressionApplier
        estimator = StatefulEstimator(
            node_id="lr", calculator=LogisticRegressionCalculator(), applier=LogisticRegressionApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert "train" in preds
        assert "test" in preds
        assert len(preds["test"]) == 40

    def test_evaluate(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import LogisticRegressionCalculator, LogisticRegressionApplier
        estimator = StatefulEstimator(
            node_id="lr", calculator=LogisticRegressionCalculator(), applier=LogisticRegressionApplier(),
        )
        estimator.fit_predict(classification_dataset, target_column="target", config={})
        report = estimator.evaluate(classification_dataset, target_column="target")
        assert report["problem_type"] == "classification"
        assert "accuracy" in report["splits"]["test"].metrics


class TestRandomForestClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import RandomForestClassifierCalculator, RandomForestClassifierApplier
        estimator = StatefulEstimator(
            node_id="rfc", calculator=RandomForestClassifierCalculator(), applier=RandomForestClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


class TestSVC:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import SVCCalculator, SVCApplier
        estimator = StatefulEstimator(
            node_id="svc", calculator=SVCCalculator(), applier=SVCApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestKNeighborsClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import KNeighborsClassifierCalculator, KNeighborsClassifierApplier
        estimator = StatefulEstimator(
            node_id="knn", calculator=KNeighborsClassifierCalculator(), applier=KNeighborsClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestDecisionTreeClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import DecisionTreeClassifierCalculator, DecisionTreeClassifierApplier
        estimator = StatefulEstimator(
            node_id="dtc", calculator=DecisionTreeClassifierCalculator(), applier=DecisionTreeClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestGradientBoostingClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import GradientBoostingClassifierCalculator, GradientBoostingClassifierApplier
        estimator = StatefulEstimator(
            node_id="gbc", calculator=GradientBoostingClassifierCalculator(), applier=GradientBoostingClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


class TestAdaBoostClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import AdaBoostClassifierCalculator, AdaBoostClassifierApplier
        estimator = StatefulEstimator(
            node_id="abc", calculator=AdaBoostClassifierCalculator(), applier=AdaBoostClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


class TestGaussianNB:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import GaussianNBCalculator, GaussianNBApplier
        estimator = StatefulEstimator(
            node_id="gnb", calculator=GaussianNBCalculator(), applier=GaussianNBApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestXGBClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        pytest.importorskip("xgboost")
        from skyulf.modeling.classification import XGBClassifierCalculator, XGBClassifierApplier
        estimator = StatefulEstimator(
            node_id="xgbc", calculator=XGBClassifierCalculator(), applier=XGBClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


# ===========================================================================
# REGRESSION MODELS
# ===========================================================================

class TestLinearRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import LinearRegressionCalculator, LinearRegressionApplier
        estimator = StatefulEstimator(
            node_id="linreg", calculator=LinearRegressionCalculator(), applier=LinearRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestRidgeRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import RidgeRegressionCalculator, RidgeRegressionApplier
        estimator = StatefulEstimator(
            node_id="ridge", calculator=RidgeRegressionCalculator(), applier=RidgeRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestLassoRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import LassoRegressionCalculator, LassoRegressionApplier
        estimator = StatefulEstimator(
            node_id="lasso", calculator=LassoRegressionCalculator(), applier=LassoRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestElasticNetRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import ElasticNetRegressionCalculator, ElasticNetRegressionApplier
        estimator = StatefulEstimator(
            node_id="enet", calculator=ElasticNetRegressionCalculator(), applier=ElasticNetRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestRandomForestRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import RandomForestRegressorCalculator, RandomForestRegressorApplier
        estimator = StatefulEstimator(
            node_id="rfr", calculator=RandomForestRegressorCalculator(), applier=RandomForestRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40

    def test_evaluate_regression_metrics(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import RandomForestRegressorCalculator, RandomForestRegressorApplier
        estimator = StatefulEstimator(
            node_id="rfr", calculator=RandomForestRegressorCalculator(), applier=RandomForestRegressorApplier(),
        )
        estimator.fit_predict(regression_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        report = estimator.evaluate(regression_dataset, target_column="target")
        assert report["problem_type"] == "regression"
        assert "mse" in report["splits"]["test"].metrics


class TestSVR:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import SVRCalculator, SVRApplier
        estimator = StatefulEstimator(
            node_id="svr", calculator=SVRCalculator(), applier=SVRApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestKNeighborsRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import KNeighborsRegressorCalculator, KNeighborsRegressorApplier
        estimator = StatefulEstimator(
            node_id="knnr", calculator=KNeighborsRegressorCalculator(), applier=KNeighborsRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestDecisionTreeRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import DecisionTreeRegressorCalculator, DecisionTreeRegressorApplier
        estimator = StatefulEstimator(
            node_id="dtr", calculator=DecisionTreeRegressorCalculator(), applier=DecisionTreeRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestGradientBoostingRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import GradientBoostingRegressorCalculator, GradientBoostingRegressorApplier
        estimator = StatefulEstimator(
            node_id="gbr", calculator=GradientBoostingRegressorCalculator(), applier=GradientBoostingRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


class TestAdaBoostRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import AdaBoostRegressorCalculator, AdaBoostRegressorApplier
        estimator = StatefulEstimator(
            node_id="abr", calculator=AdaBoostRegressorCalculator(), applier=AdaBoostRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40


class TestXGBRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        pytest.importorskip("xgboost")
        from skyulf.modeling.regression import XGBRegressorCalculator, XGBRegressorApplier
        estimator = StatefulEstimator(
            node_id="xgbr", calculator=XGBRegressorCalculator(), applier=XGBRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={"params": {"n_estimators": 10}})
        assert len(preds["test"]) == 40
