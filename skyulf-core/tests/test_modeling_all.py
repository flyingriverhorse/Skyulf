"""Comprehensive unit tests for all modeling Calculator/Applier wrappers.

Tests each model: instantiate → fit → predict → verify metrics.
Covers classification (9 models) and regression (11 models).
"""

from typing import Any, cast

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.data.dataset import SplitDataset
from skyulf.modeling.base import StatefulEstimator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_dataset() -> SplitDataset:
    """Binary classification dataset split into train/test."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    df = pd.DataFrame(cast(Any, X), columns=cast(Any, [f"f{i}" for i in range(5)]))
    df["target"] = y
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


@pytest.fixture
def regression_dataset() -> SplitDataset:
    """Regression dataset split into train/test."""
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )
    df = pd.DataFrame(cast(Any, X), columns=cast(Any, [f"f{i}" for i in range(5)]))
    df["target"] = y
    return SplitDataset(train=df.iloc[:160], test=df.iloc[160:], validation=None)


# ===========================================================================
# CLASSIFICATION MODELS
# ===========================================================================


class TestLogisticRegression:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            LogisticRegressionApplier,
            LogisticRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="lr",
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert "train" in preds
        assert "test" in preds
        assert len(preds["test"]) == 40

    def test_evaluate(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            LogisticRegressionApplier,
            LogisticRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="lr",
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
        )
        estimator.fit_predict(classification_dataset, target_column="target", config={})
        report = estimator.evaluate(classification_dataset, target_column="target")
        assert report["problem_type"] == "classification"
        assert "accuracy" in report["splits"]["test"].metrics

    def test_fit_raises_clear_error_on_incompatible_solver_penalty(
        self, classification_dataset: SplitDataset
    ) -> None:
        """solver='lbfgs' + penalty='l1' is invalid in sklearn; the calculator
        must fail fast with an actionable message rather than surfacing
        sklearn's own deep ValueError from inside LogisticRegression.fit."""
        from skyulf.modeling.classification import (
            LogisticRegressionApplier,
            LogisticRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="lr",
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
        )
        with pytest.raises(ValueError, match="does not support penalty"):
            estimator.fit_predict(
                classification_dataset,
                target_column="target",
                config={"params": {"solver": "lbfgs", "penalty": "l1"}},
            )

    def test_fit_allows_compatible_solver_penalty(
        self, classification_dataset: SplitDataset
    ) -> None:
        """saga supports l1/l2/elasticnet/None -- must not raise."""
        from skyulf.modeling.classification import (
            LogisticRegressionApplier,
            LogisticRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="lr",
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset,
            target_column="target",
            config={"params": {"solver": "saga", "penalty": "l1", "max_iter": 1000}},
        )
        assert len(preds["test"]) == 40

    def test_fit_allows_solver_without_penalty_override(
        self, classification_dataset: SplitDataset
    ) -> None:
        """Overriding only `solver` (no explicit `penalty` key) must not raise
        -- validation only runs when both keys are present in the same
        config so we don't reject a partial config the model defaults fill in."""
        from skyulf.modeling.classification import (
            LogisticRegressionApplier,
            LogisticRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="lr",
            calculator=LogisticRegressionCalculator(),
            applier=LogisticRegressionApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset,
            target_column="target",
            config={"params": {"solver": "liblinear"}},
        )
        assert len(preds["test"]) == 40


class TestRandomForestClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            RandomForestClassifierApplier,
            RandomForestClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="rfc",
            calculator=RandomForestClassifierCalculator(),
            applier=RandomForestClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


class TestSVC:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import SVCApplier, SVCCalculator

        estimator = StatefulEstimator(
            node_id="svc",
            calculator=SVCCalculator(),
            applier=SVCApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestKNeighborsClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            KNeighborsClassifierApplier,
            KNeighborsClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="knn",
            calculator=KNeighborsClassifierCalculator(),
            applier=KNeighborsClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestDecisionTreeClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            DecisionTreeClassifierApplier,
            DecisionTreeClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="dtc",
            calculator=DecisionTreeClassifierCalculator(),
            applier=DecisionTreeClassifierApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestGradientBoostingClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            GradientBoostingClassifierApplier,
            GradientBoostingClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="gbc",
            calculator=GradientBoostingClassifierCalculator(),
            applier=GradientBoostingClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


class TestAdaBoostClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            AdaBoostClassifierApplier,
            AdaBoostClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="abc",
            calculator=AdaBoostClassifierCalculator(),
            applier=AdaBoostClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


class TestGaussianNB:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import GaussianNBApplier, GaussianNBCalculator

        estimator = StatefulEstimator(
            node_id="gnb",
            calculator=GaussianNBCalculator(),
            applier=GaussianNBApplier(),
        )
        preds = estimator.fit_predict(classification_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestXGBClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        pytest.importorskip("xgboost")
        from skyulf.modeling.classification import XGBClassifierApplier, XGBClassifierCalculator

        estimator = StatefulEstimator(
            node_id="xgbc",
            calculator=XGBClassifierCalculator(),
            applier=XGBClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


# ===========================================================================
# REGRESSION MODELS
# ===========================================================================


class TestLinearRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import LinearRegressionApplier, LinearRegressionCalculator

        estimator = StatefulEstimator(
            node_id="linreg",
            calculator=LinearRegressionCalculator(),
            applier=LinearRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestRidgeRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import RidgeRegressionApplier, RidgeRegressionCalculator

        estimator = StatefulEstimator(
            node_id="ridge",
            calculator=RidgeRegressionCalculator(),
            applier=RidgeRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestLassoRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import LassoRegressionApplier, LassoRegressionCalculator

        estimator = StatefulEstimator(
            node_id="lasso",
            calculator=LassoRegressionCalculator(),
            applier=LassoRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestElasticNetRegression:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            ElasticNetRegressionApplier,
            ElasticNetRegressionCalculator,
        )

        estimator = StatefulEstimator(
            node_id="enet",
            calculator=ElasticNetRegressionCalculator(),
            applier=ElasticNetRegressionApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestRandomForestRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            RandomForestRegressorApplier,
            RandomForestRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="rfr",
            calculator=RandomForestRegressorCalculator(),
            applier=RandomForestRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate_regression_metrics(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            RandomForestRegressorApplier,
            RandomForestRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="rfr",
            calculator=RandomForestRegressorCalculator(),
            applier=RandomForestRegressorApplier(),
        )
        estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        report = estimator.evaluate(regression_dataset, target_column="target")
        assert report["problem_type"] == "regression"
        assert "mse" in report["splits"]["test"].metrics


class TestSVR:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import SVRApplier, SVRCalculator

        estimator = StatefulEstimator(
            node_id="svr",
            calculator=SVRCalculator(),
            applier=SVRApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestKNeighborsRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            KNeighborsRegressorApplier,
            KNeighborsRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="knnr",
            calculator=KNeighborsRegressorCalculator(),
            applier=KNeighborsRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestDecisionTreeRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            DecisionTreeRegressorApplier,
            DecisionTreeRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="dtr",
            calculator=DecisionTreeRegressorCalculator(),
            applier=DecisionTreeRegressorApplier(),
        )
        preds = estimator.fit_predict(regression_dataset, target_column="target", config={})
        assert len(preds["test"]) == 40


class TestGradientBoostingRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            GradientBoostingRegressorApplier,
            GradientBoostingRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="gbr",
            calculator=GradientBoostingRegressorCalculator(),
            applier=GradientBoostingRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


class TestAdaBoostRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import AdaBoostRegressorApplier, AdaBoostRegressorCalculator

        estimator = StatefulEstimator(
            node_id="abr",
            calculator=AdaBoostRegressorCalculator(),
            applier=AdaBoostRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


class TestXGBRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        pytest.importorskip("xgboost")
        from skyulf.modeling.regression import XGBRegressorApplier, XGBRegressorCalculator

        estimator = StatefulEstimator(
            node_id="xgbr",
            calculator=XGBRegressorCalculator(),
            applier=XGBRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40


# ===========================================================================
# NEW MODELS — ExtraTrees, HistGradientBoosting, LightGBM
# ===========================================================================


class TestExtraTreesClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            ExtraTreesClassifierApplier,
            ExtraTreesClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="etc",
            calculator=ExtraTreesClassifierCalculator(),
            applier=ExtraTreesClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            ExtraTreesClassifierApplier,
            ExtraTreesClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="etc",
            calculator=ExtraTreesClassifierCalculator(),
            applier=ExtraTreesClassifierApplier(),
        )
        estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        report = estimator.evaluate(classification_dataset, target_column="target")
        assert report["problem_type"] == "classification"
        assert "accuracy" in report["splits"]["test"].metrics


class TestExtraTreesRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            ExtraTreesRegressorApplier,
            ExtraTreesRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="etr",
            calculator=ExtraTreesRegressorCalculator(),
            applier=ExtraTreesRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            ExtraTreesRegressorApplier,
            ExtraTreesRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="etr",
            calculator=ExtraTreesRegressorCalculator(),
            applier=ExtraTreesRegressorApplier(),
        )
        estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        report = estimator.evaluate(regression_dataset, target_column="target")
        assert report["problem_type"] == "regression"
        assert "mse" in report["splits"]["test"].metrics


class TestHistGradientBoostingClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        from skyulf.modeling.classification import (
            HistGradientBoostingClassifierApplier,
            HistGradientBoostingClassifierCalculator,
        )

        estimator = StatefulEstimator(
            node_id="hgbc",
            calculator=HistGradientBoostingClassifierCalculator(),
            applier=HistGradientBoostingClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"max_iter": 20}}
        )
        assert len(preds["test"]) == 40

    def test_handles_nan(self, classification_dataset: SplitDataset) -> None:
        """HistGradientBoosting is natively NaN-tolerant — no imputation needed."""
        import numpy as np

        from skyulf.modeling.classification import (
            HistGradientBoostingClassifierApplier,
            HistGradientBoostingClassifierCalculator,
        )

        ds = classification_dataset
        # Inject NaN into training set
        from typing import cast as _cast

        import pandas as _pd

        train_with_nan = _pd.DataFrame(_cast(_pd.DataFrame, ds.train)).copy()
        train_with_nan.iloc[0, 0] = np.nan

        from skyulf.data.dataset import SplitDataset

        ds_nan = SplitDataset(train=train_with_nan, test=ds.test, validation=None)

        estimator = StatefulEstimator(
            node_id="hgbc_nan",
            calculator=HistGradientBoostingClassifierCalculator(),
            applier=HistGradientBoostingClassifierApplier(),
        )
        preds = estimator.fit_predict(
            ds_nan, target_column="target", config={"params": {"max_iter": 20}}
        )
        assert len(preds["test"]) == 40


class TestHistGradientBoostingRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            HistGradientBoostingRegressorApplier,
            HistGradientBoostingRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="hgbr",
            calculator=HistGradientBoostingRegressorCalculator(),
            applier=HistGradientBoostingRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"max_iter": 20}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate(self, regression_dataset: SplitDataset) -> None:
        from skyulf.modeling.regression import (
            HistGradientBoostingRegressorApplier,
            HistGradientBoostingRegressorCalculator,
        )

        estimator = StatefulEstimator(
            node_id="hgbr",
            calculator=HistGradientBoostingRegressorCalculator(),
            applier=HistGradientBoostingRegressorApplier(),
        )
        estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"max_iter": 20}}
        )
        report = estimator.evaluate(regression_dataset, target_column="target")
        assert report["problem_type"] == "regression"
        assert "mse" in report["splits"]["test"].metrics


class TestLGBMClassifier:
    def test_fit_predict(self, classification_dataset: SplitDataset) -> None:
        pytest.importorskip("lightgbm")
        from skyulf.modeling.classification import LGBMClassifierApplier, LGBMClassifierCalculator

        estimator = StatefulEstimator(
            node_id="lgbmc",
            calculator=LGBMClassifierCalculator(),
            applier=LGBMClassifierApplier(),
        )
        preds = estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate(self, classification_dataset: SplitDataset) -> None:
        pytest.importorskip("lightgbm")
        from skyulf.modeling.classification import LGBMClassifierApplier, LGBMClassifierCalculator

        estimator = StatefulEstimator(
            node_id="lgbmc",
            calculator=LGBMClassifierCalculator(),
            applier=LGBMClassifierApplier(),
        )
        estimator.fit_predict(
            classification_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        report = estimator.evaluate(classification_dataset, target_column="target")
        assert report["problem_type"] == "classification"
        assert "accuracy" in report["splits"]["test"].metrics


class TestLGBMRegressor:
    def test_fit_predict(self, regression_dataset: SplitDataset) -> None:
        pytest.importorskip("lightgbm")
        from skyulf.modeling.regression import LGBMRegressorApplier, LGBMRegressorCalculator

        estimator = StatefulEstimator(
            node_id="lgbmr",
            calculator=LGBMRegressorCalculator(),
            applier=LGBMRegressorApplier(),
        )
        preds = estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == 40

    def test_evaluate(self, regression_dataset: SplitDataset) -> None:
        pytest.importorskip("lightgbm")
        from skyulf.modeling.regression import LGBMRegressorApplier, LGBMRegressorCalculator

        estimator = StatefulEstimator(
            node_id="lgbmr",
            calculator=LGBMRegressorCalculator(),
            applier=LGBMRegressorApplier(),
        )
        estimator.fit_predict(
            regression_dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        report = estimator.evaluate(regression_dataset, target_column="target")
        assert report["problem_type"] == "regression"
        assert "mse" in report["splits"]["test"].metrics


# ===========================================================================
# Hyperparameter definitions — smoke test registry completeness
# ===========================================================================


class TestHyperparameterRegistry:
    NEW_MODELS = [
        "extra_trees_classifier",
        "extra_trees_regressor",
        "hist_gradient_boosting_classifier",
        "hist_gradient_boosting_regressor",
        "lgbm_classifier",
        "lgbm_regressor",
    ]

    def test_all_new_models_have_hyperparams(self) -> None:
        from skyulf.modeling.hyperparameters import get_hyperparameters

        for key in self.NEW_MODELS:
            params = get_hyperparameters(key)
            assert len(params) > 0, f"No hyperparameters defined for {key!r}"

    def test_all_new_models_have_search_spaces(self) -> None:
        from skyulf.modeling.hyperparameters import get_default_search_space

        for key in self.NEW_MODELS:
            space = get_default_search_space(key)
            assert len(space) > 0, f"No search space defined for {key!r}"


# ===========================================================================
# Real-shaped dataset — integration check against customers.csv
# ===========================================================================


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values and mixed dtypes — closer to production data than
    the synthetic ``make_classification``/``make_regression`` fixtures used
    elsewhere in this file.
    """

    def test_random_forest_classifier_predicts_churn(self) -> None:
        from skyulf.modeling.classification import (
            RandomForestClassifierApplier,
            RandomForestClassifierCalculator,
        )

        df = load_sample_dataset("customers")
        # RandomForestClassifier can't handle NaN directly, so rows with
        # missing age/income are dropped rather than assumed clean.
        df = df.dropna(subset=["age", "income"])
        data = df[["age", "income", "churned"]].rename(columns={"churned": "target"})
        split = len(data) - 3
        dataset = SplitDataset(train=data.iloc[:split], test=data.iloc[split:], validation=None)

        estimator = StatefulEstimator(
            node_id="rfc_real",
            calculator=RandomForestClassifierCalculator(),
            applier=RandomForestClassifierApplier(),
        )
        preds = estimator.fit_predict(
            dataset, target_column="target", config={"params": {"n_estimators": 10}}
        )
        assert len(preds["test"]) == len(dataset.test)

        report = estimator.evaluate(dataset, target_column="target")
        assert report["problem_type"] == "classification"
        assert "accuracy" in report["splits"]["test"].metrics
