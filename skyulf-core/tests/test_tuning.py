"""Tests for hyperparameter tuning."""

from typing import Any, cast

from tests.utils.dataset_loader import load_sample_dataset

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator


def test_tuner_grid_search(sample_classification_data):
    """Test Grid Search Tuning."""
    data = sample_classification_data.fillna(0).drop(columns=["category"])
    X = data.drop(columns=["target"])
    y = data["target"]

    base_calc = LogisticRegressionCalculator()
    tuner = TuningCalculator(base_calc)

    config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0], "solver": ["lbfgs"]},  # Keep it simple
        cv_folds=3,
    )

    _cfg = cast(Any, config)
    result_tuple = tuner.fit(
        X,
        y,
        config=(_cfg.to_dict() if hasattr(_cfg, "to_dict") else _cfg.__dict__),
    )

    # Unpack tuple (model, tuning_result)
    model, result = result_tuple

    assert result.best_score > 0
    assert "C" in result.best_params
    assert len(result.trials) == 3

    # Verify model is fitted
    assert hasattr(model, "predict")


def test_tuner_strategy_params(sample_classification_data):
    """Test passing strategy_params dynamically to halving."""
    data = sample_classification_data.fillna(0).drop(columns=["category"])
    X = data.drop(columns=["target"])
    y = data["target"]

    base_calc = LogisticRegressionCalculator()
    tuner = TuningCalculator(base_calc)

    config = TuningConfig(
        strategy="halving_grid",
        metric="accuracy",
        search_space={"C": [0.1, 1.0, 10.0]},
        strategy_params={"factor": 2, "min_resources": "exhaust"},
        cv_folds=2,
    )

    model, result = tuner.fit(X, y, config=config.__dict__)

    assert result.best_score > 0
    assert "C" in result.best_params


class TestRealShapedDataset:
    """Integration-style check against the checked-in ``customers.csv`` sample,
    which has missing values — closer to production data than the synthetic
    ``sample_classification_data`` fixture used elsewhere in this file.
    """

    def test_tune_churn_classifier_on_customers_data(self) -> None:
        df = load_sample_dataset("customers")
        # LogisticRegression can't handle NaN, so rows with missing
        # age/income are dropped rather than assumed clean.
        df = df.dropna(subset=["age", "income"])
        X = df[["age", "income"]]
        y = df["churned"]

        base_calc = LogisticRegressionCalculator()
        tuner = TuningCalculator(base_calc)

        config = TuningConfig(
            strategy="grid",
            metric="accuracy",
            search_space={"C": [0.1, 1.0]},
            cv_folds=2,
        )
        model, result = tuner.fit(X, y, config=config.__dict__)

        assert result.best_score > 0
        assert "C" in result.best_params
        assert hasattr(model, "predict")
