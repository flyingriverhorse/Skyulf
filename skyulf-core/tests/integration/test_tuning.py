"""Tests for hyperparameter tuning."""

from typing import Any, cast

from sklearn.ensemble import RandomForestClassifier
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import (
    CalibratedClassifierCalculator,
    LogisticRegressionCalculator,
)


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

    def test_tune_threshold_on_customers_data(self) -> None:
        """F-13 end-to-end: validation split → threshold selected on it →
        applier predictions honour the cutoff."""
        from skyulf.modeling._tuning.engine import TuningApplier
        from skyulf.modeling.classification import LogisticRegressionApplier

        df = load_sample_dataset("customers")
        df = df.dropna(subset=["age", "income"])
        X = df[["age", "income"]]
        y = df["churned"]

        n_train = int(len(df) * 0.8)
        X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
        X_val, y_val = X.iloc[n_train:], y.iloc[n_train:]

        config = TuningConfig(
            strategy="grid",
            metric="f1",
            search_space={"C": [0.1, 1.0]},
            cv_folds=2,
            tune_threshold=True,
        )
        model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
            X_train, y_train, config=config.__dict__, validation_data=(X_val, y_val)
        )

        assert result.decision_thresholds is not None
        assert result.decision_threshold_metric == "f1"
        positive = model.classes_[1]
        assert 0.0 < result.decision_thresholds[positive] < 1.0

        preds = TuningApplier(LogisticRegressionApplier()).predict(X_val, (model, result))
        assert len(preds) == len(X_val)
        assert set(preds.unique()).issubset(set(model.classes_))


def test_tuner_resolves_base_estimator_for_calibrated_classifier(sample_classification_data):
    """OC-66: the user-selected ``base_estimator`` must survive tuning.

    Before the fix, ``CalibratedClassifierCV``'s ``base_estimator`` string was
    dropped by ``filter_params_to_signature`` (it is not a constructor param),
    so every tuned model silently fell back to the default LogisticRegression.
    """
    data = sample_classification_data.fillna(0).drop(columns=["category"])
    X = data.drop(columns=["target"])
    y = data["target"]

    base_calc = CalibratedClassifierCalculator()
    base_calc.prepare_tuning_params({"base_estimator": "random_forest"})
    tuner = TuningCalculator(base_calc)

    config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"method": ["sigmoid", "isotonic"]},
        cv_folds=3,
    )
    model, result = tuner.fit(X, y, config=config.__dict__)

    assert result.best_score > 0
    # The tuned model's base estimator must be the user's choice, not the default.
    assert isinstance(model.estimator, RandomForestClassifier)
