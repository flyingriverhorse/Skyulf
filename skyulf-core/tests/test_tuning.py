"""Tests for hyperparameter tuning."""

from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.modeling.tuning.schemas import TuningConfig
from skyulf.modeling.tuning.engine import TuningCalculator


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

    result_tuple = tuner.fit(
        X, y, config=config.to_dict() if hasattr(config, "to_dict") else config.__dict__
    )

    # Unpack tuple (model, tuning_result)
    model, result = result_tuple

    assert result.best_score > 0
    assert "C" in result.best_params
    assert len(result.trials) == 3

    # Verify model is fitted
    assert hasattr(model, "predict")
