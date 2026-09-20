"""Temporal validation belongs after fold preprocessing at each estimator boundary."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.sklearn_wrapper import SklearnCalculator


class _DatePreprocessor:
    """Keep raw dates or explicitly extract numeric calendar features inside each fold."""

    def __init__(self, extract: bool = False) -> None:
        """Choose whether the model receives extracted features or retained dates."""
        self.extract = extract

    def fit_transform(self, X, y):
        """Apply the same feature contract to each fold's training rows."""
        return self.transform(X, y)

    def transform(self, X, y):
        """Expose numeric calendar parts only when extraction was explicitly selected."""
        if self.extract:
            return pd.DataFrame({"day": X["date"].dt.day}, index=X.index), y
        return X, y


def _data():
    """Use typed dates that sklearn trees previously accepted implicitly."""
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=18)}), np.arange(18)


def test_fold_step_rejects_retained_temporal_features():
    """A fold must reject dates before a tree silently accepts their physical values."""
    X, y = _data()
    step = FoldAwareModelStep(DecisionTreeRegressor(), _DatePreprocessor())
    with pytest.raises(ValueError, match="Raw temporal features"):
        step.fit(X, y)


def test_fold_step_predict_rejects_temporal_features():
    """Prediction must validate the final feature payload even without preprocessing."""
    X, y = _data()
    step = FoldAwareModelStep(DecisionTreeRegressor())
    step.fit(pd.DataFrame({"date": np.arange(18)}), y)
    with pytest.raises(ValueError, match="Raw temporal features"):
        step.predict(X)


@pytest.mark.parametrize("strategy", ["grid", "optuna"])
@pytest.mark.parametrize("extract", [False, True])
def test_tuning_validates_temporal_features_after_fold_preprocessing(strategy, extract):
    """Raw dates reach fold extraction, but retained dates cannot yield successful trials."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        pytest.importorskip("optuna_integration")
    X, y = _data()
    tuner = TuningCalculator(SklearnCalculator(DecisionTreeRegressor, {}, "regression"))
    config = TuningConfig(
        strategy=strategy,
        metric="neg_mean_squared_error",
        search_space={"max_depth": [2]},
        n_trials=1,
        cv_folds=2,
    )
    messages = []
    if extract:
        model, result = tuner.fit(
            X, y, config, preprocessing=_DatePreprocessor(True), log_callback=messages.append
        )
        assert np.isfinite(result.best_score)
        assert len(model.predict(X["date"].dt.day.to_numpy().reshape(-1, 1))) == len(y)
    else:
        with pytest.raises(ValueError):
            tuner.fit(X, y, config, preprocessing=_DatePreprocessor(), log_callback=messages.append)
        assert any("Raw temporal features" in message for message in messages)
