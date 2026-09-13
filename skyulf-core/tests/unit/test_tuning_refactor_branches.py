"""Preserve tuning diagnostics and leakage decisions after helper extraction."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from skyulf import validate_leakage_safety
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.modeling.regression import RidgeRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.mark.parametrize("with_callback", [False, True])
def test_search_and_refit_convergence_are_reported_without_aborting(caplog, with_callback):
    """SDK callers without callbacks still receive separate search and refit diagnostics."""
    features, target = make_classification(
        n_samples=60, n_features=5, class_sep=0.2, random_state=42
    )
    logs = []
    config = TuningConfig(
        strategy="grid", metric="accuracy", search_space={"max_iter": [1]}, cv_folds=2
    )

    with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
            pd.DataFrame(features * 1000.0),
            pd.Series(target),
            config,
            log_callback=logs.append if with_callback else None,
        )

    search_messages = [
        record.message for record in caplog.records if "candidate fit(s)" in record.message
    ]
    refit_messages = [
        record.message for record in caplog.records if "Final refit of" in record.message
    ]
    assert len(search_messages) == len(refit_messages) == 1
    assert search_messages[0].startswith("2 candidate fit(s)")
    assert "LogisticRegression" in refit_messages[0]
    assert not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
    assert all(message in logs for message in search_messages + refit_messages) == with_callback
    assert result.best_params == {"max_iter": 1}
    assert result.n_trials == 1
    assert 0.0 <= result.best_score <= 1.0
    assert model.predict(features).shape == target.shape


def test_final_refit_reemits_unrelated_warning_with_original_origin(monkeypatch):
    """A successful final refit must not swallow estimator diagnostics other than convergence."""
    features = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    target = features["feature"] * 2.0 + 1.0
    original_fit = Ridge.fit

    def warn_on_full_refit(self, X, y, sample_weight=None):
        """Emit a deterministic diagnostic only when fitting the final full dataset."""
        if len(X) == len(features):
            warnings.warn("Final training diagnostic", UserWarning, stacklevel=1)
        return original_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", warn_on_full_refit)
    with pytest.warns(UserWarning, match="Final training diagnostic") as caught:
        model, result = TuningCalculator(RidgeRegressionCalculator()).fit(
            features,
            target,
            TuningConfig(strategy="grid", metric="mse", search_space={"alpha": [0.0]}, cv_folds=2),
        )

    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    assert result.best_params == {"alpha": 0.0}
    assert result.best_score == pytest.approx(0.0)
    np.testing.assert_allclose(model.predict(features.to_numpy()), target)


@pytest.mark.parametrize("named_training_frames", [False, True])
def test_halving_without_required_frames_scores_raw_holdout_without_callback(named_training_frames):
    """Missing training or validation frames must preserve the documented raw-score fallback."""
    features = pd.DataFrame({"feature": np.arange(30, dtype=float) * 10.0})
    target = features["feature"] * 2.0 + 1.0
    train_x, val_x = features.iloc[:20], features.iloc[20:]
    train_y, val_y = target.iloc[:20], target.iloc[20:]
    preprocessing = FeatureEngineerFoldAdapter(
        [
            {
                "name": "scale_feature",
                "transformer": "StandardScaler",
                "params": {"columns": ["feature"]},
            }
        ],
        target_column="target",
    )

    result = TuningCalculator(RidgeRegressionCalculator()).tune(
        train_x.to_numpy(),
        train_y.to_numpy(),
        TuningConfig(
            strategy="halving_grid",
            metric="mse",
            search_space={"alpha": [1.0, 10.0]},
            strategy_params={"min_resources": len(features)},
        ),
        validation_data=(val_x.to_numpy(), val_y.to_numpy()),
        preprocessing=preprocessing,
        preprocessing_frames=(train_x, train_y) if named_training_frames else None,
    )

    expected_scores = {
        alpha: -mean_squared_error(val_y, Ridge(alpha=alpha).fit(train_x, train_y).predict(val_x))
        for alpha in (1.0, 10.0)
    }
    assert result.best_params == {"alpha": 1.0}
    assert result.best_score == pytest.approx(expected_scores[1.0])
    assert result.n_trials == 2
    assert {trial["params"]["alpha"]: trial["score"] for trial in result.trials} == pytest.approx(
        expected_scores
    )


@pytest.mark.parametrize("column_types", [None, ["feature"]])
def test_leakage_diagnostics_keep_order_and_reject_unrecognized_cast_mapping(column_types):
    """Malformed casts remain unsafe and warnings retain the first split and declared order."""
    config = {
        "preprocessing": [
            {"transformer": "Casting", "params": {"column_types": {"feature": "float64"}}},
            {"transformer": "LabelEncoder", "params": {"columns": ["target"]}},
            {"transformer": "SimpleImputer", "params": {"strategy": "mean"}},
            {"transformer": "Casting", "params": {"column_types": column_types}},
            {"transformer": "UnregisteredLearnedStep"},
            {"transformer": "TrainTestSplitter", "params": {"target_column": "target"}},
            {"transformer": "StandardScaler"},
            {"transformer": "TrainTestSplitter", "params": {"target_column": "other_target"}},
        ]
    }

    diagnostics = validate_leakage_safety(config, on_leakage="warn")

    assert len(diagnostics) == 3
    assert [message.split(" is configured")[0] for message in diagnostics] == [
        "Step 2 ('SimpleImputer')",
        "Step 3 ('Casting')",
        "Step 4 ('UnregisteredLearnedStep')",
    ]
    assert all("step 5, 'TrainTestSplitter'" in message for message in diagnostics)
    assert "not a known node" in diagnostics[-1]
    with pytest.raises(ValueError) as caught:
        validate_leakage_safety(config)
    assert str(caught.value) == "Data leakage risk:\n" + "\n".join(diagnostics)
    assert validate_leakage_safety(config, on_leakage="ignore") == []
