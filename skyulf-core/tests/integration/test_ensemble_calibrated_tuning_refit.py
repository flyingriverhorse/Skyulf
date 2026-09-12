"""Tuned calibration parameters must survive calculator refits and later CV."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.modeling.ensemble import (
    StackingClassifierCalculator,
    VotingClassifierApplier,
    VotingClassifierCalculator,
)


@pytest.mark.parametrize(
    "calculator_type", [VotingClassifierCalculator, StackingClassifierCalculator]
)
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_calibrated_best_parameters_survive_refit_and_cross_validation(
    calculator_type: type[Any], nested: bool
) -> None:
    """Post-tuning evaluation must use the selected base model, preserving reusable settings."""
    values, labels = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=73
    )
    X, y = pd.DataFrame(values), pd.Series(labels)
    calculator = calculator_type()
    settings = {
        "base_estimators": ["logistic_regression"],
        "base_estimator_params": {"logistic_regression": {"C": 2.0, "random_state": 19}},
        "calibrate_base_models": True,
        "calibration_method": "sigmoid",
        "calibration_cv": 2,
        "cv": 2,
        "n_jobs": 1,
    }
    original_settings = deepcopy(settings)
    calculator.prepare_tuning_params(settings)
    tuned, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            search_space={
                "logistic_regression__estimator__C": [0.01],
                "logistic_regression__method": ["isotonic"],
            },
            cv_folds=2,
            cv_type="stratified_k_fold",
        ),
    )
    best_params = deepcopy(result.best_params)
    config = {"params": result.best_params} if nested else result.best_params

    refitted = calculator.fit(X, y, config)

    base = refitted.named_estimators_["logistic_regression"]
    assert base.estimator.C == 0.01
    assert base.estimator.random_state == 19
    assert base.method == "isotonic"
    np.testing.assert_allclose(refitted.predict_proba(values), tuned.predict_proba(values))
    cv_result = perform_cross_validation(
        calculator,
        VotingClassifierApplier(),
        X,
        y,
        {"params": result.best_params},
        n_folds=2,
        cv_type="stratified_k_fold",
        random_state=42,
    )
    expected_loss = -cross_val_score(
        tuned,
        values,
        labels,
        scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
    ).mean()
    assert cv_result["aggregated_metrics"]["log_loss"]["mean"] == pytest.approx(expected_loss)
    assert result.best_params == best_params
    assert settings == original_settings
    fresh = calculator.fit(X, y, {})
    assert fresh.named_estimators_["logistic_regression"].estimator.C == 2.0
    assert fresh.named_estimators_["logistic_regression"].method == "sigmoid"
