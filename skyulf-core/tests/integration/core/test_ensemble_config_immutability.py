"""Ensemble fits must not persist temporary overrides into caller-owned settings."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.ensemble import (
    StackingClassifierCalculator,
    StackingRegressorCalculator,
    VotingClassifierCalculator,
    VotingRegressorCalculator,
)


@pytest.mark.parametrize(
    "calculator_type",
    [
        VotingClassifierCalculator,
        VotingRegressorCalculator,
        StackingClassifierCalculator,
        StackingRegressorCalculator,
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("captured", [False, True], ids=["direct", "tuning-refit"])
def test_temporary_overrides_leave_ensemble_config_reusable(
    calculator_type: type[Any], nested: bool, captured: bool
) -> None:
    """A later fit without overrides must recover the original base-model parameters."""
    calculator = calculator_type()
    params: dict[str, Any] = {
        "base_estimators": ["decision_tree"],
        "base_estimator_params": {"decision_tree": {"max_depth": 2, "min_samples_leaf": 1}},
        "n_jobs": 1,
    }
    if calculator.IS_STACKING:
        params.update(
            {
                "cv": 2,
                "final_estimator": "decision_tree",
                "final_estimator_params": {"max_depth": 2},
            }
        )
    config = {"params": params} if nested else params
    overrides = {"decision_tree__max_depth": 4, "decision_tree__min_samples_leaf": 3}
    if calculator.IS_STACKING:
        overrides["final_estimator__max_depth"] = 3
    if captured:
        calculator.prepare_tuning_params(config)
        fit_config = {"params": overrides} if nested else overrides
    else:
        params.update(overrides)
        fit_config = config
    original_config, original_fit_config = deepcopy(config), deepcopy(fit_config)
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series([0] * 20 + [1] * 20)

    model = calculator.fit(X, y, fit_config)

    assert model.named_estimators_["decision_tree"].max_depth == 4
    assert model.named_estimators_["decision_tree"].min_samples_leaf == 3
    if calculator.IS_STACKING:
        assert model.final_estimator_.max_depth == 3
    assert config == original_config
    assert fit_config == original_fit_config

    if captured:
        second_config: dict[str, Any] = {"params": {}} if nested else {}
    else:
        for key in overrides:
            params.pop(key)
        second_config = config
    expected_config, expected_second_config = deepcopy(config), deepcopy(second_config)
    second_model = calculator.fit(X, y, second_config)

    assert second_model.named_estimators_["decision_tree"].max_depth == 2
    assert second_model.named_estimators_["decision_tree"].min_samples_leaf == 1
    if calculator.IS_STACKING:
        assert second_model.final_estimator_.max_depth == 2
    assert config == expected_config
    assert second_config == expected_second_config
