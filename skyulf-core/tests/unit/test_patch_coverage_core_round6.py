"""Round-6 patch-coverage tests for the tuning engine's multi-fold-error branch.

Closes the single partial left by the sixth Codecov patch report: both
directions of ``len(fold_errors) > 1`` in the all-trials-failed error message
of ``_run_grid_or_random_search`` (the "(N more fold failures suppressed)"
note).
"""

from typing import Any, cast

import pandas as pd
import pytest
from sklearn.model_selection import KFold

from skyulf.modeling._tuning import grid_random as grid_random_mod
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator

_X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
_Y = pd.Series([0, 1, 0, 1])


def test_all_trials_failed_with_single_fold_error(monkeypatch):
    tuner = TuningCalculator(LogisticRegressionCalculator())

    def fake_evaluate(*args, **kwargs):
        # fold_errors is the 10th positional arg of evaluate_search_candidates
        fold_errors = args[9] if len(args) >= 10 else kwargs["fold_errors"]
        fold_errors.append("single fold boom")
        return [], -float("inf"), None

    monkeypatch.setattr(grid_random_mod, "evaluate_search_candidates", fake_evaluate)
    config = TuningConfig(strategy="grid", search_space={"max_iter": [2]}, cv_folds=2)
    with pytest.raises(ValueError, match="First trial error: single fold boom") as excinfo:
        tuner._run_grid_or_random_search(
            _X, _Y, config, LogisticRegressionCalculator, KFold(n_splits=2), "accuracy", None, None
        )
    assert "more fold failures suppressed" not in str(excinfo.value)


def test_all_trials_failed_with_multiple_fold_errors():
    tuner = TuningCalculator(LogisticRegressionCalculator())
    config = TuningConfig(strategy="grid", search_space={"max_iter": [2, 3]}, cv_folds=2)
    # `object` cannot be instantiated with params, so every candidate/fold
    # fails and appends to fold_errors -> the suppressed-failures note renders.
    with pytest.raises(ValueError, match="more fold failures suppressed"):
        tuner._run_grid_or_random_search(
            _X, _Y, config, cast("Any", object), KFold(n_splits=2), "accuracy", None, None
        )
