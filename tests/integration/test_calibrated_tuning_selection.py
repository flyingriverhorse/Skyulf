"""Exercise Canvas calibration search choices through backend preparation and real folds."""

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.config import get_settings
from backend.ml_pipeline._execution.engine._node_runners import NodeRunnersMixin
from backend.ml_pipeline._execution.schemas import NodeConfig
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling.classification import CalibratedClassifierCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.mark.parametrize("strategy", ["grid", "optuna"])
def test_canvas_calibration_search_selection_survives_fold_refit(strategy, monkeypatch):
    """Canvas-selected bases must survive both direct candidate fitting and nested search routing."""
    settings = get_settings()
    monkeypatch.setattr(settings, "TUNING_N_JOBS", 1)
    monkeypatch.setattr(settings, "TUNING_PARALLEL_BACKEND", "")
    node = NodeConfig(
        node_id="classification",
        step_type="training",
        inputs=["scale"],
        params={
            "run_mode": "tuned",
            "target_column": "target",
            "algorithm": "calibrated_classifier",
            "tuning_config": {
                "strategy": strategy,
                "metric": "accuracy",
                "n_trials": 1,
                "cv_enabled": True,
                "cv_folds": 2,
                "random_state": 7,
                "search_space": {
                    "base_estimator": ["random_forest"],
                    "method": ["isotonic"],
                    "cv": [3],
                },
            },
        },
    )
    original_params = deepcopy(node.params)
    calculator = CalibratedClassifierCalculator()
    tuning_config = NodeRunnersMixin()._prepare_tuning_config(node, calculator)

    rng = np.random.default_rng(42)
    y = pd.Series(np.tile([0, 1], 48), name="target")
    X = pd.DataFrame({"signal": y + rng.normal(size=96), "noise": rng.normal(size=96)})
    preprocessing = FeatureEngineerFoldAdapter(
        steps_config=[
            {
                "name": "scale",
                "transformer": "StandardScaler",
                "params": {"columns": ["signal", "noise"]},
            }
        ],
        target_column="target",
    )
    logs: list[str] = []
    model, result = TuningCalculator(calculator).fit(
        X, y, tuning_config, preprocessing=preprocessing, log_callback=logs.append
    )

    assert all(
        isinstance(fold.estimator, RandomForestClassifier) for fold in model.calibrated_classifiers_
    )
    assert {fold.estimator.random_state for fold in model.calibrated_classifiers_} == {7}
    assert len(model.calibrated_classifiers_) == 3
    assert model.method == "isotonic"
    assert result.n_trials == 1
    assert np.isfinite(result.best_score)
    assert len(result.trials) == 1
    assert np.isfinite(result.trials[0]["score"])
    public_result = json.loads(
        json.dumps({"best_params": result.best_params, "trials": result.trials}, allow_nan=False)
    )
    expected_params = {"base_estimator": "random_forest", "method": "isotonic", "cv": 3}
    assert public_result["best_params"] == expected_params
    assert public_result["trials"][0]["params"] == expected_params
    if strategy == "optuna":
        assert any("fold-aware estimator" in message for message in logs)
    assert node.params == original_params
