import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering.modeling import hyperparameter_tuning_tasks as tuning_tasks
    from core.feature_engineering.modeling.model_training_tasks import CrossValidationConfig
except ImportError:  # pragma: no cover - allow running without installation
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering.modeling import hyperparameter_tuning_tasks as tuning_tasks
    from core.feature_engineering.modeling.model_training_tasks import CrossValidationConfig


class _StubSpec:
    key = "logistic_regression"
    default_params = {"solver": "lbfgs", "penalty": "l2"}

    @staticmethod
    def factory(**_kwargs):  # pragma: no cover - factory unused in tested function
        raise AssertionError("factory should not be called")


class _StubJob:
    baseline_hyperparameters = None
    random_state = 123
    n_iterations = 5
    scoring = "accuracy"
    search_strategy = "grid"
    cross_validation = {"enabled": True}
    search_space = {"solver": ["lbfgs", "saga"]}
    model_type = "logistic_regression"
    dataset_source_id = "dataset"
    pipeline_id = "pipeline"
    node_id = "node"


def _search_config(search_space: dict[str, list[object]]) -> tuning_tasks.SearchConfiguration:
    return tuning_tasks.SearchConfiguration(
        strategy="grid",
        selected_strategy="grid",
        search_space=search_space,
        n_iterations=10,
        scoring="accuracy",
        random_state=42,
        cross_validation=CrossValidationConfig(
            enabled=True,
            strategy="kfold",
            folds=3,
            shuffle=True,
            random_state=7,
            refit_strategy="train_only",
        ),
    )


def test_prepare_search_parameters_sanitizes_logistic(monkeypatch):
    node_config = {
        "baseline_hyperparameters": json.dumps({"solver": "sag", "penalty": "elasticnet", "l1_ratio": 0.7})
    }
    search_space = {
        "solver": ["sag", "lbfgs", "newton-cg"],
        "penalty": ["elasticnet", "l1", "l2"],
        "l1_ratio": [0.2, 0.6],
        "unused_param": [1],
    }

    base_params, filtered_space = tuning_tasks._prepare_search_parameters(
        _StubJob(),
        node_config,
        _StubSpec(),
        _search_config(search_space),
    )

    assert base_params["solver"] in {"lbfgs", "saga"}
    assert base_params["penalty"] in {"l2", "none"}
    assert "l1_ratio" not in filtered_space
    assert "unused_param" in filtered_space


def test_prepare_search_parameters_raises_when_space_empty():
    node_config: dict[str, object] = {}
    search_space: dict[str, list[int]] = {}

    with pytest.raises(ValueError, match="Search space is empty"):
        tuning_tasks._prepare_search_parameters(
            _StubJob(),
            node_config,
            _StubSpec(),
            _search_config(search_space),
        )

