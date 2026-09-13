"""Pruning metadata must match the training graph without reading or fitting its data."""

from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.ml_pipeline.api import router


@pytest.fixture
def client():
    """Mount the production pipeline routes without application startup services."""
    app = FastAPI()
    app.include_router(router, prefix="/pipeline")
    with TestClient(app) as session:
        yield session


def _body(model_type="sgd_classifier", steps=None, **params):
    """Describe the same linear graph for API inspection and actual fold resolution."""
    nodes = [{"node_id": "load", "step_type": "data_loader"}]
    source = "load"
    if steps is not None:
        nodes.append(
            {
                "node_id": "features",
                "step_type": "feature_engineering",
                "inputs": [source],
                "params": {
                    "steps": [
                        {"name": f"step_{i}", "transformer": name, "params": {"columns": ["x"]}}
                        for i, name in enumerate(steps)
                    ]
                },
            }
        )
        source = "features"
    nodes.append(
        {
            "node_id": "model",
            "step_type": "training",
            "inputs": [source],
            "params": {"target_column": "target", "algorithm": model_type, **params},
        }
    )
    return {
        "model_type": model_type,
        "node_id": "model",
        "search_space": {},
        "pipeline": {"pipeline_id": "pruning-support", "nodes": nodes},
    }


@pytest.mark.parametrize(
    ("model_type", "space", "mode", "reason"),
    [
        ("sgd_classifier", {}, "iterations", "incremental training"),
        ("SGD Classifier", {}, "iterations", "incremental training"),
        ("sgd-classifier", {}, "iterations", "incremental training"),
        ("random_forest_classifier", {}, "folds", "folds"),
        ("RandomForestClassifier", {}, "folds", "folds"),
        ("random_forest", {}, "none", "Ambiguous"),
        ("gaussian_nb", {}, "folds", "folds"),
        ("missing_model", {}, "none", "Unknown algorithm"),
        ("sgd_classifier", {"max_iter": [5]}, "folds", "folds"),
        ("sgd_classifier", {"early_stopping": [True, False]}, "folds", "folds"),
        ("sgd_classifier", {"class_weight": ["balanced"]}, "folds", "folds"),
    ],
)
def test_model_and_search_support(client, model_type, space, mode, reason):
    """The API must expose runtime eligibility and registry aliases, not a second model list."""
    body = _body(model_type)
    body["search_space"] = space
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["supported"] is (mode != "none")
    assert result["mode"] == mode
    assert reason in result["reason"]


@pytest.mark.parametrize("cv_enabled", [False, True])
@pytest.mark.parametrize(
    "steps",
    [
        None,
        [],
        ["Split"],
        ["TrainTestSplitter", "feature_target_split"],
        ["DropMissingRows", "Split"],
        ["Split", "DropMissingRows"],
        ["Split", "StandardScaler"],
        ["StandardScaler"],
    ],
)
def test_graph_support_matches_actual_fold_resolution(client, steps, cv_enabled):
    """Stateless pre-split exclusions and post-split wrappers must match actual execution."""
    from backend.ml_pipeline._execution.engine import PipelineEngine
    from backend.ml_pipeline._execution.schemas import NodeConfig
    from skyulf.data.dataset import SplitDataset
    from skyulf.modeling._tuning.engine import TuningCalculator
    from skyulf.modeling._tuning.schemas import TuningConfig
    from skyulf.modeling._tuning.splitters import select_cv_by_type
    from skyulf.modeling.classification import SGDClassifierCalculator

    body = _body(steps=steps, tuning_config={"cv_enabled": cv_enabled})
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})

    class Store:
        """Supply controlled in-memory snapshots to the production fold resolver."""

        def load(self, key):
            """Keep raw and split snapshots consistent without fitting any transformers."""
            return frame if key == "load" else SplitDataset(train=frame, test=frame.iloc[:0])

    engine = PipelineEngine(MagicMock(load=Store().load), MagicMock())
    engine._node_configs = {row["node_id"]: NodeConfig(**row) for row in body["pipeline"]["nodes"]}
    resolved, fallback = engine._resolve_fold_preprocessing(engine._node_configs["model"], "target")
    assert fallback is None
    preprocessing, frames, validation = resolved if resolved else (None, None, None)
    calculator = SGDClassifierCalculator()
    config = TuningConfig(strategy="optuna", cv_enabled=cv_enabled)
    assert frames is not None if preprocessing is not None else validation is None
    runtime_support = TuningCalculator(calculator).pruning_support(
        config,
        preprocessing=preprocessing is not None,
        n_splits=select_cv_by_type(config, "classification").get_n_splits(),
    )
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json() == runtime_support


def test_query_preserves_saved_values_and_uses_advanced_defaults(client, monkeypatch):
    """Inspection may neither train nor apply stale Basic-mode hyperparameters."""
    from sklearn.linear_model import SGDClassifier

    from backend.ml_pipeline._execution.engine import PipelineEngine
    from backend.ml_pipeline._internal._routers.pruning import (
        PruningSupportRequest,
        get_pruning_support,
    )

    def fail_io(*args, **kwargs):
        """Capability checks must not initialize execution services or fit data."""
        pytest.fail("capability check attempted execution")

    monkeypatch.setattr(PipelineEngine, "__init__", fail_io)
    monkeypatch.setattr(SGDClassifier, "fit", fail_io)
    monkeypatch.setattr(SGDClassifier, "partial_fit", fail_io)
    body = _body(hyperparameters={"early_stopping": True, "max_iter": 0})
    body["strategy_params"] = {"pruner": "none"}
    before = deepcopy(body)
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is True
    assert response.json()["mode"] == "iterations"
    assert body == before
    request = PruningSupportRequest(**body)
    saved_request = deepcopy(request.model_dump())
    assert get_pruning_support(request).supported is True
    assert request.model_dump() == saved_request


@pytest.mark.parametrize(
    "defect", ["absent", "missing_input", "cycle", "duplicate", "merged", "bad_steps"]
)
def test_malformed_or_merged_graph_never_claims_support(client, defect):
    """Incomplete or ambiguous graphs must produce a visible reason instead of HTTP 500."""
    body = _body()
    nodes = body["pipeline"]["nodes"]
    if defect == "absent":
        body["node_id"] = "absent"
    elif defect == "missing_input":
        nodes[-1]["inputs"] = ["absent"]
    elif defect == "cycle":
        nodes[0]["inputs"] = ["model"]
    elif defect == "duplicate":
        nodes.append(deepcopy(nodes[-1]))
    elif defect == "merged":
        nodes.insert(1, {"node_id": "other", "step_type": "data_loader"})
        nodes[-1]["inputs"] = ["load", "other"]
    else:
        body = _body(steps=[])
        body["pipeline"]["nodes"][1]["params"]["steps"] = [None]
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert response.json()["reason"]


def test_explicit_pruning_opt_out_is_reported(client):
    """A compatibility opt-out must remain visible in the dropdown's reason."""
    body = _body()
    body["strategy_params"] = {"pruning": False}
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert "pruning=False" in response.json()["reason"]


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_ambiguous_alias_uses_the_same_task_resolution_as_training(client, task_type):
    """Legacy model aliases must resolve to the proper task rather than become unknown."""
    body = _body("random_forest", task_type=task_type)
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is True
    assert response.json()["mode"] == "folds"
    assert "Ambiguous" not in response.json()["reason"]


@pytest.mark.parametrize(
    "value",
    [
        {"type": "float", "low": 0.001, "high": 0.1},
        {"type": "categorical", "choices": [False, True]},
        None,
        "0.001",
    ],
)
def test_non_array_search_values_are_not_a_supported_canvas_wire_format(client, value):
    """The editor and execution accept candidate arrays, so invalid shapes must stay rejected."""
    body = _body()
    body["search_space"] = {"alpha": value}
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "search_space", "alpha"]


def test_learned_pre_split_graph_has_a_visible_safety_reason(client):
    """A graph blocked before training cannot be advertised as eligible for pruning."""
    response = client.post(
        "/pipeline/pruning-support",
        json=_body(steps=["StandardScaler", "Split"]),
    )
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert "before the first split" in response.json()["reason"]


@pytest.mark.parametrize(
    ("model_type", "cv_enabled", "mode"),
    [
        ("random_forest_classifier", True, "folds"),
        ("random_forest_classifier", False, "none"),
        ("logistic_regression", True, "folds"),
        ("svc", True, "folds"),
        ("gaussian_nb", True, "folds"),
        ("xgboost_classifier", False, "iterations"),
        ("lgbm_classifier", False, "iterations"),
    ],
)
def test_cv_and_preprocessing_select_the_actual_pruning_mode(client, model_type, cv_enabled, mode):
    """Ordinary preprocessing must retain native boosting or between-fold pruning."""
    body = _body(
        model_type,
        steps=["Split", "StandardScaler"],
        tuning_config={"cv_enabled": cv_enabled, "cv_folds": 3, "cv_type": "k_fold"},
    )
    body["strategy_params"] = {"pruner": "none"}
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == mode
    assert result["supported"] is (mode != "none")
    assert result["reason"]


@pytest.mark.parametrize("validation_size", [0.0, 0.2])
def test_validation_holdout_takes_precedence_over_post_tuning_cv(client, validation_size):
    """A single validation split cannot advertise between-fold stopping from the CV toggle."""
    body = _body(
        "random_forest_classifier",
        steps=["Split", "StandardScaler"],
        tuning_config={"cv_enabled": True, "cv_folds": 3},
    )
    body["pipeline"]["nodes"][1]["params"]["steps"][0]["params"]["validation_size"] = (
        validation_size
    )
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["supported"] is (validation_size == 0)
    assert result["mode"] == ("folds" if validation_size == 0 else "none")


@pytest.mark.parametrize("model_type", ["voting_classifier", "stacking_classifier"])
def test_ensemble_structure_is_prepared_like_advanced_training(client, model_type):
    """Ensemble choices must be resolved before inspecting their required estimator list."""
    body = _body(
        model_type,
        tuning_config={
            "cv_enabled": True,
            "cv_folds": 3,
            "base_estimators": ["random_forest", "svc"],
            "voting": "hard",
            "tune_base_models": True,
        },
    )
    before = deepcopy(body)
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["mode"] == "folds"
    assert response.json()["supported"] is True
    assert body == before


@pytest.mark.parametrize(
    ("model_type", "mode"),
    [
        ("random_forest_classifier", "folds"),
        ("ridge_regression", "folds"),
        ("sgd_classifier", "folds"),
        ("xgboost_classifier", "iterations"),
        ("lgbm_classifier", "iterations"),
    ],
)
def test_canvas_default_search_candidates_match_runtime_capability(client, model_type, mode):
    """The real provider's candidate lists must remain eligible with CV and preprocessing."""
    from skyulf.modeling.hyperparameters import get_default_search_space

    body = _body(
        model_type,
        steps=["Split", "StandardScaler"],
        tuning_config={"cv_enabled": True, "cv_folds": 3},
    )
    body["search_space"] = get_default_search_space(model_type, "optuna")
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["mode"] == mode
    assert response.json()["supported"] is True


@pytest.mark.parametrize("transformer", ["DropMissingRows", "StandardScaler"])
def test_separate_transformer_nodes_also_require_fold_preprocessing(client, transformer):
    """Ordinary converted Canvas nodes must match composite feature-engineering steps."""
    body = _body()
    nodes = body["pipeline"]["nodes"]
    nodes.insert(1, {"node_id": "feature", "step_type": transformer, "inputs": ["load"]})
    nodes[-1]["inputs"] = ["feature"]
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is True
    assert response.json()["mode"] == "folds"
