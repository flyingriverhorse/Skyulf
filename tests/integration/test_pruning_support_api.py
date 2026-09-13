"""Pruning metadata must match the training graph without reading or fitting its data."""

from copy import deepcopy

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
    ("model_type", "space", "supported", "reason"),
    [
        ("sgd_classifier", {}, True, None),
        ("SGD Classifier", {}, True, None),
        ("sgd-classifier", {}, True, None),
        ("random_forest_classifier", {}, False, "incremental training"),
        ("RandomForestClassifier", {}, False, "incremental training"),
        ("random_forest", {}, False, "Ambiguous"),
        ("gaussian_nb", {}, False, "epoch budget"),
        ("missing_model", {}, False, "Unknown algorithm"),
        ("sgd_classifier", {"max_iter": [5]}, False, "searched max_iter"),
        ("sgd_classifier", {"early_stopping": [True, False]}, False, "early_stopping"),
        ("sgd_classifier", {"class_weight": ["balanced"]}, False, "balanced"),
    ],
)
def test_model_and_search_support(client, model_type, space, supported, reason):
    """The API must expose runtime eligibility and registry aliases, not a second model list."""
    body = _body(model_type)
    body["search_space"] = space
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["supported"] is supported
    assert result["reason"] is None if reason is None else reason in result["reason"]


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
    from skyulf.modeling._tuning.strategies.optuna import _unsupported_pruning_reason
    from skyulf.modeling.classification import SGDClassifierCalculator

    body = _body(steps=steps, cv_enabled=cv_enabled)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})

    class Store:
        """Supply controlled in-memory snapshots to the production fold resolver."""

        def load(self, key):
            """Keep raw and split snapshots consistent without fitting any transformers."""
            return frame if key == "load" else SplitDataset(train=frame, test=None)

    engine = PipelineEngine(Store(), None)
    engine._node_configs = {row["node_id"]: NodeConfig(**row) for row in body["pipeline"]["nodes"]}
    resolved, fallback = engine._resolve_fold_preprocessing(engine._node_configs["model"], "target")
    assert fallback is None
    preprocessing, frames, validation = resolved if resolved else (None, None, None)
    calculator = SGDClassifierCalculator()
    config = TuningConfig(strategy="optuna", cv_enabled=cv_enabled)
    estimator, search_config, _, _ = TuningCalculator(calculator)._prepare_search_estimator(
        calculator.model_class,
        config,
        preprocessing,
        frames,
        None,
        validation,
        None,
    )
    runtime_reason = _unsupported_pruning_reason(estimator, search_config.search_space)
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is (runtime_reason is None)


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
    assert response.json() == {"supported": True, "reason": None}
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
    assert response.json()["supported"] is False
    assert "incremental training" in response.json()["reason"]
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


@pytest.mark.parametrize("transformer", ["DropMissingRows", "StandardScaler"])
def test_separate_transformer_nodes_also_require_fold_preprocessing(client, transformer):
    """Ordinary converted Canvas nodes must match composite feature-engineering steps."""
    body = _body()
    nodes = body["pipeline"]["nodes"]
    nodes.insert(1, {"node_id": "feature", "step_type": transformer, "inputs": ["load"]})
    nodes[-1]["inputs"] = ["feature"]
    response = client.post("/pipeline/pruning-support", json=body)
    assert response.status_code == 200
    assert response.json()["supported"] is False
    assert "preprocessing runs inside each validation fold" in response.json()["reason"]
