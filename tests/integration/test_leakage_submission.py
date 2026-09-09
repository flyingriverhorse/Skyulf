"""Exercise leakage validation through the pipeline submission HTTP boundary."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.database.engine import get_async_session
from backend.ml_pipeline._internal._routers import run_pipeline as run_pipeline_mod
from skyulf.registry import NodeRegistry

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "skyulf-core/tests/test_cases/leakage/registry_nodes.json"
)
_FIXTURE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_TRANSFORMER_CASES = (
    [(node_type, params, True) for node_type, params in _FIXTURE["learned_transformers"].items()]
    + [
        (node_type, params, False)
        for node_type, params in _FIXTURE["stateless_transformers"].items()
    ]
    + [(node_type, {}, False) for node_type in _FIXTURE["splitters"]]
)


@pytest.fixture
def submission_client(monkeypatch):
    """Keep graph processing real while isolating database and task submission effects."""
    app = FastAPI()
    app.include_router(run_pipeline_mod.router, prefix="/pipeline")
    app.dependency_overrides[get_async_session] = lambda: None
    submit = AsyncMock(return_value=(["review-job"], []))
    monkeypatch.setattr(run_pipeline_mod.limiter, "enabled", False)
    monkeypatch.setattr(run_pipeline_mod, "resolve_pipeline_nodes", AsyncMock(return_value=None))
    monkeypatch.setattr(run_pipeline_mod, "_submit_branch_jobs", submit)
    monkeypatch.setattr(run_pipeline_mod, "_dispatch_branch_tasks", AsyncMock())
    with TestClient(app) as client:
        yield client, submit


def _branched_payload():
    """Build one protected training path and one full-dataset scaler path."""
    return deepcopy(_FIXTURE["backend_graph"])


def _place_candidate(payload, placement):
    """Place the JSON candidate relative to the graph's training boundaries."""
    nodes = {node["node_id"]: node for node in payload["nodes"]}
    if placement == "before_split":
        nodes["split"]["inputs"] = ["scale"]
        payload["nodes"] = [
            node for node in payload["nodes"] if node["node_id"] != "unsafe_training"
        ]
    elif placement == "after_split":
        nodes["scale"]["inputs"] = ["split"]


def test_submission_rejects_unprotected_branch_before_creating_jobs(submission_client):
    """Partitioning must not hide an unsafe training branch from the default leakage gate."""
    client, submit = submission_client

    response = client.post("/pipeline/run", json=_branched_payload())

    assert response.status_code == 400
    assert "unsafe_training" in response.json()["detail"]
    submit.assert_not_awaited()


@pytest.mark.parametrize("on_leakage", ["warn", "ignore"])
def test_submission_preserves_nonblocking_leakage_modes(submission_client, on_leakage):
    """Users who explicitly allow leakage must still be able to submit all branches."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["metadata"] = {"on_leakage": on_leakage}

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 200
    assert len(submit.await_args.args[1]) == 2


def test_submission_of_safe_target_ignores_unexecuted_unsafe_branch(submission_client):
    """Training one selected node must not validate a sibling that will not execute."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["target_node_id"] = "safe_training"

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 200
    branches = submit.await_args.args[1]
    assert [[node.node_id for node in branch.nodes] for branch in branches] == [
        ["load", "split", "safe_training"]
    ]


def test_submission_of_unsafe_target_retains_full_graph_split_context(submission_client):
    """Selecting the unsafe branch must not turn its missing split into a no-split advisory."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["target_node_id"] = "unsafe_training"

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 400
    assert "unsafe_training" in response.json()["detail"]
    submit.assert_not_awaited()


def test_submission_preserves_advisory_for_a_graph_with_no_splitters(submission_client):
    """A genuinely splitter-free graph keeps its existing nonblocking advisory behavior."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["nodes"] = [
        node for node in payload["nodes"] if node["node_id"] not in {"split", "safe_training"}
    ]

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 200
    assert len(submit.await_args.args[1]) == 1


def test_submission_accepts_linear_unsplit_preprocessing_with_cv(submission_client):
    """Supported per-fold preprocessing must remain usable beside a split training branch."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["nodes"][-1]["params"] = {"cv_enabled": True}

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 200
    assert len(submit.await_args.args[1]) == 2


def test_submission_rejects_cv_with_unsupported_merged_preprocessing(submission_client):
    """The HTTP gate must reject CV that would score globally fitted preprocessing."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["nodes"].insert(
        -1,
        {
            "node_id": "other_scale",
            "step_type": "MinMaxScaler",
            "params": {},
            "inputs": ["load"],
        },
    )
    payload["nodes"][-1]["params"] = {"cv_enabled": True}
    payload["nodes"][-1]["inputs"] = ["scale", "other_scale"]

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 400
    assert "unsafe_training" in response.json()["detail"]
    submit.assert_not_awaited()


def test_json_fixture_covers_every_registered_library_node():
    """Adding a production node must require an explicit leakage expectation in the fixture."""
    metadata = NodeRegistry.get_all_metadata()
    expected = (
        set(_FIXTURE["learned_transformers"])
        | set(_FIXTURE["stateless_transformers"])
        | set(_FIXTURE["splitters"])
        | set(_FIXTURE["models"])
    )
    registered = {
        node_type
        for node_type in metadata
        if NodeRegistry.get_calculator(node_type).__module__.startswith("skyulf.")
    }

    assert expected == registered


@pytest.mark.parametrize(
    "node_type,params,learned", _TRANSFORMER_CASES, ids=lambda value: str(value)
)
@pytest.mark.parametrize("placement", ["before_split", "after_split", "unprotected_branch"])
@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
def test_json_transformer_submission_matrix(
    submission_client, node_type, params, learned, placement, mode
):
    """Every registered transformer must obey the leakage contract at the HTTP boundary."""
    client, submit = submission_client
    payload = _branched_payload()
    candidate = next(node for node in payload["nodes"] if node["node_id"] == "scale")
    candidate.update(step_type=node_type, params=deepcopy(params))
    payload["metadata"] = {"on_leakage": mode}
    _place_candidate(payload, placement)

    response = client.post("/pipeline/run", json=payload)

    blocked = learned and placement != "after_split" and mode == "raise"
    assert response.status_code == (400 if blocked else 200)
    if blocked:
        assert node_type in response.json()["detail"]
        submit.assert_not_awaited()
    else:
        submit.assert_awaited_once()


@pytest.mark.parametrize("model_type", _FIXTURE["models"])
@pytest.mark.parametrize("placement", ["after_split", "unprotected_branch"])
def test_json_model_submission_matrix(submission_client, model_type, placement):
    """Every model family must be protected by the same branch-level submission gate."""
    client, submit = submission_client
    payload = _branched_payload()
    payload["nodes"][-1]["params"]["algorithm"] = model_type
    _place_candidate(payload, placement)

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == (200 if placement == "after_split" else 400)
    if placement == "after_split":
        submit.assert_awaited_once()
    else:
        assert "unsafe_training" in response.json()["detail"]
        submit.assert_not_awaited()


@pytest.mark.parametrize("case", _FIXTURE["parameter_exemptions"], ids=lambda case: case["id"])
@pytest.mark.parametrize("placement", ["before_split", "unprotected_branch"])
def test_json_parameter_exemptions_remain_usable(submission_client, case, placement):
    """Fixed-parameter and target-only variants must not be blocked as learned feature fits."""
    client, submit = submission_client
    payload = _branched_payload()
    candidate = next(node for node in payload["nodes"] if node["node_id"] == "scale")
    candidate.update(step_type=case["transformer"], params=deepcopy(case["params"]))
    _place_candidate(payload, placement)

    response = client.post("/pipeline/run", json=payload)

    assert response.status_code == 200
    submit.assert_awaited_once()
