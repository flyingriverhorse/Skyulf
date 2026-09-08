"""Exercise operation-aware leakage at real API and fold-refit boundaries."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from backend.ml_pipeline._execution.engine._feature_eng import _step_learns_from_data
from tests.integration.test_leakage_submission import (
    submission_client,  # noqa: F401 - shared real-router pytest fixture
)

_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "skyulf-core/tests/test_cases/leakage/operation_modes.json"
    ).read_text(encoding="utf-8")
)
_CASES = [case for case in _FIXTURE["cases"] if case["transformer"] != "UnknownLeakagePlugin"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
@pytest.mark.parametrize("placement", ["before_split", "after_split", "unprotected_branch"])
def test_submission_uses_operation_semantics(submission_client, case, mode, placement):
    """Unsafe parameter modes must be stopped at submission before jobs exist."""
    client, submit = submission_client
    nodes = [
        {"node_id": "load", "step_type": "data_loader", "params": {}, "inputs": []},
        {
            "node_id": "split",
            "step_type": "TrainTestSplitter",
            "params": {"target_column": "target"},
            "inputs": ["load"],
        },
        {
            "node_id": "candidate",
            "step_type": case["transformer"],
            "params": deepcopy(case["params"]),
            "inputs": ["load"],
        },
        {
            "node_id": "train",
            "step_type": "training",
            "params": {"target_column": "target"},
            "inputs": ["candidate"],
        },
    ]
    if placement == "before_split":
        nodes[1]["inputs"] = ["candidate"]
        nodes[3]["inputs"] = ["split"]
    elif placement == "after_split":
        nodes[2]["inputs"] = ["split"]
    payload = {
        "pipeline_id": "leakage-submission",
        "nodes": nodes,
        "metadata": {"on_leakage": mode},
    }
    response = client.post("/pipeline/run", json=payload)
    blocked = case["learns"] and placement != "after_split" and mode == "raise"
    if blocked:
        assert response.status_code == 400, response.text
        assert "Data leakage risk" in response.json()["detail"]
        submit.assert_not_awaited()
    else:
        assert response.status_code == 200, response.text
        submit.assert_awaited_once()


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["id"])
def test_fold_refit_uses_the_same_operation_contract(case):
    """CV trunk reconstruction must not use a stale copy of admission rules."""
    step = {"transformer": case["transformer"], "params": deepcopy(case["params"])}
    assert _step_learns_from_data(step, target_column="target") is case["learns"]
