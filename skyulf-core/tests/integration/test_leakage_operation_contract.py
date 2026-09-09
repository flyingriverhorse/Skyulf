"""Check operation-level leakage decisions independently of registry flags."""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from skyulf import SkyulfPipeline, validate_leakage_safety

_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "test_cases/leakage/operation_modes.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["id"])
@pytest.mark.parametrize("mode", ["raise", "warn", "ignore"])
@pytest.mark.parametrize("placement", ["before_split", "after_split", "no_split"])
def test_operation_decision_matches_actual_learning(case, mode, placement):
    """A coarse registry label must not hide learned modes or reject fixed rules."""
    candidate = {
        "name": "candidate",
        "transformer": case["transformer"],
        "params": deepcopy(case["params"]),
    }
    splitter = {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {"target_column": "target", "test_size": 0.25},
    }
    steps = [candidate, splitter]
    if placement == "after_split":
        steps.reverse()
    elif placement == "no_split":
        steps = [candidate]
    config = {"preprocessing": steps, "modeling": {}}
    violation = case["learns"] and placement == "before_split"
    if violation and mode == "raise":
        with pytest.raises(ValueError, match="Data leakage risk"):
            validate_leakage_safety(config, on_leakage=mode)
    else:
        messages = validate_leakage_safety(config, on_leakage=mode)
        expected_warning = mode != "ignore" and (violation or placement == "no_split")
        assert bool(messages) is expected_warning


@pytest.mark.parametrize(
    "case", [case for case in _FIXTURE["cases"] if case["learns"]], ids=lambda case: case["id"]
)
@pytest.mark.parametrize("method", ["fit", "get_fitted_split"])
def test_learned_operation_is_rejected_before_any_fit(case, method):
    """Both public fitting entry points must stop before an unsafe node learns."""
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "candidate", "transformer": case["transformer"], "params": case["params"]},
                {"name": "split", "transformer": "TrainTestSplitter", "params": {}},
            ],
            "modeling": {},
        }
    )
    data = pd.DataFrame({"x": range(12), "target": [0, 1] * 6})
    with pytest.raises(ValueError, match="Data leakage risk"):
        getattr(pipeline, method)(data, target_column="target")
    assert not pipeline.is_fitted()
