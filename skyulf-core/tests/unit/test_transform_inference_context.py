"""Saved power transforms expose batch fallback without learning request statistics."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import PowerTransformer as SkPowerTransformer

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values=(1.0, 2.0, 4.0, 8.0)):
    """Use stable floats while retaining duplicate pandas row labels."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
    frame.index = [i // 2 for i in range(len(frame))]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, method="yeo-johnson", standardize=True):
    """Exercise genuine fit output for both public configuration shapes."""
    config = (
        {"columns": ["x"], "method": method, "standardize": standardize}
        if node == "PowerTransformer"
        else {"transformations": [{"column": "x", "method": method, "standardize": standardize}]}
    )
    return {
        "name": "transform",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(_frame(engine), config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _probe(record, sample, engine):
    """Use the same context-aware diagnostic as saved-model callers."""
    return _probe_step(
        record,
        {"name": "transform", "transformer": record["type"], "params": record["params"]},
        sample,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )


@pytest.mark.parametrize("node", ["GeneralTransformation", "PowerTransformer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_contract_inspection_does_not_execute(node, engine, monkeypatch):
    """Inspection must neither fit nor run transforms while describing global fallback."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Make hidden inspection-time execution fail immediately."""
        raise AssertionError("Unexpected execution")

    applier: Any = type(record["applier"])
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    monkeypatch.setattr(SkPowerTransformer, "fit", forbidden)
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability(node, {}, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", "preserve", "global"
    )
    assert get_inference_capability(node, {}, state, engine="spark") is None
    assert artifact_digest(state) == before


@pytest.mark.parametrize("node", ["GeneralTransformation", "PowerTransformer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method,bad", [("box-cox", -1.0), ("yeo-johnson", np.inf)])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_bad_neighbor_changes_full_request_fallback(
    node, engine, method, bad, standardize, monkeypatch
):
    """A bad neighbor leaves valid rows untransformed, so independent chunks are unsafe."""
    record = _record(node, engine, method, standardize)
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Saved inference must reconstruct learned parameters instead of fitting."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(SkPowerTransformer, "fit", forbidden)
    sample = _frame(engine, [2.0, bad])
    full = record["applier"].apply(sample, record["artifact"])
    single = record["applier"].apply(_frame(engine, [2.0]), record["artifact"])
    assert full["x"].to_numpy()[0] == 2.0
    assert single["x"].to_numpy()[0] != full["x"].to_numpy()[0]
    detail, output = _probe(record, sample, engine)
    assert detail["context"] == "global" and detail["status"] == "requires_context", detail
    assert output is sample
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "method",
    ["log", "sqrt", "square", "cube_root", "reciprocal", "exp", "square_root", "exponential"],
)
def test_general_simple_modes_replay_pointwise(engine, method, monkeypatch):
    """Simple GeneralTransformation rules retain exact row-local saved replay."""
    record = _record("GeneralTransformation", engine, method)

    def forbidden(*args, **kwargs):
        """A fixed formula must never call a calculator at prediction time."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("GeneralTransformation"), "fit", forbidden)
    detail, _ = _probe(record, _frame(engine, [None, -1, 0, 2, 4]), engine)
    assert detail["context"] == "row" and detail["status"] == "passed", detail


@pytest.mark.parametrize("node", ["GeneralTransformation", "PowerTransformer"])
@pytest.mark.parametrize(
    "change", ["type", "extra", "lambdas", "scale", "standardize", "missing_lambdas"]
)
def test_corrupt_power_state_is_not_a_context_contract(node, change):
    """Malformed learned vectors cannot silently obtain context metadata."""
    state: Any = deepcopy(_record(node, "pandas")["artifact"])
    rule = state if node == "PowerTransformer" else state["transformations"][0]
    if change == "type":
        state["type"] = "other"
    elif change == "extra":
        state["extra"] = True
    elif change == "lambdas":
        rule["lambdas"] = []
    elif change == "scale":
        rule["scaler_params"]["scale"] = ["broken"]
    elif change == "missing_lambdas":
        del rule["lambdas"]
    else:
        rule["standardize"] = "yes"
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_genuine_numpy_selection_and_flag(engine):
    """Real fit-produced NumPy names and flags remain inspectable without normalization."""
    config = {"columns": [np.str_("x")], "standardize": np.bool_(True)}
    state = NodeRegistry.get_calculator("PowerTransformer")().fit(_frame(engine), config)
    capability = get_inference_capability("PowerTransformer", config, state, engine=engine)
    assert capability is not None and capability.context == "global"


@pytest.mark.parametrize("node", ["GeneralTransformation", "PowerTransformer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_fitted_noop_has_row_context(node, engine):
    """Explicit empty selections contain no active transform or whole-batch fallback."""
    config = {"columns": []} if node == "PowerTransformer" else {"transformations": []}
    state = NodeRegistry.get_calculator(node)().fit(_frame(engine), config)
    record = {
        "name": "transform",
        "type": node,
        "params": config,
        "artifact": state,
        "applier": NodeRegistry.get_applier(node)(),
    }
    detail, output = _probe(record, _frame(engine, [None, 1, 2]), engine)
    assert detail["status"] == "passed" and detail["context"] == "row", detail
    assert len(output) == 3
