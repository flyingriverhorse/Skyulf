"""Inspection replay must reuse saved reports without inspecting prediction data again."""

from copy import deepcopy
from typing import Any, cast

import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

NODES = ("DatasetProfile", "DataSnapshot")


def _record(node, engine):
    """Save real training reports with nulls and a bounded snapshot."""
    training = pd.DataFrame({"x": [1.0, None, 3.0], "label": ["a", "b", "c"]})
    if engine == "polars":
        training = pl.from_pandas(training)
    config = {"n_rows": 2} if node == "DataSnapshot" else {}
    return {
        "name": "inspect",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(training, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_inspection_metadata_never_executes_or_rewrites_saved_reports(node, engine, monkeypatch):
    """Reading context must preserve the fitted report without invoking fit or apply."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)
    owner: Any = NodeRegistry.get_applier(node)

    def forbidden(*args, **kwargs):
        """Detect accidental statistics collection or data transformation."""
        raise AssertionError("Inspection metadata executed model code")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(owner, "apply", forbidden)
    assert get_inference_capability(node, record["params"], state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert owner.validate_inference_state(state) is state
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_inspection_probes_are_pure_passthroughs_with_different_prediction_data(
    node, engine, monkeypatch
):
    """Prediction values, order and empty schemas cannot replace the saved training report."""
    record = _record(node, engine)
    sample = pd.DataFrame({"x": [9.0, None, -2.0], "label": ["unseen", "a", None]})
    if engine == "polars":
        sample = pl.from_pandas(sample)
    state = deepcopy(record["artifact"])

    def forbidden(*args, **kwargs):
        """Make report recomputation during prediction observable."""
        raise AssertionError("Prediction must not regenerate the report")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    assert record["applier"].apply(sample, record["artifact"]) is sample
    target = object()
    payload = (sample, target)
    assert record["applier"].apply(payload, record["artifact"]) is payload
    report, result = _probe_step(
        record,
        {"name": "inspect", "transformer": node, "params": record["params"]},
        sample,
        engine,
        (1, 2),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert report["status"] == "passed", report
    assert report["state_validation"] == "node_owned" and report["context"] == "row"
    assert report["checks"] == [
        {"name": name, "status": "passed"}
        for name in ("full", "repeat", "chunks:1", "chunks:2", "reverse", "empty")
    ]
    assert len(result) == len(sample)
    assert artifact_digest(record["artifact"]) == artifact_digest(state)


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("mutation", ["type", "missing", "extra", "not_dict"])
def test_inspection_requires_its_own_saved_envelope(node, mutation):
    """Incorrect artifact identity must not obtain context from a different owner."""
    record = _record(node, "pandas")
    state: Any = record["artifact"]
    field = "profile" if node == "DatasetProfile" else "snapshot"
    if mutation == "type":
        state["type"] = "other"
    elif mutation == "missing":
        del state[field]
    elif mutation == "extra":
        state["new_behavior"] = True
    else:
        state = None
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, cast(Any, state), engine="pandas")


@pytest.mark.parametrize("node", NODES)
def test_inspection_ignores_opaque_report_content_and_declines_unknown_engine(node):
    """Unused fit reports need no new value codec or arbitrary runtime inspection."""
    state = _record(node, "pandas")["artifact"]
    field = "profile" if node == "DatasetProfile" else "snapshot"
    state[field] = object()
    owner: Any = NodeRegistry.get_applier(node)
    assert owner.validate_inference_state(state) is state
    assert get_inference_capability(node, {}, state, engine="pandas") is not None
    assert get_inference_capability(node, {}, state, engine="spark") is None
