"""Training filters describe native apply separately from prediction skips."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import (
    ExecutionCapability,
    UnsupportedExecutionError,
    require_capability,
)
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry


def _frame(engine, data=None):
    """Use numeric features supported by filtering and optional native resampling."""
    data = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0]} if data is None else data
    return pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)


def _record(node, engine, config=None):
    """Snapshot real calculator state without assembling a pretend fitted artifact."""
    defaults = {"Oversampling": {"method": "random_over"}, "Undersampling": {}}
    config = defaults.get(node, {}) if config is None else config
    return {
        "name": "training",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(_frame(engine), config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _forbidden(*args, **kwargs):
    """Fail if metadata or an ordinary prediction skip executes trusted training code."""
    raise AssertionError("Unexpected fit/apply/callback")


def _probe(record, sample, engine, *, active):
    """Use the unchanged strict probe with the real pipeline's active/skip decision."""
    return _probe_step(
        record,
        {"name": record["name"], "transformer": record["type"], "params": record["params"]},
        sample,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=active,
        project_sha=None,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,effect,context",
    [
        ("DropMissingRows", "filter", "row"),
        ("Oversampling", "expand", "global"),
        ("Undersampling", "filter", "global"),
    ],
)
def test_real_training_state_declares_native_context(node, effect, context, engine, monkeypatch):
    """Skipped prediction nodes still own truthful descriptions of direct native apply."""
    record = _record(node, engine)
    before = pickle.dumps(record["artifact"])
    owner: Any = type(record["applier"])
    monkeypatch.setattr(owner, "apply", _forbidden)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    assert owner.validate_inference_state(record["artifact"]) is record["artifact"]
    assert get_inference_capability(node, {}, record["artifact"], engine=engine) == (
        ExecutionCapability(engine, "apply", "local", effect, context)
    )
    assert get_inference_capability(node, {}, record["artifact"], engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", engine, config=record["params"])
    assert pickle.dumps(record["artifact"]) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "config,positions",
    [
        ({}, [0]),
        ({"how": "all"}, [0, 1, 2]),
        ({"threshold": np.int64(1), "missing_threshold": "unused"}, [0, 1, 2]),
        ({"missing_threshold": np.float64(50)}, [0, 1, 2]),
        ({"subset": (np.str_("x"),)}, [0, 2]),
        ({"subset": ["unseen"]}, [0]),
    ],
)
def test_native_missing_filter_replays_chunks_order_empty_and_target_alignment(
    engine, config, positions, monkeypatch
):
    """Row filtering must retain native threshold precedence, missing-column fallback and X/y order."""
    record = _record("DropMissingRows", engine, config)
    sample = _frame(engine, {"x": [1.0, None, 3.0, None], "z": [1.0, 2.0, None, None]})
    if engine == "pandas":
        sample.index = [0, 0, 1, 1]
    before = artifact_digest(record)
    monkeypatch.setattr(NodeRegistry.get_calculator("DropMissingRows"), "fit", _forbidden)
    apply = record["applier"].apply
    state = record["artifact"]
    full, selected = apply((sample, np.arange(4)), state)
    assert selected.tolist() == positions
    parts = [
        apply(sample.iloc[i : i + 1] if engine == "pandas" else sample.slice(i, 1), state)
        for i in range(4)
    ]
    chunks = pd.concat(parts) if engine == "pandas" else pl.concat(parts)
    assert chunks.equals(full)
    reversed_sample = sample.iloc[::-1] if engine == "pandas" else sample.reverse()
    reversed_output = apply(reversed_sample, state)
    restored = reversed_output.iloc[::-1] if engine == "pandas" else reversed_output.reverse()
    assert restored.equals(full)
    empty = sample.iloc[:0] if engine == "pandas" else sample.head(0)
    assert apply(empty, state).equals(empty)
    capability = get_inference_capability("DropMissingRows", {}, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("missing", [False, True])
def test_active_missing_filter_keeps_strict_prediction_row_guard(engine, missing, monkeypatch):
    """An explicitly active diagnostic must report row loss instead of disguising it as a skip."""
    record = _record("DropMissingRows", engine)
    monkeypatch.setattr(NodeRegistry.get_calculator("DropMissingRows"), "fit", _forbidden)
    sample = _frame(engine, {"x": [1.0, None if missing else 2.0, 3.0]})
    detail, output = _probe(record, sample, engine, active=True)
    assert detail["context"] == "row" and detail["row_effect"] == "filter", detail
    assert detail["action"] == "apply" and detail["state_validation"] == "node_owned", detail
    if missing:
        assert detail["status"] == "failed" and output is sample, detail
        assert detail["checks"][0]["error_type"] == "ValueError", detail
    else:
        assert output.equals(sample)
        assert detail["status"] == "passed", detail


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,rows", [("DropMissingRows", 8), ("Oversampling", 12), ("Undersampling", 6)]
)
def test_real_training_execution_then_prediction_skips_without_apply(
    node, rows, engine, monkeypatch
):
    """Saved pipeline prediction skips training transforms, including nullable and empty requests."""
    if node != "DropMissingRows":
        pytest.importorskip("imblearn")
    config = {"method": "random_over"} if node == "Oversampling" else {}
    step = {"name": "training", "transformer": node, "params": config}
    engineer = FeatureEngineer([step])
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0]
    if node == "DropMissingRows":
        values[0] = None
    frame = _frame(engine, {"x": values})
    target = pd.Series([0] * 6 + [1] * 3, name="target")
    target = target if engine == "pandas" else pl.from_pandas(target)
    (trained, trained_y), _ = engineer.fit_transform((frame, target))
    assert len(trained) == len(trained_y) == rows
    record = engineer.fitted_steps[0]
    before = artifact_digest(record)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier(node), "apply", _forbidden)
    sample = _frame(engine, {"x": [None, 99.0, -2.0]})
    for request in (sample, sample.iloc[:0] if engine == "pandas" else sample.head(0)):
        assert engineer.transform(request, preserve_rows=True).equals(request)
        detail, output = _probe(record, request, engine, active=bool(engineer._transform_steps()))
        assert output is request and detail["status"] == "skipped", detail
        assert detail["action"] == "skip_preserve_rows" and detail["checks"] == [], detail
        assert detail["state_validation"] == "unavailable", detail
    if node != "DropMissingRows":
        assert engineer.transform(sample, preserve_rows=False).equals(sample)
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["Oversampling", "Undersampling"])
def test_active_sampler_requires_global_context_and_no_target_native_noop(
    node, engine, monkeypatch
):
    """No-target apply is a no-op, but state alone cannot prove the request has no target."""
    record = _record(node, engine)
    sample = _frame(engine, {"x": [None, 1.0, 2.0]})
    before = artifact_digest(record)
    assert record["applier"].apply(sample, record["artifact"]).equals(sample)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    monkeypatch.setattr(NodeRegistry.get_applier(node), "apply", _forbidden)
    detail, output = _probe(record, sample, engine, active=True)
    assert detail["status"] == "requires_context" and detail["context"] == "global", detail
    assert detail["state_validation"] == "node_owned" and detail["checks"] == [], detail
    assert output is sample and artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["Oversampling", "Undersampling"])
def test_mixed_sampling_effects_remain_unknown_after_real_execution(node, engine):
    """Tomek cleanup can shrink oversampling output and replacement can duplicate selected rows."""
    pytest.importorskip("imblearn")
    if node == "Oversampling":
        config = {"method": "smote_tomek", "k_neighbors": 1}
        values, labels = [0.0, 0.1, 1.0, 1.1, 2.0, 2.1], [0, 1, 0, 1, 0, 1]
    else:
        config = {"replacement": True, "random_state": 12}
        values, labels = list(range(8)), [0] * 6 + [1] * 2
    frame = _frame(engine, {"x": values})
    target = pd.Series(labels, name="target")
    target = target if engine == "pandas" else pl.from_pandas(target)
    record = _record(node, engine, config)
    before = artifact_digest(record)
    result, target_out = record["applier"].apply((frame, target), record["artifact"])
    assert len(result) == len(target_out) == (0 if node == "Oversampling" else 4)
    if node == "Undersampling":
        assert result["x"].to_list() == [3, 3, 6, 7]
    assert get_inference_capability(node, {}, record["artifact"], engine=engine) is None
    assert artifact_digest(record) == before


@pytest.mark.parametrize(
    "node,config",
    [
        ("Oversampling", {"sampling_strategy": _forbidden}),
        ("Oversampling", {"method": "smote", "k_neighbors": _forbidden}),
        ("Oversampling", {"method": "svm_smote", "svm_estimator": _forbidden}),
        ("Oversampling", {"method": "kmeans_smote", "kmeans_estimator": _forbidden}),
        ("Undersampling", {"method": "nearmiss", "n_neighbors": _forbidden}),
        ("Undersampling", {"sampling_strategy": _forbidden}),
    ],
)
def test_custom_sampler_callbacks_are_not_certified_by_class_identity(node, config):
    """An unexecuted custom sampler dependency cannot inherit a built-in context promise."""
    record = _record(node, "pandas", config)
    before = pickle.dumps(record["artifact"])
    assert get_inference_capability(node, {}, record["artifact"], engine="pandas") is None
    assert pickle.dumps(record["artifact"]) == before


@pytest.mark.parametrize("node", ["DropMissingRows", "Oversampling", "Undersampling"])
@pytest.mark.parametrize("defect", ["container", "type", "dispatch"])
def test_malformed_effective_state_is_rejected(node, defect):
    """Broken saved dispatch must fail inspection without executing the transformation."""
    record = _record(node, "pandas")
    state = record["artifact"]
    if defect == "container":
        state = []
    elif defect == "type":
        state["type"] = "different"
    else:
        state["how" if node == "DropMissingRows" else "method"] = ["invalid"]
    owner: Any = type(record["applier"])
    with pytest.raises(ValueError):
        owner.validate_inference_state(state)


@pytest.mark.parametrize("node", ["Oversampling", "Undersampling"])
def test_legacy_default_sampler_and_unused_options_keep_native_meaning(node):
    """Optional sampler defaults and unused fit metadata must not become a new transport gate."""
    config = (
        {
            "method": "random_over",
            "svm_estimator": _forbidden,
            "synthetic_weight": {"report": "unused"},
        }
        if node == "Oversampling"
        else {}
    )
    record = _record(node, "pandas", config)
    for state in (record["artifact"], {"type": record["artifact"]["type"]}):
        before = pickle.dumps(state)
        capability = get_inference_capability(node, {}, state, engine="pandas")
        assert capability is not None and capability.context == "global"
        assert pickle.dumps(state) == before


def test_positional_missing_threshold_does_not_claim_row_context():
    """A pandas vector threshold is request-position dependent even though native apply accepts it."""
    record = _record("DropMissingRows", "pandas", {"threshold": np.array([0, 1])})
    sample = _frame("pandas", {"x": [None, 1.0]})
    assert record["applier"].apply(sample, record["artifact"]).equals(sample)
    assert (
        get_inference_capability("DropMissingRows", {}, record["artifact"], engine="pandas") is None
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("field", ["threshold", "missing_threshold"])
def test_native_numpy_boolean_threshold_keeps_row_context(engine, field):
    """NumPy boolean scalars retain the native filtering contract of Python booleans."""
    record = _record("DropMissingRows", engine, {field: np.bool_(True)})
    sample = _frame(engine, {"x": [1.0, None, 3.0]})
    before = pickle.dumps(record["artifact"])
    output, target = record["applier"].apply((sample, np.arange(3)), record["artifact"])
    assert len(output) == 2 and target.tolist() == [0, 2]
    capability = get_inference_capability("DropMissingRows", {}, record["artifact"], engine=engine)
    assert capability == ExecutionCapability(engine, "apply", "local", "filter", "row")
    assert pickle.dumps(record["artifact"]) == before
