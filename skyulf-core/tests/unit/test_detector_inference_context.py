"""Saved outlier detectors expose filtering without silently skipping prediction checks."""

import pickle
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.covariance import EllipticEnvelope

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

NODES = ["EllipticEnvelope", "IQR", "ZScore"]


def _frame(engine, values):
    """Retain a fixed float dtype and duplicate indexes through filtering probes."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
    frame.index = [index // 2 for index in range(len(frame))]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config=None, values=None):
    """Create actual artifacts, including sklearn's complete learned covariance state."""
    config = {"columns": ["x"]} if config is None else config
    values = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5] if values is None else values
    return {
        "name": "detector",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(_frame(engine, values), config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _poison_fit(monkeypatch, node):
    """Disable every learning path while preserving the real saved predict implementation."""

    def forbidden(*args, **kwargs):
        """Make inference-time learning immediately observable."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(EllipticEnvelope, "fit", forbidden)


def _probe(record, sample, engine):
    """Use the real prediction row guard, exact replay checks and mutation diagnostics."""
    return _probe_step(
        record,
        {"name": "detector", "transformer": record["type"], "params": record["params"]},
        sample,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_detectors_inspect_real_saved_state_without_execution(node, engine, monkeypatch):
    """A fitted filter contract must be pure and must never grant worker admission."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)
    _poison_fit(monkeypatch, node)

    def forbidden(*args, **kwargs):
        """Metadata inspection must not run the detector or covariance distances."""
        raise AssertionError("Unexpected apply")

    applier: Any = type(record["applier"])
    monkeypatch.setattr(applier, "apply", forbidden)
    monkeypatch.setattr(EllipticEnvelope, "predict", forbidden)
    assert get_inference_capability(node, {}, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", "filter", "row"
    )
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability(node, {}, state, engine="spark") is None
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_benign_predictions_replay_saved_detectors_exactly(node, engine, monkeypatch):
    """Filtering metadata must still execute benign requests and strict empty/chunk probes."""
    record = _record(node, engine)
    before = artifact_digest(record)
    sample = _frame(engine, [0, 0.5, None, 1])
    _poison_fit(monkeypatch, node)
    detail, output = _probe(record, sample, engine)
    assert detail["context"] == "row" and detail["row_effect"] == "filter", detail
    assert detail["action"] == "apply" and detail["status"] == "passed", detail
    assert detail["state_validation"] == "node_owned"
    assert output.equals(sample)
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_outlier_predictions_fail_row_guard_while_native_filter_aligns_targets(
    node, engine, monkeypatch
):
    """Saved native filters align X/y, but unkeyed prediction may not silently lose rows."""
    record = _record(node, engine)
    before = artifact_digest(record)
    sample = _frame(engine, [0, 100, None, 0.5])
    labels = np.array([10, 20, 30, 40])
    _poison_fit(monkeypatch, node)
    filtered, targets = record["applier"].apply((sample, labels), record["artifact"])
    assert targets.tolist() == [10, 30, 40]
    assert len(filtered) == 3
    detail, output = _probe(record, sample, engine)
    assert detail["context"] == "row" and detail["row_effect"] == "filter", detail
    assert detail["action"] == "apply" and detail["status"] == "failed", detail
    assert detail["checks"][0]["reason"] == "apply_error"
    assert detail["checks"][0]["error_type"] == "ValueError"
    assert output is sample
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_native_filter_chunks_order_and_nonfinite_policy(node, engine, monkeypatch):
    """Finite outliers cannot disable filtering for companions; null/infinity policies remain native."""
    record = _record(node, engine)
    sample = _frame(engine, [0.5, 100, None, np.inf, -np.inf])
    before = artifact_digest(record)
    _poison_fit(monkeypatch, node)
    apply = record["applier"].apply
    state = record["artifact"]
    full, positions = apply((sample, np.arange(len(sample))), state)
    expected_positions = [0, 2, 3, 4] if node == "EllipticEnvelope" else [0, 2]
    assert positions.tolist() == expected_positions
    parts = [
        apply(sample.iloc[i : i + 1] if engine == "pandas" else sample.slice(i, 1), state)
        for i in range(len(sample))
    ]
    chunks = pd.concat(parts) if engine == "pandas" else pl.concat(parts)
    assert chunks.equals(full)
    reversed_sample = sample.iloc[::-1] if engine == "pandas" else sample.reverse()
    reverse = apply(reversed_sample, state)
    reverse = reverse.iloc[::-1] if engine == "pandas" else reverse.reverse()
    assert reverse.equals(full)
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("reason", ["explicit_empty", "missing", "no_numeric"])
def test_genuine_detector_noops_preserve_requests(node, engine, reason, monkeypatch):
    """Real empty artifacts and populated warning-only artifacts must remain valid no-ops."""
    config = {
        "columns": [] if reason == "explicit_empty" else ["missing" if reason == "missing" else "x"]
    }
    record = _record(node, engine, config, [None] * 8 if reason == "no_numeric" else None)
    before = artifact_digest(record)
    _poison_fit(monkeypatch, node)
    sample = _frame(engine, [100, None, 0])
    detail, output = _probe(record, sample, engine)
    assert detail["context"] == "row" and detail["row_effect"] == "preserve", detail
    assert detail["status"] == "passed", detail
    assert output.equals(sample)
    assert artifact_digest(record) == before


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("change", ["type", "extra", "warnings", "mapping", "entry"])
def test_detector_outer_state_rejects_malformed_contracts(node, change):
    """Corrupt discriminator, map ownership and diagnostic shapes must not acquire context."""
    state = deepcopy(_record(node, "pandas")["artifact"])
    field = {"EllipticEnvelope": "models", "IQR": "bounds", "ZScore": "stats"}[node]
    if change == "type":
        state["type"] = "other"
    elif change == "extra":
        state["extra"] = True
    elif change == "warnings":
        state["warnings"] = [lambda: None]
    elif change == "mapping":
        state[field] = []
    else:
        state[field]["x"] = []
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize(
    "node,change",
    [
        ("IQR", "missing_bound"),
        ("IQR", "text_bound"),
        ("ZScore", "missing_stat"),
        ("ZScore", "negative_std"),
        ("ZScore", "text_stat"),
        ("EllipticEnvelope", "unfitted"),
        ("EllipticEnvelope", "feature_count"),
        ("EllipticEnvelope", "location"),
        ("EllipticEnvelope", "covariance"),
        ("EllipticEnvelope", "precision"),
        ("EllipticEnvelope", "offset"),
        ("EllipticEnvelope", "callback"),
    ],
)
def test_detector_learned_values_are_inspected(node, change):
    """Learned numeric state must stay aligned with its per-column native apply contract."""
    state = deepcopy(_record(node, "pandas")["artifact"])
    if node == "IQR":
        if change == "missing_bound":
            del state["bounds"]["x"]["lower"]
        else:
            state["bounds"]["x"]["upper"] = "4"
    elif node == "ZScore":
        if change == "missing_stat":
            del state["stats"]["x"]["mean"]
        else:
            state["stats"]["x"]["std"] = -1 if change == "negative_std" else "1"
    else:
        changes = {
            "feature_count": ("n_features_in_", 2),
            "location": ("location_", np.array([0, 1.0])),
            "covariance": ("covariance_", np.array([["bad"]])),
            "precision": ("precision_", None),
            "offset": ("offset_", None),
            "callback": ("predict", lambda values: np.ones(len(values))),
        }
        if change == "unfitted":
            state["models"]["x"] = EllipticEnvelope()
        else:
            field, value = changes[change]
            setattr(state["models"]["x"], field, value)
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,parameter,value",
    [
        ("EllipticEnvelope", "contamination", np.float64(0.1)),
        ("IQR", "multiplier", np.float64(1.5)),
        ("ZScore", "threshold", np.float64(3)),
    ],
)
def test_real_numpy_and_tuple_detector_configuration(node, parameter, value, engine, monkeypatch):
    """Locally supported NumPy parameters and column labels must survive pure saved inspection."""
    record = _record(node, engine, {"columns": (np.str_("x"),), parameter: value})
    _poison_fit(monkeypatch, node)
    detail, output = _probe(record, _frame(engine, [0, 0.5]), engine)
    assert detail["status"] == "passed", detail
    assert record["artifact"][parameter] is value
    assert len(output) == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("threshold,expected", [(None, 0), (True, 1), (Decimal("3"), 2)])
def test_zscore_native_nullable_boolean_and_decimal_thresholds(engine, threshold, expected):
    """Saved thresholds retain native comparison behavior, including Decimal's existing seal limit."""
    record = _record("ZScore", engine, {"columns": ["x"], "threshold": threshold})
    state = record["artifact"]
    output = record["applier"].apply(_frame(engine, [0, 3]), state)
    capability = get_inference_capability("ZScore", {}, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert len(output) == expected
    assert state["threshold"] is threshold
    if isinstance(threshold, Decimal):
        with pytest.raises(TypeError):
            artifact_digest(state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["IQR", "Winsorize", "ManualBounds"])
def test_native_empty_bound_column_names_remain_inspectable(node, engine):
    """The shared local bound guard must accept an empty name already supported by fit/apply."""
    frame = _frame(engine, [-1, 0, 1, 2, 3, 4, 5, 6])
    frame = frame.rename(columns={"x": ""}) if engine == "pandas" else frame.rename({"x": ""})
    config = (
        {"bounds": {"": {"lower": -10, "upper": 10}}}
        if node == "ManualBounds"
        else {"columns": [""]}
    )
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    before = artifact_digest(state)
    output = NodeRegistry.get_applier(node)().apply(frame, state)
    assert len(output) == len(frame) and list(output.columns) == [""]
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,parameter,value",
    [
        ("IQR", "multiplier", Fraction(3, 2)),
        ("ZScore", "threshold", Fraction(3, 1)),
        ("EllipticEnvelope", "contamination", Decimal("0.1")),
        ("EllipticEnvelope", "contamination", Fraction(1, 10)),
    ],
)
def test_native_fraction_and_decimal_detector_settings_remain_saved(node, parameter, value, engine):
    """Native numeric scalars and warning-only fits are local states despite their seal limit."""
    record = _record(node, engine, {"columns": ["x"], parameter: value})
    state = record["artifact"]
    before = pickle.dumps(state)
    sample = _frame(engine, [0, 0.5])
    output = record["applier"].apply(sample, state)
    assert output.equals(sample)
    if node == "EllipticEnvelope":
        assert state["models"] == {} and state["warnings"]
    capability = get_inference_capability(node, {}, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert state[parameter] is value
    assert pickle.dumps(state) == before
    with pytest.raises(TypeError):
        artifact_digest(state)


@pytest.mark.parametrize("node", [*NODES, "ManualBounds", "Winsorize"])
@pytest.mark.parametrize("column", [1, ("metric", "x")])
def test_native_hashable_pandas_detector_column_names(node, column):
    """Local detector inspection must preserve native integer and tuple column identities."""
    frame = pd.DataFrame({column: [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]})
    config = (
        {"bounds": {column: {"lower": -10, "upper": 10}}}
        if node == "ManualBounds"
        else {"columns": [column]}
    )
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    before = pickle.dumps(state)
    output = NodeRegistry.get_applier(node)().apply(frame.iloc[2:4], state)
    assert output.equals(frame.iloc[2:4])
    capability = get_inference_capability(node, config, state, engine="pandas")
    assert capability is not None and capability.context == "row"
    assert pickle.dumps(state) == before
