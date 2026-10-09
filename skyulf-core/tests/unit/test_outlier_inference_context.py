"""Saved outlier bounds expose clipping and prediction row-loss boundaries."""

from copy import deepcopy
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values):
    """Keep a stable numeric dtype and duplicate pandas indexes across requests."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
    frame.index = [index // 2 for index in range(len(frame))]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config=None):
    """Inspect an actual fitted state rather than a handwritten approximation."""
    config = (
        config
        if config is not None
        else (
            {"bounds": {"x": {"lower": 0.0, "upper": 10.0}}}
            if node == "ManualBounds"
            else {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75}
        )
    )
    return {
        "name": "bounds",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(_frame(engine, [0, 2, 4, 6]), config),
        "applier": NodeRegistry.get_applier(node)(),
    }


@pytest.mark.parametrize("node,effect", [("ManualBounds", "filter"), ("Winsorize", "preserve")])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_bounds_inspection_never_executes(node, effect, engine, monkeypatch):
    """Metadata inspection must not fit, transform, normalize or mutate saved bounds."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Make accidental fitting or application during inspection fail immediately."""
        raise AssertionError("Inspection executed a transform")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    applier: Any = NodeRegistry.get_applier(node)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, {}, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", effect, "row"
    )
    assert applier.validate_inference_state(state) is state
    assert artifact_digest(state) == before
    assert get_inference_capability(node, {}, state, engine="spark") is None


@pytest.mark.parametrize("node", ["ManualBounds", "Winsorize"])
@pytest.mark.parametrize("change", ["type", "extra", "bounds", "column", "field", "value"])
def test_invalid_bound_state_has_no_declaration(node, change):
    """Corrupt bounds cannot silently acquire trusted context metadata."""
    state: Any = deepcopy(_record(node, "pandas")["artifact"])
    if change == "type":
        state["type"] = "other"
    elif change == "extra":
        state["unreviewed"] = True
    elif change == "bounds":
        state["bounds"] = []
    elif change == "column":
        state["bounds"] = {1: {"lower": 0, "upper": 1}}
    elif change == "field":
        state["bounds"]["x"]["inclusive"] = False
    else:
        state["bounds"]["x"]["upper"] = "10"
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_winsorize_reuses_saved_quantiles_across_requests(engine, monkeypatch):
    """Clipping must use training limits even for reordered, empty or extreme requests."""
    record = _record("Winsorize", engine)
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Prediction must not relearn request quantiles."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("Winsorize"), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": "bounds", "transformer": "Winsorize", "params": record["params"]},
        _frame(engine, [-100, 2, None, 1000, 3]),
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert detail["context"] == "row" and detail["status"] == "passed", detail
    assert detail["state_validation"] == "node_owned"
    np.testing.assert_allclose(output["x"].to_numpy(), [1.5, 2, np.nan, 4.5, 3], equal_nan=True)
    assert artifact_digest(record) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_bounds_reports_prediction_row_loss(engine, monkeypatch):
    """A row-local filter must not silently drop inputs from an unkeyed prediction request."""
    record = _record("ManualBounds", engine)
    frame = _frame(engine, [-100, 2, None, 1000, 3])

    def forbidden(*args, **kwargs):
        """Prediction may reuse saved bounds but must never fit them again."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator("ManualBounds"), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": "bounds", "transformer": "ManualBounds", "params": record["params"]},
        frame,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert detail["context"] == "row" and detail["row_effect"] == "filter"
    assert detail["action"] == "apply" and detail["status"] == "failed"
    assert detail["checks"][0]["reason"] == "apply_error"
    assert detail["checks"][0]["error_type"] == "ValueError"
    assert output is frame


@pytest.mark.parametrize(
    "node,config",
    [
        ("Winsorize", {"columns": []}),
        ("ManualBounds", {"bounds": {}}),
        ("ManualBounds", {"bounds": {"x": {"lower": None, "upper": np.float64(4)}}}),
    ],
)
def test_genuine_noop_and_numpy_bound_state_remains_inspectable(node, config):
    """No-op fits and NumPy scalar bounds must not be mistaken for corrupt artifacts."""
    state = _record(node, "pandas", config)["artifact"]
    capability = get_inference_capability(node, config, state, engine="pandas")
    assert capability is not None and capability.context == "row"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_decimal_bounds_preserve_native_fit_apply(engine):
    """Inspection must accept Decimal limits already supported by the saved applier."""
    config = {"bounds": {"x": {"lower": Decimal("0.5"), "upper": Decimal("1.5")}}}
    record = _record("ManualBounds", engine, config)
    output = record["applier"].apply(_frame(engine, [0, 1, 2]), record["artifact"])
    assert list(output["x"]) == [1]
    capability = get_inference_capability("ManualBounds", config, record["artifact"], engine=engine)
    assert capability is not None and capability.context == "row"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_integer_winsorize_keeps_strict_dtype_probe(engine):
    """A row declaration must not conceal pandas integer clipping's chunk dtype drift."""
    record = _record("Winsorize", engine)
    sample = pd.DataFrame({"x": [2, 10]})
    if engine == "polars":
        sample = pl.from_pandas(sample)
    detail, _ = _probe_step(
        record,
        {"name": "bounds", "transformer": "Winsorize", "params": record["params"]},
        sample,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert detail["context"] == "row"
    if engine == "polars":
        assert detail["status"] == "passed", detail
    else:
        assert detail["status"] == "failed", detail
        assert any(check.get("reason") == "output_mismatch" for check in detail["checks"]), detail
