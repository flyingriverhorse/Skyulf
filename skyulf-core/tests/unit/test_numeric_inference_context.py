"""Saved numeric transformations expose local context while reusing their native apply."""

from copy import deepcopy
from decimal import Decimal
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values=(0.0, 1.0, 2.0, 4.0)):
    """Keep test values and row identity equivalent on both local engines."""
    frame = pd.DataFrame({"x": values, "keep": list(range(len(values)))})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _config(node):
    """Supply the public configuration shape actually consumed by each calculator."""
    if node == "SimpleTransformation":
        return {"transformations": [{"column": "x", "method": "square"}]}
    return {"columns": ["x"], "n_bins": 2}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer", "SimpleTransformation"])
def test_fitted_numeric_nodes_own_local_context(node, engine, monkeypatch):
    """Inspection must recognize the saved owner without refitting or applying data."""
    config = _config(node)
    calculator = NodeRegistry.get_calculator(node)
    applier: Any = NodeRegistry.get_applier(node)
    state = calculator().fit(_frame(engine), config)
    before = deepcopy(state)

    def forbidden(*args, **kwargs):
        """Metadata inspection must not execute the transform or learn new parameters."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(calculator, "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, config, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", "preserve", "row"
    )
    assert applier.validate_inference_state(state) is state
    assert state == before


def _record(node, engine, config=None, frame=None):
    """Produce the genuine fitted-record shape consumed by diagnostics."""
    config = _config(node) if config is None else config
    frame = _frame(engine) if frame is None else frame
    return {
        "name": "numeric",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(frame, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _probe(record, frame, engine, monkeypatch):
    """Use the existing exact probe, including per-chunk schema and mutation checks."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Inference must replay fitted state rather than learn from this request."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(record["type"]), "fit", forbidden)
    detail, output = _probe_step(
        record,
        {"name": record["name"], "transformer": record["type"], "params": record["params"]},
        frame,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    assert detail["context"] == "row"
    assert detail["state_validation"] == "node_owned"
    assert artifact_digest(record) == before
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "strategy"),
    [
        ("GeneralBinning", method)
        for method in ("uniform", "equal_width", "equal_frequency", "kmeans", "kbins")
    ]
    + [
        ("KBinsDiscretizer", method)
        for method in ("uniform", "quantile", "kmeans", "equal_width", "equal_frequency")
    ],
)
def test_binning_fitted_strategies_replay_fixed_edges(node, strategy, engine, monkeypatch):
    """Every fitted strategy must reuse saved edges across null and out-of-range requests."""
    record = _record(node, engine, {"columns": ["x"], "n_bins": 2, "strategy": strategy})
    detail, output = _probe(
        record, _frame(engine, [-1.0, 0.0, 0.5, 1.5, 4.0, 5.0, None]), engine, monkeypatch
    )
    assert detail["status"] == "passed"
    values = output["x_binned"].to_list()
    assert pd.isna(values[0]) and pd.isna(values[-2]) and pd.isna(values[-1])
    assert values[1:3] == [0, 0]
    assert values[4] == 1


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer"])
@pytest.mark.parametrize("labels", ["ordinal", "bin_index", "range"])
@pytest.mark.parametrize("missing", ["keep", "label"])
def test_binning_saved_display_options_keep_exact_schema(
    node, labels, missing, engine, monkeypatch
):
    """Inherited bin rendering must retain saved NumPy flags and stable empty/chunk schemas."""
    config = {
        "columns": ["x"],
        "strategy": "uniform",
        "n_bins": 2,
        "label_format": labels,
        "missing_strategy": missing,
        "drop_original": np.bool_(True),
        "include_lowest": np.bool_(False),
        "output_suffix": "_bucket",
    }
    record = _record(node, engine, config)
    detail, output = _probe(record, _frame(engine, [0.0, 1.0, 4.0, None]), engine, monkeypatch)
    assert detail["status"] == "passed"
    assert list(output.columns) == ["keep", "x_bucket"]
    assert type(record["artifact"]["drop_original"]) is np.bool_


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "labels", [["low", "high"], ("low", "high"), [np.str_("low"), np.str_("high")], ["ignored"]]
)
def test_general_binning_custom_label_containers_and_fallback(engine, labels, monkeypatch):
    """Custom labels retain their container and documented wrong-length fallback."""
    config = {
        "columns": ["x"],
        "strategy": "custom",
        "custom_bins": {"x": list(np.linspace(0, 4, 3))},
        "custom_labels": {"x": labels},
    }
    record = _record("GeneralBinning", engine, config)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed"
    expected = [0, 0, 0, 1] if len(labels) == 1 else ["low", "low", "low", "high"]
    assert output["x_binned"].to_list() == expected
    assert type(record["artifact"]["custom_labels"]["x"]) is type(labels)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_general_binning_numeric_labels_keep_engine_limit_visible(engine, monkeypatch):
    """Valid pandas numeric categories must not disguise native Polars label restrictions."""
    config = {
        "columns": ["x"],
        "strategy": "custom",
        "custom_bins": {"x": [0.0, 2.0, 4.0]},
        "custom_labels": {"x": [0, 1]},
    }
    record = _record("GeneralBinning", engine, config)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    if engine == "pandas":
        assert detail["status"] == "passed"
        assert output["x_binned"].to_list() == [0, 0, 0, 1]
    else:
        assert detail["checks"][0] == {
            "name": "full",
            "status": "failed",
            "reason": "apply_error",
            "error_type": "TypeError",
        }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer"])
@pytest.mark.parametrize("config", [{}, {"columns": []}, {"columns": ["absent"]}])
def test_binning_defaults_and_real_noops_remain_inspectable(node, config, engine, monkeypatch):
    """Default auto-selection and real empty fitted artifacts must retain their semantics."""
    record = _record(node, engine, config)
    frame = _frame(engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output.equals(frame) if config else "x_binned" in output.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer"])
def test_binning_legacy_range_state_keeps_existing_rendering(node, engine, monkeypatch):
    """Older states without optional labels must not require refitting merely for inspection."""
    record = _record(node, engine, {"columns": ["x"], "n_bins": 2, "label_format": "range"})
    record["artifact"].pop("range_labels")
    record["artifact"].pop("custom_labels")
    direct = record["applier"].apply(_frame(engine), record["artifact"])
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["checks"][0]["status"] == "passed"
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "method",
    ["log", "sqrt", "square_root", "cube_root", "reciprocal", "square", "exp", "exponential"],
)
def test_simple_modes_reuse_domain_null_and_clipping_rules(engine, method, monkeypatch):
    """Each native math operation must preserve exact values and schema across batch shapes."""
    config = {"transformations": [{"column": "x", "method": method}]}
    record = _record("SimpleTransformation", engine, config)
    detail, output = _probe(
        record, _frame(engine, [-8.0, -1.0, 0.0, 1.0, 8.0, 1000.0, None]), engine, monkeypatch
    )
    assert detail["status"] == "passed"
    values = output["x"].to_list()
    assert pd.isna(values[-1])
    if method in ("log", "sqrt", "square_root"):
        assert pd.isna(values[0]) and pd.isna(values[1]) and values[2] == 0
    elif method == "reciprocal":
        assert pd.isna(values[2]) and values[3] == 1
    elif method == "cube_root":
        assert values[0] == pytest.approx(-2)
    elif method == "square":
        assert values[:5] == [64, 1, 0, 1, 64]
    else:
        assert values[2] == 1 and values[-2] == pytest.approx(1.0142320547350045e304)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "transformations",
    [
        [],
        (),
        None,
        [{}],
        [{"column": "absent", "method": "log"}],
        [{"column": "x", "method": "unknown"}],
        [{"column": "x"}],
    ],
)
def test_simple_real_noop_states_remain_noops(engine, transformations, monkeypatch):
    """Existing missing-column, unknown-method and empty-sequence skips must remain valid."""
    record = _record("SimpleTransformation", engine, {"transformations": transformations})
    frame = _frame(engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("threshold", [np.float64(2.0), np.int64(2), None, float("inf")])
def test_simple_tuple_sequence_preserves_numpy_clipping_state(engine, threshold, monkeypatch):
    """Ordered tuple rules and native numeric thresholds must be inspected without coercion."""
    config = {
        "transformations": (
            {"column": "x", "method": "square"},
            {"column": "x", "method": "exp", "clip_threshold": threshold},
        )
    }
    record = _record("SimpleTransformation", engine, config)
    detail, output = _probe(record, _frame(engine, [0.0, -1.0, 2.0, None]), engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output["x"].to_list()[0] == 1
    assert type(record["artifact"]["transformations"]) is tuple
    assert type(record["artifact"]["transformations"][1]["clip_threshold"]) is type(threshold)


@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer", "SimpleTransformation"])
@pytest.mark.parametrize("change", ["extra", "missing", "type", "container"])
def test_numeric_invalid_state_shape_cannot_gain_context(node, change):
    """Corrupted state fields must fail before native apply can hide the problem."""
    state: Any = _record(node, "pandas")["artifact"]
    if change == "extra":
        state["new_behavior"] = True
    elif change == "missing":
        state.pop("type")
    elif change == "type":
        state["type"] = "other"
    else:
        state = cast(Any, [])
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize(
    "transformations",
    [
        "square",
        [1],
        [{"column": ["x"], "method": "square"}],
        [{"column": "x", "method": []}],
        [{"column": "x", "method": "exp", "clip_threshold": object()}],
        [{"column": "x", "method": "square", "callback": object()}],
    ],
)
def test_simple_malformed_rules_are_rejected(transformations):
    """Inspection must reject unsupported containers or behavior without invoking callbacks."""
    state = _record("SimpleTransformation", "pandas", {"transformations": transformations})[
        "artifact"
    ]
    with pytest.raises(ValueError):
        get_inference_capability("SimpleTransformation", {}, state, engine="pandas")


def test_simple_iterator_state_is_not_consumed():
    """One-shot rules must be rejected before inspection changes later inference behavior."""
    rule = {"column": "x", "method": "square"}
    rules = iter([rule])
    state = _record("SimpleTransformation", "pandas", {"transformations": rules})["artifact"]
    with pytest.raises(ValueError):
        get_inference_capability("SimpleTransformation", {}, state, engine="pandas")
    assert next(rules) is rule


@pytest.mark.parametrize("labels", [[], {"x": object()}, {"x": [object()]}])
def test_general_binning_invalid_label_state_is_rejected(labels):
    """A recognized label field must not conceal non-scalar or executable payloads."""
    state = _record("GeneralBinning", "pandas")["artifact"]
    state["custom_labels"] = labels
    with pytest.raises(ValueError):
        get_inference_capability("GeneralBinning", {}, state, engine="pandas")


@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer", "SimpleTransformation"])
def test_numeric_local_hooks_do_not_admit_spark(node):
    """Metadata for local apply must not grant distributed execution."""
    assert get_inference_capability(node, {}, {}, engine="spark") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "options",
    [
        {"n_bins": 2, "label_format": "range", "precision": np.int64(2)},
        {
            "strategy": "custom",
            "custom_bins": {"x": [0.0, 4.0]},
            "custom_labels": {"x": np.array(["only"])},
        },
    ],
)
def test_binning_retains_numpy_precision_and_label_array(engine, options, monkeypatch):
    """Valid NumPy fitted containers must remain available to native saved apply."""
    record = _record("GeneralBinning", engine, {"columns": ["x"], **options})
    state = record["artifact"]
    direct = record["applier"].apply(_frame(engine), state)
    assert "x_binned" in direct.columns
    assert record["applier"].validate_inference_state(state) is state
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed"
    if "custom_labels" in options:
        assert isinstance(state["custom_labels"]["x"], np.ndarray)
    else:
        assert type(state["precision"]) is np.int64
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_simple_numpy_boolean_clip_retains_native_engine_boundary(engine, monkeypatch):
    """Numeric scalar compatibility must leave genuine engine apply errors observable."""
    config = {
        "transformations": [{"column": "x", "method": "exp", "clip_threshold": np.bool_(True)}]
    }
    record = _record("SimpleTransformation", engine, config)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    if engine == "polars":
        assert detail["status"] == "passed"
        assert output["x"].to_list()[0] == 1
    else:
        assert detail["checks"][0]["reason"] == "apply_error"
    assert type(record["artifact"]["transformations"][0]["clip_threshold"]) is np.bool_


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_simple_numeric_text_coercion_keeps_input_unchanged(engine, monkeypatch):
    """Saved pointwise math must keep existing bad-text coercion and null handling."""
    frame = _frame(engine, ["bad", " 2 ", "-3", None])
    record = _record("SimpleTransformation", engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    values = output["x"].to_list()
    assert pd.isna(values[0]) and pd.isna(values[3])
    assert values[1:3] == [4.0, 9.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_simple_decimal_threshold_retains_native_fitted_state(engine):
    """Decimal clipping supported by native apply must remain inspectable without coercion."""
    threshold = Decimal("2")
    config = {"transformations": [{"column": "x", "method": "exp", "clip_threshold": threshold}]}
    frame = _frame(engine, [0.0, 1.0])
    record = _record("SimpleTransformation", engine, config)
    direct = record["applier"].apply(frame, record["artifact"])
    assert direct["x"].to_list() == pytest.approx([1.0, np.exp(1)])
    assert get_inference_capability(
        "SimpleTransformation", config, record["artifact"], engine=engine
    ) == ExecutionCapability(engine, "apply", "local", "preserve", "row")
    assert record["applier"].validate_inference_state(record["artifact"]) is record["artifact"]
    assert record["artifact"]["transformations"][0]["clip_threshold"] is threshold
