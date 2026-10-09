"""Node-owned local context describes existing stateless preprocessing apply paths."""

from copy import deepcopy
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


def _frame(engine):
    """Use real values for resolving columns and geographic inputs during fitting."""
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.5, 2.0],
            "lat1": [0.0, 10.0, 40.0, 90.0],
            "lon1": [0.0, 20.0, 30.0, 0.0],
            "lat2": [0.0, 10.0, -40.0, -90.0],
            "lon2": [1.0, 21.0, -150.0, 0.0],
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


def _config(node):
    """Return literal public configurations for each reviewed node."""
    return {
        "CustomBinning": {"columns": ["x"], "bins": [0.0, 1.0, 2.0]},
        "ValueReplacement": {"columns": ["x"], "mapping": {0.0: 9.0}},
        "GeoDistance": {
            "lat1_col": "lat1",
            "lon1_col": "lon1",
            "lat2_col": "lat2",
            "lon2_col": "lon2",
        },
    }[node]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["CustomBinning", "ValueReplacement", "GeoDistance"])
def test_real_fitted_state_has_owned_local_context(node, engine, monkeypatch):
    """Fitted diagnostics must recognize local row dependency without fitting again."""
    config = _config(node)
    calculator = NodeRegistry.get_calculator(node)
    state = calculator().fit(_frame(engine), config)
    original = deepcopy(state)

    def forbidden(*args, **kwargs):
        """Reject accidental execution while inspecting saved metadata."""
        raise AssertionError("Inspection must not fit or apply")

    applier: Any = NodeRegistry.get_applier(node)
    monkeypatch.setattr(calculator, "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability == ExecutionCapability(engine, "apply", "local", "preserve", "row")
    assert applier.validate_inference_state(state) is state
    assert state == original


def _record(node, engine, config=None, frame=None):
    """Fit a real record with the same structure consumed by preprocessing diagnostics."""
    config = _config(node) if config is None else config
    frame = _frame(engine) if frame is None else frame
    return {
        "name": "reviewed",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(frame, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _probe(record, frame, engine, monkeypatch):
    """Exercise actual apply, mutation checks and exact per-chunk schema comparisons."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """A diagnostic must never relearn on any input chunk."""
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


@pytest.mark.parametrize("node", ["CustomBinning", "ValueReplacement", "GeoDistance"])
@pytest.mark.parametrize("engine", ["spark", "unknown"])
def test_other_engines_gain_no_capability(node, engine):
    """Local metadata must not grant unsupported or distributed execution."""
    assert get_inference_capability(node, {}, {}, engine=engine) is None


@pytest.mark.parametrize("node", ["CustomBinning", "ValueReplacement", "GeoDistance"])
@pytest.mark.parametrize("change", ["extra", "missing", "wrong_type", "not_dict"])
def test_structurally_invalid_state_is_rejected(node, change):
    """Unknown, missing or mismatched artifact fields must not receive a promise."""
    state: Any = _record(node, "pandas")["artifact"]
    if change == "extra":
        state["unreviewed"] = True
    elif change == "missing":
        state.pop("type")
    elif change == "wrong_type":
        state["type"] = "other"
    else:
        state = cast(Any, [])
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        ("CustomBinning", "bin_edges", {"x": [0.0, float("nan"), 2.0]}),
        ("CustomBinning", "bin_edges", {"x": [2.0, 1.0, 0.0]}),
        ("CustomBinning", "bin_edges", {"x": [False, 1.0]}),
        ("CustomBinning", "bin_edges", {"x": "0,1,2"}),
        ("CustomBinning", "drop_original", 1),
        ("CustomBinning", "include_lowest", "yes"),
        ("CustomBinning", "label_format", "unknown"),
        ("CustomBinning", "missing_strategy", "unknown"),
        ("CustomBinning", "missing_label", None),
        ("CustomBinning", "output_suffix", 4),
        ("CustomBinning", "precision", True),
        ("ValueReplacement", "columns", ["x", "x"]),
        ("ValueReplacement", "columns", "x"),
        ("ValueReplacement", "mapping", {"x": {0: 1}, "lat1": 5}),
        ("ValueReplacement", "mapping", {"x": {0: [1]}}),
        ("ValueReplacement", "mapping", [0, 1]),
        ("ValueReplacement", "to_replace", object()),
        ("ValueReplacement", "value", object()),
        ("GeoDistance", "lat1_col", ""),
        ("GeoDistance", "lon2_col", ["lon2"]),
        ("GeoDistance", "method", "manhattan"),
        ("GeoDistance", "unit", "m"),
        ("GeoDistance", "output_column", ""),
    ],
)
def test_invalid_owned_values_are_rejected(node, field, value):
    """Recognized field names alone must not conceal malformed apply semantics."""
    state = _record(node, "pandas")["artifact"]
    state[field] = value
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("label_format", ["ordinal", "bin_index", "range"])
@pytest.mark.parametrize("missing_strategy", ["keep", "label"])
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize("drop_original", [True, False])
def test_custom_binning_modes_report_exact_batch_behavior(
    engine, label_format, missing_strategy, include_lowest, drop_original, monkeypatch
):
    """Fixed bins must keep values and dtypes across chunks, nulls and empty requests."""
    config = {
        **_config("CustomBinning"),
        "label_format": label_format,
        "missing_strategy": missing_strategy,
        "include_lowest": include_lowest,
        "drop_original": drop_original,
        "output_suffix": "_bucket",
    }
    record = _record("CustomBinning", engine, config)
    frame = pd.DataFrame(
        {"x": [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, None]}, index=[8, 4, 4, 6, 2, 9, 1, 0]
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert all(check["status"] == "passed" for check in detail["checks"])
    assert list(output.columns) == (["x_bucket"] if drop_original else ["x", "x_bucket"])
    actual = output["x_bucket"].to_list()
    if label_format == "range":
        left, right = (
            ("[0.0, 1.0]", "[1.0, 2.0]") if include_lowest else ("(0.0, 1.0]", "(1.0, 2.0]")
        )
    elif engine == "polars" and missing_strategy == "label":
        left, right = "0", "1"
    else:
        left, right = 0, 1
    expected = [None, left if include_lowest else None, left, left, right, right, None, None]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert (
                actual_value == "Missing" if missing_strategy == "label" else pd.isna(actual_value)
            )
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "config",
    [{"columns": []}, {"columns": ["x"], "bins": []}, {"columns": ["x"], "bins": [1.0, 1.0]}],
)
def test_custom_binning_fitted_noops_remain_noops(engine, config, monkeypatch):
    """Real empty or degenerate fitted bins retain every row and the original schema."""
    frame = _frame(engine)
    record = _record("CustomBinning", engine, config)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output.equals(frame)


@pytest.mark.parametrize(
    "labels", [{"x": ["one"]}, {"x": ["same", "same"]}, {"other": ["one", "two"]}]
)
def test_custom_binning_rejects_corrupt_saved_range_labels(labels):
    """Saved labels must match the fitted intervals and selected columns unambiguously."""
    state = _record(
        "CustomBinning", "pandas", {**_config("CustomBinning"), "label_format": "range"}
    )["artifact"]
    state["range_labels"] = labels
    with pytest.raises(ValueError):
        get_inference_capability("CustomBinning", {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        ({"mapping": {"0": 9.0}}, [9.0, 1.0, 2.0, None]),
        ({"mapping": {"x": {"0": 9.0}}}, [9.0, 1.0, 2.0, None]),
        ({"to_replace": "0", "value": 9.0}, [9.0, 1.0, 2.0, None]),
        ({"to_replace": ["0", "1"], "value": 9.0}, [9.0, 9.0, 2.0, None]),
        ({"to_replace": ["0", "1"], "value": [9.0, 8.0]}, [9.0, 8.0, 2.0, None]),
        ({"to_replace": {"x": {"0": 9.0}}}, [9.0, 1.0, 2.0, None]),
        (
            {"replacements": [{"old": "0", "new": 9.0}], "mapping": {"0": -1.0}},
            [9.0, 1.0, 2.0, None],
        ),
        ({"mapping": {"0": 9.0}, "to_replace": "0", "value": -1.0}, [9.0, 1.0, 2.0, None]),
        ({}, [0.0, 1.0, 2.0, None]),
    ],
)
def test_value_replacement_rules_preserve_saved_execution(engine, rules, expected, monkeypatch):
    """Mapping precedence and scalar/list rules must execute unchanged on saved state."""
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, None], "untouched": [1, 2, 3, 4]}, index=[9, 3, 3, 1]
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    record = _record("ValueReplacement", engine, {"columns": ["x"], **rules}, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    actual = output["x"].to_list()
    assert actual[:3] == expected[:3]
    assert pd.isna(actual[3])
    assert output["untouched"].to_list() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "rules", [{"to_replace": 0, "value": [1]}, {"to_replace": [0, 1], "value": [2]}]
)
def test_value_replacement_rejects_invalid_pair_cardinality(rules):
    """Malformed scalar/list rules must fail inspection before executing apply."""
    state = _record("ValueReplacement", "pandas", {"columns": ["x"], **rules})["artifact"]
    with pytest.raises(ValueError):
        get_inference_capability("ValueReplacement", {}, state, engine="pandas")


@pytest.mark.parametrize(
    "rules",
    [
        {"mapping": {"one": 1}},
        {"mapping": {"x": {"one": 1}}},
        {"to_replace": "one", "value": 1},
        {"to_replace": ["one"], "value": [1]},
    ],
)
def test_value_replacement_preserves_object_dtype_across_chunks(rules, monkeypatch):
    """Replacing a string with a number must not infer types from request neighbors."""
    frame = pd.DataFrame({"x": ["one", "unmapped", None]})
    record = _record("ValueReplacement", "pandas", {"columns": ["x"], **rules}, frame)
    detail, output = _probe(record, frame, "pandas", monkeypatch)
    assert detail["status"] == "passed", detail
    assert all(check["status"] == "passed" for check in detail["checks"])
    assert output["x"].dtype == object
    assert output["x"].to_list() == [1, "unmapped", None]


def test_value_replacement_numeric_widening_remains_reported(monkeypatch):
    """Prevent object parity fixes from concealing unresolved integer-to-float widening."""
    frame = pd.DataFrame({"x": [1, 2]})
    record = _record("ValueReplacement", "pandas", {"columns": ["x"], "mapping": {1: 0.5}}, frame)
    detail, output = _probe(record, frame, "pandas", monkeypatch)
    checks = {check["name"]: check for check in detail["checks"]}
    assert detail["status"] == "failed"
    assert checks["chunks:1"]["reason"] == "output_mismatch"
    assert output["x"].to_list() == [0.5, 2.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["haversine", "euclidean"])
@pytest.mark.parametrize("unit", ["km", "mi"])
def test_geodistance_modes_preserve_nulls_and_batch_values(engine, method, unit, monkeypatch):
    """Both distance methods reuse fixed units without learning from request neighbors."""
    config = {**_config("GeoDistance"), "method": method, "unit": unit}
    frame = pd.DataFrame(
        {
            "lat1": [0.0, None, 0.0, 100.0],
            "lon1": [0.0, 0.0, 0.0, 200.0],
            "lat2": [0.0, 0.0, 0.0, 100.0],
            "lon2": [1.0, 1.0, 0.0, 200.0],
        },
        index=[9, 3, 3, 1],
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    record = _record("GeoDistance", engine, config)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    values = output[f"geo_distance_{unit}"].to_list()
    assert values[0] == pytest.approx(111.1950802335329 if unit == "km" else 69.09341954914815)
    assert pd.isna(values[1])
    assert values[2:] == [0.0, 0.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_geodistance_reused_coordinates_and_explicit_name(engine, monkeypatch):
    """Repeated coordinate names and an explicit output column are legitimate fitted state."""
    config = {
        **_config("GeoDistance"),
        "lat2_col": "lat1",
        "lon2_col": "lon1",
        "output_column": "distance",
    }
    record = _record("GeoDistance", engine, config)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output["distance"].to_list() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["missing", "nonnumeric"])
def test_geodistance_invalid_input_uses_original_apply_behavior(engine, kind, monkeypatch):
    """Valid saved metadata cannot hide missing-input noops or bad coordinate failures."""
    record = _record("GeoDistance", engine)
    frame = _frame("pandas")
    if kind == "missing":
        frame = frame.drop(columns="lat1")
    else:
        frame["lat1"] = ["bad", "0", "40", "90"]
    if engine == "polars":
        frame = pl.from_pandas(frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == ("passed" if kind == "missing" else "failed")
    if kind == "missing":
        assert output.equals(frame)
    else:
        assert detail["checks"][0]["reason"] == "apply_error"


@pytest.mark.parametrize("node", ["ValueReplacement", "GeoDistance"])
def test_empty_state_is_not_a_fitted_artifact_for_other_nodes(node):
    """Only nodes that actually fit an empty dictionary can declare that no-op state."""
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, {}, engine="pandas")


@pytest.mark.parametrize("node", ["GeneralBinning", "KBinsDiscretizer"])
def test_other_binning_nodes_declare_their_own_local_context(node):
    """Each binning calculator must own its declaration without granting Spark execution."""
    config = {"columns": ["x"], "n_bins": 2, "strategy": "uniform"}
    state = NodeRegistry.get_calculator(node)().fit(_frame("pandas"), config)
    assert get_inference_capability(node, config, state, engine="pandas") == ExecutionCapability(
        "pandas", "apply", "local", "preserve", "row"
    )
    assert get_inference_capability(node, config, state, engine="spark") is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("rules", [{"columns": []}, {"columns": ["absent"]}])
def test_value_replacement_resolved_noop_preserves_schema(engine, rules, monkeypatch):
    """Explicitly empty or absent-column selection is a typed fitted no-op."""
    record = _record("ValueReplacement", engine, {**rules, "mapping": {"0": 7.0}})
    frame = _frame(engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "bins", [[float("-inf"), 0.0, float("inf")], [0.0, 1.0, 1.0, 2.0], [0.00001, 0.00002, 0.00003]]
)
def test_custom_binning_retains_infinite_duplicate_and_precise_edges(engine, bins, monkeypatch):
    """Valid fixed bins retain canonical labels without the validator refitting their edges."""
    config = {"columns": ["x"], "bins": bins, "label_format": "range"}
    record = _record("CustomBinning", engine, config)
    labels = record["artifact"]["range_labels"]["x"]
    assert len(labels) == len(set(labels)) == 2
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed"
    assert "x_binned" in output.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
def test_value_replacement_nullable_rules_preserve_native_types(engine, dtype, monkeypatch):
    """JSON keys must retain the existing boolean and nullable numeric replacement behavior."""
    is_boolean = dtype == "boolean"
    frame = pd.DataFrame(
        {"x": pd.Series([True, False, None] if is_boolean else [1, 2, None], dtype=dtype)}
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["x"], "mapping": {"true": False} if is_boolean else {"1": 9}}
    record = _record("ValueReplacement", engine, config, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    values = output["x"].to_list()
    assert values[:2] == ([False, False] if is_boolean else [9, 2])
    assert pd.isna(values[2])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("edge_type", [np.float64, np.float32, np.int64, np.uint64])
def test_custom_binning_inspects_numpy_edges_without_coercion(engine, edge_type, monkeypatch):
    """Real NumPy-generated bin lists must remain usable after diagnostic inspection."""
    config = {
        "columns": ["x"],
        "bins": list(np.linspace(0, 2, 3).astype(edge_type)),
        "label_format": "range",
    }
    record = _record("CustomBinning", engine, config)
    state = record["artifact"]
    applier = record["applier"]
    direct = applier.apply(_frame(engine), state)
    assert direct["x_binned"].to_list() == ["[0.0, 1.0]", "[0.0, 1.0]", "[1.0, 2.0]", "[1.0, 2.0]"]
    assert applier.validate_inference_state(state) is state
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed"
    assert all(type(edge) is edge_type for edge in state["bin_edges"]["x"])
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_custom_binning_inspects_legacy_range_state_without_adding_labels(engine, monkeypatch):
    """Artifacts saved before canonical labels must retain their legacy rendering path."""
    config = {**_config("CustomBinning"), "label_format": "range"}
    record = _record("CustomBinning", engine, config)
    state = record["artifact"]
    state.pop("range_labels")
    direct = record["applier"].apply(_frame(engine), state)
    assert "x_binned" in direct.columns
    assert record["applier"].validate_inference_state(state) is state
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail.get("reason") != "invalid_step_contract"
    assert detail["checks"][0]["status"] == "passed"
    assert "range_labels" not in state
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "rules",
    [
        {"to_replace": (0.0, 0.5), "value": (9.0, 8.0)},
        {"to_replace": (0.0, 0.5), "value": [9.0, 8.0]},
        {"to_replace": [0.0, 0.5], "value": (9.0, 8.0)},
        {"mapping": {np.float64(0): np.float64(9), np.float32(0.5): np.float32(8)}},
        {"mapping": {"x": {np.float64(0): np.int64(9), np.float64(0.5): np.int64(8)}}},
    ],
)
def test_value_replacement_inspects_tuple_and_numpy_rules_unchanged(engine, rules, monkeypatch):
    """Valid tuple and NumPy rules must not fail merely because fit preserves their types."""
    frame = _frame(engine)
    record = _record("ValueReplacement", engine, {"columns": ["x"], **rules})
    state = record["artifact"]
    direct = record["applier"].apply(frame, state)
    assert direct["x"].to_list() == [9.0, 8.0, 1.5, 2.0]
    before = deepcopy(state)
    assert record["applier"].validate_inference_state(state) is state
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed"
    assert type(state["to_replace"]) is type(before["to_replace"])
    assert type(state["value"]) is type(before["value"])
    assert state == before
    assert output.equals(direct)


@pytest.mark.parametrize("bad_edge", [np.bool_(True), np.complex64(1 + 2j), np.float64("nan")])
def test_custom_binning_still_rejects_nonreal_numpy_edges(bad_edge):
    """NumPy compatibility must not admit boolean, complex or NaN bin coordinates."""
    state = _record("CustomBinning", "pandas")["artifact"]
    state["bin_edges"]["x"] = [0.0, bad_edge, 2.0]
    with pytest.raises(ValueError):
        get_inference_capability("CustomBinning", {}, state, engine="pandas")


@pytest.mark.parametrize(
    "rules", [{"to_replace": 0.0, "value": (9.0,)}, {"to_replace": (0.0, 0.5), "value": (9.0,)}]
)
def test_value_replacement_rejects_malformed_tuple_cardinality(rules):
    """Tuple support must enforce the same cardinality contract as list rules."""
    state = _record("ValueReplacement", "pandas", {"columns": ["x"], **rules})["artifact"]
    with pytest.raises(ValueError):
        get_inference_capability("ValueReplacement", {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_custom_binning_rejects_intrinsic_saved_output_collision(engine):
    """Corrupted suffix/drop choices must not advertise an artifact that always collides."""
    config = {**_config("CustomBinning"), "drop_original": True, "output_suffix": ""}
    record = _record("CustomBinning", engine, config)
    state = record["artifact"]
    state["drop_original"] = False
    with pytest.raises(ValueError):
        record["applier"].apply(_frame(engine), state)
    with pytest.raises(ValueError):
        get_inference_capability("CustomBinning", {}, state, engine=engine)
