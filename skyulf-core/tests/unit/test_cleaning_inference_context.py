"""Saved cleaning state owns context without bypassing exact inference probes."""

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

NODES = ["Casting", "AliasReplacement", "InvalidValueReplacement", "TextCleaning"]
CONFIGS = {
    "Casting": {"column_types": {"text": "string"}},
    "AliasReplacement": {"columns": ["text"], "alias_type": "boolean"},
    "InvalidValueReplacement": {"columns": ["number"], "rule": "negative"},
    "TextCleaning": {"columns": ["text"], "operations": [{"op": "trim"}]},
}


def _native(frame, engine):
    """Preserve pandas indices and explicit dtypes while selecting the native engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _frame(engine):
    """Include nulls, unmatched text and duplicate indices in ordinary saved applies."""
    return _native(
        pd.DataFrame(
            {"text": [" Yes! ", "no", None, "unseen"], "number": [-1.0, 0.0, None, 2.0]},
            index=[5, 2, 2, 1],
        ),
        engine,
    )


def _record(node, engine, config=None, frame=None):
    """Capture the real fitted record consumed by preprocessing diagnostics."""
    config = CONFIGS[node] if config is None else config
    frame = _frame(engine) if frame is None else frame
    return {
        "name": "cleaning",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(frame, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _probe(record, frame, engine, monkeypatch, context="row"):
    """Exercise saved apply with fit poisoned and retain mutation and exact dtype checks."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """A saved inference diagnostic must never fit on request rows."""
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
    assert detail["state_validation"] == "node_owned", detail
    assert detail["context"] == context, detail
    assert artifact_digest(record) == before
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_fitted_cleaning_has_owned_context_without_execution(node, engine, monkeypatch):
    """Metadata inspection must use saved state without running fit or apply."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Inspection must never execute a transformation."""
        raise AssertionError("Unexpected execution")

    applier: Any = type(record["applier"])
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, {}, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", "preserve", "row"
    )
    assert applier.validate_inference_state(state) is state
    assert artifact_digest(state) == before


@pytest.mark.parametrize("node", NODES)
def test_cleaning_does_not_admit_other_engines(node):
    """Local inspection must not grant Spark or unknown execution engines a capability."""
    assert get_inference_capability(node, {}, {}, engine="spark") is None
    assert get_inference_capability(node, {}, {}, engine="unknown") is None


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("change", ["extra", "missing", "wrong_type", "not_dict"])
def test_cleaning_rejects_malformed_state_structure(node, change):
    """Unknown fields and missing discriminators cannot advertise reviewed saved behavior."""
    state: Any = _record(node, "pandas")["artifact"]
    if change == "extra":
        state["future_mode"] = True
    elif change == "missing":
        state.pop("type")
    elif change == "wrong_type":
        state["type"] = "wrong"
    else:
        state = cast(Any, [])
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        ("Casting", "type_map", {"text": []}),
        ("Casting", "type_map", {2: "float64"}),
        ("Casting", "coerce_on_error", 1),
        ("Casting", "categories", {"absent": ["a"]}),
        ("AliasReplacement", "columns", ["text", "text"]),
        ("AliasReplacement", "alias_type", "unknown"),
        ("AliasReplacement", "custom_map", {"yes": [1]}),
        ("InvalidValueReplacement", "columns", "number"),
        ("InvalidValueReplacement", "rule", "unknown"),
        ("InvalidValueReplacement", "replace_inf", 1),
        ("InvalidValueReplacement", "min_value", "zero"),
        ("InvalidValueReplacement", "replacement", [0]),
        ("TextCleaning", "operations", [{"op": "unknown"}]),
        ("TextCleaning", "operations", [{"op": "trim", "mode": []}]),
        ("TextCleaning", "operations", [{"op": "regex", "pattern": "["}]),
        ("TextCleaning", "operations", [{"op": "regex", "repl": lambda match: ""}]),
        ("TextCleaning", "operations", [{"op": "trim", "future": True}]),
    ],
)
def test_cleaning_rejects_malformed_owned_values(node, field, value):
    """Recognized fields must still contain valid fixed configuration and learned state."""
    state = _record(node, "pandas")["artifact"]
    state[field] = value
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_saved_cleaning_apply_keeps_probe_checks_strict(node, engine, monkeypatch):
    """Saved applies must survive repeated, chunked, reversed and empty requests unchanged."""
    record = _record(node, engine)
    frame = _frame(engine)
    direct = record["applier"].apply(frame, record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert all(check["status"] == "passed" for check in detail["checks"]), detail
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("selection", [[], ["absent"]])
def test_real_cleaning_noops_preserve_schema(node, engine, selection, monkeypatch):
    """Fit-produced empty and missing-column states must retain the unchanged request."""
    record = _record(node, engine, {"columns": selection})
    frame = _frame(engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mode", ["boolean", "country", "custom", "punctuation"])
def test_alias_modes_replay_saved_rules(engine, mode, monkeypatch):
    """Every alias mode must preserve unmatched and null values without refitting maps."""
    frame = _native(pd.DataFrame({"text": [" Yes! ", "U.S.A.", "unseen", None]}), engine)
    config = {"columns": ["text"], "alias_type": mode, "custom_map": {"YES!": "matched"}}
    record = _record("AliasReplacement", engine, config, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    expected = {
        "boolean": ["Yes", "U.S.A.", "unseen"],
        "country": [" Yes! ", "USA", "unseen"],
        "custom": ["matched", "U.S.A.", "unseen"],
        "punctuation": [" Yes ", "USA", "unseen"],
    }[mode]
    assert output["text"].to_list()[:3] == expected
    assert pd.isna(output["text"].to_list()[3])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "rules",
    [
        {"rule": "negative_to_nan"},
        {"mode": "zero_to_nan"},
        {"mode": "percentage_bounds"},
        {"mode": "age_bounds"},
        {"rule": "custom_range", "min_value": np.int64(0)},
        {"rule": "custom_range", "max_value": np.float32(100)},
        {"replace_inf": np.bool_(True), "replace_neg_inf": np.bool_(True)},
    ],
)
def test_invalid_rules_preserve_numpy_state_and_numeric_values(engine, rules, monkeypatch):
    """Normalized presets and infinity flags must replay fixed numeric rules with nulls."""
    frame = _native(
        pd.DataFrame({"number": [-1.0, 0.0, 50.0, 150.0, np.inf, -np.inf, None]}), engine
    )
    config = {"columns": ["number"], "replacement": np.float32(-5), **rules}
    record = _record("InvalidValueReplacement", engine, config, frame)
    direct = record["applier"].apply(frame, record["artifact"])
    state = deepcopy(record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert type(record["artifact"]["replacement"]) is type(state["replacement"])
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "operation",
    [
        {"op": "trim", "mode": "leading"},
        {"op": "trim", "mode": "trailing"},
        {"op": "case", "mode": "lower"},
        {"op": "case", "mode": "upper"},
        {"op": "case", "mode": "title"},
        {"op": "case", "mode": "sentence"},
        {"op": "remove_special", "mode": "keep_alphanumeric"},
        {"op": "remove_special", "mode": "keep_alphanumeric_space"},
        {"op": "remove_special", "mode": "letters_only"},
        {"op": "remove_special", "mode": "digits_only"},
        {"op": "regex", "mode": "collapse_whitespace"},
        {"op": "regex", "mode": "extract_digits"},
        {"op": "regex", "mode": "normalize_slash_dates"},
        {"op": "regex", "mode": "custom", "pattern": r"[0-9]+", "repl": "N"},
    ],
)
def test_text_modes_replay_ordered_operations(engine, operation, monkeypatch):
    """Supported text modes must expose their real full/chunk/empty semantics."""
    frame = _native(pd.DataFrame({"text": ["  Ab! 12/3/2024 ", "no digits", None, ""]}), engine)
    config = {"columns": ["text"], "operations": [operation, {"op": "trim"}]}
    record = _record("TextCleaning", engine, config, frame)
    direct = record["applier"].apply(frame, record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    null_map_inference = engine == "pandas" and operation.get("mode") == "normalize_slash_dates"
    assert detail["status"] == ("failed" if null_map_inference else "passed"), detail
    if null_map_inference:
        checks = {check["name"]: check for check in detail["checks"]}
        assert checks["chunks:1"]["reason"] == "apply_error"
        assert checks["chunks:1"]["error_type"] == "AttributeError"
    assert output.equals(direct)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("float32", ["1.5", "bad", None, "2"], [1.5, None, None, 2.0]),
        ("int16", ["1", "bad", None, "2"], [1, None, None, 2]),
        ("boolean", ["yes", "no", None, "unknown"], [True, False, None, None]),
        ("string", ["a", "b", None, "new"], ["a", "b", None, "new"]),
        (
            "datetime",
            ["2024-01-01", "2/3/2024", None, "bad"],
            [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-03"), None, None],
        ),
    ],
)
def test_casting_families_report_exact_saved_apply(engine, dtype, values, expected, monkeypatch):
    """Row casts must retain nulls while exposing pandas integer dtype changes per chunk."""
    frame = _native(pd.DataFrame({"text": values}), engine)
    config = {"column_types": {"text": dtype}, "coerce_on_error": np.bool_(True)}
    record = _record("Casting", engine, config, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    varying_integer_dtype = engine == "pandas" and dtype == "int16"
    assert detail["status"] == ("failed" if varying_integer_dtype else "passed"), detail
    if varying_integer_dtype:
        checks = {check["name"]: check for check in detail["checks"]}
        assert checks["chunks:1"]["reason"] == "output_mismatch"
    for actual, wanted in zip(output["text"].to_list(), expected, strict=True):
        assert pd.isna(actual) if wanted is None else actual == wanted


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("legacy", [False, True])
def test_casting_vocabulary_distinguishes_legacy_request_context(engine, legacy, monkeypatch):
    """Unfrozen pandas category vocabularies require complete request context."""
    config = {"columns": ["text"], "target_type": "category"}
    training = _native(pd.DataFrame({"text": ["a", "b", None]}), engine)
    frame = _native(pd.DataFrame({"text": ["b", "unseen", None, "a"]}), engine)
    record = _record("Casting", engine, config, training)
    state = record["artifact"]
    if legacy:
        state.pop("categories")
    before = artifact_digest(state)
    output = record["applier"].apply(frame, state)
    capability = get_inference_capability("Casting", config, state, engine=engine)
    assert capability is not None
    assert capability.context == ("global" if legacy and engine == "pandas" else "row")
    if capability.context == "row":
        detail, observed = _probe(record, frame, engine, monkeypatch)
        assert detail["status"] == "passed", detail
        assert observed.equals(output)
    else:
        assert list(output["text"].cat.categories) == ["a", "b", "unseen"]
        singleton = record["applier"].apply(frame.iloc[:1], state)
        assert list(singleton["text"].cat.categories) == ["b"]
        detail, _ = _probe(record, frame, engine, monkeypatch, context="global")
        assert detail["status"] == "requires_context"
    values = output["text"].to_list()
    assert values[0] == "b"
    assert (values[1] == "unseen") if legacy else pd.isna(values[1])
    assert artifact_digest(state) == before


@pytest.mark.parametrize("categories", [["a", "a"], ["a", None], "a"])
def test_casting_rejects_corrupt_saved_vocabulary(categories):
    """Malformed learned categories cannot be silently replaced by request-derived labels."""
    state = _record("Casting", "pandas", {"columns": ["text"], "target_type": "category"})[
        "artifact"
    ]
    state["categories"]["text"] = categories
    with pytest.raises(ValueError):
        get_inference_capability("Casting", {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_casting_strict_invalid_values_keep_original_apply_errors(engine, monkeypatch):
    """Local context must never turn a strict saved cast failure into a successful probe."""
    frame = _native(pd.DataFrame({"text": ["1.5", "bad"]}), engine)
    config = {"column_types": {"text": "float64"}, "coerce_on_error": False}
    record = _record("Casting", engine, config, frame)
    detail, _ = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "failed"
    assert detail["checks"][0]["reason"] == "apply_error"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_text_custom_regex_retains_engine_limitations(engine, monkeypatch):
    """Python lookbehind remains executable locally and unsupported by Polars regex."""
    frame = _native(pd.DataFrame({"text": ["ab", "xb", None]}), engine)
    config = {
        "columns": ["text"],
        "operations": [{"op": "regex", "pattern": r"(?<=a)b", "repl": "X"}],
    }
    record = _record("TextCleaning", engine, config, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == ("failed" if engine == "polars" else "passed"), detail
    if engine == "polars":
        assert detail["checks"][0]["reason"] == "apply_error"
    else:
        assert output["text"].to_list()[:2] == ["aX", "xb"]


def test_invalid_integer_widening_on_empty_requests_remains_visible(monkeypatch):
    """Empty integer requests must still expose dtype differences from a replaced full batch."""
    frame = pd.DataFrame({"number": [-1, 2]})
    config = {"columns": ["number"], "rule": "negative", "replacement": np.nan}
    record = _record("InvalidValueReplacement", "pandas", config, frame)
    detail, output = _probe(record, frame, "pandas", monkeypatch)
    checks = {check["name"]: check for check in detail["checks"]}
    assert detail["status"] == "failed", detail
    assert checks["empty"]["reason"] == "output_mismatch"
    assert pd.isna(output["number"].iloc[0])
    assert output["number"].iloc[1] == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_alias_numpy_rules_are_inspected_without_rewriting(engine, monkeypatch):
    """Local alias inspection must retain genuine NumPy scalar values saved by fit."""
    config = {
        "columns": ["text"],
        "alias_type": "custom",
        "custom_map": {np.str_("YES!"): np.str_("accepted")},
    }
    record = _record("AliasReplacement", engine, config)
    state = record["artifact"]
    assert record["applier"].validate_inference_state(state) is state
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert type(state["custom_map"]["yes"]) is np.str_
    assert output["text"].to_list()[0] == "accepted"


def test_casting_inspects_native_dtype_without_normalizing_saved_state():
    """Native dtype instances accepted by fit must remain available for local replay."""
    frame = pd.DataFrame({"number": [1.0, 2.0]})
    dtype = np.dtype("int16")
    record = _record("Casting", "pandas", {"column_types": {"number": dtype}}, frame)
    state = record["artifact"]
    assert record["applier"].validate_inference_state(state) is state
    output = record["applier"].apply(frame, state)
    assert state["type_map"]["number"] is dtype
    assert output["number"].dtype == dtype
    assert output["number"].to_list() == [1, 2]


@pytest.mark.parametrize("target", [int, np.int64, "complex128", "timedelta64[ns]"])
def test_coercive_pandas_fallback_casts_require_request_context(target, monkeypatch):
    """A bad neighbor can prevent an entire fallback cast, changing successful row values."""
    frame = pd.DataFrame({"text": ["1", "bad"]})
    config = {"column_types": {"text": target}, "coerce_on_error": True}
    record = _record("Casting", "pandas", config, frame)
    state = record["artifact"]
    full = record["applier"].apply(frame, state)
    singleton = record["applier"].apply(frame.iloc[:1], state)
    assert full["text"].iloc[0] == "1"
    assert not isinstance(singleton["text"].iloc[0], str)
    detail, _ = _probe(record, frame, "pandas", monkeypatch, context="global")
    assert detail["status"] == "requires_context"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("operations", [None, [{"op": "trim", "mode": None}]])
def test_text_saved_null_options_keep_native_noop_and_default(engine, operations, monkeypatch):
    """Fit-preserved null options already mean no operations or default trimming."""
    record = _record("TextCleaning", engine, {"columns": ["text"], "operations": operations})
    state = record["artifact"]
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert state["operations"] is operations
    assert output["text"].to_list()[0] == (" Yes! " if operations is None else "Yes!")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node", ["AliasReplacement", "InvalidValueReplacement", "ValueReplacement"]
)
def test_saved_pandas_na_rules_retain_engine_boundaries(node, engine, monkeypatch):
    """Nullable sentinels must be inspected locally while actual engine errors stay visible."""
    configs = {
        "AliasReplacement": {
            "columns": ["text"],
            "alias_type": "custom",
            "custom_map": {"yes": pd.NA},
        },
        "InvalidValueReplacement": {
            "columns": ["number"],
            "rule": "negative",
            "replacement": pd.NA,
        },
        "ValueReplacement": {"columns": ["text"], "mapping": {" Yes! ": pd.NA}},
    }
    record = _record(node, engine, configs[node])
    state = record["artifact"]
    assert record["applier"].validate_inference_state(state) is state
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == ("passed" if engine == "pandas" else "failed"), detail
    if engine == "pandas":
        column = "number" if node == "InvalidValueReplacement" else "text"
        assert pd.isna(output[column].to_list()[0])
    else:
        assert detail["checks"][0]["reason"] == "apply_error"
