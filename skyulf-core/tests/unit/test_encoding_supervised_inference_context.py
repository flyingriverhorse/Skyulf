"""Saved ordinal and supervised encodings retain their inference-only local context."""

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

NODES = ["OrdinalEncoder", "TargetEncoder", "WOEEncoder"]


def _native(frame, engine):
    """Retain native input types and pandas indices for exact probe comparisons."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _training(engine):
    """Supply complete binary labels and a learned null feature category."""
    frame = pd.DataFrame({"cat": ["a", "a", "b", "b", None, "c", "a", "c"]})
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return _native(frame, engine), pl.from_pandas(labels) if engine == "polars" else labels


def _config(node):
    """Choose literal public recipes with a hand-checkable supervised fallback."""
    return {
        "OrdinalEncoder": {"columns": ["cat"]},
        "TargetEncoder": {"columns": ["cat"], "target_type": "binary", "smooth": 0.0},
        "WOEEncoder": {"columns": ["cat"], "regularization": 0.5},
    }[node]


def _record(node, engine, config=None, training=None):
    """Create the real fitted record consumed by local preprocessing diagnostics."""
    config = _config(node) if config is None else config
    training = _training(engine) if training is None else training
    return {
        "name": "encoded",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(training, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _probe(record, frame, engine, monkeypatch, context="row"):
    """Disable all learning and inspect exact saved apply and state/input mutation checks."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Inference must never fit a node or rebuild a supervised estimator."""
        raise AssertionError("Unexpected fit")

    calculator = NodeRegistry.get_calculator(record["type"])
    monkeypatch.setattr(calculator, "fit", forbidden)
    if hasattr(calculator, "fit_transform_train"):
        monkeypatch.setattr(calculator, "fit_transform_train", forbidden)
    monkeypatch.setattr(OrdinalEncoder, "fit", forbidden)
    monkeypatch.setattr(TargetEncoder, "fit", forbidden)
    monkeypatch.setattr(TargetEncoder, "fit_transform", forbidden)
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
    assert detail.get("context") == context, detail
    assert detail.get("state_validation") == "node_owned", detail
    assert artifact_digest(record) == before
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_supervised_saved_state_has_pure_owned_context(node, engine, monkeypatch):
    """Metadata inspection must accept genuine state without fitting or applying it."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)

    def forbidden(*args, **kwargs):
        """Expose hidden transformation calls during a metadata query."""
        raise AssertionError("Unexpected execution")

    applier: Any = type(record["applier"])
    monkeypatch.setattr(applier, "apply", forbidden)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(OrdinalEncoder, "fit", forbidden)
    monkeypatch.setattr(TargetEncoder, "fit", forbidden)
    assert applier.validate_inference_state(state) is state
    assert get_inference_capability(node, {}, state, engine=engine) == ExecutionCapability(
        engine, "apply", "local", "preserve", "row"
    )
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_saved_encoders_keep_null_unseen_chunk_reverse_and_empty_behavior(
    node, engine, monkeypatch
):
    """A saved mapping must not change when an unknown or null arrives beside another row."""
    record = _record(node, engine)
    frame = _native(pd.DataFrame({"cat": ["a", "unseen", None, "c"]}, index=[7, 2, 2, 1]), engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert all(check["status"] == "passed" for check in detail["checks"]), detail
    expected_unknown = {"OrdinalEncoder": -1.0, "TargetEncoder": 0.5, "WOEEncoder": 0.0}[node]
    assert output["cat"].to_list()[1] == expected_unknown


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("change", ["missing", "extra", "type", "not_dict"])
def test_supervised_context_rejects_malformed_outer_state(node, change):
    """Unrecognized or incomplete artifacts must fail before advertising local behavior."""
    state: Any = _record(node, "pandas")["artifact"]
    if change == "missing":
        state.pop("type")
    elif change == "extra":
        state["future_mode"] = True
    elif change == "type":
        state["type"] = "wrong"
    else:
        state = cast(Any, [])
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("node", NODES)
def test_supervised_context_does_not_enable_other_engines(node):
    """Diagnostic declarations must not imply Spark or arbitrary engine support."""
    assert get_inference_capability(node, {}, {}, engine="spark") is None
    assert get_inference_capability(node, {}, {}, engine="other") is None


@pytest.mark.parametrize("node", ["OrdinalEncoder", "TargetEncoder"])
@pytest.mark.parametrize("damage", ["unfitted", "feature_count", "category_width", "override"])
def test_saved_estimator_contract_rejects_malformed_execution_state(node, damage):
    """An estimator identity alone cannot certify unfitted, inconsistent or overridden execution."""
    state = _record(node, "pandas")["artifact"]
    encoder = state["encoder_object"]
    if damage == "unfitted":
        state["encoder_object"] = type(encoder)()
    elif damage == "feature_count":
        encoder.n_features_in_ = 2
    elif damage == "category_width":
        encoder.categories_ = []
    else:
        encoder.transform = lambda values: values
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize(
    ("node", "field", "value"),
    [
        ("OrdinalEncoder", "categories_count", [999]),
        ("OrdinalEncoder", "encoders", {"unknown": None}),
        ("OrdinalEncoder", "category_key_version", 2),
        ("TargetEncoder", "columns", ["cat", "other"]),
        ("WOEEncoder", "mappings", {"cat": {"a": "bad"}}),
        ("WOEEncoder", "mappings", {}),
        ("WOEEncoder", "information_value", {"other": 0.5}),
        ("WOEEncoder", "default", object()),
        ("WOEEncoder", "category_key_version", True),
    ],
)
def test_supervised_owned_values_are_checked(node, field, value):
    """Malformed maps, counts and lookup-version fields cannot masquerade as fitted state."""
    state = _record(node, "pandas")["artifact"]
    state[field] = value
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["TargetEncoder", "WOEEncoder"])
def test_cross_fitted_training_saves_full_inference_mapping(node, engine, monkeypatch):
    """Training encodings exclude each row's label while saved apply uses the full-data map."""
    frame = _native(pd.DataFrame({"cat": list("abcdefgh")}), engine)
    labels = _training(engine)[1]
    config = _config(node)
    state, transformed = NodeRegistry.get_calculator(node)().fit_transform_train(
        (frame, labels), config
    )
    record = {
        "name": "encoded",
        "type": node,
        "params": config,
        "artifact": state,
        "applier": NodeRegistry.get_applier(node)(),
    }
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    training_values = transformed[0]["cat"].to_list()
    assert training_values != output["cat"].to_list()
    assert list(output.columns) == ["cat"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_type", ["continuous", "multiclass"])
def test_target_encoder_resolved_output_shape_and_defaults(engine, target_type, monkeypatch):
    """Saved class order and global means determine regression or expanded multiclass output."""
    training = _training(engine)[0]
    labels = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if target_type == "continuous"
        else [0, 1, 2, 0, 1, 2, 0, 1],
        name="target",
    )
    if engine == "polars":
        labels = pl.from_pandas(labels)
    config = {"columns": (np.str_("cat"),), "target_type": target_type, "smooth": np.float64(0)}
    record = _record("TargetEncoder", engine, config, (training, labels))
    sample = _native(pd.DataFrame({"cat": ["new", "a", None, "c"]}), engine)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    if target_type == "continuous":
        assert output["cat"].to_list()[0] == pytest.approx(0.45)
    else:
        assert list(output.columns) == ["cat_cls0", "cat_cls1", "cat_cls2"]
        assert [output[col].to_list()[0] for col in output.columns] == [3 / 8, 3 / 8, 2 / 8]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_real_empty_feature_artifacts_preserve_requests(node, engine, monkeypatch):
    """True fitted no-op and ordinal target-only state must remain usable without y."""
    record = _record(node, engine, {"columns": []})
    frame = _training(engine)[0]
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ordinal_embedded_target_is_optional_at_inference(engine, monkeypatch):
    """An encoded training target must not become a required feature for prediction."""
    frame = pd.DataFrame({"cat": ["a", "b", "a", "b"], "target": ["no", "yes", "no", "yes"]})
    config = {
        "columns": ["target", "cat"],
        "target_column": "target",
        "categories_order": ["no,yes", "a,b"],
    }
    record = _record("OrdinalEncoder", engine, config, _native(frame, engine))
    state = record["artifact"]
    assert state["target_column"] == "target"
    sample = _native(pd.DataFrame({"cat": ["b", "new", None, "a"]}), engine)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["cat"].to_list() == [1.0, -1.0, -1.0, 0.0]
    embedded = record["applier"].apply(_native(frame, engine), state)
    assert list(embedded.columns) == ["cat", "target"]
    assert embedded["target"].to_list() == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["OrdinalEncoder", "WOEEncoder"])
def test_legacy_saved_key_rules_remain_unmodified(node, engine, monkeypatch):
    """Artifacts without a key-version field retain their original text lookup contract."""
    record = _record(node, engine)
    state = record["artifact"]
    state.pop("category_key_version")
    sample = _native(pd.DataFrame({"cat": ["a", "new", None, "c"]}), engine)
    before = deepcopy(state)
    context = "global" if engine == "pandas" else "row"
    detail, _ = _probe(record, sample, engine, monkeypatch, context=context)
    assert detail["status"] == ("requires_context" if context == "global" else "passed"), detail
    assert "category_key_version" not in state
    assert artifact_digest(state) == artifact_digest(before)


@pytest.mark.parametrize("node", ["OrdinalEncoder", "WOEEncoder"])
def test_legacy_pandas_datetime_keys_need_complete_request_context(node, monkeypatch):
    """Legacy string conversion changes a midnight key when another row contains a time."""
    frame = pd.DataFrame({"cat": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-02 01:00:00"])})
    record = _record(
        node, "pandas", {"columns": ["cat"]}, (frame, pd.Series([0, 1], name="target"))
    )
    record["artifact"].pop("category_key_version")
    full = record["applier"].apply(frame, record["artifact"])
    singleton = record["applier"].apply(frame.iloc[:1], record["artifact"])
    assert full["cat"].iloc[0] != singleton["cat"].iloc[0]
    detail, _ = _probe(record, frame, "pandas", monkeypatch, context="global")
    assert detail["status"] == "requires_context"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("unknown", [np.int64(-7), np.float64(np.nan)])
def test_ordinal_numpy_unknown_codes_are_saved_without_coercion(engine, unknown, monkeypatch):
    """Supported NumPy unknown codes must remain distinct from learned positions."""
    record = _record(
        "OrdinalEncoder", engine, {"columns": (np.str_("cat"),), "unknown_value": unknown}
    )
    detail, output = _probe(
        record, _native(pd.DataFrame({"cat": ["new", "a"]}), engine), engine, monkeypatch
    )
    assert detail["status"] == "passed", detail
    assert record["artifact"]["encoder_object"].unknown_value is unknown
    actual = output["cat"].to_list()[0]
    assert pd.isna(actual) if pd.isna(unknown) else actual == unknown


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ordinal_python_bool_unknown_matches_fitted_integer_contract(engine, monkeypatch):
    """Python True is a native integer code when it does not overlap the learned category."""
    training = _native(pd.DataFrame({"cat": ["a", "a"]}), engine)
    record = _record(
        "OrdinalEncoder", engine, {"columns": ["cat"], "unknown_value": True}, training
    )
    sample = _native(pd.DataFrame({"cat": ["a", "unknown"]}), engine)
    assert record["applier"].apply(sample, record["artifact"])["cat"].to_list() == [0.0, 1.0]
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert record["artifact"]["encoder_object"].unknown_value is True
    assert output["cat"].to_list() == [0.0, 1.0]


@pytest.mark.parametrize("unknown", [False, np.bool_(True)])
def test_ordinal_bool_unknown_keeps_native_type_and_overlap_rejections(unknown):
    """Accepting Python True must not permit a colliding code or a NumPy boolean."""
    training = pd.DataFrame({"cat": ["a", "a"]})
    with pytest.raises(ValueError):
        _record(
            "OrdinalEncoder", "pandas", {"columns": ["cat"], "unknown_value": unknown}, training
        )
    record = _record("OrdinalEncoder", "pandas", {"columns": ["cat"]}, training)
    record["artifact"]["encoder_object"].unknown_value = unknown
    with pytest.raises(ValueError):
        get_inference_capability("OrdinalEncoder", {}, record["artifact"], engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["OrdinalEncoder", "TargetEncoder"])
@pytest.mark.parametrize("missing", ["all", "partial"])
def test_saved_encoder_missing_inputs_keep_native_dimension_contract(
    node, engine, missing, monkeypatch
):
    """Dropping every selected input is a no-op; partial feature matrices still fail sklearn."""
    frame, labels = _training("pandas")
    frame["other"] = ["x", "y"] * 4
    config = {**_config(node), "columns": ["cat", "other"]}
    record = _record(
        node,
        engine,
        config,
        (_native(frame, engine), pl.from_pandas(labels) if engine == "polars" else labels),
    )
    sample = frame.drop(columns=["cat", "other"] if missing == "all" else ["other"])
    sample["keep"] = list(range(len(sample)))
    sample = _native(sample, engine)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == ("passed" if missing == "all" else "failed"), detail
    if missing == "partial":
        assert detail["checks"][0]["reason"] == "apply_error"
    else:
        assert output.equals(sample)
