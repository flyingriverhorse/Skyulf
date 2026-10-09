"""Inspect basic fitted encoders without relearning or duplicating their native apply."""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import LabelEncoder

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values=("b", "a", "b", None)):
    """Keep row identity and categorical data equal on the two native engines."""
    frame = pd.DataFrame({"category": values, "keep": list(range(len(values)))})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config=None, frame=None):
    """Build a real fitted record matching the inference diagnostic input."""
    config = {"columns": ["category"]} if config is None else config
    frame = _frame(engine) if frame is None else frame
    return {
        "name": "encoding",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(frame, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
def test_fitted_basic_encoder_owns_local_context(node, engine, monkeypatch):
    """Saved metadata must be inspected without executing data or refitting encoders."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)
    applier: Any = NodeRegistry.get_applier(node)

    def forbidden(*args, **kwargs):
        """Metadata inspection must only inspect state."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, record["params"], state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert applier.validate_inference_state(state) is state
    assert artifact_digest(state) == before


def _probe(record, frame, engine, monkeypatch, context="row"):
    """Run exact chunk, permutation, empty, schema and mutation checks against native apply."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Saved inference must never relearn a vocabulary."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(record["type"]), "fit", forbidden)
    monkeypatch.setattr(LabelEncoder, "fit", forbidden)
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
    assert detail["context"] == context
    assert detail["state_validation"] == "node_owned"
    assert artifact_digest(record) == before
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
@pytest.mark.parametrize(
    "values",
    [
        ["b", "a", "unseen", None],
        [1.0, 2.0, 2.5, None],
        [True, False, True, None],
        list(
            pd.to_datetime(
                pd.Series(["2024-01-01", "2024-01-02 12:00:00", "2024-01-03", None]), format="mixed"
            )
        ),
    ],
)
def test_basic_saved_encoders_keep_exact_native_observations(node, engine, values, monkeypatch):
    """Typed categories, unseen values and nulls must replay without silently weakening checks."""
    frame = _frame(engine, values)
    record = _record(node, engine, frame=frame.head(2))
    detail, output = _probe(record, frame, engine, monkeypatch)
    checks = {check["name"]: check for check in detail["checks"]}
    assert checks["full"]["status"] == "passed"
    if node == "HashEncoder":
        assert detail["status"] == "failed", detail
        assert checks["empty"]["reason"] == "output_mismatch", detail
    else:
        assert detail["status"] == "passed", detail
    assert len(output) == len(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drop_first", [False, np.bool_(True)])
@pytest.mark.parametrize("values", [["a", "b", None], [None, None, None], ["a", "a", "a"]])
def test_dummy_fitted_vocabulary_handles_empty_and_dropped_categories(
    engine, drop_first, values, monkeypatch
):
    """Zero-indicator vocabularies must retain rows and the saved drop-first scalar."""
    frame = _frame(engine, values)
    config = {"columns": (np.str_("category"),), "drop_first": drop_first}
    record = _record("DummyEncoder", engine, config, frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["keep"].to_list() == [0, 1, 2]
    assert record["artifact"]["drop_first"] is drop_first


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
@pytest.mark.parametrize("config", [{}, {"columns": []}, {"columns": ["absent"]}])
def test_basic_encoder_defaults_and_real_noops(node, engine, config, monkeypatch):
    """Actual default fits and absent or empty column selections remain inspectable."""
    record = _record(node, engine, config)
    frame = _frame(engine)
    detail, output = _probe(record, frame, engine, monkeypatch)
    expected = "failed" if node == "HashEncoder" and not config else "passed"
    assert detail["status"] == expected, detail
    if config or node == "LabelEncoder":
        assert output.equals(frame)
    else:
        assert len(output) == len(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
def test_basic_missing_inference_columns_remain_native_noops(node, engine, monkeypatch):
    """Resolved training selections must not be rediscovered from a different request schema."""
    record = _record(node, engine)
    frame = (
        _frame(engine).drop("category", axis=1)
        if engine == "pandas"
        else _frame(engine).drop("category")
    )
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
def test_legacy_keys_keep_native_context_and_values(node, engine, monkeypatch):
    """Unversioned pandas datetime rendering must not receive a false row declaration."""
    frame = _frame(
        engine, list(pd.to_datetime(["2024-01-01", "2024-01-02 12:00:00"], format="mixed"))
    )
    record = _record(node, engine, frame=frame)
    version = "numeric_normalization_version" if node == "HashEncoder" else "category_key_version"
    record["artifact"].pop(version)
    direct = record["applier"].apply(frame, record["artifact"])
    singleton = record["applier"].apply(frame.head(1), record["artifact"])
    context = "global" if engine == "pandas" else "row"
    detail, _ = _probe(record, frame, engine, monkeypatch, context=context)
    if engine == "pandas":
        assert detail["status"] == "requires_context", detail
        assert not direct.head(1).equals(singleton)
    else:
        assert direct.head(1).equals(singleton)
        assert detail["status"] == ("failed" if node == "HashEncoder" else "passed"), detail


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("missing_code", [-9, np.int64(-3), np.float64(-2.0)])
def test_label_tuple_rules_and_numpy_missing_code(engine, missing_code, monkeypatch):
    """Native local scalar and ordered column containers must survive unchanged."""
    config = {"columns": (np.str_("category"),), "missing_code": missing_code}
    record = _record("LabelEncoder", engine, config, _frame(engine, ["a", "b"]))
    detail, output = _probe(record, _frame(engine, ["a", "new", None]), engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["category"].to_list() == [0, missing_code, missing_code]
    assert record["artifact"]["columns"] is config["columns"]
    assert record["artifact"]["missing_code"] is missing_code


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("feature_columns", [[], ["category", "target"]])
def test_label_embedded_target_replays_its_fitted_vocabulary(engine, feature_columns, monkeypatch):
    """Target-only and feature-plus-target artifacts must preserve target routing and schema."""
    train = pd.DataFrame({"category": ["a", "b", "a"], "target": ["yes", "no", "yes"]})
    sample = pd.DataFrame({"category": ["a", "new", None], "target": ["yes", "other", None]})
    if engine == "polars":
        train, sample = pl.from_pandas(train), pl.from_pandas(sample)
    config = {"columns": feature_columns, "target_column": "target"}
    record = _record("LabelEncoder", engine, config, train)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["target"].to_list() == [1, -1, -1]
    assert record["artifact"]["target_column"] == "target"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
@pytest.mark.parametrize("change", ["extra", "missing", "kind", "columns", "version"])
def test_basic_encoder_malformed_state_rejected(node, engine, change):
    """Malformed saved shape or unknown key versions cannot acquire local context."""
    state = _record(node, engine)["artifact"]
    if change == "extra":
        state["callback"] = object()
    elif change == "missing":
        state.pop("type")
    elif change == "kind":
        state["type"] = "other"
    elif change == "columns":
        state["columns"] = "category"
    else:
        key = "numeric_normalization_version" if node == "HashEncoder" else "category_key_version"
        state[key] = 99
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine=engine)


@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
def test_basic_local_hooks_do_not_admit_spark(node):
    """Adding local metadata must not grant a distributed execution path."""
    assert get_inference_capability(node, {}, {}, engine="spark") is None


@pytest.mark.parametrize(
    "change", ["unfitted", "count", "dimension", "key_type", "duplicate", "unknown"]
)
def test_label_malformed_learned_encoder_is_rejected(change):
    """Inspect sklearn vocabulary structure without executing a mutated learned object."""
    state = _record("LabelEncoder", "pandas")["artifact"]
    encoder = state["encoders"]["category"]
    if change == "unfitted":
        state["encoders"]["category"] = LabelEncoder()
    elif change == "count":
        state["classes_count"]["category"] = 99
    elif change == "dimension":
        encoder.classes_ = encoder.classes_.reshape(1, -1)
    elif change == "key_type":
        encoder.classes_ = np.array([1, 2])
    elif change == "duplicate":
        encoder.classes_ = np.array(["a", "a", "b"])
    else:
        state["encoders"]["other"] = encoder
    with pytest.raises(ValueError):
        get_inference_capability("LabelEncoder", {}, state, engine="pandas")


@pytest.mark.parametrize(
    "categories", [{"category": ["a", "a"]}, {"other": ["a"]}, {"category": [object()]}]
)
def test_dummy_malformed_vocabulary_is_rejected(categories):
    """Invalid categories must fail before pandas or generated-name construction sees them."""
    state = _record("DummyEncoder", "pandas")["artifact"]
    state["categories"] = categories
    with pytest.raises(ValueError):
        get_inference_capability("DummyEncoder", {}, state, engine="pandas")


@pytest.mark.parametrize("count", [0, -1, "8", None, object()])
def test_hash_invalid_bucket_count_is_rejected(count):
    """An unusable modulo divisor must fail at saved-state inspection."""
    state = _record("HashEncoder", "pandas")["artifact"]
    state["n_features"] = count
    with pytest.raises(ValueError):
        get_inference_capability("HashEncoder", {}, state, engine="pandas")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["DummyEncoder", "HashEncoder", "LabelEncoder"])
def test_empty_training_frames_produce_inspectable_native_states(node, engine, monkeypatch):
    """An empty learned vocabulary must retain the current native empty replay behavior."""
    frame = _frame(engine).head(0)
    record = _record(node, engine, frame=frame)
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert len(output) == 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_hash_numpy_bucket_scalar_keeps_native_engine_boundary(engine, monkeypatch):
    """NumPy modulo dtype differences must not be hidden by normalizing saved parameters."""
    config = {"columns": (np.str_("category"),), "n_features": np.int64(8)}
    record = _record("HashEncoder", engine, config, _frame(engine, ["a", "b"]))
    frame = _frame(engine, ["a", "b", "unseen", None])
    # NumPy 2 rejects oversized Python integers instead of promoting modulo to float.
    error = OverflowError if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else None
    if error is not None:
        with pytest.raises(error):
            record["applier"].apply(frame, record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert record["artifact"]["n_features"] is config["n_features"]
    assert detail["status"] == "failed", detail
    if error is not None:
        assert detail["checks"] == [
            {
                "name": "full",
                "status": "failed",
                "reason": "apply_error",
                "error_type": error.__name__,
            }
        ]
        assert output.equals(frame)
    elif engine == "polars":
        # Unordered unique values can expose mixed bucket types at different checks.
        failures = [check for check in detail["checks"] if check["status"] == "failed"]
        assert failures
        for check in failures:
            assert check["reason"] in {"apply_error", "output_mismatch"}
            if check["reason"] == "apply_error":
                assert check["error_type"] == "TypeError"
    else:
        assert len(output) == 4
        checks = {check["name"]: check for check in detail["checks"]}
        assert checks["full"]["status"] == "passed"
        assert checks["chunks:1"]["reason"] == "output_mismatch"
        assert checks["chunks:3"]["reason"] == "output_mismatch"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_hash_version_one_keeps_legacy_batch_context(engine, monkeypatch):
    """Numeric normalization alone did not repair pandas datetime batch rendering."""
    frame = _frame(
        engine, list(pd.to_datetime(["2024-01-01", "2024-01-02 12:00:00"], format="mixed"))
    )
    record = _record("HashEncoder", engine, frame=frame)
    record["artifact"]["numeric_normalization_version"] = 1
    context = "global" if engine == "pandas" else "row"
    detail, _ = _probe(record, frame, engine, monkeypatch, context=context)
    assert detail["status"] == ("requires_context" if engine == "pandas" else "failed"), detail


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_label_explicit_target_replays_without_fitting_or_mutating(engine, monkeypatch):
    """Explicit target-only records must inspect and replay their original native classes."""
    frame = _frame(engine, ["a", "b", "a"])
    target = pd.Series(["yes", "no", "yes"], name="target")
    if engine == "polars":
        target = pl.Series("target", target.to_list())
    config = {"columns": ()}
    state = NodeRegistry.get_calculator("LabelEncoder")().fit((frame, target), config)
    before = artifact_digest(state)
    applier: Any = NodeRegistry.get_applier("LabelEncoder")()

    def forbidden(*args, **kwargs):
        """Target inference must never learn classes from the request."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(LabelEncoder, "fit", forbidden)
    assert applier.validate_inference_state(state) is state
    features, encoded = applier.apply((frame, target), state)
    assert features.equals(frame)
    assert encoded.to_list() == [1, 0, 1]
    assert encoded.name == "target"
    assert artifact_digest(state) == before


def test_dummy_intrinsic_generated_name_collision_is_rejected():
    """Saved vocabularies cannot declare success when their generated indicator names collide."""
    frame = pd.DataFrame({"category": ["a"], "category_a": ["b"]})
    record = _record("DummyEncoder", "pandas", {"columns": ["category", "category_a"]}, frame)
    record["artifact"]["categories"] = {"category": ["a_b"], "category_a": ["b"]}
    with pytest.raises(ValueError, match="DummyEncoder"):
        get_inference_capability("DummyEncoder", {}, record["artifact"], engine="pandas")


def test_label_metadata_never_calls_native_transform(monkeypatch):
    """Learned-state inspection must read classes rather than execute the sklearn estimator."""
    record = _record("LabelEncoder", "pandas")

    def forbidden(*args, **kwargs):
        """Metadata queries cannot execute estimator behavior."""
        raise AssertionError("Unexpected transform")

    monkeypatch.setattr(LabelEncoder, "transform", forbidden)
    assert get_inference_capability(
        "LabelEncoder", {}, record["artifact"], engine="pandas"
    ) == ExecutionCapability("pandas", "apply", "local", "preserve", "row")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_label_legacy_numeric_string_keys_retain_dtype_boundary(engine):
    """Old string-key models must not silently acquire the new typed numeric lookup semantics."""
    training = _frame(engine, [1, 2])
    frame = _frame(engine, [1.0, 2.0, None])
    record = _record("LabelEncoder", engine, frame=training)
    canonical = record["applier"].apply(frame, record["artifact"])
    assert canonical["category"].to_list() == [0, 1, -1]
    legacy = record["artifact"]
    legacy.pop("category_key_version")
    legacy["encoders"]["category"] = LabelEncoder().fit(np.array(["1", "2"]))
    before = artifact_digest(legacy)
    output = record["applier"].apply(frame, legacy)
    assert output["category"].to_list() == [-1, -1, -1]
    capability = get_inference_capability("LabelEncoder", record["params"], legacy, engine=engine)
    assert capability is not None
    assert capability.context == ("global" if engine == "pandas" else "row")
    assert artifact_digest(legacy) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_label_decimal_missing_code_retains_native_state(engine):
    """Native Decimal fallback codes must inspect without coercion or a seal-support promise."""
    missing_code = Decimal("-1")
    config = {"columns": ["category"], "missing_code": missing_code}
    record = _record("LabelEncoder", engine, config, _frame(engine, ["a", "b"]))
    state = record["artifact"]
    output = record["applier"].apply(_frame(engine, ["a", "unknown"]), state)
    assert output["category"].to_list() == [0, -1]
    assert get_inference_capability("LabelEncoder", config, state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert record["applier"].validate_inference_state(state) is state
    assert state["missing_code"] is missing_code
    with pytest.raises(TypeError, match="Decimal"):
        artifact_digest(state)
