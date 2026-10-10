"""Saved imputation estimators retain their actual context and native limitations."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _frame(engine, values=None):
    """Keep numeric nulls and nontrivial pandas labels in both local engines."""
    values = values or {"a": [0.0, 1.0, 2.0, None, 4.0, 5.0], "b": [0.0, 2.0, None, 6.0, 8.0, 10.0]}
    frame = pd.DataFrame(values)
    frame.index = [9, 2, 9, -1, 8, 0][: len(frame)]
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config=None, train=None):
    """Capture a real calculator result, including the learned sklearn estimator."""
    config = config if config is not None else {"columns": ["a", "b"]}
    return {
        "name": "impute",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(
            _frame(engine) if train is None else train, config
        ),
        "applier": NodeRegistry.get_applier(node)(),
    }


def _forbidden(*args, **kwargs):
    """Expose metadata execution or inference-time fitting immediately."""
    raise AssertionError("Unexpected fit or transform during inspection")


@pytest.mark.parametrize("node,context", [("KNNImputer", "row"), ("IterativeImputer", "global")])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_estimator_metadata_inspection_is_pure(node, context, engine, monkeypatch):
    """A fitted estimator must be inspected without learning, transforming or worker admission."""
    record = _record(node, engine)
    state = record["artifact"]
    saved = pickle.dumps(state)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    monkeypatch.setattr(type(state["imputer_object"]), "fit", _forbidden)
    monkeypatch.setattr(type(state["imputer_object"]), "transform", _forbidden)
    assert record["applier"].validate_inference_state(state) is state
    capability = get_inference_capability(node, {}, state, engine=engine)
    assert capability is not None
    assert (capability.context, capability.execution_kind, capability.row_effect) == (
        context,
        "local",
        "preserve",
    )
    assert get_inference_capability(node, {}, state, engine="spark") is None
    with pytest.raises(UnsupportedExecutionError):
        require_capability(node, "apply", engine, config={}, execution_kind="python_batch")
    assert pickle.dumps(state) == saved


def _slice(frame, positions):
    """Select native rows without changing pandas index labels."""
    return frame.iloc[positions] if isinstance(frame, pd.DataFrame) else frame[positions]


def _assert_equal(actual, expected):
    """Keep values, native dtypes, columns and row ordering exact."""
    if isinstance(actual, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


def _probe(record, frame, engine, monkeypatch):
    """Use the existing strict probe with calculator and sklearn fitting disabled."""
    node = record["type"]
    before = artifact_digest(record)
    original = deepcopy(frame)
    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", _forbidden)
    if record["artifact"]:
        monkeypatch.setattr(type(record["artifact"]["imputer_object"]), "fit", _forbidden)
    detail, output = _probe_step(
        record,
        {"name": record["name"], "transformer": node, "params": record["params"]},
        frame,
        engine,
        (1, 3),
        (256, 1024 * 1024),
        active=True,
        project_sha=None,
    )
    _assert_equal(frame, original)
    assert artifact_digest(record) == before
    assert detail["state_validation"] == "node_owned", detail
    return detail, output


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_replays_saved_donors_for_unseen_null_singleton_and_empty(engine, weights, monkeypatch):
    """Neighbor imputation must reuse training donors even when the scoring batch changes."""
    record = _record(
        "KNNImputer", engine, {"columns": ["a", "b"], "n_neighbors": 2, "weights": weights}
    )
    sample = _frame(
        engine, {"b": [None, None, 3.0, 14.0], "keep": [3, 2, 1, 0], "a": [None, 2.0, None, 7.0]}
    )
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert detail["context"] == "row"
    assert list(output.columns) == ["b", "keep", "a"]
    assert output["a"].to_list()[0] == 2.4
    assert output["b"].to_list()[0] == 5.2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_iterative_saved_rounds_require_global_context_for_all_null_request(engine, monkeypatch):
    """An entirely missing singleton takes sklearn's initial-fill shortcut, unlike a mixed batch."""
    record = _record("IterativeImputer", engine)
    state = record["artifact"]
    sample = _frame(engine, {"a": [None, 2.0, None, 7.0], "b": [None, None, 3.0, 14.0]})
    before = artifact_digest(state)
    detail, _ = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "requires_context"
    assert detail["context"] == "global"
    full = record["applier"].apply(sample, state)
    singleton = record["applier"].apply(_slice(sample, [0]), state)
    assert singleton["a"].to_list() == [2.4]
    assert singleton["b"].to_list() == [5.2]
    assert full["a"].to_list()[0] != singleton["a"].to_list()[0]
    _assert_equal(record["applier"].apply(sample, state), full)
    _assert_equal(record["applier"].apply(_slice(sample, []), state), _slice(full, []))
    assert artifact_digest(state) == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_zero_iterative_rounds_use_saved_initial_statistics_for_every_partition(
    engine, monkeypatch
):
    """A zero-round real fit has only fixed initial fills and can declare row context."""
    record = _record("IterativeImputer", engine, {"columns": ["a", "b"], "max_iter": np.int64(0)})
    sample = _frame(engine, {"a": [None, 0.0, None, 7.0], "b": [None, None, 3.0, 14.0]})
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert detail["context"] == "row"
    assert output["a"].to_list() == [2.4, 0.0, 2.4, 7.0]


@pytest.mark.parametrize("node", ["KNNImputer", "IterativeImputer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_all_missing_fit_columns_and_zero_columns_are_genuine_noops(node, engine, monkeypatch):
    """The existing empty artifacts must preserve frames rather than fabricate estimators."""
    train = _frame(engine, {"a": [None, None], "b": [None, None]})
    # Preserve numeric nulls instead of from_pandas' all-None String inference.
    train = train.astype(float) if engine == "pandas" else train.cast(pl.Float64)
    record = _record(node, engine, train=train)
    explicit = _record(node, engine, {"columns": []})
    assert record["artifact"] == explicit["artifact"] == {}
    sample = _frame(engine)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    _assert_equal(output, sample)


@pytest.mark.parametrize("node", ["KNNImputer", "IterativeImputer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_real_fit_keeps_all_null_columns_untouched_and_accepts_numpy_settings(
    node, engine, monkeypatch
):
    """Dropped all-null fit columns stay outside the estimator's learned feature width."""
    train = _frame(engine, {"a": np.array([0, 0, 0, 0], dtype=np.float32), "b": [None] * 4})
    if engine == "pandas":
        train["b"] = train["b"].astype(float)
    else:
        train = train.with_columns(pl.col("b").cast(pl.Float64))
    config = {"columns": (np.str_("a"), "b"), "n_neighbors": np.int64(2), "max_iter": np.int64(0)}
    record = _record(node, engine, config, train)
    assert record["artifact"]["columns"] == ["a"]
    sample = _frame(engine, {"a": [None, 0.0, 8.0], "b": [None, None, None]})
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output["a"].to_list() == [0.0, 0.0, 8.0]
    assert output["b"].to_list() == [None, None, None]


@pytest.mark.parametrize(
    "estimator", ["DecisionTree", "ExtraTrees", "KNeighbors", np.str_("unknown-fallback")]
)
def test_iterative_real_estimator_choices_replay_without_predictor_fit(estimator, monkeypatch):
    """Every existing estimator alias keeps its fitted predictors instead of fitting at apply."""
    record = _record("IterativeImputer", "pandas", {"columns": ["a", "b"], "estimator": estimator})
    imputer = record["artifact"]["imputer_object"]
    for step in imputer.imputation_sequence_:
        monkeypatch.setattr(type(step.estimator), "fit", _forbidden)
    sample = _frame("pandas", {"a": [None, 2.0], "b": [3.0, None]})
    before = artifact_digest(record)
    assert record["applier"].validate_inference_state(record["artifact"]) is record["artifact"]
    output = record["applier"].apply(sample, record["artifact"])
    assert np.isfinite(output.to_numpy()).all()
    assert artifact_digest(record) == before


def _batch_weights(distances):
    """Represent a valid sklearn weight callback whose result depends on request size."""
    weights = np.zeros_like(distances)
    weights[:, 0 if len(distances) > 1 else -1] = 1.0
    return weights


def test_callable_knn_weights_remain_usable_without_a_row_promise(monkeypatch):
    """Arbitrary sklearn callbacks can use other request rows and must stay undeclared."""
    record = _record(
        "KNNImputer", "pandas", {"columns": ["a", "b"], "n_neighbors": 2, "weights": _batch_weights}
    )
    sample = _frame("pandas", {"a": [None, None], "b": [1.0, 9.0]})
    assert record["applier"].validate_inference_state(record["artifact"]) is record["artifact"]
    assert get_inference_capability("KNNImputer", {}, record["artifact"], engine="pandas") is None
    detail, _ = _probe(record, sample, "pandas", monkeypatch)
    assert detail["context"] == "unknown"
    assert detail["status"] == "failed"
    assert any(check.get("reason") == "output_mismatch" for check in detail["checks"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_posterior_estimator_has_no_context_promise_and_changes_rng(engine, monkeypatch):
    """A genuinely fitted posterior sampler consumes random state even with a fixed seed."""
    record = _record("IterativeImputer", engine)
    imputer_type = type(record["artifact"]["imputer_object"])
    # The node currently does not expose sample_posterior; inspect a real saved sklearn fit.
    record["artifact"]["imputer_object"] = imputer_type(
        sample_posterior=True, random_state=0, max_iter=2
    ).fit(_frame("pandas").to_numpy())
    state = record["artifact"]
    before = artifact_digest(state)
    assert get_inference_capability("IterativeImputer", {}, state, engine=engine) is None
    assert artifact_digest(state) == before
    sample = _frame(engine, {"a": [None, 2.0], "b": [3.0, None]})
    detail, _ = _probe(record, sample, engine, monkeypatch)
    assert detail["context"] == "unknown"
    assert detail["checks"][0]["reason"] == "state_mutation"
    record["applier"].apply(sample, state)
    assert artifact_digest(state) != before


@pytest.mark.parametrize("node", ["KNNImputer", "IterativeImputer"])
@pytest.mark.parametrize("damage", ["fields", "columns", "estimator", "width", "array"])
def test_malformed_saved_imputation_state_is_rejected_without_mutation(node, damage):
    """A declaration must not accept missing estimators or mismatched learned dimensions."""
    record = _record(node, "pandas")
    state = record["artifact"]
    imputer = state["imputer_object"]
    if damage == "fields":
        state["extra"] = 1
    elif damage == "columns":
        state["columns"] = "a"
    elif damage == "estimator":
        state["imputer_object"] = type(imputer)()
    elif damage == "width":
        imputer.n_features_in_ = 10
    elif node == "KNNImputer":
        imputer._mask_fit_X = np.zeros((1, 1), dtype=bool)
    else:
        imputer.initial_imputer_.statistics_ = np.zeros(1)
    before = pickle.dumps(state)
    with pytest.raises(ValueError):
        get_inference_capability(node, {}, state, engine="pandas")
    assert pickle.dumps(state) == before


def test_iterative_invalid_learned_feature_index_is_rejected():
    """A corrupted prediction sequence must fail inspection before indexing scoring data."""
    record = _record("IterativeImputer", "pandas")
    sequence = record["artifact"]["imputer_object"].imputation_sequence_
    sequence[0] = sequence[0]._replace(neighbor_feat_idx=np.array([10]))
    with pytest.raises(ValueError, match="neighbor index"):
        record["applier"].validate_inference_state(record["artifact"])


@pytest.mark.parametrize("bound", ["_min_value", "_max_value"])
def test_iterative_nonzero_rounds_require_saved_bounds(bound):
    """A missing learned clipping array fails inspection before the first prediction."""
    record = _record("IterativeImputer", "pandas")
    delattr(record["artifact"]["imputer_object"], bound)
    with pytest.raises(ValueError, match="array"):
        record["applier"].validate_inference_state(record["artifact"])


@pytest.mark.parametrize("node", ["KNNImputer", "IterativeImputer"])
def test_polars_string_null_fit_keeps_existing_numpy_boundary(node):
    """Explicit String columns still reach the existing unsupported NumPy object boundary."""
    with pytest.raises(TypeError, match="isnan"):
        NodeRegistry.get_calculator(node)().fit(
            pl.DataFrame({"a": [None, None]}, schema={"a": pl.String}), {"columns": ["a"]}
        )


@pytest.mark.parametrize("node", ["KNNImputer", "IterativeImputer"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_float32_and_nullable_input_replay_native_output_dtype(node, engine, monkeypatch):
    """Native float32 and nullable pandas conversion retain their own exact output schemas."""
    train = _frame(engine, {"a": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)})
    record = _record(node, engine, {"columns": ["a"], "max_iter": 0}, train)
    sample = _frame(engine, {"a": [None, 0.0, 8.0]})
    sample = sample.astype("Float32") if engine == "pandas" else sample.cast(pl.Float32)
    detail, output = _probe(record, sample, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert str(output["a"].dtype) == ("float64" if engine == "pandas" else "Float32")
