"""Separate text vocabularies preserve source identity and old joined artifact replay."""

import copy
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from skyulf import SkyulfPipeline
from skyulf.inference._partition_builtins import fresh_builtin_record
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.inference.probed_steps import REQUIRED_CHECKS
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry

NODES = ["count_vectorizer", "tfidf_vectorizer"]


def _train():
    """Keep source-specific tokens, null documents and duplicate indexes observable."""
    return pd.DataFrame(
        {
            "title": ["red blue", "red", None, "blue green"],
            "body": ["blue", "blue blue", "red blue", None],
            "keep": [7, 3, 9, 1],
        },
        index=[5, 5, 2, 2],
    )


def _native(frame, engine):
    """Use the public native dataframe inputs without wrapper-specific test shortcuts."""
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drop", [False, True])
def test_separate_vocabularies_match_independent_sklearn_and_disk_replay(
    node, engine, drop, tmp_path
):
    """Per-source statistics, output order, unknowns and nulls must survive batch and engine changes."""
    train = _train()
    config = {"columns": ["body", "title"], "column_mode": "per_column", "drop_original": drop}
    state = NodeRegistry.get_calculator(node)().fit(_native(train, engine), config)
    assert state["column_mode"] == "per_column"
    assert list(state["column_artifacts"]) == config["columns"]
    path = tmp_path / "text.pkl"
    path.write_bytes(pickle.dumps(state))
    state = pickle.loads(path.read_bytes())
    digest = artifact_digest(state)
    sample = pd.DataFrame(
        {
            "title": ["green", None, "unseen", "red red"],
            "body": ["green", "red", None, "blue"],
            "keep": [1, 2, 3, 4],
        },
        index=[4, 4, 0, 0],
    )
    expected = []
    names = []
    estimator = CountVectorizer if node == "count_vectorizer" else TfidfVectorizer
    suffix = "count" if node == "count_vectorizer" else "tfidf"
    for col in config["columns"]:
        fitted = estimator().fit(train[col].fillna(""))
        expected.append(fitted.transform(sample[col].fillna("")).toarray())
        names.extend(f"{col}__{suffix}__{term}" for term in fitted.get_feature_names_out())
    assert state["output_columns"] == names
    for other_engine in ("pandas", "polars"):
        request = _native(sample, other_engine)
        applier = NodeRegistry.get_applier(node)()
        result = applier.apply(request, state)
        output = (
            result.select(names).to_numpy()
            if other_engine == "polars"
            else result[names].to_numpy()
        )
        np.testing.assert_allclose(output, np.concatenate(expected, axis=1))
        for index in range(len(sample)):
            part = request[index : index + 1] if other_engine == "polars" else request.iloc[[index]]
            actual = applier.apply(part, state)
            values = (
                actual.select(names).to_numpy()
                if other_engine == "polars"
                else actual[names].to_numpy()
            )
            np.testing.assert_allclose(values, output[index : index + 1])
        empty = applier.apply(request.head(0), state)
        assert list(empty.columns) == list(result.columns)
        assert list(empty.dtypes) == list(result.dtypes)
        assert list(result.columns) == (["keep"] if drop else list(sample.columns)) + names
        if other_engine == "pandas":
            pd.testing.assert_frame_equal(request, sample)
    assert NodeRegistry.get_applier(node)().validate_inference_state(state) is state
    assert artifact_digest(state) == digest


@pytest.mark.parametrize("node", NODES)
def test_per_column_ngrams_never_cross_document_fields(node):
    """Separate vocabularies cannot invent the last-title/first-body bigram of joined mode."""
    frame = pd.DataFrame({"title": ["red fox", "red fox"], "body": ["blue bird", "blue bird"]})
    config = {"columns": ["title", "body"], "ngram_range": [2, 2], "column_mode": "per_column"}
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    names = state["output_columns"]
    assert len(names) == 2
    assert all("fox blue" not in name for name in names)


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_per_column_empty_vocabulary_and_missing_sources_fail_clearly(node, engine):
    """No partial fitted recipe may hide a source that produced no usable vocabulary."""
    frame = _train()
    frame["body"] = ""
    calculator = NodeRegistry.get_calculator(node)()
    with pytest.raises(ValueError, match="body.*empty vocabulary"):
        calculator.fit(
            _native(frame, engine), {"columns": ["title", "body"], "column_mode": "per_column"}
        )
    state = calculator.fit(
        _native(_train(), engine), {"columns": ["title", "body"], "column_mode": "per_column"}
    )
    with pytest.raises(ValueError, match="body"):
        NodeRegistry.get_applier(node)().apply(
            _native(_train().drop(columns="body"), engine), state
        )


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("mode", [None, "unknown", True])
def test_vectorizer_rejects_unknown_column_mode(node, mode):
    """Malformed explicit mode values cannot silently fall back to joined text."""
    with pytest.raises(ValueError, match="column_mode"):
        NodeRegistry.get_calculator(node)().fit(
            _train(), {"columns": ["title", "body"], "column_mode": mode}
        )


@pytest.mark.parametrize("node", NODES)
def test_joined_legacy_artifact_shape_and_replay_stay_unchanged(node, tmp_path):
    """Omitted and explicit joined modes must keep the old flat serialization contract."""
    calc = NodeRegistry.get_calculator(node)()
    frame = _train()
    legacy = calc.fit(frame, {"columns": ["title", "body"]})
    explicit = calc.fit(frame, {"columns": ["title", "body"], "column_mode": "joined"})
    assert "column_mode" not in legacy and "column_artifacts" not in legacy
    assert set(legacy) == set(explicit)
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(legacy))
    applier = NodeRegistry.get_applier(node)()
    pd.testing.assert_frame_equal(
        applier.apply(frame, pickle.loads(path.read_bytes())), applier.apply(frame, explicit)
    )


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("damage", ["mode", "child", "names", "vocabulary", "drop", "nested"])
def test_per_column_saved_state_rejects_inconsistent_child_bindings(node, damage):
    """Outer axes and nested learned statistics must describe exactly the same output."""
    state = copy.deepcopy(
        NodeRegistry.get_calculator(node)().fit(
            _train(), {"columns": ["title", "body"], "column_mode": "per_column"}
        )
    )
    if damage == "mode":
        state["column_mode"] = "other"
    elif damage == "child":
        state["column_artifacts"].pop("title")
    elif damage == "names":
        state["output_columns"] = state["output_columns"][::-1]
    elif damage == "vocabulary":
        state["column_artifacts"]["title"]["vocabulary"] = {"tampered": 0}
    elif damage == "drop":
        state["column_artifacts"]["title"]["drop_original"] = True
    else:
        state["column_artifacts"]["title"]["column_mode"] = "per_column"
    with pytest.raises(ValueError):
        NodeRegistry.get_applier(node)().validate_inference_state(state)


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_per_column_actual_no_refit_partition_probe(node, engine):
    """Opt-in learned state must satisfy the real sealed-probe checks under its existing owner."""
    frame = _native(_train(), engine)
    params = {"columns": ["title", "body"], "column_mode": "per_column", "drop_original": True}
    state = NodeRegistry.get_calculator(node)().fit(frame, params)
    record = {
        "name": "text",
        "type": node,
        "artifact": pickle.loads(pickle.dumps(state)),
        "params": params,
        "applier": NodeRegistry.get_applier(node)(),
    }
    config = {"name": "text", "transformer": node, "params": params}
    detail, _ = _probe_step(
        record,
        config,
        frame,
        engine,
        (1, 2, 7),
        (256, 8 * 1024 * 1024),
        active=True,
        project_sha=None,
        perturb=True,
        fresh=fresh_builtin_record,
    )
    assert {item["name"]: item["status"] for item in detail["checks"]} == dict.fromkeys(
        REQUIRED_CHECKS, "passed"
    ), detail


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_per_column_pipeline_save_load_predict(node, engine, tmp_path, monkeypatch):
    """Real exported models retain independent vocabularies and IDF across inference engines."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    frame = pd.concat([_train()] * 3, ignore_index=True).drop(columns="keep")
    frame["target"] = [1.0, 3.0, 7.0, 5.0] * 3
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "text",
                    "transformer": node,
                    "params": {
                        "columns": ["title", "body"],
                        "column_mode": "per_column",
                        "drop_original": True,
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(_native(frame, engine), target_column="target")
    save_pipeline(pipeline, tmp_path / "pipeline")
    restored = load_pipeline(tmp_path / "pipeline")
    sample = frame.drop(columns="target")
    expected = predict_pipeline(sample, restored).to_numpy()
    np.testing.assert_allclose(
        predict_pipeline(pl.from_pandas(sample), restored).to_numpy(), expected
    )
    np.testing.assert_allclose(
        predict_pipeline(sample.iloc[::-1], restored).to_numpy(), expected[::-1]
    )
    assert predict_pipeline(_native(sample, engine).head(0), restored).shape[0] == 0


@pytest.mark.parametrize("node", NODES)
def test_per_column_target_exclusion_feature_limits_and_collision_guards(node):
    """Separate vocabularies keep target protection and validate names against the full frame."""
    frame = _train()
    frame["label"] = ["leak"] * len(frame)
    config = {
        "columns": ["title", "label", "body"],
        "target_column": "label",
        "column_mode": "per_column",
        "max_features": 1,
    }
    state = NodeRegistry.get_calculator(node)().fit(frame, config)
    assert state["columns"] == ["title", "body"]
    assert len(state["output_columns"]) == 2
    for child in state["column_artifacts"].values():
        assert len(child["vocabulary"]) == 1
    frame[state["output_columns"][0]] = 1
    with pytest.raises(ValueError, match="collid|collision|already exist"):
        NodeRegistry.get_calculator(node)().fit(frame, config)


@pytest.mark.parametrize("node", NODES)
def test_per_column_callbacks_do_not_acquire_row_capability(node):
    """A custom saved child analyzer must retain the existing vectorizer admission boundary."""
    state = NodeRegistry.get_calculator(node)().fit(
        _train(), {"columns": ["title", "body"], "column_mode": "per_column"}
    )
    state["column_artifacts"]["title"]["vectorizer_object"].analyzer = lambda text: text.split()
    assert NodeRegistry.get_applier(node)().inference_capability(state, engine="pandas") is None


def test_per_column_tfidf_rejects_modified_saved_idf():
    """A nested estimator must retain the same IDF integrity checks as the legacy flat state."""
    state = NodeRegistry.get_calculator("tfidf_vectorizer")().fit(
        _train(), {"columns": ["title", "body"], "column_mode": "per_column"}
    )
    state["column_artifacts"]["body"]["idf"][0] += 1
    with pytest.raises(ValueError, match="IDF.*disagree"):
        NodeRegistry.get_applier("tfidf_vectorizer")().validate_inference_state(state)


@pytest.mark.parametrize("node", NODES)
def test_per_column_polars_non_string_fallback_retains_source_blocks(node):
    """Numeric source rendering must use the existing pandas fallback without losing other blocks."""
    frame = pd.DataFrame({"title": ["red blue", "blue"], "body": [123, 456]})
    state = NodeRegistry.get_calculator(node)().fit(
        frame, {"columns": ["title", "body"], "column_mode": "per_column"}
    )
    applier = NodeRegistry.get_applier(node)()
    expected = applier.apply(frame, state)
    actual = applier.apply(pl.from_pandas(frame), state)
    pd.testing.assert_frame_equal(actual.to_pandas(), expected)
