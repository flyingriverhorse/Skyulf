"""Saved text owners expose local context without refitting or duplicating transforms."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

from skyulf.core.capabilities import ExecutionCapability
from skyulf.inference.preprocessing_probe import _probe_step
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry

NODES = ("count_vectorizer", "hashing_vectorizer", "tfidf_vectorizer", "tokenizer")


def _frame(engine, values=("red blue", "blue green", "red red", None)):
    """Retain null text and row identity in both supported native frames."""
    frame = pd.DataFrame({"text": values, "keep": list(range(len(values)))})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _record(node, engine, config=None, frame=None):
    """Build the genuine fitted record consumed by inference diagnostics."""
    config = {"columns": ["text"], "n_features": 8} if config is None else config
    frame = _frame(engine) if frame is None else frame
    return {
        "name": "text",
        "type": node,
        "params": config,
        "artifact": NodeRegistry.get_calculator(node)().fit(frame, config),
        "applier": NodeRegistry.get_applier(node)(),
    }


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_text_owner_inspects_without_execution(node, engine, monkeypatch):
    """A context query must inspect actual saved state without applying or fitting text."""
    record = _record(node, engine)
    state = record["artifact"]
    before = artifact_digest(state)
    applier: Any = NodeRegistry.get_applier(node)

    def forbidden(*args, **kwargs):
        """Metadata inspection may not execute a calculator or transform."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(NodeRegistry.get_calculator(node), "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    if "vectorizer_object" in state:
        monkeypatch.setattr(type(state["vectorizer_object"]), "transform", forbidden)
    assert get_inference_capability(node, record["params"], state, engine=engine) == (
        ExecutionCapability(engine, "apply", "local", "preserve", "row")
    )
    assert applier.validate_inference_state(state) is state
    assert artifact_digest(state) == before


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_text_only_prediction_keeps_generated_rows_when_original_is_dropped(node, engine):
    """Scoring text without the training target must preserve rows after dropping its only source."""
    training = pd.DataFrame({"text": ["red blue", "blue green", "red red"], "target": [1, 2, 3]})
    sample = pd.DataFrame({"text": ["red blue", None, "unseen"]})
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    config = {
        "columns": ["text"],
        "target_column": "target",
        "drop_original": True,
        "n_features": 8,
        "add_token_count": True,
    }
    record = _record(node, engine, config, training)
    output = record["applier"].apply(sample, record["artifact"])
    empty = record["applier"].apply(sample.head(0), record["artifact"])
    assert len(output) == 3
    assert list(output.columns) == record["artifact"]["output_columns"]
    assert len(empty) == 0
    assert list(empty.columns) == list(output.columns)


def _probe(record, frame, engine, monkeypatch):
    """Exercise native full, chunk, reverse, empty, schema and state checks without fitting."""
    before = artifact_digest(record)

    def forbidden(*args, **kwargs):
        """Inference must retain learned vocabulary and corpus weights."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(NodeRegistry.get_calculator(record["type"]), "fit", forbidden)
    for estimator in (CountVectorizer, TfidfVectorizer, HashingVectorizer):
        monkeypatch.setattr(estimator, "fit", forbidden)
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


def _assert_native_checks(detail, node, *, cached_stop_words=False, nullable_polars=False):
    """Keep exact diagnostics for native cache mutation and nullable boolean rendering."""
    failures = [check for check in detail["checks"] if check["status"] != "passed"]
    if cached_stop_words:
        assert detail["checks"] == [
            {"name": "full", "status": "failed", "reason": "state_mutation"}
        ], detail
    elif nullable_polars:
        assert [(check["name"], check["reason"]) for check in failures] == [
            ("chunks:1", "output_mismatch"),
            ("chunks:3", "output_mismatch"),
        ], detail
    else:
        assert detail["status"] == "passed", detail


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("add_token_count", [False, True])
@pytest.mark.parametrize("drop_original", [False, True])
def test_tokenizer_empty_outputs_keep_populated_schema(engine, add_token_count, drop_original):
    """Empty requests retain string/count types without changing nulls, targets or row identity."""
    original = pd.DataFrame({"text": ["Red blue", None, ""], "target": [2, 4, 6]}, index=[8, 2, 2])
    frame = pl.from_pandas(original) if engine == "polars" else original.copy(deep=True)
    record = _record(
        "tokenizer",
        engine,
        {"columns": ["text"], "add_token_count": add_token_count, "drop_original": drop_original},
        frame,
    )
    output = record["applier"].apply(frame, record["artifact"])
    empty = record["applier"].apply(frame.head(0), record["artifact"])
    assert output["text__tokens"].to_list() == ["red blue", "", ""]
    if add_token_count:
        assert output["text__token_count"].to_list() == [2, 0, 0]
    if isinstance(frame, pl.DataFrame):
        assert empty.schema == output.schema
        assert output.schema["text__tokens"] == pl.String
        assert not add_token_count or output.schema["text__token_count"] == pl.Int64
        assert frame.equals(pl.from_pandas(original))
    else:
        pd.testing.assert_frame_equal(empty, output.head(0))
        assert output["text__tokens"].dtype == object
        assert not add_token_count or output["text__token_count"].dtype == np.dtype("int64")
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_index_equal(output.index, original.index)
    assert output["target"].to_list() == [2, 4, 6]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "node,options",
    [
        ("count_vectorizer", {}),
        ("count_vectorizer", {"binary": True, "lowercase": False, "ngram_range": (1, 2)}),
        (
            "count_vectorizer",
            {"max_features": 2, "min_df": 1, "max_df": 0.9, "stop_words": ["green"]},
        ),
        ("tfidf_vectorizer", {}),
        (
            "tfidf_vectorizer",
            {
                "sublinear_tf": True,
                "ngram_range": (np.int64(1), np.int64(2)),
                "stop_words": "english",
            },
        ),
        ("tfidf_vectorizer", {"max_features": 2, "lowercase": False}),
        ("hashing_vectorizer", {"norm": None, "alternate_sign": False}),
        ("hashing_vectorizer", {"norm": "none", "alternate_sign": True}),
        ("hashing_vectorizer", {"norm": "l1", "stop_words": ["green"]}),
        ("hashing_vectorizer", {"norm": "l2", "lowercase": False}),
        ("tokenizer", {"analyzer": "word", "add_token_count": False}),
        ("tokenizer", {"analyzer": "word", "add_token_count": True, "stop_words": ["green"]}),
        ("tokenizer", {"analyzer": "char", "ngram_range": (1, 2), "add_token_count": True}),
        ("tokenizer", {"analyzer": "char_wb", "ngram_range": (2, 3), "lowercase": np.bool_(False)}),
    ],
)
def test_text_modes_replay_saved_outputs_without_refitting(node, options, engine, monkeypatch):
    """Public analyzer, vocabulary and normalization modes keep native exact parity evidence."""
    config = {
        "columns": (np.str_("text"),),
        "n_features": np.int64(8),
        "drop_original": np.bool_(True),
        **options,
    }
    record = _record(node, engine, config)
    frame = _frame(engine, ["Red red unseen", None, "blue green", ""])
    if node == "hashing_vectorizer":
        record["applier"].apply(frame, record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    _assert_native_checks(
        detail,
        node,
        cached_stop_words=isinstance(options.get("stop_words"), list) and node != "tokenizer",
    )
    assert len(output) == len(frame)
    native_output = record["applier"].apply(frame, record["artifact"])
    assert list(native_output.columns) == ["keep", *record["artifact"]["output_columns"]]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize(
    "values",
    [
        [12.0, 12.5, 123.0, None],
        [True, False, True, None],
        list(
            pd.to_datetime(
                pd.Series(["2024-01-01", "2024-01-02 12:00:00", "2024-01-03", None]), format="mixed"
            )
        ),
    ],
)
def test_typed_text_fallback_is_independent_of_neighboring_rows(node, values, engine, monkeypatch):
    """Object-first rendering must retain timestamp hours and numeric values in singleton chunks."""
    frame = _frame(engine, values)
    record = _record(node, engine, frame=frame)
    if node == "hashing_vectorizer":
        record["applier"].apply(frame, record["artifact"])
    detail, output = _probe(record, frame, engine, monkeypatch)
    _assert_native_checks(
        detail, node, nullable_polars=engine == "polars" and isinstance(values[0], bool)
    )
    assert len(output) == len(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("config", [{}, {"columns": []}, {"columns": ["absent"]}])
def test_text_default_and_unselected_artifacts_are_actual_noops(node, config, engine, monkeypatch):
    """Genuine empty artifacts must preserve input and stay inspectable."""
    record = _record(node, engine, config)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert record["artifact"] == {}
    assert detail["status"] == "passed", detail
    assert output.equals(_frame(engine))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pristine_hash_vectorizer_reports_native_lazy_cache_mutation(engine, monkeypatch):
    """The first native analyzer cache write must remain visible to exact diagnostics."""
    record = _record("hashing_vectorizer", engine)
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["checks"] == [{"name": "full", "status": "failed", "reason": "state_mutation"}]
    assert len(output) == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_text_missing_inference_sources_remain_native_noops(node, engine, monkeypatch):
    """The saved selection is never relearned when all source columns disappear."""
    record = _record(node, engine)
    frame = (
        _frame(engine).drop("text") if engine == "polars" else _frame(engine).drop(columns="text")
    )
    detail, output = _probe(record, frame, engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output.equals(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["count_vectorizer", "tfidf_vectorizer"])
def test_empty_training_vocabulary_preserves_actual_noop(node, engine, monkeypatch):
    """All-null training has no vocabulary and must not fabricate fitted features."""
    record = _record(node, engine, frame=_frame(engine, [None, "", "a"]))
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert record["artifact"] == {}
    assert detail["status"] == "passed", detail
    assert output.equals(_frame(engine))


@pytest.mark.parametrize("node", NODES)
@pytest.mark.parametrize("change", ["type", "missing", "extra", "collision", "names", "columns"])
def test_malformed_text_artifacts_fail_pure_owned_inspection(node, change):
    """Wrong layouts and incomplete state cannot acquire plausible context metadata."""
    record = _record(node, "pandas")
    state = record["artifact"]
    if change == "type":
        state["type"] = "different"
    elif change == "missing":
        del state["output_columns"]
    elif change == "extra":
        state["unexpected"] = True
    elif change == "collision":
        state["output_columns"][0] = "text"
    elif change == "names":
        state["output_columns"] = "output"
    else:
        state["columns"] = [object()]
    applier: Any = type(record["applier"])
    with pytest.raises(ValueError):
        applier.validate_inference_state(state)


@pytest.mark.parametrize(
    "node,field,value",
    [
        ("count_vectorizer", "binary", "yes"),
        ("count_vectorizer", "vocabulary", {"red": 2}),
        ("tfidf_vectorizer", "idf", [float("nan")]),
        ("tfidf_vectorizer", "vocabulary", {"red": 0}),
        ("hashing_vectorizer", "n_features", 9),
        ("hashing_vectorizer", "norm", "l1"),
        ("tokenizer", "analyzer", "invalid"),
        ("tokenizer", "ngram_range", [1]),
        ("tokenizer", "add_token_count", "yes"),
    ],
)
def test_learned_widths_and_analyzer_settings_are_checked(node, field, value):
    """Inconsistent duplicated parameters and learned dimensions are rejected before probing."""
    record = _record(node, "pandas")
    state = record["artifact"]
    state[field] = value
    applier: Any = type(record["applier"])
    with pytest.raises(ValueError):
        applier.validate_inference_state(state)


def _split_tokens(text):
    """Represent an actual supported arbitrary analyzer without external dependencies."""
    return text.split()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_custom_analyzers_are_usable_but_context_is_unknown(node, engine, monkeypatch):
    """Native callbacks remain usable without claiming their dependency or purity contract."""
    config = {"columns": ["text"], "n_features": 8, "analyzer": _split_tokens}
    record = _record(node, engine, config)
    state = record["artifact"]
    if node != "tokenizer":
        state["vectorizer_object"].analyzer = _split_tokens
    assert len(record["applier"].apply(_frame(engine), state)) == 4
    applier: Any = type(record["applier"])

    def forbidden(*args, **kwargs):
        """Capability inspection must not execute the callback or build an analyzer."""
        raise AssertionError("Unexpected analyzer construction")

    monkeypatch.setattr(CountVectorizer, "build_analyzer", forbidden)
    assert applier.validate_inference_state(state) is state
    assert applier.inference_capability(state, engine=engine) is None


@pytest.mark.parametrize("node", NODES)
def test_text_hooks_do_not_admit_other_engines_or_worker_contracts(node):
    """Local declarations must not silently enable Spark worker execution."""
    applier: Any = NodeRegistry.get_applier(node)
    assert applier.inference_capability({}, engine="spark") is None
    assert "__execution_capabilities__" not in applier.__dict__
    assert "validate_fitted_state" not in applier.__dict__


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("width", [0, -1])
def test_hash_zero_output_artifacts_retain_existing_noop(engine, width, monkeypatch):
    """Native zero-width artifacts bypass hashing and must preserve that saved behavior."""
    record = _record("hashing_vectorizer", engine, {"columns": ["text"], "n_features": width})
    detail, output = _probe(record, _frame(engine), engine, monkeypatch)
    assert detail["status"] == "passed", detail
    assert output.equals(_frame(engine))


def test_wide_hash_context_does_not_allocate_dense_output(monkeypatch, caplog):
    """Local context is not a dense-memory guarantee and inspection must not allocate output."""
    record = _record("hashing_vectorizer", "pandas", {"columns": ["text"], "n_features": 10_001})

    def forbidden(*args, **kwargs):
        """Metadata inspection may not materialize the wide matrix."""
        raise AssertionError("Unexpected transform")

    monkeypatch.setattr(HashingVectorizer, "transform", forbidden)
    assert "10,001 output columns" in caplog.text
    assert (
        record["applier"].inference_capability(record["artifact"], engine="pandas").context == "row"
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", NODES)
def test_multiple_selected_columns_keep_saved_join_and_partial_missing_semantics(node, engine):
    """Partial scoring input must retain native source joining without inventing missing text."""
    training = pd.DataFrame({"text": ["red blue", "blue green"], "second": ["apple", "pear"]})
    training = pl.from_pandas(training) if engine == "polars" else training
    record = _record(node, engine, {"columns": ["text", "second"], "n_features": 8}, training)
    sample = _frame(engine, ["red apple", None, "blue pear"])
    state = record["artifact"]
    output = record["applier"].apply(sample, state)
    assert record["applier"].validate_inference_state(state) is state
    assert len(output) == 3
    if node == "tokenizer":
        assert "text__tokens" in output.columns and "second__tokens" not in output.columns
    else:
        assert all(name in output.columns for name in state["output_columns"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "option",
    [
        {"lowercase": None},
        {"drop_original": None},
        {"add_token_count": None},
        {"stop_words": np.array(["green"])},
        {"stop_words": ("green",)},
    ],
)
def test_tokenizer_retains_native_nullable_and_numpy_options(engine, option):
    """State inspection must accept values already supported by the genuine saved analyzer."""
    record = _record("tokenizer", engine, {"columns": ["text"], **option})
    state = record["artifact"]
    output = record["applier"].apply(_frame(engine), state)
    assert len(output) == 4
    assert record["applier"].validate_inference_state(state) is state
    assert record["applier"].inference_capability(state, engine=engine).context == "row"


@pytest.mark.parametrize("node", ["count_vectorizer", "hashing_vectorizer", "tfidf_vectorizer"])
def test_overridden_native_transform_has_unknown_context(node):
    """An estimator instance override may consume its entire input batch."""
    record = _record(node, "pandas")
    state = record["artifact"]

    def batch_transform(documents):
        """Return a valid matrix whose values depend on other documents in the request."""
        return np.full((len(documents), len(state["output_columns"])), len(documents))

    state["vectorizer_object"].transform = batch_transform
    full = record["applier"].apply(_frame("pandas"), state)
    singleton = record["applier"].apply(_frame("pandas").head(1), state)
    assert full[state["output_columns"][0]].iloc[0] != singleton[state["output_columns"][0]].iloc[0]
    assert record["applier"].inference_capability(state, engine="pandas") is None
