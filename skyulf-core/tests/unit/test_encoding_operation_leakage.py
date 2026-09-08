"""Semantic leakage boundaries for categorical encoding and text feature nodes."""

import sys
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.cleaning.text import TextCleaningApplier, TextCleaningCalculator
from skyulf.preprocessing.encoding.dummy import DummyEncoderApplier, DummyEncoderCalculator
from skyulf.preprocessing.encoding.hash import HashEncoderApplier, HashEncoderCalculator
from skyulf.preprocessing.encoding.label import LabelEncoderApplier, LabelEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderApplier, OrdinalEncoderCalculator
from skyulf.preprocessing.encoding.target import TargetEncoderApplier, TargetEncoderCalculator
from skyulf.preprocessing.encoding.woe import WOEEncoderApplier, WOEEncoderCalculator
from skyulf.preprocessing.vectorization import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
    HashingVectorizerApplier,
    HashingVectorizerCalculator,
    SentenceEmbedderApplier,
    SentenceEmbedderCalculator,
    TfidfVectorizerApplier,
    TfidfVectorizerCalculator,
    TokenizerApplier,
    TokenizerCalculator,
    sentence_embedder,
)

_TEXT_NODES = [
    pytest.param(CountVectorizerCalculator, CountVectorizerApplier, {}, id="count"),
    pytest.param(TfidfVectorizerCalculator, TfidfVectorizerApplier, {}, id="tfidf"),
    pytest.param(
        HashingVectorizerCalculator, HashingVectorizerApplier, {"n_features": 8}, id="hashing"
    ),
    pytest.param(
        SentenceEmbedderCalculator,
        SentenceEmbedderApplier,
        {"model_name": "leakage-audit-model"},
        id="sentence",
    ),
    pytest.param(TokenizerCalculator, TokenizerApplier, {"add_token_count": True}, id="tokenizer"),
]


class _FixedEmbeddingModel:
    """Provide deterministic pretrained inference without downloads or optional dependencies."""

    def get_embedding_dimension(self) -> int:
        """Expose a constant feature width independent of the corpus."""
        return 2

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        """Encode each row independently so changes to joined target text remain observable."""
        values = np.array([[len(text), len(text.split())] for text in texts], dtype=float)
        if options.get("normalize_embeddings"):
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1)
        return values


@pytest.fixture
def embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the embedding node's real fit/apply boundary while replacing external inference."""
    monkeypatch.setitem(
        sentence_embedder._MODEL_CACHE, "leakage-audit-model", _FixedEmbeddingModel()
    )


@pytest.mark.parametrize("calculator,applier,options", _TEXT_NODES)
@pytest.mark.parametrize("target_source", ["config", "named_y"])
@pytest.mark.parametrize("drop_original", [False, True])
def test_text_features_exclude_known_target(
    calculator: Any,
    applier: Any,
    options: dict[str, Any],
    target_source: str,
    drop_original: bool,
    embedding_model: None,
) -> None:
    """A target selected beside text must never become features that survive a target split."""
    frame = pd.DataFrame({"text": ["alpha sky", "beta sea"], "target": ["red", "blue"]})
    config = {**options, "columns": ["text", "target"], "drop_original": drop_original}
    data: Any = frame
    if target_source == "config":
        config["target_column"] = "target"
    else:
        data = frame, frame["target"]
    artifact = calculator().fit(data, config)
    output = applier().apply(data, artifact)
    result = output[0] if isinstance(output, tuple) else output
    assert "target" in result.columns
    pd.testing.assert_series_equal(result["target"], frame["target"])
    changed = frame.assign(target=["unknown long secret", "another hidden class"])
    changed_data = (changed, changed["target"]) if target_source == "named_y" else changed
    changed_output = applier().apply(changed_data, artifact)
    changed_result = changed_output[0] if isinstance(changed_output, tuple) else changed_output
    pd.testing.assert_frame_equal(
        result.drop(columns="target"), changed_result.drop(columns="target")
    )
    assert artifact["columns"] == ["text"]


@pytest.mark.parametrize("calculator,applier,options", _TEXT_NODES)
@pytest.mark.parametrize("selection", ["omitted", None, [], ["target"], ["missing"]])
def test_text_empty_or_target_only_selection_is_noop(
    calculator: Any,
    applier: Any,
    options: dict[str, Any],
    selection: Any,
    embedding_model: None,
) -> None:
    """A missing or target-only text selection cannot derive features from labels."""
    frame = pd.DataFrame({"text": ["alpha"], "target": ["secret"]})
    config = {**options, "target_column": "target"}
    if selection != "omitted":
        config["columns"] = selection
    artifact = calculator().fit(frame, config)
    pd.testing.assert_frame_equal(applier().apply(frame, artifact), frame)
    assert artifact == {}


@pytest.mark.parametrize("calculator,applier,options", _TEXT_NODES)
def test_text_replay_uses_training_artifact_and_is_batch_independent(
    calculator: Any,
    applier: Any,
    options: dict[str, Any],
    embedding_model: None,
) -> None:
    """Held-out words and labels must neither refit text state nor alter another row's features."""
    train = pd.DataFrame({"text": ["alpha beta", "alpha gamma", "beta gamma"]})
    config = {**options, "columns": ["text"], "drop_original": True}
    artifact = calculator().fit(train, config)
    original_train = applier().apply(train, artifact)
    held = pd.DataFrame({"text": ["alpha unseenword", None]})
    labels = pd.Series(["red", "blue"], name="target")
    alone, returned_labels = applier().apply((held, labels), artifact)
    together = pd.concat(
        [held, pd.DataFrame({"text": ["unrelated corpus tokens"]})], ignore_index=True
    )
    batched = applier().apply(together, artifact)
    pd.testing.assert_frame_equal(alone, batched.iloc[:2])
    pd.testing.assert_series_equal(returned_labels, labels)
    pd.testing.assert_frame_equal(applier().apply(train, artifact), original_train)
    assert not any("unseenword" in str(column) for column in alone.columns)


def test_sentence_target_only_does_not_load_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing a target-only selection must happen before optional model loading."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    artifact = SentenceEmbedderCalculator().fit(
        pd.DataFrame({"target": ["secret"]}),
        {"columns": ["target"], "target_column": "target", "model_name": "unavailable-audit-model"},
    )
    assert artifact == {}


def test_sentence_missing_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A selected real feature must fail clearly rather than silently changing embeddings."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceEmbedderCalculator().fit(
            pd.DataFrame({"text": ["alpha"]}),
            {"columns": ["text"], "model_name": "unavailable-audit-model"},
        )


@pytest.mark.parametrize("selection", ["omitted", None])
def test_ordinal_implicit_columns_really_learn_feature_categories(selection: Any) -> None:
    """Implicit columns cannot receive the target-only leakage exemption."""
    train = pd.DataFrame({"category": ["alpha", "beta"], "numeric": [1, 2]})
    config: dict[str, Any] = {}
    if selection != "omitted":
        config["columns"] = selection
    artifact = OrdinalEncoderCalculator().fit(train, config)
    held = pd.DataFrame({"category": ["gamma"], "numeric": [3]})
    encoded = OrdinalEncoderApplier().apply(held, artifact)
    assert encoded["category"].tolist() == [-1.0]
    assert artifact["columns"] == ["category"]


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (LabelEncoderCalculator, LabelEncoderApplier),
        (OrdinalEncoderCalculator, OrdinalEncoderApplier),
    ],
)
def test_explicit_empty_encoding_only_changes_target(calculator: Any, applier: Any) -> None:
    """An empty feature selection must preserve X even when categorical columns exist."""
    frame = pd.DataFrame({"category": ["alpha", "beta"]})
    target = pd.Series(["negative", "positive"], name="target")
    artifact = calculator().fit((frame, target), {"columns": []})
    result, labels = applier().apply((frame, target), artifact)
    pd.testing.assert_frame_equal(result, frame)
    assert labels.tolist() == [0, 1]


@pytest.mark.parametrize("selection", ["omitted", None, []])
def test_label_default_columns_do_not_autodetect_features(selection: Any) -> None:
    """The label encoder's default target-only behavior differs from ordinal auto-detection."""
    frame = pd.DataFrame({"category": ["alpha", "beta"]})
    labels = pd.Series(["negative", "positive"], name="target")
    config: dict[str, Any] = {}
    if selection != "omitted":
        config["columns"] = selection
    artifact = LabelEncoderCalculator().fit((frame, labels), config)
    result, encoded = LabelEncoderApplier().apply((frame, labels), artifact)
    pd.testing.assert_frame_equal(result, frame)
    assert encoded.tolist() == [0, 1]


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (LabelEncoderCalculator, LabelEncoderApplier),
        (OrdinalEncoderCalculator, OrdinalEncoderApplier),
    ],
)
def test_unseen_target_label_uses_trained_missing_code(calculator: Any, applier: Any) -> None:
    """An unseen held-out class must not extend the fitted label vocabulary."""
    features = pd.DataFrame({"value": [1, 2]})
    target = pd.Series(["negative", "positive"], name="target")
    artifact = calculator().fit((features, target), {"columns": []})
    result, encoded = applier().apply(
        (features, pd.Series(["unseen", "positive"], name="target")), artifact
    )
    pd.testing.assert_frame_equal(result, features)
    assert encoded.tolist() == [-1, 1]


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (LabelEncoderCalculator, LabelEncoderApplier),
        (OrdinalEncoderCalculator, OrdinalEncoderApplier),
        (OneHotEncoderCalculator, OneHotEncoderApplier),
        (DummyEncoderCalculator, DummyEncoderApplier),
    ],
)
@pytest.mark.parametrize("options", [{}, {"drop_first": True}])
def test_category_replay_does_not_learn_heldout_values(
    calculator: Any, applier: Any, options: dict[str, Any]
) -> None:
    """A held-out category must retain the training schema and the configured unknown encoding."""
    train = pd.DataFrame({"category": ["alpha", "beta", "alpha", "beta"]})
    artifact = calculator().fit(train, {**options, "columns": ["category"]})
    train_before = applier().apply(train, artifact)
    held = pd.DataFrame({"category": ["gamma", "alpha"]})
    alone = applier().apply(held, artifact)
    batch = pd.concat([held, pd.DataFrame({"category": ["delta"]})], ignore_index=True)
    pd.testing.assert_frame_equal(alone, applier().apply(batch, artifact).iloc[:2])
    pd.testing.assert_frame_equal(applier().apply(train, artifact), train_before)
    unknown = alone.iloc[0].to_numpy(dtype=float)
    expected = -1 if calculator in (LabelEncoderCalculator, OrdinalEncoderCalculator) else 0
    assert np.all(unknown == expected)


@pytest.mark.parametrize("selection", ["omitted", None, ["category"]])
def test_hash_column_resolution_does_not_learn_values(selection: Any) -> None:
    """Hashing a constant training category cannot depend on held-out category discovery."""
    train = pd.DataFrame({"category": ["alpha", "alpha"], "value": [1, 2]})
    other = train.assign(category=["beta", "gamma"])
    config: dict[str, Any] = {"n_features": 101}
    if selection != "omitted":
        config["columns"] = selection
    first = HashEncoderCalculator().fit(train, config)
    second = HashEncoderCalculator().fit(other, config)
    pd.testing.assert_frame_equal(
        HashEncoderApplier().apply(other, first), HashEncoderApplier().apply(other, second)
    )
    assert first == second


def test_hash_numeric_value_is_independent_of_fractional_batch_sibling() -> None:
    """An unrelated fractional row must not change the hash of an integral value."""
    alone = pd.DataFrame({"category": [1]})
    batched = pd.DataFrame({"category": [1, 2.5]})
    artifact = HashEncoderCalculator().fit(alone, {"columns": ["category"], "n_features": 65536})
    first = HashEncoderApplier().apply(alone, artifact)
    together = HashEncoderApplier().apply(batched, artifact)
    assert first["category"].iloc[0] == together["category"].iloc[0]


@pytest.mark.parametrize(
    "calculator,applier,options,expected_oof,expected_unknown",
    [
        (
            TargetEncoderCalculator,
            TargetEncoderApplier,
            {"smooth": 0, "target_type": "binary"},
            0.5,
            0.5,
        ),
        (WOEEncoderCalculator, WOEEncoderApplier, {}, 0.0, 0.0),
    ],
)
def test_supervised_encoding_training_rows_are_cross_fitted(
    calculator: Any,
    applier: Any,
    options: dict[str, Any],
    expected_oof: float,
    expected_unknown: float,
) -> None:
    """Unique training categories must not reveal their own labels through supervised encodings."""
    frame = pd.DataFrame({"category": [f"id{index}" for index in range(8)]})
    target = pd.Series([0, 1] * 4, name="target")
    artifact, (out_of_fold, returned_target) = calculator().fit_transform_train(
        (frame, target), {**options, "columns": ["category"]}
    )
    np.testing.assert_allclose(out_of_fold["category"], expected_oof)
    pd.testing.assert_series_equal(returned_target, target)
    replay = applier().apply(frame, artifact)
    assert not np.allclose(replay["category"], out_of_fold["category"])
    held = pd.DataFrame({"category": ["new", "id0"]})
    encoded, _ = applier().apply((held, pd.Series([0, 0], name="target")), artifact)
    changed, _ = applier().apply((held, pd.Series([1, 1], name="target")), artifact)
    pd.testing.assert_frame_equal(encoded, changed)
    assert encoded["category"].iloc[0] == expected_unknown


@pytest.mark.parametrize(
    "operations",
    [
        [{"op": "trim", "mode": "both"}],
        [{"op": "case", "mode": "lower"}],
        [{"op": "remove_special", "mode": "keep_alphanumeric"}],
        [{"op": "regex", "mode": "collapse_whitespace"}],
        [{"op": "regex", "mode": "normalize_slash_dates"}],
        [],
    ],
)
@pytest.mark.parametrize("selection", ["omitted", [], ["text"]])
def test_text_cleaning_is_cellwise_and_preserves_target(
    operations: list[dict[str, Any]], selection: Any
) -> None:
    """Text cleaning must not learn from sibling rows or implicitly rewrite a configured target."""
    frame = pd.DataFrame({"text": [" Alpha! 1/2/2024 "], "target": ["UPPER"]})
    config: dict[str, Any] = {"operations": operations, "target_column": "target"}
    if selection != "omitted":
        config["columns"] = selection
    artifact = TextCleaningCalculator().fit(frame, config)
    alone = TextCleaningApplier().apply(frame, artifact)
    batched = pd.concat(
        [frame, pd.DataFrame({"text": ["different"], "target": ["LOWER"]})], ignore_index=True
    )
    pd.testing.assert_frame_equal(alone, TextCleaningApplier().apply(batched, artifact).iloc[:1])
    assert alone["target"].tolist() == ["UPPER"]


@pytest.mark.parametrize("frame_type", [pd.DataFrame, pl.DataFrame])
def test_hash_legacy_artifacts_preserve_historical_numeric_buckets(frame_type: Any) -> None:
    """Loading an old model must not silently change its integral-float hash features."""
    legacy_artifact = {
        "type": "hash_encoder",
        "columns": ["category"],
        "n_features": 65536,
    }
    output = HashEncoderApplier().apply(frame_type({"category": [1.0]}), legacy_artifact)
    assert output["category"].to_list() == [51646]


@pytest.mark.parametrize("frame_type", [pd.DataFrame, pl.DataFrame])
def test_hash_new_artifacts_version_numeric_normalization(frame_type: Any) -> None:
    """New fits opt into stable numeric rendering without changing literal numeric strings."""
    integer_frame = frame_type({"category": [1]})
    artifact = HashEncoderCalculator().fit(
        integer_frame, {"columns": ["category"], "n_features": 65536}
    )
    floating_batch = frame_type({"category": [1.0, 2.5]})
    integers = HashEncoderApplier().apply(integer_frame, artifact)
    floats = HashEncoderApplier().apply(floating_batch, artifact)
    strings = HashEncoderApplier().apply(frame_type({"category": ["1.0"]}), artifact)
    assert integers["category"].to_list() == [64758]
    assert floats["category"].to_list()[0] == 64758
    assert strings["category"].to_list() == [51646]
    assert artifact["numeric_normalization_version"] == 1


@pytest.mark.parametrize("frame_type", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("version", [0, 2, None, "1", True])
def test_hash_unknown_normalization_version_is_rejected(frame_type: Any, version: Any) -> None:
    """Unsupported or malformed artifact versions must not silently choose different model inputs."""
    artifact = {
        "type": "hash_encoder",
        "columns": ["category"],
        "n_features": 65536,
        "numeric_normalization_version": version,
    }
    with pytest.raises(ValueError, match="numeric normalization version"):
        HashEncoderApplier().apply(frame_type({"category": [1.0]}), artifact)
