"""Independent encoding/text probes beyond the known numeric-category replay defect."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.preprocessing.cleaning.text import TextCleaningApplier, TextCleaningCalculator
from skyulf.preprocessing.encoding.hash import HashEncoderApplier, HashEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from skyulf.preprocessing.encoding.woe import WOEEncoderCalculator
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


class _AuditEmbeddingModel:
    """Supply only external pretrained inference while exercising the real node boundaries."""

    def get_embedding_dimension(self) -> int:
        """Keep output width independent of the fitted or held-out corpus."""
        return 2

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        """Derive deterministic row-local features without downloading a model."""
        output = np.asarray([[len(text), len(text.split())] for text in texts], dtype=float)
        if options.get("normalize_embeddings"):
            output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1)
        return output


@pytest.fixture
def fixed_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent optional dependency loading while keeping text normalization and replay real."""
    monkeypatch.setitem(
        sentence_embedder._MODEL_CACHE, "deep-audit-20260908", _AuditEmbeddingModel()
    )


_TEXT_NODES = [
    pytest.param(CountVectorizerCalculator, CountVectorizerApplier, {}, id="count"),
    pytest.param(TfidfVectorizerCalculator, TfidfVectorizerApplier, {}, id="tfidf"),
    pytest.param(
        HashingVectorizerCalculator, HashingVectorizerApplier, {"n_features": 16}, id="hashing"
    ),
    pytest.param(TokenizerCalculator, TokenizerApplier, {"add_token_count": True}, id="tokenizer"),
    pytest.param(
        SentenceEmbedderCalculator,
        SentenceEmbedderApplier,
        {"model_name": "deep-audit-20260908"},
        id="sentence",
    ),
]


@pytest.mark.parametrize("calculator,applier,options", _TEXT_NODES)
def test_fitted_text_node_accepts_equivalent_nullable_categorical_text(
    calculator: Any,
    applier: Any,
    options: dict[str, Any],
    fixed_embedding_model: None,
) -> None:
    """Casting the same held-out text to category must not break fitted-artifact replay."""
    train = pd.DataFrame({"text": ["alpha beta", "beta gamma", "alpha gamma"]})
    artifact = calculator().fit(train, {**options, "columns": ["text"], "drop_original": True})
    held = pd.DataFrame({"text": ["alpha", None, "unseenword"]})
    expected = applier().apply(held, artifact)
    categorical = held.astype({"text": "category"})

    actual = applier().apply(categorical, artifact)

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("calculator", [CountVectorizerCalculator, TfidfVectorizerCalculator])
def test_vocabulary_fit_accepts_nullable_categorical_text(calculator: Any) -> None:
    """Categorical text with missing values must fit the same observed training vocabulary."""
    train = pd.DataFrame({"text": pd.Series(["alpha", None, "beta"], dtype="category")})

    artifact = calculator().fit(train, {"columns": ["text"]})

    assert set(artifact["vocabulary"]) == {"alpha", "beta"}


def test_onehot_include_missing_fits_categorical_dtype() -> None:
    """Explicit missing-category encoding must accept pandas categorical source columns."""
    train = pd.DataFrame({"category": pd.Series(["alpha", None, "beta"], dtype="category")})

    artifact = OneHotEncoderCalculator().fit(
        train, {"columns": ["category"], "include_missing": True}
    )
    encoded = OneHotEncoderApplier().apply(train, artifact)

    np.testing.assert_array_equal(encoded.sum(axis=1).to_numpy(), [1, 1, 1])


def test_onehot_include_missing_replays_categorical_dtype() -> None:
    """A previously fitted missing category must remain usable after a dtype-only change."""
    train = pd.DataFrame({"category": ["alpha", None, "beta"]})
    artifact = OneHotEncoderCalculator().fit(
        train, {"columns": ["category"], "include_missing": True}
    )
    expected = OneHotEncoderApplier().apply(train, artifact)

    encoded = OneHotEncoderApplier().apply(train.astype({"category": "category"}), artifact)

    pd.testing.assert_frame_equal(encoded, expected)


@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
def test_woe_does_not_silently_turn_missing_targets_into_negative_labels(method: str) -> None:
    """Missing ground truth must not silently contribute fabricated negative-class counts."""
    frame = pd.DataFrame({"category": ["a", "b", "a", "b", "a", "a"]})
    target = pd.Series([0, 1, 0, 1, np.nan, np.nan], name="target")

    with pytest.raises(ValueError):
        getattr(WOEEncoderCalculator(), method)((frame, target), {"columns": ["category"]})


def test_hash_missing_value_is_independent_of_numeric_batch_sibling() -> None:
    """An unrelated numeric row must not change the bucket assigned to the same missing value."""
    artifact = HashEncoderCalculator().fit(
        pd.DataFrame({"category": ["known"]}),
        {"columns": ["category"], "n_features": 65536},
    )
    alone = HashEncoderApplier().apply(pd.DataFrame({"category": [None]}), artifact)
    batched = HashEncoderApplier().apply(pd.DataFrame({"category": [None, 2.5]}), artifact)

    assert alone.loc[0, "category"] == batched.loc[0, "category"]


def test_text_cleaning_preserves_nullable_categorical_rows() -> None:
    """Fixed cleaning provides a control for the same categorical text representation."""
    train = pd.DataFrame({"text": [" ALPHA ", None, " BETA "]})
    artifact = TextCleaningCalculator().fit(
        train,
        {"columns": ["text"], "operations": [{"op": "trim"}, {"op": "case", "mode": "lower"}]},
    )

    encoded = TextCleaningApplier().apply(train.astype({"text": "category"}), artifact)

    assert encoded["text"].dropna().tolist() == ["alpha", "beta"]
    assert encoded["text"].isna().tolist() == [False, True, False]
