"""F-09: ``SkyulfPolarsWrapper`` is a documented public input type.

Passing a wrapped Polars frame used to crash OneHotEncoder,
PolynomialFeatures and the text vectorizers, while the same nodes handled a
wrapped *pandas* frame fine — an asymmetry, not an inherent limitation. Each
node must accept the wrapper for fit and apply and produce the same values as
the raw Polars path.
"""

import sys
from typing import Any

import numpy as np
import polars as pl
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.preprocessing.encoding.one_hot import (
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
)
from skyulf.preprocessing.feature_generation.polynomial import (
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.vectorization.count_vectorizer import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
)
from skyulf.preprocessing.vectorization.hashing_vectorizer import (
    HashingVectorizerApplier,
    HashingVectorizerCalculator,
)
from skyulf.preprocessing.vectorization.tfidf_vectorizer import (
    TfidfVectorizerApplier,
    TfidfVectorizerCalculator,
)


def _unwrap(frame: Any) -> pl.DataFrame:
    return frame._df if isinstance(frame, SkyulfPolarsWrapper) else frame


def _fit_apply_roundtrip(calculator: Any, applier: Any, frame: Any, config: dict[str, Any]) -> Any:
    params = calculator.fit(frame, dict(config))
    return applier.apply(frame, dict(params))


def test_one_hot_accepts_wrapped_polars_frame() -> None:
    raw = pl.DataFrame({"color": ["red", "blue", "red", "green"]})
    config = {"columns": ["color"]}

    expected = _unwrap(
        _fit_apply_roundtrip(OneHotEncoderCalculator(), OneHotEncoderApplier(), raw, config)
    )
    actual = _unwrap(
        _fit_apply_roundtrip(
            OneHotEncoderCalculator(), OneHotEncoderApplier(), SkyulfPolarsWrapper(raw), config
        )
    )
    assert actual.equals(expected)


def test_polynomial_features_accepts_wrapped_polars_frame() -> None:
    raw = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    config = {"columns": ["a", "b"], "degree": 2}

    expected = _unwrap(
        _fit_apply_roundtrip(
            PolynomialFeaturesCalculator(), PolynomialFeaturesApplier(), raw, config
        )
    )
    actual = _unwrap(
        _fit_apply_roundtrip(
            PolynomialFeaturesCalculator(),
            PolynomialFeaturesApplier(),
            SkyulfPolarsWrapper(raw),
            config,
        )
    )
    assert actual.equals(expected)


@pytest.mark.parametrize(
    ("calculator_cls", "applier_cls"),
    [
        (CountVectorizerCalculator, CountVectorizerApplier),
        (TfidfVectorizerCalculator, TfidfVectorizerApplier),
        (HashingVectorizerCalculator, HashingVectorizerApplier),
    ],
)
def test_vectorizers_accept_wrapped_polars_frame(calculator_cls: type, applier_cls: type) -> None:
    raw = pl.DataFrame({"text": ["the cat sat", "the dog ran", "a cat ran"]})
    config = {"columns": ["text"]}

    expected = _unwrap(_fit_apply_roundtrip(calculator_cls(), applier_cls(), raw, config))
    actual = _unwrap(
        _fit_apply_roundtrip(calculator_cls(), applier_cls(), SkyulfPolarsWrapper(raw), config)
    )
    assert actual.equals(expected)


class _SentenceEncoder:
    """Produce small text-dependent embeddings without model files or network access."""

    def get_sentence_embedding_dimension(self) -> int:
        """Declare the two columns emitted by the deterministic encoder."""
        return 2

    def encode(self, texts, *, normalize_embeddings, show_progress_bar):
        """Use text length and letter count so row and text changes affect the output."""
        assert show_progress_bar is False
        values = np.array([[len(text), text.count("a")] for text in texts], dtype=np.float32)
        if normalize_embeddings:
            values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values


@pytest.fixture
def sentence_encoder(monkeypatch):
    """Mock model loading and packaging so wrapper parity needs no optional NLP imports."""
    from skyulf.preprocessing.vectorization import sentence_embedder

    encoder = _SentenceEncoder()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(sentence_embedder, "_load_model", lambda model_name: encoder)
    monkeypatch.setattr(sentence_embedder, "_snapshot_model", lambda model: {})
    return encoder


def test_sentence_embedder_accepts_wrapped_polars_frame(sentence_encoder) -> None:
    """Wrapped input must retain the same text-derived embedding values as native Polars."""
    from skyulf.preprocessing.vectorization.sentence_embedder import (
        SentenceEmbedderApplier,
        SentenceEmbedderCalculator,
    )

    raw = pl.DataFrame({"text": ["the cat sat", "the dog ran"]})
    config = {"columns": ["text"]}

    expected = _unwrap(
        _fit_apply_roundtrip(SentenceEmbedderCalculator(), SentenceEmbedderApplier(), raw, config)
    )
    actual = _unwrap(
        _fit_apply_roundtrip(
            SentenceEmbedderCalculator(),
            SentenceEmbedderApplier(),
            SkyulfPolarsWrapper(raw),
            config,
        )
    )
    assert actual.equals(expected)
    assert actual["text"].to_list() == raw["text"].to_list()
    values = np.array([[11, 2], [11, 1]], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    np.testing.assert_allclose(actual.select("text__emb__0", "text__emb__1").to_numpy(), values)
