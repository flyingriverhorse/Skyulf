"""Empty sentence batches must retain the fitted output schema without loading weights."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.preprocessing.vectorization import sentence_embedder


class OfflineEncoder:
    """Mirror the real encoder's one-dimensional empty output without downloads."""

    def get_embedding_dimension(self):
        """Keep the artifact width independent of optional model packages."""
        return 3

    def encode(self, texts, **kwargs):
        """Sentence-transformers returns shape (0,) for an empty sequence."""
        return np.array([[len(text), 1, 2] for text in texts], dtype=np.float32)


@pytest.fixture(params=["pandas", "polars", "polars-wrapper"])
def text_frame(request, monkeypatch):
    """Exercise native/wrapped frame APIs with model loading and asset packaging mocked."""
    monkeypatch.setenv("SKYULF_ENGINE", request.param.split("-")[0])
    monkeypatch.setattr(sentence_embedder, "_load_model", lambda name: OfflineEncoder())
    monkeypatch.setattr(sentence_embedder, "_snapshot_model", lambda model: {})
    data = {"text": ["hello", "world"], "keep": [1, 2]}
    if request.param == "pandas":
        return pd.DataFrame(data, index=pd.Index([20, 10], name="row"))
    frame = pl.DataFrame(data)
    return SkyulfPolarsWrapper(frame) if request.param == "polars-wrapper" else frame


def _native(frame):
    """Expose output values without discarding wrapper assertions at the public boundary."""
    return frame.to_native() if isinstance(frame, SkyulfPolarsWrapper) else frame


def _empty(frame):
    """Keep the original native schema and wrapper when selecting zero rows."""
    empty = _native(frame)[:0]
    return SkyulfPolarsWrapper(empty) if isinstance(frame, SkyulfPolarsWrapper) else empty


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("drop_original", [False, True])
def test_embedding_batch_keeps_fitted_schema(text_frame, empty, drop_original):
    """Empty inference must preserve replayed columns, source policy, index and tuple target."""
    params = sentence_embedder.SentenceEmbedderCalculator().fit(
        text_frame, {"columns": ["text"], "drop_original": drop_original}
    )
    params = pickle.loads(pickle.dumps(params))
    batch = _empty(text_frame) if empty else text_frame
    target = pd.Series([], dtype=float) if empty else pd.Series([0, 1])

    result, output_y = sentence_embedder.SentenceEmbedderApplier().apply((batch, target), params)

    assert type(result) is type(batch)
    output = _native(result)
    expected_columns = (["keep"] if drop_original else ["text", "keep"]) + params["output_columns"]
    assert list(output.columns) == expected_columns
    assert output.shape == (0 if empty else 2, len(expected_columns))
    assert output_y is target
    assert all(output[name].to_numpy().dtype == np.float32 for name in params["output_columns"])
    if isinstance(batch, pd.DataFrame):
        pd.testing.assert_index_equal(output.index, batch.index)
    if not empty:
        np.testing.assert_array_equal(
            output[params["output_columns"]].to_numpy(), [[5, 1, 2], [5, 1, 2]]
        )
    assert list(text_frame.columns) == ["text", "keep"]


def test_empty_embedding_does_not_load_model(text_frame, monkeypatch):
    """An empty scoring partition needs only saved output names, including legacy artifacts."""
    params = sentence_embedder.SentenceEmbedderCalculator().fit(text_frame, {"columns": ["text"]})
    params.pop("embedding_dim")

    def unavailable_model(name):
        """Fail if empty scoring attempts to reload optional pretrained weights."""
        raise ImportError("optional model unavailable")

    monkeypatch.setattr(sentence_embedder, "_load_model", unavailable_model)

    output = _native(sentence_embedder.SentenceEmbedderApplier().apply(_empty(text_frame), params))

    assert output.shape == (0, 5)
    assert list(output.columns)[-3:] == params["output_columns"]


def test_empty_embedding_still_rejects_output_collisions(text_frame):
    """A zero-row fast path must retain the normal generated-column collision guard."""
    params = sentence_embedder.SentenceEmbedderCalculator().fit(text_frame, {"columns": ["text"]})
    params["output_columns"] = ["keep", "second", "third"]

    with pytest.raises(ValueError, match="(?i)(collid|collision|already|duplicate)"):
        sentence_embedder.SentenceEmbedderApplier().apply(_empty(text_frame), params)


def test_nonempty_embedding_still_propagates_model_errors(text_frame, monkeypatch):
    """Handling empty batches must not suppress encoder failures for real text."""
    params = sentence_embedder.SentenceEmbedderCalculator().fit(text_frame, {"columns": ["text"]})

    def unavailable_model(name):
        """Mirror an optional-model loading failure for a nonempty batch."""
        raise ImportError("optional model unavailable")

    monkeypatch.setattr(sentence_embedder, "_load_model", unavailable_model)

    with pytest.raises(ImportError, match="optional model unavailable"):
        sentence_embedder.SentenceEmbedderApplier().apply(text_frame, params)
