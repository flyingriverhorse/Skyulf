"""Unit tests for SentenceEmbedderCalculator and SentenceEmbedderApplier.

All tests mock ``_load_model`` at the module level to avoid downloading
sentence-transformer weights.  The mock model returns deterministic random
embeddings of a fixed dimension so the tests run without any network access or
optional ``sentence-transformers`` package.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Mock helper — creates a fake sentence-transformer model
# ---------------------------------------------------------------------------

_EMBED_DIM = 16  # small dimension for fast tests


def _make_mock_model(dim: int = _EMBED_DIM) -> MagicMock:
    """Return a MagicMock that mimics a SentenceTransformer model.

    Uses a fixed seed so embedding values are deterministic across calls.
    """
    rng = np.random.default_rng(42)
    model = MagicMock()
    model.get_embedding_dimension.return_value = dim
    model.get_sentence_embedding_dimension.return_value = dim
    # encode() returns (n_texts, dim) float32 array
    model.encode.side_effect = lambda texts, **kwargs: rng.random((len(texts), dim)).astype(
        np.float32
    )
    return model


_MOCK_MODEL = _make_mock_model()

_MODULE_PATH = "skyulf.preprocessing.vectorization.sentence_embedder._load_model"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def text_df() -> pd.DataFrame:
    """Small DataFrame with a single text column."""
    return pd.DataFrame(
        {
            "text": [
                "the quick brown fox",
                "hello world",
                "machine learning is fun",
                "natural language processing",
                "deep neural networks",
            ]
        }
    )


@pytest.fixture
def multi_col_df() -> pd.DataFrame:
    """DataFrame with two text columns that can be concatenated."""
    return pd.DataFrame(
        {
            "title": ["foo bar", "hello world", "test case"],
            "body": ["extra context", "some more text", "final words"],
        }
    )


# ---------------------------------------------------------------------------
# Calculator — fit()
# ---------------------------------------------------------------------------


class TestSentenceEmbedderCalculatorFit:
    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_returns_artifact_type(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """fit() must return an artifact with type='sentence_embedder'."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        assert art["type"] == "sentence_embedder"

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_stores_embedding_dim(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """fit() must store the model's embedding dimension in the artifact."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        assert art["embedding_dim"] == _EMBED_DIM

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_generates_output_columns(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Output column names must follow the '{src}__emb__{i}' naming convention."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        assert len(art["output_columns"]) == _EMBED_DIM
        assert art["output_columns"][0] == "text__emb__0"
        assert art["output_columns"][-1] == f"text__emb__{_EMBED_DIM - 1}"

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_stores_model_name(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """The configured model_name must be persisted in the artifact."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(
            text_df, {"columns": ["text"], "model_name": "custom-model-v1"}
        )
        assert art["model_name"] == "custom-model-v1"

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_stores_normalize_flag(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """The normalize flag must be persisted in the artifact."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"], "normalize": False})
        assert art["normalize"] is False

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_stores_drop_original(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """The drop_original flag must be stored in the artifact."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(
            text_df, {"columns": ["text"], "drop_original": True}
        )
        assert art["drop_original"] is True

    def test_fit_empty_columns_returns_empty_dict(self, text_df: pd.DataFrame) -> None:
        """fit() with no columns config must return an empty dict without calling the model."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": []})
        assert art == {}

    def test_fit_missing_column_returns_empty_dict(self, text_df: pd.DataFrame) -> None:
        """fit() with all-missing column names must return an empty dict."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["no_such_column"]})
        assert art == {}

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_multi_column_prefix_is_joined(
        self, _mock: Any, multi_col_df: pd.DataFrame
    ) -> None:
        """Multiple columns produce a prefix joined by '_' in output column names."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(multi_col_df, {"columns": ["title", "body"]})
        assert art["output_columns"][0].startswith("title_body__emb__")

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_polars_input(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Polars DataFrame input must be accepted and produce the same artifact as pandas."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        pl_df = pl.from_pandas(text_df)
        art = SentenceEmbedderCalculator().fit(pl_df, {"columns": ["text"]})
        assert art["embedding_dim"] == _EMBED_DIM
        assert len(art["output_columns"]) == _EMBED_DIM


# ---------------------------------------------------------------------------
# Applier — apply()
# ---------------------------------------------------------------------------


class TestSentenceEmbedderApplierApply:
    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_adds_embedding_columns(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """apply() must append embedding columns to the DataFrame."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(text_df, art)
        for col in art["output_columns"]:
            assert col in result.columns

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_preserves_row_count(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """apply() must not change the number of rows."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(text_df, art)
        assert len(result) == len(text_df)

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_embedding_dtype_is_float(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Embedding columns must have a floating-point dtype."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(text_df, art)
        for col in art["output_columns"]:
            assert np.issubdtype(result[col].dtype, np.floating), (
                f"Column {col!r} has non-float dtype: {result[col].dtype}"
            )

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_keeps_original_column_by_default(
        self, _mock: Any, text_df: pd.DataFrame
    ) -> None:
        """Source text column must be retained unless drop_original=True."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(text_df, art)
        assert "text" in result.columns

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_drop_original_removes_source_column(
        self, _mock: Any, text_df: pd.DataFrame
    ) -> None:
        """drop_original=True must remove the source text column from output."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(
            text_df, {"columns": ["text"], "drop_original": True}
        )
        result = SentenceEmbedderApplier().apply(text_df, art)
        assert "text" not in result.columns

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_output_shape(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Output DataFrame shape must be (n_rows, original_cols + embedding_dim)."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(text_df, art)
        expected_cols = len(text_df.columns) + _EMBED_DIM
        assert result.shape == (len(text_df), expected_cols)

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_empty_output_columns_returns_unchanged(
        self, _mock: Any, text_df: pd.DataFrame
    ) -> None:
        """If output_columns is empty in params, apply() returns the frame unchanged."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderApplier

        params = {
            "columns": ["text"],
            "output_columns": [],  # empty → early return path
            "model_name": "any",
            "normalize": True,
            "drop_original": False,
        }
        result = SentenceEmbedderApplier().apply(text_df, params)
        pd.testing.assert_frame_equal(result, text_df)

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_missing_column_in_params_returns_unchanged(
        self, _mock: Any, text_df: pd.DataFrame
    ) -> None:
        """If params.columns reference a missing column, apply() returns unchanged."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderApplier

        params = {
            "columns": ["nonexistent"],
            "output_columns": ["nonexistent__emb__0"],
            "model_name": "any",
            "normalize": True,
            "drop_original": False,
        }
        result = SentenceEmbedderApplier().apply(text_df, params)
        pd.testing.assert_frame_equal(result, text_df)

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_polars_input_returns_polars(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Polars DataFrame input must produce a polars DataFrame output."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        pl_df = pl.from_pandas(text_df)
        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        result = SentenceEmbedderApplier().apply(pl_df, art)
        assert isinstance(result, pl.DataFrame)

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_apply_multi_column_concatenates_text(
        self, _mock: Any, multi_col_df: pd.DataFrame
    ) -> None:
        """Applying with multiple columns must concatenate them and produce correct output shape."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(multi_col_df, {"columns": ["title", "body"]})
        result = SentenceEmbedderApplier().apply(multi_col_df, art)
        # Both source columns retained + embedding columns
        assert "title" in result.columns
        assert "body" in result.columns
        assert len([c for c in result.columns if "__emb__" in c]) == _EMBED_DIM

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_apply_normalize_false(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """normalize=False must be forwarded to model.encode without error."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"], "normalize": False})
        result = SentenceEmbedderApplier().apply(text_df, art)
        assert len(result) == len(text_df)
        # Verify encode was called with normalize_embeddings=False
        _MOCK_MODEL.encode.assert_called()
        call_kwargs = _MOCK_MODEL.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is False


# ---------------------------------------------------------------------------
# _embedding_dimension helper
# ---------------------------------------------------------------------------


class TestEmbeddingDimensionHelper:
    def test_uses_get_embedding_dimension_if_available(self) -> None:
        """_embedding_dimension must prefer the newer get_embedding_dimension API."""
        from skyulf.preprocessing.vectorization.sentence_embedder import _embedding_dimension

        model = MagicMock()
        model.get_embedding_dimension.return_value = 256
        assert _embedding_dimension(model) == 256

    def test_falls_back_to_get_sentence_embedding_dimension(self) -> None:
        """_embedding_dimension falls back to the old API when the new one is absent."""
        from skyulf.preprocessing.vectorization.sentence_embedder import _embedding_dimension

        model = MagicMock(spec=["get_sentence_embedding_dimension"])
        model.get_sentence_embedding_dimension.return_value = 384
        assert _embedding_dimension(model) == 384


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


class TestModelCache:
    @patch(_MODULE_PATH)
    def test_load_model_is_called_on_first_fit(
        self, mock_loader: Any, text_df: pd.DataFrame
    ) -> None:
        """_load_model must be called once when fitting for the first time."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        mock_loader.return_value = _make_mock_model()
        SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        mock_loader.assert_called_once()

    @patch(_MODULE_PATH)
    def test_load_model_called_per_apply(self, mock_loader: Any, text_df: pd.DataFrame) -> None:
        """_load_model is called during apply (model is looked up by name each time)."""
        from skyulf.preprocessing.vectorization.sentence_embedder import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        mock_loader.return_value = _make_mock_model()
        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        SentenceEmbedderApplier().apply(text_df, art)
        assert mock_loader.call_count >= 2  # once in fit, once in apply
