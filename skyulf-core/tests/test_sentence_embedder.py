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
from tests.utils.test_case_loader import TestCaseLoader

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

_fit_artifact_field_cases = TestCaseLoader(
    "preprocessing/sentence_embedder_fit_artifact_fields"
).load()
_fit_empty_cases = TestCaseLoader("preprocessing/sentence_embedder_fit_empty").load()
_apply_noop_cases = TestCaseLoader("preprocessing/sentence_embedder_apply_noop").load()
_dimension_helper_cases = TestCaseLoader("preprocessing/sentence_embedder_dimension_helper").load()

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
    @pytest.mark.parametrize(*_fit_artifact_field_cases)
    def test_fit_artifact_field(
        self,
        _mock: Any,
        text_df: pd.DataFrame,
        extra_config: dict[str, Any],
        artifact_key: str,
        expected: Any,
    ) -> None:
        """fit() must persist the expected artifact field for each config scenario."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"], **extra_config})
        assert dict(art)[artifact_key] == expected

    @patch(_MODULE_PATH, return_value=_MOCK_MODEL)
    def test_fit_generates_output_columns(self, _mock: Any, text_df: pd.DataFrame) -> None:
        """Output column names must follow the '{src}__emb__{i}' naming convention."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": ["text"]})
        assert len(art["output_columns"]) == _EMBED_DIM
        assert art["output_columns"][0] == "text__emb__0"
        assert art["output_columns"][-1] == f"text__emb__{_EMBED_DIM - 1}"

    @pytest.mark.parametrize(*_fit_empty_cases)
    def test_fit_returns_empty_dict(
        self, text_df: pd.DataFrame, columns_config: list[str], expected: dict[str, Any]
    ) -> None:
        """fit() with no resolvable columns must return an empty dict without calling the model."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderCalculator

        art = SentenceEmbedderCalculator().fit(text_df, {"columns": columns_config})
        assert art == expected

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
    @pytest.mark.parametrize(*_apply_noop_cases)
    def test_apply_returns_unchanged(
        self, _mock: Any, text_df: pd.DataFrame, params: dict[str, Any], expect_unchanged: bool
    ) -> None:
        """apply() with no resolvable output must return the frame unchanged."""
        from skyulf.preprocessing.vectorization.sentence_embedder import SentenceEmbedderApplier

        result = SentenceEmbedderApplier().apply(text_df, params)
        assert expect_unchanged
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
    @pytest.mark.parametrize(*_dimension_helper_cases)
    def test_embedding_dimension(self, spec: list[str] | None, attr: str, dim: int) -> None:
        """_embedding_dimension must prefer the newer API and fall back to the older one."""
        from skyulf.preprocessing.vectorization.sentence_embedder import _embedding_dimension

        model = MagicMock(spec=spec)
        getattr(model, attr).return_value = dim
        assert _embedding_dimension(model) == dim


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
