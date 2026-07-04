"""Coverage-gap tests for vectorization: hashing, tfidf, count, tokenizer, _common.

Complements ``test_text_vectorization.py`` by covering edge cases: empty
corpora, single-character/very short strings, drop_original interplay, the
"norm=None" hashing path, and the shared ``_common.py`` helpers directly.
"""

from typing import Any

import pandas as pd
import polars as pl

from skyulf.preprocessing.vectorization._common import (
    _join_text_columns,
    _warn_large_output,
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
from skyulf.preprocessing.vectorization.tokenizer import TokenizerApplier, TokenizerCalculator

SHORT_CORPUS = ["a", "b b", ""]


# ---------------------------------------------------------------------------
# _common.py
# ---------------------------------------------------------------------------


def test_join_text_columns_single_column() -> None:
    """A single text column is returned as-is (NaN filled to empty string)."""
    df = pd.DataFrame({"x": ["hello", None]})
    joined = _join_text_columns(df, ["x"])
    assert joined.tolist() == ["hello", ""]


def test_join_text_columns_multiple_columns_joined_with_space() -> None:
    """Multiple text columns are concatenated with a single space separator."""
    df = pd.DataFrame({"x": ["hello"], "y": ["world"]})
    joined = _join_text_columns(df, ["x", "y"])
    assert joined.tolist() == ["hello world"]


def test_warn_large_output_below_threshold_returns_none() -> None:
    """Output counts at/below the threshold produce no warning."""
    assert _warn_large_output(100, threshold=10_000) is None


def test_warn_large_output_above_threshold_returns_message() -> None:
    """Exceeding the threshold returns a human-readable warning string."""
    msg = _warn_large_output(20_000, threshold=10_000)
    assert msg is not None
    assert "20,000" in msg


# ---------------------------------------------------------------------------
# HashingVectorizer edge cases
# ---------------------------------------------------------------------------


def test_hashing_vectorizer_empty_columns_config_returns_empty_artifact() -> None:
    """No configured columns yields an empty artifact (no-op)."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = HashingVectorizerCalculator().fit(df, {"columns": []})
    assert art == {}


def test_hashing_vectorizer_missing_column_returns_empty_artifact() -> None:
    """Configuring a column absent from the frame yields an empty artifact."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = HashingVectorizerCalculator().fit(df, {"columns": ["nonexistent"]})
    assert art == {}


def test_hashing_vectorizer_apply_missing_vectorizer_object_is_noop() -> None:
    """Applying an artifact without a vectorizer object returns the frame unchanged."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    out = HashingVectorizerApplier().apply(df, {"columns": ["text"]})
    assert list(out.columns) == ["text"]


def test_hashing_vectorizer_norm_none_config() -> None:
    """``norm=None`` (falsy) is normalised to no L2 normalisation."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = HashingVectorizerCalculator().fit(
        df, {"columns": ["text"], "n_features": 16, "norm": None}
    )
    assert art["norm"] is None


def test_hashing_vectorizer_short_strings_produce_dense_output() -> None:
    """Very short/empty strings are still hashed into a fixed-width dense output."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = HashingVectorizerCalculator().fit(df, {"columns": ["text"], "n_features": 8})
    out = HashingVectorizerApplier().apply(df, art)
    hash_cols = [c for c in out.columns if "__hash__" in c]
    assert len(hash_cols) == 8
    assert len(out) == 3


def test_hashing_vectorizer_polars_input_at_fit_time() -> None:
    """Fitting directly on a polars frame converts internally to pandas (line 104)."""
    df = pl.DataFrame({"text": SHORT_CORPUS})
    art = HashingVectorizerCalculator().fit(df, {"columns": ["text"], "n_features": 8})
    assert art["type"] == "hashing_vectorizer"


def test_hashing_vectorizer_apply_all_columns_missing_is_noop() -> None:
    """If none of the fitted columns are present at apply time, no-op (line 43)."""
    fit_df = pd.DataFrame({"text": ["hello world"]})
    art = HashingVectorizerCalculator().fit(fit_df, {"columns": ["text"], "n_features": 8})
    apply_df = pd.DataFrame({"other": ["unrelated"]})
    out = HashingVectorizerApplier().apply(apply_df, art)
    assert list(out.columns) == ["other"]


def test_hashing_vectorizer_drop_original_removes_source_column() -> None:
    """``drop_original=True`` removes the source text column after vectorizing (line 57)."""
    df = pd.DataFrame({"text": ["hello world", "foo bar baz"]})
    art = HashingVectorizerCalculator().fit(
        df, {"columns": ["text"], "n_features": 8, "drop_original": True}
    )
    out = HashingVectorizerApplier().apply(df, art)
    assert "text" not in out.columns


def test_hashing_vectorizer_large_n_features_logs_warning(caplog: Any) -> None:
    """A very large ``n_features`` triggers the ``logger.warning`` branch (line 129)."""
    import logging

    df = pd.DataFrame({"text": ["hello world"]})
    with caplog.at_level(logging.WARNING):
        HashingVectorizerCalculator().fit(df, {"columns": ["text"], "n_features": 20_000})
    assert "output columns" in caplog.text


# ---------------------------------------------------------------------------
# TfidfVectorizer edge cases
# ---------------------------------------------------------------------------


def test_tfidf_vectorizer_empty_columns_returns_empty_artifact() -> None:
    """No configured columns yields an empty artifact."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = TfidfVectorizerCalculator().fit(df, {"columns": []})
    assert art == {}


def test_tfidf_vectorizer_apply_missing_vectorizer_is_noop() -> None:
    """Applying without a fitted vectorizer object leaves the frame unchanged."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    out = TfidfVectorizerApplier().apply(df, {"columns": ["text"]})
    assert list(out.columns) == ["text"]


def test_tfidf_vectorizer_drop_original_removes_source_column() -> None:
    """``drop_original=True`` removes the source text column after vectorizing."""
    df = pd.DataFrame({"text": ["hello world", "foo bar baz"]})
    art = TfidfVectorizerCalculator().fit(
        df, {"columns": ["text"], "max_features": 10, "drop_original": True}
    )
    out = TfidfVectorizerApplier().apply(df, art)
    assert "text" not in out.columns


def test_tfidf_vectorizer_multi_column_join() -> None:
    """Two source columns are joined into one text stream before fitting."""
    df = pd.DataFrame({"title": ["hello"], "body": ["world"]})
    art = TfidfVectorizerCalculator().fit(df, {"columns": ["title", "body"], "max_features": 10})
    assert art["columns"] == ["title", "body"]
    assert all(c.startswith("title_body__tfidf__") for c in art["output_columns"])


def test_tfidf_vectorizer_apply_all_columns_missing_is_noop() -> None:
    """If none of the fitted columns are present at apply time, no-op (line 39)."""
    fit_df = pd.DataFrame({"text": ["hello world"]})
    art = TfidfVectorizerCalculator().fit(fit_df, {"columns": ["text"], "max_features": 10})
    apply_df = pd.DataFrame({"other": ["unrelated"]})
    out = TfidfVectorizerApplier().apply(apply_df, art)
    assert list(out.columns) == ["other"]


def test_tfidf_vectorizer_fit_accepts_polars_input() -> None:
    """Fitting directly on a polars frame must route through ``to_pandas`` (line 99)."""
    df = pl.DataFrame({"text": ["hello world", "world of tokens"]})
    art = TfidfVectorizerCalculator().fit(df, {"columns": ["text"], "max_features": 10})
    assert art["columns"] == ["text"]
    assert len(art["output_columns"]) > 0


def test_tfidf_vectorizer_missing_column_at_fit_returns_empty() -> None:
    """A configured column absent from the frame yields an empty artifact (line 103)."""
    df = pd.DataFrame({"text": ["hello world"]})
    art = TfidfVectorizerCalculator().fit(df, {"columns": ["nonexistent"]})
    assert art == {}


def test_tfidf_vectorizer_large_vocabulary_logs_warning(caplog: Any, monkeypatch: Any) -> None:
    """A very large vocabulary triggers the ``logger.warning`` branch (line 132)."""
    import logging

    from skyulf.preprocessing.vectorization import tfidf_vectorizer as tfidf_module

    monkeypatch.setattr(
        tfidf_module, "_warn_large_output", lambda output_cols: "forced warning for test"
    )
    with caplog.at_level(logging.WARNING):
        df = pd.DataFrame({"text": ["hello world"]})
        TfidfVectorizerCalculator().fit(df, {"columns": ["text"]})
    assert "forced warning for test" in caplog.text


# ---------------------------------------------------------------------------
# CountVectorizer edge cases
# ---------------------------------------------------------------------------


def test_count_vectorizer_binary_mode_caps_counts_at_one() -> None:
    """``binary=True`` caps token counts at 1 even with repeated tokens."""
    df = pd.DataFrame({"text": ["hello hello hello", "world"]})
    art = CountVectorizerCalculator().fit(
        df, {"columns": ["text"], "binary": True, "max_features": 10}
    )
    out = CountVectorizerApplier().apply(df, art)
    hello_col = next(c for c in art["output_columns"] if c.endswith("__hello"))
    assert out[hello_col].iloc[0] == 1


def test_count_vectorizer_missing_vectorizer_is_noop() -> None:
    """Applying without a fitted vectorizer object leaves the frame unchanged."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    out = CountVectorizerApplier().apply(df, {"columns": ["text"]})
    assert list(out.columns) == ["text"]


def test_count_vectorizer_missing_column_at_fit_returns_empty() -> None:
    """A configured column absent from the frame yields an empty artifact."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = CountVectorizerCalculator().fit(df, {"columns": ["nonexistent"]})
    assert art == {}


def test_count_vectorizer_apply_all_columns_missing_is_noop() -> None:
    """If none of the fitted columns are present at apply time, no-op (line 45)."""
    fit_df = pd.DataFrame({"text": ["hello world"]})
    art = CountVectorizerCalculator().fit(fit_df, {"columns": ["text"]})
    apply_df = pd.DataFrame({"other": ["unrelated"]})
    out = CountVectorizerApplier().apply(apply_df, art)
    assert list(out.columns) == ["other"]


def test_count_vectorizer_fit_accepts_polars_input() -> None:
    """Fitting directly on a polars frame must route through ``to_pandas`` (line 107)."""
    df = pl.DataFrame({"text": ["hello world", "world of tokens"]})
    art = CountVectorizerCalculator().fit(df, {"columns": ["text"], "max_features": 10})
    assert art["columns"] == ["text"]
    assert len(art["output_columns"]) > 0


def test_count_vectorizer_large_vocabulary_logs_warning(caplog: Any, monkeypatch: Any) -> None:
    """A very large vocabulary triggers the ``logger.warning`` branch (line 140)."""
    import logging

    from skyulf.preprocessing.vectorization import count_vectorizer as cv_module

    monkeypatch.setattr(
        cv_module, "_warn_large_output", lambda output_cols: "forced warning for test"
    )
    with caplog.at_level(logging.WARNING):
        df = pd.DataFrame({"text": ["hello world"]})
        CountVectorizerCalculator().fit(df, {"columns": ["text"]})
    assert "forced warning for test" in caplog.text


# ---------------------------------------------------------------------------
# Tokenizer edge cases
# ---------------------------------------------------------------------------


def test_tokenizer_missing_column_at_fit_returns_empty() -> None:
    """A configured column absent from the frame yields an empty artifact."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    art = TokenizerCalculator().fit(df, {"columns": ["nonexistent"]})
    assert art == {}


def test_tokenizer_apply_missing_columns_is_noop() -> None:
    """Applying with no valid columns leaves the frame unchanged."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    out = TokenizerApplier().apply(df, {"columns": ["nonexistent"]})
    assert list(out.columns) == ["text"]


def test_tokenizer_char_analyzer() -> None:
    """The ``char`` analyzer tokenizes at the character level."""
    df = pd.DataFrame({"text": ["ab"]})
    art = TokenizerCalculator().fit(
        df, {"columns": ["text"], "analyzer": "char", "ngram_range": [1, 1]}
    )
    out = TokenizerApplier().apply(df, art)
    assert out["text__tokens"].iloc[0] == "a b"


def test_tokenizer_empty_string_produces_empty_token_string() -> None:
    """An empty source string tokenizes to an empty joined-token string."""
    df = pd.DataFrame({"text": [""]})
    art = TokenizerCalculator().fit(df, {"columns": ["text"], "add_token_count": True})
    out = TokenizerApplier().apply(df, art)
    assert out["text__tokens"].iloc[0] == ""
    assert out["text__token_count"].iloc[0] == 0
