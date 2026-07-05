"""Coverage-gap tests for vectorization: hashing, tfidf, count, tokenizer, _common.

Complements ``test_text_vectorization.py`` by covering edge cases: empty
corpora, single-character/very short strings, drop_original interplay, the
"norm=None" hashing path, and the shared ``_common.py`` helpers directly.
"""

import importlib
import logging
from typing import Any

import pandas as pd
import polars as pl
import pytest
from tests.utils.test_case_loader import TestCaseLoader

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

# Registry mapping a node name (as used in JSON fixtures) to its
# Calculator/Applier pair, so parametrized tests can look up the right
# classes for a given scenario without encoding class objects in JSON.
_VECTORIZER_NODES: dict[str, tuple[type, type]] = {
    "hashing_vectorizer": (HashingVectorizerCalculator, HashingVectorizerApplier),
    "tfidf_vectorizer": (TfidfVectorizerCalculator, TfidfVectorizerApplier),
    "count_vectorizer": (CountVectorizerCalculator, CountVectorizerApplier),
    "tokenizer": (TokenizerCalculator, TokenizerApplier),
}


def _load_single_param(source_path: str, group: str | None = None) -> list[Any]:
    """Load a JSON fixture with exactly one param, unwrapping 1-tuples.

    ``pytest.mark.parametrize`` treats a single (comma-less) param name
    specially: argvalues must be bare scalars, not 1-tuples, or the whole
    tuple gets bound to the parameter instead of its single element.
    """
    params_string, scenarios = TestCaseLoader(source_path, group=group).load()
    return [params_string, [scenario[0] for scenario in scenarios]]


_invalid_fit_columns_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="invalid_fit_columns"
).load()
_apply_missing_vectorizer_cases = _load_single_param(
    "preprocessing/vectorization_gaps", group="apply_missing_vectorizer"
)
_apply_all_columns_missing_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="apply_all_columns_missing"
).load()
_drop_original_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="drop_original"
).load()
_large_vocab_warning_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="large_vocab_warning"
).load()
_polars_fit_input_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="polars_fit_input"
).load()
_join_text_columns_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="common_join_text_columns"
).load()
_warn_large_output_cases = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="common_warn_large_output"
).load()


@pytest.mark.parametrize(*_invalid_fit_columns_cases)
def test_invalid_fit_columns_returns_empty_artifact(node: str, columns: list[str]) -> None:
    """Empty or unmatched ``columns`` config yields an empty artifact at fit time."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    calculator_cls, _ = _VECTORIZER_NODES[node]
    art = calculator_cls().fit(df, {"columns": columns})
    assert art == {}


@pytest.mark.parametrize(*_apply_missing_vectorizer_cases)
def test_apply_missing_vectorizer_object_is_noop(node: str) -> None:
    """Applying an artifact without a fitted vectorizer object leaves the frame unchanged."""
    df = pd.DataFrame({"text": SHORT_CORPUS})
    _, applier_cls = _VECTORIZER_NODES[node]
    out = applier_cls().apply(df, {"columns": ["text"]})
    assert list(out.columns) == ["text"]


@pytest.mark.parametrize(*_apply_all_columns_missing_cases)
def test_apply_all_fitted_columns_missing_is_noop(node: str, fit_extra: dict[str, Any]) -> None:
    """If none of the fitted columns are present at apply time, the frame is unchanged."""
    calculator_cls, applier_cls = _VECTORIZER_NODES[node]
    fit_df = pd.DataFrame({"text": ["hello world"]})
    art = calculator_cls().fit(fit_df, {"columns": ["text"], **fit_extra})
    apply_df = pd.DataFrame({"other": ["unrelated"]})
    out = applier_cls().apply(apply_df, art)
    assert list(out.columns) == ["other"]


@pytest.mark.parametrize(*_drop_original_cases)
def test_drop_original_removes_source_column(node: str, fit_extra: dict[str, Any]) -> None:
    """``drop_original=True`` removes the source text column after vectorizing."""
    calculator_cls, applier_cls = _VECTORIZER_NODES[node]
    df = pd.DataFrame({"text": ["hello world", "foo bar baz"]})
    art = calculator_cls().fit(df, {"columns": ["text"], "drop_original": True, **fit_extra})
    out = applier_cls().apply(df, art)
    assert "text" not in out.columns


@pytest.mark.parametrize(*_large_vocab_warning_cases)
def test_large_vocabulary_logs_warning(
    node: str, module_path: str, caplog: Any, monkeypatch: Any
) -> None:
    """A forced ``_warn_large_output`` result must surface through ``logger.warning``."""
    calculator_cls, _ = _VECTORIZER_NODES[node]
    module = importlib.import_module(module_path)
    monkeypatch.setattr(module, "_warn_large_output", lambda output_cols: "forced warning for test")
    with caplog.at_level(logging.WARNING):
        df = pd.DataFrame({"text": ["hello world"]})
        calculator_cls().fit(df, {"columns": ["text"]})
    assert "forced warning for test" in caplog.text


@pytest.mark.parametrize(*_polars_fit_input_cases)
def test_fit_accepts_polars_input(node: str, fit_extra: dict[str, Any]) -> None:
    """Fitting directly on a polars frame must route internally through pandas."""
    calculator_cls, _ = _VECTORIZER_NODES[node]
    df = pl.DataFrame({"text": ["hello world", "world of tokens"]})
    art = calculator_cls().fit(df, {"columns": ["text"], **fit_extra})
    assert art["columns"] == ["text"]
    assert len(art["output_columns"]) > 0


# ---------------------------------------------------------------------------
# _common.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(*_join_text_columns_cases)
def test_join_text_columns(
    columns_data: dict[str, Any], columns: list[str], expected: list[str]
) -> None:
    """Single and multiple text columns are joined (NaN filled to empty string)."""
    df = pd.DataFrame(columns_data)
    joined = _join_text_columns(df, columns)
    assert joined.tolist() == expected


@pytest.mark.parametrize(*_warn_large_output_cases)
def test_warn_large_output(
    count: int, threshold: int, expect_none: bool, expect_substring: str | None
) -> None:
    """Output counts below threshold produce no warning; above, a human-readable message."""
    msg = _warn_large_output(count, threshold=threshold)
    if expect_none:
        assert msg is None
    else:
        assert msg is not None
        assert expect_substring is not None
        assert expect_substring in msg


# ---------------------------------------------------------------------------
# HashingVectorizer edge cases
# ---------------------------------------------------------------------------


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


def test_hashing_vectorizer_large_n_features_logs_warning(caplog: Any) -> None:
    """A very large ``n_features`` triggers the ``logger.warning`` branch (line 129)."""
    df = pd.DataFrame({"text": ["hello world"]})
    with caplog.at_level(logging.WARNING):
        HashingVectorizerCalculator().fit(df, {"columns": ["text"], "n_features": 20_000})
    assert "output columns" in caplog.text


# ---------------------------------------------------------------------------
# TfidfVectorizer edge cases
# ---------------------------------------------------------------------------


def test_tfidf_vectorizer_multi_column_join() -> None:
    """Two source columns are joined into one text stream before fitting."""
    df = pd.DataFrame({"title": ["hello"], "body": ["world"]})
    art = TfidfVectorizerCalculator().fit(df, {"columns": ["title", "body"], "max_features": 10})
    assert art["columns"] == ["title", "body"]
    assert all(c.startswith("title_body__tfidf__") for c in art["output_columns"])


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


# ---------------------------------------------------------------------------
# Tokenizer edge cases
# ---------------------------------------------------------------------------


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
