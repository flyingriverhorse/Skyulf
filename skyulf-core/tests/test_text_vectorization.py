"""Tests for text vectorization nodes (CountVectorizer, TfidfVectorizer, HashingVectorizer)."""

import importlib.util
from typing import Any

import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.vectorization import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
    HashingVectorizerApplier,
    HashingVectorizerCalculator,
    TfidfVectorizerApplier,
    TfidfVectorizerCalculator,
    TokenizerApplier,
    TokenizerCalculator,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

CORPUS = [
    "the quick brown fox",
    "the fox jumped over the lazy dog",
    "hello world hello",
    "machine learning is fun",
    "natural language processing with python",
]

CONFIG_COL = "text"


@pytest.fixture
def df_pandas() -> pd.DataFrame:
    return pd.DataFrame({CONFIG_COL: CORPUS})


@pytest.fixture
def df_polars() -> Any:
    return pl.DataFrame({CONFIG_COL: CORPUS})


# ── CountVectorizer ───────────────────────────────────────────────────────────


class TestCountVectorizer:
    def test_fit_produces_artifact(self, df_pandas: pd.DataFrame) -> None:
        calc = CountVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "max_features": 20})
        assert art["type"] == "count_vectorizer"
        assert art["columns"] == [CONFIG_COL]
        assert len(art["output_columns"]) <= 20
        assert all(c.startswith(f"{CONFIG_COL}__count__") for c in art["output_columns"])
        assert len(art["vocabulary"]) == len(art["output_columns"])

    def test_apply_adds_columns_pandas(self, df_pandas: pd.DataFrame) -> None:
        calc = CountVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "max_features": 10})
        appl = CountVectorizerApplier()
        result = appl.apply(df_pandas, art)
        assert isinstance(result, pd.DataFrame)
        for col in art["output_columns"]:
            assert col in result.columns
        # Source column retained by default
        assert CONFIG_COL in result.columns

    def test_apply_drop_original(self, df_pandas: pd.DataFrame) -> None:
        calc = CountVectorizerCalculator()
        art = calc.fit(
            df_pandas, {"columns": [CONFIG_COL], "max_features": 5, "drop_original": True}
        )
        assert art["drop_original"] is True
        appl = CountVectorizerApplier()
        result = appl.apply(df_pandas, art)
        assert CONFIG_COL not in result.columns

    def test_apply_polars_roundtrip(self, df_polars: Any) -> None:
        df_pd = df_polars.to_pandas()
        calc = CountVectorizerCalculator()
        art = calc.fit(df_pd, {"columns": [CONFIG_COL], "max_features": 8})
        appl = CountVectorizerApplier()
        result = appl.apply(df_polars, art)
        assert isinstance(result, pl.DataFrame)
        for col in art["output_columns"]:
            assert col in result.columns

    def test_output_values_non_negative(self, df_pandas: pd.DataFrame) -> None:
        calc = CountVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "max_features": 10})
        appl = CountVectorizerApplier()
        result = appl.apply(df_pandas, art)
        numeric_cols = art["output_columns"]
        assert (result[numeric_cols] >= 0).all().all()

    def test_empty_columns_config_returns_unchanged(self, df_pandas: pd.DataFrame) -> None:
        calc = CountVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": []})
        assert art == {}


# ── TfidfVectorizer ───────────────────────────────────────────────────────────


class TestTfidfVectorizer:
    def test_fit_produces_artifact(self, df_pandas: pd.DataFrame) -> None:
        calc = TfidfVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "max_features": 15})
        assert art["type"] == "tfidf_vectorizer"
        assert len(art["idf"]) == len(art["output_columns"])
        assert all(c.startswith(f"{CONFIG_COL}__tfidf__") for c in art["output_columns"])

    def test_apply_values_in_range(self, df_pandas: pd.DataFrame) -> None:
        calc = TfidfVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "max_features": 10})
        appl = TfidfVectorizerApplier()
        result = appl.apply(df_pandas, art)
        numeric_cols = art["output_columns"]
        assert (result[numeric_cols] >= 0).all().all()
        assert (result[numeric_cols] <= 1.01).all().all()  # TF-IDF normalised to ≤1

    def test_apply_polars_roundtrip(self, df_polars: Any) -> None:
        df_pd = df_polars.to_pandas()
        calc = TfidfVectorizerCalculator()
        art = calc.fit(df_pd, {"columns": [CONFIG_COL], "max_features": 8})
        appl = TfidfVectorizerApplier()
        result = appl.apply(df_polars, art)
        assert isinstance(result, pl.DataFrame)

    def test_ngram_range(self, df_pandas: pd.DataFrame) -> None:
        calc = TfidfVectorizerCalculator()
        art = calc.fit(
            df_pandas,
            {"columns": [CONFIG_COL], "max_features": 20, "ngram_range": [1, 2]},
        )
        # Bigrams should be present (e.g. "quick brown")
        assert any(" " in c.split("__tfidf__")[-1] for c in art["output_columns"])


# ── HashingVectorizer ─────────────────────────────────────────────────────────


class TestHashingVectorizer:
    def test_fit_produces_artifact(self, df_pandas: pd.DataFrame) -> None:
        calc = HashingVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "n_features": 64})
        assert art["type"] == "hashing_vectorizer"
        assert art["n_features"] == 64
        assert len(art["output_columns"]) == 64
        assert art["output_columns"][0] == f"{CONFIG_COL}__hash__0"
        assert art["output_columns"][-1] == f"{CONFIG_COL}__hash__63"

    def test_apply_output_shape(self, df_pandas: pd.DataFrame) -> None:
        calc = HashingVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "n_features": 32})
        appl = HashingVectorizerApplier()
        result = appl.apply(df_pandas, art)
        assert len([c for c in result.columns if c.startswith(f"{CONFIG_COL}__hash__")]) == 32

    def test_apply_polars_roundtrip(self, df_polars: Any) -> None:
        df_pd = df_polars.to_pandas()
        calc = HashingVectorizerCalculator()
        art = calc.fit(df_pd, {"columns": [CONFIG_COL], "n_features": 16})
        appl = HashingVectorizerApplier()
        result = appl.apply(df_polars, art)
        assert isinstance(result, pl.DataFrame)

    def test_stateless_same_result_on_new_data(self, df_pandas: pd.DataFrame) -> None:
        """HashingVectorizer gives identical results regardless of training data."""
        calc = HashingVectorizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "n_features": 32})
        appl = HashingVectorizerApplier()
        result1 = appl.apply(df_pandas.iloc[:2], art)
        result2 = appl.apply(df_pandas.iloc[:2].copy(), art)
        pd.testing.assert_frame_equal(result1, result2)


# ── Registry integration ──────────────────────────────────────────────────────


def test_vectorizers_registered() -> None:
    from skyulf.registry import NodeRegistry

    meta = NodeRegistry.get_all_metadata()
    for node_id in (
        "count_vectorizer",
        "tfidf_vectorizer",
        "hashing_vectorizer",
        "tokenizer",
        "sentence_embedder",
    ):
        assert node_id in meta, f"{node_id!r} not registered"
        assert meta[node_id]["category"] == "Text"


def test_naive_bayes_registered() -> None:
    from skyulf.registry import NodeRegistry

    meta = NodeRegistry.get_all_metadata()
    for node_id in ("multinomial_nb", "bernoulli_nb"):
        assert node_id in meta, f"{node_id!r} not registered"
        assert meta[node_id]["category"] == "Modeling"


# ── End-to-end: TF-IDF → MultinomialNB ───────────────────────────────────────


def test_tfidf_multinomialnb_pipeline() -> None:
    """Smoke test: TF-IDF features fed into MultinomialNB should predict classes."""
    texts = [
        "great food amazing taste",
        "delicious meal wonderful",
        "terrible service awful food",
        "horrible experience bad",
        "fantastic wonderful amazing",
    ]
    labels = [1, 1, 0, 0, 1]

    df = pd.DataFrame({"review": texts, "label": labels})
    config = {"columns": ["review"], "max_features": 20}

    # Step 1: TF-IDF
    tfidf_calc = TfidfVectorizerCalculator()
    art = tfidf_calc.fit(df, config)
    tfidf_appl = TfidfVectorizerApplier()
    df_vec = tfidf_appl.apply(df, art)

    # Step 2: MultinomialNB — non-negative TF-IDF values, works with MultinomialNB
    from skyulf.modeling.naive_bayes import MultinomialNBApplier, MultinomialNBCalculator

    feature_cols = art["output_columns"]
    X_train = df_vec[feature_cols]
    y_train = df_vec["label"]

    nb_calc = MultinomialNBCalculator()
    model_art = nb_calc.fit(X_train, y_train, config={})
    assert model_art is not None

    nb_appl = MultinomialNBApplier()
    preds = nb_appl.predict(X_train, model_art)
    assert len(preds) == len(texts)
    # All predictions should be 0 or 1
    assert set(preds).issubset({0, 1})


# ── Tokenizer ─────────────────────────────────────────────────────────────────


class TestTokenizer:
    def test_word_tokens_joined(self, df_pandas: pd.DataFrame) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL]})
        assert art["output_columns"] == [f"{CONFIG_COL}__tokens"]

        appl = TokenizerApplier()
        result = appl.apply(df_pandas, art)
        assert f"{CONFIG_COL}__tokens" in result.columns
        # "the quick brown fox" → lowercase word tokens joined by space
        assert result[f"{CONFIG_COL}__tokens"].iloc[0] == "the quick brown fox"

    def test_stop_words_removed(self, df_pandas: pd.DataFrame) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "stop_words": "english"})
        appl = TokenizerApplier()
        result = appl.apply(df_pandas, art)
        # "the" is an English stop word and should be dropped
        assert "the" not in result[f"{CONFIG_COL}__tokens"].iloc[0].split()

    def test_token_count_column(self, df_pandas: pd.DataFrame) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "add_token_count": True})
        appl = TokenizerApplier()
        result = appl.apply(df_pandas, art)
        assert f"{CONFIG_COL}__token_count" in result.columns
        assert result[f"{CONFIG_COL}__token_count"].iloc[0] == 4  # quick brown fox + the

    def test_drop_original(self, df_pandas: pd.DataFrame) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL], "drop_original": True})
        appl = TokenizerApplier()
        result = appl.apply(df_pandas, art)
        assert CONFIG_COL not in result.columns
        assert f"{CONFIG_COL}__tokens" in result.columns

    def test_polars_input(self, df_polars: Any) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_polars, {"columns": [CONFIG_COL]})
        appl = TokenizerApplier()
        result = appl.apply(df_polars, art)
        assert isinstance(result, pl.DataFrame)
        assert f"{CONFIG_COL}__tokens" in result.columns

    def test_empty_columns_noop(self, df_pandas: pd.DataFrame) -> None:
        calc = TokenizerCalculator()
        art = calc.fit(df_pandas, {"columns": []})
        assert art == {}


# ── SentenceEmbedder (optional dependency) ────────────────────────────────────

_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not _HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed (optional 'nlp' extra)",
)
class TestSentenceEmbedder:
    def test_embeddings_shape(self, df_pandas: pd.DataFrame) -> None:
        from skyulf.preprocessing.vectorization import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        calc = SentenceEmbedderCalculator()
        art = calc.fit(df_pandas, {"columns": [CONFIG_COL]})
        assert art["embedding_dim"] > 0
        assert len(art["output_columns"]) == art["embedding_dim"]

        appl = SentenceEmbedderApplier()
        result = appl.apply(df_pandas, art)
        for col in art["output_columns"]:
            assert col in result.columns
        assert len(result) == len(df_pandas)
