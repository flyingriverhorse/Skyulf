"""
Text / NLP Vectorization Demo — Skyulf Core.

Demonstrates the three vectorizer nodes and two Naive Bayes classifiers
added in v0.6.2.  Run from the repo root:

    python skyulf-core/examples/06_text_nlp_vectorization.py

No network access needed — everything runs on synthetic data.

Sections
--------
1. CountVectorizer   — bag-of-words counts
2. TfidfVectorizer   — TF-IDF weights, including bigrams
3. HashingVectorizer — stateless hash trick (no vocabulary)
4. Polars input      — all nodes accept polars DataFrames transparently
5. MultinomialNB     — sentiment classification on TF-IDF features
6. BernoulliNB       — spam detection on binary features
7. Multi-column text — join two text columns before vectorising
8. Tokenizer         — split text into tokens (stateless)
9. SentenceEmbedder  — dense semantic embeddings (optional dependency)
"""

# pylint: disable=no-value-for-parameter
# The calculator/applier `fit`/`apply` calls below use the public 2-arg
# wrapper signature (see `fit_method`/`apply_method` in `preprocessing/base.py`).
# Pylint statically inspects the pre-decoration signature and misreports a
# missing argument; the decorator supplies it at runtime.

import pandas as pd
import polars as pl

# ── Node imports ──────────────────────────────────────────────────────────────
from skyulf.modeling.naive_bayes import (
    BernoulliNBApplier,
    BernoulliNBCalculator,
    MultinomialNBApplier,
    MultinomialNBCalculator,
)
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

# ── Shared demo data ──────────────────────────────────────────────────────────
REVIEWS_TRAIN = pd.DataFrame(
    {
        "review": [
            "great product amazing quality love it",
            "wonderful experience highly recommend",
            "fantastic service will buy again",
            "excellent value for money superb",
            "terrible quality broke after one day",
            "awful experience never buying again",
            "horrible product complete waste of money",
            "very bad service extremely disappointed",
        ],
        "subject": [
            "item quality review",
            "customer service feedback",
            "product purchase opinion",
            "value assessment report",
            "quality complaint issue",
            "service complaint feedback",
            "product complaint waste",
            "service complaint disappointment",
        ],
        "label": [1, 1, 1, 1, 0, 0, 0, 0],
    }
)

REVIEWS_TEST = pd.DataFrame(
    {
        "review": [
            "amazing quality really loved the product",
            "terrible waste of money bad quality",
        ],
        "subject": [
            "product quality opinion",
            "complaint issue waste",
        ],
        "label": [1, 0],
    }
)


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CountVectorizer
# ─────────────────────────────────────────────────────────────────────────────
def demo_count_vectorizer() -> None:
    separator("1. CountVectorizer — bag-of-words counts")

    config = {
        "columns": ["review"],
        "max_features": 10,  # keep only the 10 most frequent tokens
        "min_df": 1,
        "ngram_range": [1, 1],
    }

    calc = CountVectorizerCalculator()
    artifact = calc.fit(REVIEWS_TRAIN, config)

    print(f"  Vocabulary ({len(artifact['vocabulary'])} tokens):")
    for token, idx in sorted(artifact["vocabulary"].items(), key=lambda x: x[1]):
        print(f"    [{idx}] {token}")

    applier = CountVectorizerApplier()
    result = applier.apply(REVIEWS_TRAIN, artifact)

    print(f"\n  Output shape: {result.shape}")
    print("  First row counts (non-zero):")
    first = result.iloc[0]
    non_zero = {c: int(first[c]) for c in artifact["output_columns"] if first[c] > 0}
    for col, val in sorted(non_zero.items()):
        print(f"    {col}: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TfidfVectorizer (with bigrams)
# ─────────────────────────────────────────────────────────────────────────────
def demo_tfidf_vectorizer() -> None:
    separator("2. TfidfVectorizer — TF-IDF weights + bigrams")

    config = {
        "columns": ["review"],
        "max_features": 15,
        "ngram_range": [1, 2],  # unigrams AND bigrams
        "sublinear_tf": True,  # apply log(1+tf) scaling
    }

    calc = TfidfVectorizerCalculator()
    artifact = calc.fit(REVIEWS_TRAIN, config)

    bigrams = [t for t in artifact["vocabulary"] if " " in t]
    print(f"  Total features: {len(artifact['vocabulary'])}")
    print(f"  Bigram examples: {bigrams[:5]}")

    applier = TfidfVectorizerApplier()
    result = applier.apply(REVIEWS_TRAIN, artifact)

    print(f"\n  Output shape: {result.shape}")
    print(
        f"  Value range: [{result[artifact['output_columns']].min().min():.3f}, "
        f"{result[artifact['output_columns']].max().max():.3f}]"
    )
    print("  (TF-IDF is ≤ 1.0 with L2 normalisation)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. HashingVectorizer (stateless)
# ─────────────────────────────────────────────────────────────────────────────
def demo_hashing_vectorizer() -> None:
    separator("3. HashingVectorizer — stateless hash trick")

    config = {
        "columns": ["review"],
        "n_features": 32,  # number of hash buckets (power of 2 recommended)
        "norm": "l2",
    }

    # HashingVectorizer has no vocabulary — Calculator just stores config.
    calc = HashingVectorizerCalculator()
    artifact = calc.fit(REVIEWS_TRAIN, config)

    print(f"  n_features : {artifact['n_features']}")
    print(f"  norm       : {artifact['norm']}")
    print(f"  Output cols: {artifact['output_columns'][:4]} … {artifact['output_columns'][-1]}")

    applier = HashingVectorizerApplier()
    result = applier.apply(REVIEWS_TRAIN, artifact)
    print(f"\n  Output shape: {result.shape}")

    # Apply to unseen data without re-fitting — that's the whole point.
    result_test = applier.apply(REVIEWS_TEST, artifact)
    print(f"  Test  shape: {result_test.shape}  (same artifact — no refit needed)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Polars input
# ─────────────────────────────────────────────────────────────────────────────
def demo_polars_input() -> None:
    separator("4. Polars input — transparent polars ↔ pandas bridge")

    df_train_pl = pl.from_pandas(REVIEWS_TRAIN)
    df_test_pl = pl.from_pandas(REVIEWS_TEST)

    config = {"columns": ["review"], "max_features": 8}

    # Fit on polars (internally converts to pandas for sklearn, then back)
    calc = TfidfVectorizerCalculator()
    artifact = calc.fit(df_train_pl.to_pandas(), config)

    applier = TfidfVectorizerApplier()
    result = applier.apply(df_train_pl, artifact)

    print(f"  Input type : {type(df_train_pl).__name__}")
    print(f"  Output type: {type(result).__name__}")
    print(f"  Output shape: {result.shape}")

    # Test set also returns polars
    result_test = applier.apply(df_test_pl, artifact)
    print(f"  Test output type: {type(result_test).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MultinomialNB (sentiment)
# ─────────────────────────────────────────────────────────────────────────────
def demo_multinomial_nb() -> None:
    separator("5. MultinomialNB — sentiment classification")

    # Step 1: Vectorize
    vec_config = {"columns": ["review"], "max_features": 20}
    calc = TfidfVectorizerCalculator()
    art_vec = calc.fit(REVIEWS_TRAIN, vec_config)

    applier_vec = TfidfVectorizerApplier()
    train_vec = applier_vec.apply(REVIEWS_TRAIN, art_vec)
    test_vec = applier_vec.apply(REVIEWS_TEST, art_vec)

    feature_cols = art_vec["output_columns"]

    # Step 2: Train MultinomialNB
    nb_calc = MultinomialNBCalculator()
    model_art = nb_calc.fit(
        train_vec[feature_cols],
        train_vec["label"],
        config={"alpha": 1.0},
    )

    # Step 3: Predict
    nb_appl = MultinomialNBApplier()
    preds = nb_appl.predict(test_vec[feature_cols], model_art)

    print("  Predictions vs ground truth:")
    for i, (pred, actual) in enumerate(zip(preds, REVIEWS_TEST["label"], strict=True)):
        label = "positive" if pred == 1 else "negative"
        match = "✓" if pred == actual else "✗"
        print(f"    [{match}] '{REVIEWS_TEST['review'].iloc[i][:40]}…'  → {label}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BernoulliNB (spam detection)
# ─────────────────────────────────────────────────────────────────────────────
def demo_bernoulli_nb() -> None:
    separator("6. BernoulliNB — spam detection on binary features")

    spam_train = pd.DataFrame(
        {
            "message": [
                "win a free prize click now",
                "you have won a lottery claim your prize",
                "free money limited offer click here",
                "meeting at 10am see you there",
                "can we reschedule the call tomorrow",
                "project update attached please review",
            ],
            "spam": [1, 1, 1, 0, 0, 0],
        }
    )
    spam_test = pd.DataFrame(
        {
            "message": [
                "click here to claim your free prize",
                "let me know when you are available",
            ]
        }
    )

    # BernoulliNB works well with binary (0/1) features — use binarize threshold
    config_vec = {"columns": ["message"], "max_features": 15}
    calc = CountVectorizerCalculator()
    art_vec = calc.fit(spam_train, config_vec)

    applier_vec = CountVectorizerApplier()
    train_vec = applier_vec.apply(spam_train, art_vec)
    test_vec = applier_vec.apply(spam_test, art_vec)

    feature_cols = art_vec["output_columns"]

    nb_calc = BernoulliNBCalculator()
    model_art = nb_calc.fit(
        train_vec[feature_cols],
        train_vec["spam"],
        config={"alpha": 1.0, "binarize": 0.0},
    )

    nb_appl = BernoulliNBApplier()
    preds = nb_appl.predict(test_vec[feature_cols], model_art)

    print("  Predictions:")
    for msg, pred in zip(spam_test["message"], preds, strict=True):
        label = "SPAM" if pred == 1 else "HAM "
        print(f"    [{label}] '{msg}'")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multi-column text join
# ─────────────────────────────────────────────────────────────────────────────
def demo_multi_column() -> None:
    separator("7. Multi-column join — review + subject combined")

    # When columns = ["review", "subject"], the node concatenates them with a space
    config = {"columns": ["review", "subject"], "max_features": 20}

    calc = TfidfVectorizerCalculator()
    artifact = calc.fit(REVIEWS_TRAIN, config)

    # Output columns are prefixed with the first source column name
    sample_cols = artifact["output_columns"][:5]
    print(f"  Sample output columns: {sample_cols}")

    applier = TfidfVectorizerApplier()
    result = applier.apply(REVIEWS_TRAIN, artifact)
    print(f"  Output shape: {result.shape}")
    print("  (Both 'review' and 'subject' columns are still in the output by default.)")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def demo_tokenizer() -> None:
    separator("8. Tokenizer — split text into tokens (stateless)")

    config = {
        "columns": ["review"],
        "analyzer": "word",
        "stop_words": "english",
        "add_token_count": True,
    }

    calc = TokenizerCalculator()
    artifact = calc.fit(REVIEWS_TRAIN, config)
    print(f"  Output columns: {artifact['output_columns']}")

    applier = TokenizerApplier()
    result = applier.apply(REVIEWS_TRAIN, artifact)

    print("  Tokenized text (English stop words removed):")
    for i in range(3):
        tokens = result["review__tokens"].iloc[i]
        count = result["review__token_count"].iloc[i]
        print(f"    [{count:>2} tokens] {tokens}")
    print("  Tip: feed the '__tokens' column into a vectorizer for custom pipelines.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SentenceEmbedder (optional dependency)
# ─────────────────────────────────────────────────────────────────────────────
def demo_sentence_embedder() -> None:
    separator("9. SentenceEmbedder — dense semantic embeddings")

    try:
        from skyulf.preprocessing.vectorization import (
            SentenceEmbedderApplier,
            SentenceEmbedderCalculator,
        )

        config = {"columns": ["review"], "model_name": "all-MiniLM-L6-v2"}
        calc = SentenceEmbedderCalculator()
        artifact = calc.fit(REVIEWS_TRAIN, config)
        print(f"  Embedding dimension: {artifact['embedding_dim']}")

        applier = SentenceEmbedderApplier()
        result = applier.apply(REVIEWS_TRAIN, artifact)
        print(f"  Output shape: {result.shape}")
        print(f"  First 5 embedding columns: {artifact['output_columns'][:5]}")
    except ImportError as exc:
        print(f"  Skipped — {exc}")
        print("  Install with: pip install skyulf[nlp]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_count_vectorizer()
    demo_tfidf_vectorizer()
    demo_hashing_vectorizer()
    demo_polars_input()
    demo_multinomial_nb()
    demo_bernoulli_nb()
    demo_multi_column()
    demo_tokenizer()
    demo_sentence_embedder()

    print("\n" + "=" * 60)
    print("  All demos completed successfully.")
    print("=" * 60)
