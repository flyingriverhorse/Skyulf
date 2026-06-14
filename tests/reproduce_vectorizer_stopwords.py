"""Verify new vectorizer params: lowercase, stop_words, binary.

Reproduces the original bug context (text -> vectorizer -> model) and confirms
the new sklearn passthrough knobs take effect:
  * stop_words="english" removes common words from the vocabulary
  * binary=True emits 1/0 presence instead of raw counts
  * lowercase=False preserves case in tokens
"""

import pandas as pd

from skyulf.preprocessing.vectorization.count_vectorizer import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
)
from skyulf.preprocessing.vectorization.hashing_vectorizer import (
    HashingVectorizerCalculator,
)
from skyulf.preprocessing.vectorization.tfidf_vectorizer import (
    TfidfVectorizerCalculator,
)

CORPUS = pd.DataFrame(
    {
        "text": [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a Cat and a Dog played",
            "the the the cat cat",
        ]
    }
)


def test_stop_words_removed() -> None:
    cfg = {"columns": ["text"], "stop_words": "english", "drop_original": True}
    art = CountVectorizerCalculator().fit(CORPUS, cfg)
    vocab = set(art["vocabulary"].keys())
    assert "the" not in vocab, "stop word 'the' should be removed"
    assert "cat" in vocab
    print("[ok] stop_words removed common tokens; vocab:", sorted(vocab))


def test_binary_counts() -> None:
    cfg = {"columns": ["text"], "binary": True, "drop_original": True}
    art = CountVectorizerCalculator().fit(CORPUS, cfg)
    out = CountVectorizerApplier().apply(CORPUS, art)
    df = out[0] if isinstance(out, tuple) else out
    # Row 3 has "the the the cat cat" -> binary means max value is 1.
    cat_col = [c for c in df.columns if c.endswith("__count__cat")][0]
    assert df[cat_col].max() == 1, "binary=True must clip counts to 1"
    print("[ok] binary=True produced 1/0 presence values")


def test_lowercase_false_preserves_case() -> None:
    cfg = {"columns": ["text"], "lowercase": False, "drop_original": True}
    art = CountVectorizerCalculator().fit(CORPUS, cfg)
    vocab = set(art["vocabulary"].keys())
    assert "Cat" in vocab and "cat" in vocab, "case should be preserved"
    print("[ok] lowercase=False kept distinct 'Cat'/'cat' tokens")


def test_tfidf_and_hashing_accept_stopwords() -> None:
    tfidf = TfidfVectorizerCalculator().fit(CORPUS, {"columns": ["text"], "stop_words": "english"})
    assert "the" not in tfidf["vocabulary"]
    hashing = HashingVectorizerCalculator().fit(
        CORPUS, {"columns": ["text"], "stop_words": "english", "n_features": 64}
    )
    assert hashing["stop_words"] == "english"
    print("[ok] tfidf + hashing accept stop_words")


if __name__ == "__main__":
    test_stop_words_removed()
    test_binary_counts()
    test_lowercase_false_preserves_case()
    test_tfidf_and_hashing_accept_stopwords()
    print("\nAll vectorizer param checks passed.")
