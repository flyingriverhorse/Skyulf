"""Text classification with Skyulf vectorizers, tokenizer, and Naive Bayes.

Run from the repository root:
    python skyulf-core/examples/06_text_nlp_vectorization.py

Count and TF-IDF learn vocabulary/IDF and must be fit on training text only.
HashingVectorizer and Tokenizer are stateless.  SentenceEmbedder is optional:
install ``skyulf-core[nlp]`` and use its registered ``sentence_embedder`` node
when downloading a transformer model is appropriate for your environment.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skyulf.modeling.naive_bayes import (
    BernoulliNBApplier,
    BernoulliNBCalculator,
    MultinomialNBApplier,
    MultinomialNBCalculator,
)
from skyulf.preprocessing.pipeline import FeatureEngineer

TEXTS = [
    ("The team scored a goal and won the match", "sport"),
    ("A goalkeeper saved the final penalty", "sport"),
    ("The coach planned a better football defense", "sport"),
    ("The new graphics card renders pixels quickly", "tech"),
    ("Software developers fixed a production bug", "tech"),
    ("The database query needs a faster index", "tech"),
    ("The player trained for the championship match", "sport"),
    ("Our application deploys a machine learning model", "tech"),
]


def vectorize_and_score(name: str, params: dict, calculator, applier) -> None:
    """Fit one vectorizer on train text and score its matching Naive Bayes model."""
    frame = pd.DataFrame(TEXTS, columns=["text", "label"])
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame["label"])
    engineer = FeatureEngineer([{"name": name, "transformer": name, "params": params}])
    X_train, _ = engineer.fit_transform(train[["text"]])
    X_test = engineer.transform(test[["text"]])
    model = calculator.fit(X_train, train["label"], {"params": {"alpha": 1.0}})
    predictions = applier.predict(X_test, model)
    print(
        f"{name}: matrix={X_train.shape}, accuracy={accuracy_score(test['label'], predictions):.3f}"
    )


def main() -> None:
    """Run tokenizer inspection and three vectorizer/model combinations."""
    tokenized, _ = FeatureEngineer(
        [
            {
                "name": "tokenize",
                "transformer": "tokenizer",
                "params": {"columns": ["text"], "add_token_count": True},
            }
        ]
    ).fit_transform(pd.DataFrame(TEXTS[:2], columns=["text", "label"])[["text"]])
    print("Tokenizer columns:", tokenized.columns.tolist())

    vectorize_and_score(
        "count_vectorizer",
        {"columns": ["text"], "max_features": 40, "drop_original": True},
        MultinomialNBCalculator(),
        MultinomialNBApplier(),
    )
    vectorize_and_score(
        "tfidf_vectorizer",
        {"columns": ["text"], "max_features": 40, "drop_original": True},
        MultinomialNBCalculator(),
        MultinomialNBApplier(),
    )
    vectorize_and_score(
        "hashing_vectorizer",
        {
            "columns": ["text"],
            "n_features": 32,
            "alternate_sign": False,
            "drop_original": True,
        },
        BernoulliNBCalculator(),
        BernoulliNBApplier(),
    )
    print("Text/NLP example completed. SentenceEmbedder is documented in this module docstring.")


if __name__ == "__main__":
    main()
