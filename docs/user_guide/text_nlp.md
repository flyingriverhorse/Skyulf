# Text & NLP

Machine-learning models only understand **numbers**. A column of raw text
(`"the pitcher threw a curveball"`) has to be turned into numeric features before any
model can learn from it. Skyulf ships a small, composable set of **text nodes** that do
exactly that, plus two text-friendly classifiers.

This page explains each node, when to reach for it, and how they combine. For a runnable,
end-to-end tour on real data see the notebook
[`examples/07_text_nlp_real_data.ipynb`](https://github.com/flyingriverhorse/Skyulf/blob/main/skyulf-core/examples/07_text_nlp_real_data.ipynb)
and the script `examples/06_text_nlp_vectorization.py`.

---

## The text toolbox at a glance

| Node | id | What it does | Needs a fit? |
|------|----|--------------|--------------|
| Count Vectorizer | `count_vectorizer` | Counts how often each word appears ("bag of words") | Yes (learns a vocabulary) |
| TF-IDF Vectorizer | `tfidf_vectorizer` | Like counts, but down-weights words common to every document | Yes (vocabulary + IDF weights) |
| Hashing Vectorizer | `hashing_vectorizer` | Hashes words into a fixed number of columns — no vocabulary | No (stateless) |
| Tokenizer | `tokenizer` | Splits / cleans text into tokens (lowercase, stop-words, n-grams) | No (stateless) |
| Sentence Embedder | `sentence_embedder` | Encodes text into dense semantic vectors (needs `skyulf[nlp]`) | Yes (loads a model) |
| Multinomial NB | `multinomial_nb` | Fast, strong classifier for word-count / TF-IDF features | Yes |
| Bernoulli NB | `bernoulli_nb` | Classifier for "word present / absent" features | Yes |

All vectorizers live in `skyulf.preprocessing.vectorization`; the Naive Bayes models live in
`skyulf.modeling.naive_bayes`. Every node follows the standard Skyulf
**Calculator → Applier** contract (see [Overview](overview.md)): the Calculator *learns* from
the training data and returns an artifact; the Applier *uses* that artifact to transform any
DataFrame. Vectorizers keep the original columns and **add** new feature columns whose names
are listed in `artifact['output_columns']`.

!!! tip "Fit on train only"
    Vectorizers learn a vocabulary, so they are subject to data leakage just like scalers and
    encoders. Always `fit()` on the **training** split and reuse the same artifact to `apply()`
    on test / inference data.

---

## Count Vectorizer — bag of words

The simplest representation: build a vocabulary of every word seen in training, then count how
many times each word appears in each document.

```python
from skyulf.preprocessing.vectorization import (
    CountVectorizerCalculator,
    CountVectorizerApplier,
)

cfg = {
    "columns": ["text"],     # one or more text columns (joined with a space)
    "max_features": 5000,    # keep the 5000 most frequent words
    "min_df": 2,             # ignore words appearing in < 2 documents
    "ngram_range": [1, 1],   # unigrams only
}

art = CountVectorizerCalculator().fit(train_df, cfg)   # learn vocabulary on TRAIN
train_X = CountVectorizerApplier().apply(train_df, art)
test_X = CountVectorizerApplier().apply(test_df, art)  # same vocabulary

feature_cols = art["output_columns"]   # e.g. ['text__count__game', ...]
vocab = art["vocabulary"]              # {word: column index}
```

**Reach for it when** you want a dead-simple, interpretable baseline.

---

## TF-IDF Vectorizer — counts, but smarter

Plain counts over-reward words that appear everywhere ("the", "and"). **TF-IDF**
(Term Frequency × Inverse Document Frequency) multiplies each count by how *rare* the word is
across the corpus, so distinctive words score higher. This is the **best default** for most
text-classification tasks.

```python
from skyulf.preprocessing.vectorization import (
    TfidfVectorizerCalculator,
    TfidfVectorizerApplier,
)

cfg = {
    "columns": ["text"],
    "max_features": 20000,
    "min_df": 2,
    "ngram_range": [1, 2],   # unigrams + bigrams capture short phrases
    "sublinear_tf": True,    # dampen very frequent words (log scaling)
}

art = TfidfVectorizerCalculator().fit(train_df, cfg)
app = TfidfVectorizerApplier()
train_X = app.apply(train_df, art)
test_X = app.apply(test_df, art)
```

The artifact stores both the vocabulary and the learned IDF weights, so test data is
transformed with exactly the weights learned on train.

---

## Hashing Vectorizer — no vocabulary needed

Both vectorizers above must **store** a vocabulary, which can be large and must be learned up
front. The Hashing Vectorizer instead pushes every word through a hash function into a fixed
number of columns (`n_features`). It is **stateless** — there is nothing to learn — so it is
ideal for streaming data or vocabularies too big to keep in memory.

```python
from skyulf.preprocessing.vectorization import (
    HashingVectorizerCalculator,
    HashingVectorizerApplier,
)

cfg = {"columns": ["text"], "n_features": 4096, "norm": "l2"}

art = HashingVectorizerCalculator().fit(train_df, cfg)   # no vocabulary learned
train_X = HashingVectorizerApplier().apply(train_df, art)
```

The trade-off: output columns are anonymous hash buckets (`text__hash__0 …`), so you lose the
word ↔ column mapping, and different words can occasionally collide into the same bucket.

!!! warning "Hashing + Multinomial Naive Bayes"
    Multinomial Naive Bayes needs **non-negative** features. The Hashing Vectorizer uses a
    signed hash by default (`alternate_sign=True`), producing negatives. Set
    `alternate_sign=False` if you intend to feed it to Multinomial NB.

---

## Tokenizer — clean the text *before* vectorizing

Sometimes you want to pre-process text once — lowercase it, drop common filler ("stop") words,
split on characters instead of words — and then feed the cleaned result into a vectorizer. The
Tokenizer emits a space-joined token string column `{col}__tokens` (and optionally
`{col}__token_count`) that any vectorizer can consume.

```python
from skyulf.preprocessing.vectorization import (
    TokenizerCalculator,
    TokenizerApplier,
)

cfg = {
    "columns": ["text"],
    "analyzer": "word",        # 'word' | 'char' | 'char_wb'
    "stop_words": "english",   # drop common filler words ('the', 'is', ...)
    "add_token_count": True,   # also emit text__token_count
}

art = TokenizerCalculator().fit(train_df, cfg)
train_tok = TokenizerApplier().apply(train_df, art)
# -> new columns: ['text__tokens', 'text__token_count']
```

### Chaining: Tokenizer → TF-IDF

Because every node speaks the same DataFrame-in / DataFrame-out language, you can simply point
the next vectorizer at the Tokenizer's output column:

```python
chain_cfg = {
    "columns": ["text__tokens"],   # the Tokenizer's output column
    "max_features": 20000,
    "ngram_range": [1, 2],
    "sublinear_tf": True,
}
chain_art = TfidfVectorizerCalculator().fit(train_tok, chain_cfg)
train_X = TfidfVectorizerApplier().apply(train_tok, chain_art)
```

In practice this `text → Tokenizer → TF-IDF → model` chain often beats vectorizing raw text,
because stop-word removal focuses the vocabulary on meaningful words.

---

## Sentence Embedder — modern dense embeddings (optional)

Bag-of-words and TF-IDF treat "great" and "excellent" as completely unrelated. **Sentence
embeddings** map text into a dense vector space where similar *meanings* land close together,
which can help on short text. This node is powered by
[`sentence-transformers`](https://www.sbert.net/) and is an **optional** dependency.

```bash
pip install skyulf[nlp]          # or: pip install -r requirements-nlp.txt
```

```python
from skyulf.preprocessing.vectorization import (
    SentenceEmbedderCalculator,
    SentenceEmbedderApplier,
)

cfg = {"columns": ["text"], "model_name": "all-MiniLM-L6-v2", "normalize": True}

art = SentenceEmbedderCalculator().fit(train_df, cfg)
train_X = SentenceEmbedderApplier().apply(train_df, art)
# output columns: text__emb__0 … text__emb__{embedding_dim-1}
```

The node imports `sentence-transformers` lazily and raises a clear install hint if the extra is
missing, so pipelines that don't use it never pay the (large, PyTorch-based) dependency cost.

!!! warning "Embeddings are dense and can be negative"
    Embedding values can be negative, so they **cannot** be fed to Multinomial / Bernoulli
    Naive Bayes. Pair embeddings with Logistic Regression, SVC, Random Forest, or boosting.

---

## Which model can I use?

**Any of them.** Once text is vectorized it is just numeric columns, so the output feeds any
Skyulf classifier or regressor exactly like ordinary tabular data — `logistic_regression`,
`random_forest_classifier`, `svc`, `gradient_boosting_classifier`,
`hist_gradient_boosting_classifier`, `extra_trees_classifier`, `xgboost_classifier`,
`lgbm_classifier`, the ensemble nodes (`voting_classifier`, `stacking_classifier`), and so on.

There is only **one real constraint**, and it comes from the *model*, not from Skyulf:

| Features | Good models | Avoid | Why |
|----------|-------------|-------|-----|
| Counts / TF-IDF (non-negative) | **Multinomial NB**, Logistic Regression, linear SVC, trees, boosting | — | NB is the fast, strong text baseline; linear models thrive on high-dimensional sparse text |
| Binary word presence | **Bernoulli NB**, plus any of the above | — | Bernoulli models "word present / absent" |
| Dense embeddings (can be negative) | Logistic Regression, SVC, Random Forest, boosting | **Multinomial / Bernoulli NB** | Naive Bayes requires non-negative inputs |

### Multinomial & Bernoulli Naive Bayes

These two text-friendly classifiers live in `skyulf.modeling.naive_bayes`:

```python
from skyulf.modeling.naive_bayes import (
    MultinomialNBCalculator, MultinomialNBApplier,
    BernoulliNBCalculator, BernoulliNBApplier,
)
from sklearn.metrics import accuracy_score

# TF-IDF (or counts) -> Multinomial NB: the classic strong baseline
model = MultinomialNBCalculator().fit(
    train_X[feature_cols], train_df["label"], config={"alpha": 0.1}
)
preds = MultinomialNBApplier().predict(test_X[feature_cols], model)
print(accuracy_score(test_df["label"], preds))
```

Bernoulli NB binarizes its inputs internally at the `binarize` threshold, so it pairs naturally
with raw counts: `BernoulliNBCalculator().fit(X, y, config={"binarize": 0.0})`.

---

## The universal recipe

Every text pipeline follows the same shape:

```python
# 1. LEARN the transform on TRAIN only
art = SomeVectorizerCalculator().fit(train_df, config)

# 2. APPLY the same transform to train and test
train_X = SomeVectorizerApplier().apply(train_df, art)
test_X = SomeVectorizerApplier().apply(test_df, art)
cols = art["output_columns"]

# 3. Train a model on the new numeric columns
model = SomeModelCalculator().fit(train_X[cols], train_df["label"])
preds = SomeModelApplier().predict(test_X[cols], model)
```

---

## In the visual canvas

All of these nodes appear in the **Preprocessing** section of the node sidebar in the ML Canvas.
The built-in **"Text Classification"** starter template wires the standard strong baseline for
you — `dataset → Text Cleaning → TF-IDF → Train/Test Split → Logistic Regression`. Load it, point
the Text Cleaning and TF-IDF nodes at your raw-text column, pick the label column on the split,
and **Run All**.

## See also

- [Overview](overview.md) — the Calculator → Applier pattern.
- [SplitDataset & Leakage](splitdataset_and_leakage.md) — why vectorizers must fit on train only.
- [Preprocessing Nodes](../reference/preprocessing_nodes.md) — full node reference.
- [Modeling Nodes](../reference/modeling_nodes.md) — every available classifier and regressor.
