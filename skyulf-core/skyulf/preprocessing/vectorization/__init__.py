"""Text vectorization preprocessing nodes.

Each module registers its node via ``@NodeRegistry.register`` at import time;
importing them here wires the nodes into the registry and re-exports the
public Calculator/Applier pairs.

  count_vectorizer.py   — CountVectorizer (bag-of-words, fitted vocabulary)
  tfidf_vectorizer.py   — TfidfVectorizer (TF-IDF weights, fitted vocabulary)
  hashing_vectorizer.py — HashingVectorizer (stateless, hashing trick)
  tokenizer.py          — Tokenizer (word/char tokens, stateless)
  sentence_embedder.py  — SentenceEmbedder (dense embeddings, optional dep)
"""

from .count_vectorizer import CountVectorizerApplier, CountVectorizerCalculator
from .hashing_vectorizer import HashingVectorizerApplier, HashingVectorizerCalculator
from .sentence_embedder import SentenceEmbedderApplier, SentenceEmbedderCalculator
from .tfidf_vectorizer import TfidfVectorizerApplier, TfidfVectorizerCalculator
from .tokenizer import TokenizerApplier, TokenizerCalculator

__all__ = [
    "CountVectorizerCalculator",
    "CountVectorizerApplier",
    "TfidfVectorizerCalculator",
    "TfidfVectorizerApplier",
    "HashingVectorizerCalculator",
    "HashingVectorizerApplier",
    "TokenizerCalculator",
    "TokenizerApplier",
    "SentenceEmbedderCalculator",
    "SentenceEmbedderApplier",
]
