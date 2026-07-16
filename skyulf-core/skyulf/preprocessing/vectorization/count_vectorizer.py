"""CountVectorizer node — converts text to token-count feature columns.

Calculator fits a ``sklearn.feature_extraction.text.CountVectorizer`` on the
training corpus and records the vocabulary.  Applier transforms any text input
using the fitted vocabulary.

Output is **always dense** (``toarray()`` called on the sparse sklearn output).
For very large vocabularies a ``node_warnings`` entry is emitted instead of
raising an error.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import CountVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _join_text_columns,
    _sklearn_vectorizer_apply_pandas,
    _warn_large_output,
    apply_text_pandas_only,
    resolve_fit_text_columns,
)

logger = logging.getLogger(__name__)

_LARGE_VOCAB_THRESHOLD = 10_000


# ── Apply ─────────────────────────────────────────────────────────────────────


def _count_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Transform text columns using the fitted ``CountVectorizer``."""
    return _sklearn_vectorizer_apply_pandas(X, y, params)


class CountVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_text_pandas_only(X, params, _count_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("count_vectorizer", CountVectorizerApplier)
@node_meta(
    id="count_vectorizer",
    name="Count Vectorizer",
    category="Text",
    description=(
        "Convert text columns to token-count feature columns (bag-of-words). "
        "Fits a vocabulary on training data; apply uses that vocabulary."
    ),
    params={
        "columns": [],
        "max_features": None,
        "min_df": 1,
        "max_df": 1.0,
        "ngram_range": [1, 1],
        "lowercase": True,
        "stop_words": None,
        "binary": False,
        "drop_original": False,
    },
    tags=["text", "nlp", "vectorizer", "bag-of-words"],
)
def _build_count_artifact(
    config: dict[str, Any], X: pd.DataFrame, valid_cols: list[str]
) -> CountVectorizerArtifact:
    """Fit a ``CountVectorizer`` on the resolved text columns and build its artifact dict."""
    max_features: int | None = config.get("max_features") or None
    min_df: Any = config.get("min_df", 1)
    max_df: Any = config.get("max_df", 1.0)
    ngram_min, ngram_max = config.get("ngram_range", [1, 1])
    lowercase: bool = bool(config.get("lowercase", True))
    stop_words: str | None = config.get("stop_words") or None
    binary: bool = bool(config.get("binary", False))

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
        lowercase=lowercase,
        stop_words=stop_words,
        binary=binary,
    )

    text = _join_text_columns(X, valid_cols)
    vectorizer.fit(text)

    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
    prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
    output_columns = [f"{prefix}__count__{name}" for name in feature_names]

    warn = _warn_large_output(len(output_columns), threshold=_LARGE_VOCAB_THRESHOLD)
    if warn:
        logger.warning(warn)

    return {
        "type": "count_vectorizer",
        "columns": valid_cols,
        "output_columns": output_columns,
        "vocabulary": vectorizer.vocabulary_,
        "max_features": max_features,
        "lowercase": lowercase,
        "stop_words": stop_words,
        "binary": binary,
        "vectorizer_object": vectorizer,
        "drop_original": bool(config.get("drop_original", False)),
    }


class CountVectorizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: dict[str, Any]) -> None:
        # Vocabulary size is data-dependent — return None to signal unknown.
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> CountVectorizerArtifact:  # pylint: disable=arguments-differ
        resolved = resolve_fit_text_columns(X, config)
        if resolved is None:
            return {}
        X, valid_cols = resolved

        return _build_count_artifact(config, X, valid_cols)
