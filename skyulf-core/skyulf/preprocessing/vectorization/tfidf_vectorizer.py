"""TF-IDF Vectorizer node — converts text to TF-IDF weighted feature columns.

Wraps ``sklearn.feature_extraction.text.TfidfVectorizer``.  The fitted IDF
weights are stored in the artifact so they can be inspected or serialised.
Output is **always dense**.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import TfidfVectorizerArtifact
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ._common import (
    _join_text_columns,
    _sklearn_vectorizer_apply_pandas,
    _warn_large_output,
    apply_text_pandas_only,
    resolve_fit_text_columns,
)

logger = logging.getLogger(__name__)


# ── Apply ─────────────────────────────────────────────────────────────────────


def _tfidf_apply_pandas(
    X: pd.DataFrame, y: Any, params: dict[str, Any]
) -> tuple[pd.DataFrame, Any]:
    """Transform text columns using the fitted ``TfidfVectorizer``."""
    return _sklearn_vectorizer_apply_pandas(X, y, params)


class TfidfVectorizerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_text_pandas_only(X, params, _tfidf_apply_pandas)


# ── Calculate ─────────────────────────────────────────────────────────────────


@NodeRegistry.register("tfidf_vectorizer", TfidfVectorizerApplier)
@node_meta(
    id="tfidf_vectorizer",
    name="TF-IDF Vectorizer",
    category="Text",
    description=(
        "Convert text columns to TF-IDF weighted feature columns. "
        "Penalises very common tokens and rewards rare-but-informative ones."
    ),
    params={
        "columns": [],
        "max_features": None,
        "min_df": 1,
        "max_df": 1.0,
        "ngram_range": [1, 1],
        "sublinear_tf": False,
        "lowercase": True,
        "stop_words": None,
        "drop_original": False,
    },
    tags=["text", "nlp", "tfidf", "vectorizer"],
)
def _build_tfidf_artifact(
    config: dict[str, Any], X: pd.DataFrame, valid_cols: list[str]
) -> TfidfVectorizerArtifact:
    """Fit a ``TfidfVectorizer`` on the resolved text columns and build its artifact dict."""
    max_features: int | None = config.get("max_features") or None
    min_df: Any = config.get("min_df", 1)
    max_df: Any = config.get("max_df", 1.0)
    ngram_min, ngram_max = config.get("ngram_range", [1, 1])
    sublinear_tf: bool = bool(config.get("sublinear_tf", False))
    lowercase: bool = bool(config.get("lowercase", True))
    stop_words: str | None = config.get("stop_words") or None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
        sublinear_tf=sublinear_tf,
        lowercase=lowercase,
        stop_words=stop_words,
    )

    text = _join_text_columns(X, valid_cols)
    vectorizer.fit(text)

    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
    prefix = valid_cols[0] if len(valid_cols) == 1 else "_".join(valid_cols)
    output_columns = [f"{prefix}__tfidf__{name}" for name in feature_names]

    warn = _warn_large_output(len(output_columns))
    if warn:
        logger.warning(warn)

    return {
        "type": "tfidf_vectorizer",
        "columns": valid_cols,
        "output_columns": output_columns,
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "max_features": max_features,
        "lowercase": lowercase,
        "stop_words": stop_words,
        "vectorizer_object": vectorizer,
        "drop_original": bool(config.get("drop_original", False)),
    }


class TfidfVectorizerCalculator(BaseCalculator):
    def infer_output_schema(self, input_schema: Any, config: dict[str, Any]) -> None:
        return None

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> TfidfVectorizerArtifact:  # pylint: disable=arguments-differ
        resolved = resolve_fit_text_columns(X, config)
        if resolved is None:
            return {}
        X, valid_cols = resolved

        return _build_tfidf_artifact(config, X, valid_cols)
